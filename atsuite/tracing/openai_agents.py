"""Observe native Agents SDK runs without replacing tools, models or the agent loop."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from importlib.metadata import version
from threading import Lock
from typing import Any
from uuid import uuid4

from agents import RunConfig, RunHooks, Runner
from agents.tracing import (
    TracingProcessor,
    add_trace_processor,
    gen_trace_id,
    get_current_span,
    get_current_trace,
    trace,
)

from atsuite.tracing.context import active_adapter
from atsuite.tracing.core import TraceCollector, TracingAdapter, json_value


class _Processor(TracingProcessor):
    """A stateless SDK extension; route events only to the active request's adapter."""

    def on_trace_start(self, native) -> None:
        pass

    def on_trace_end(self, native) -> None:
        pass

    def on_span_start(self, native) -> None:
        adapter = active_adapter.get()
        if isinstance(adapter, OpenAIAgentsTracingAdapter) and native.trace_id == adapter.trace_id:
            adapter._native_start(native)

    def on_span_end(self, native) -> None:
        adapter = active_adapter.get()
        if isinstance(adapter, OpenAIAgentsTracingAdapter) and native.trace_id == adapter.trace_id:
            adapter._native_end(native)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


_processor = _Processor()
_processor_lock = Lock()
_registered = False


class OpenAIAgentsTracingAdapter(TracingAdapter, RunHooks):
    """Collect native containment and tool-call causality, preserving user hooks."""

    def __init__(self, collector: TraceCollector):
        super().__init__(collector)
        global _registered
        with _processor_lock:
            if not _registered:
                add_trace_processor(_processor)
                _registered = True
        self.running = False
        self.trace_id = None
        self.session_run = None
        self._models = {}
        self._native_models = set()
        self._calls = {}
        self._user_hooks = None

    def current_span_id(self) -> str | None:
        """The SDK propagates its native span into function tools and executor tasks."""
        native = get_current_span()
        return self._owner(native.span_id) if native else self.session_run

    def _native_start(self, native) -> None:
        data = native.span_data
        parent = native.parent_id or self.session_run
        self.parents[native.span_id] = parent
        if data.type in {"generation", "response"}:
            # Lifecycle hooks also cover custom models that emit no SDK model spans.
            model = self._models.get(parent)
            if model in self.active:
                self.parents[native.span_id] = model
            else:
                # Nested Runner calls need not inherit the outer run's lifecycle hooks.
                self._start_call(
                    native.span_id,
                    "llm",
                    getattr(data, "model", None) or "model",
                    parent,
                    None,
                )
                self._models[parent] = native.span_id
                self._native_models.add(native.span_id)
            return
        kind = "tool" if data.type == "function" else "workflow"
        name = getattr(data, "name", None) or data.type
        if data.type == "turn":
            name = f"turn:{data.turn}"
        self._start_call(native.span_id, kind, name, parent, None, {"sdk_span_type": data.type})
        if kind == "tool" and parent in self._models:
            self.collector.spans[native.span_id]["dependencies"] = [self._models[parent]]

    def _native_end(self, native) -> None:
        rid = self._owner(native.span_id)
        if rid is None:
            self.collector.diagnostics.append(f"Unmatched Agents SDK span: {native.span_id}")
            return
        data = native.span_data
        if data.type in {"generation", "response"}:
            if self.include_data:
                self.collector.spans[rid]["metadata"]["sdk_model_span"] = json_value(data.export())
            if native.span_id in self._native_models:
                self.collector.spans[rid]["metadata"]["usage"] = json_value(getattr(data, "usage", None))
                output = getattr(data, "output", None) or getattr(data, "response", None)
                self._end_call(rid, output if self.include_data else None, native.error)
            elif native.error:
                self._end_call(rid, error=native.error)
            return
        span = self.collector.spans[rid]
        if self.include_data:
            inputs = getattr(data, "input", None)
            if data.type == "function" and isinstance(inputs, str):
                try:
                    inputs = json.loads(inputs)
                except ValueError:
                    pass
            span["input"] = json_value(inputs)
            span["metadata"]["sdk_span"] = json_value(data.export())
        self._end_call(
            rid,
            getattr(data, "output", None) if self.include_data else None,
            native.error,
        )

    async def _forward(self, name, *args) -> None:
        if self._user_hooks is not None:
            await getattr(self._user_hooks, name)(*args)

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        parent = self.current_span_id()
        rid = str(uuid4())
        self._models[parent] = rid
        inputs = {"system": system_prompt, "items": input_items} if self.include_data else None
        self._start_call(
            rid,
            "llm",
            getattr(agent.model, "model", None) or agent.name,
            parent,
            inputs,
        )
        await self._forward("on_llm_start", context, agent, system_prompt, input_items)

    async def on_llm_end(self, context, agent, response) -> None:
        rid = self._models[self.current_span_id()]
        for item in response.output:
            call_id = getattr(item, "call_id", None)
            if call_id:
                self._calls[call_id] = rid
        self.collector.spans[rid]["metadata"].update(
            usage=json_value(response.usage),
            response_id=response.response_id,
            request_id=response.request_id,
        )
        self._end_call(rid, response.output if self.include_data else None)
        await self._forward("on_llm_end", context, agent, response)

    async def on_tool_start(self, context, agent, tool) -> None:
        rid = self.current_span_id()
        call_id = getattr(context, "tool_call_id", None)
        producer = self._calls.get(call_id)
        if producer is None:
            self.collector.diagnostics.append(f"Missing model origin for SDK tool call: {call_id}")
        else:
            # Siblings depend on their generating model, even if a fast sibling already finished.
            self.collector.spans[rid]["dependencies"] = [producer]
        self.collector.spans[rid]["metadata"]["tool_call_id"] = call_id
        await self._forward("on_tool_start", context, agent, tool)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        await self._forward("on_tool_end", context, agent, tool, result)

    async def on_agent_start(self, context, agent) -> None:
        await self._forward("on_agent_start", context, agent)

    async def on_agent_end(self, context, agent, output) -> None:
        await self._forward("on_agent_end", context, agent, output)

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        await self._forward("on_handoff", context, from_agent, to_agent)

    async def astream(self, agent: Any, inputs: Any, config=None, **kwargs):
        """Yield the native RunResult; Runner options and sessions pass through unchanged."""
        if self.running or self.collector.spans:
            raise ValueError("Use a new Agents SDK adapter/collector for each run")
        run_config = kwargs.pop("run_config", config)
        if isinstance(run_config, dict) or run_config is None:
            run_config = RunConfig(**(run_config or {}))
        if run_config.tracing_disabled:
            raise ValueError("Agents SDK tracing must be enabled for the Wabber adapter")
        self.include_data = run_config.trace_include_sensitive_data
        self._user_hooks = kwargs.pop("hooks", None)
        existing = get_current_trace()
        self.trace_id = existing.trace_id if existing else run_config.trace_id or gen_trace_id()
        self.session_run = str(uuid4())
        self.running = True
        token = active_adapter.set(self)
        self.collector.harness_state["in_progress"] = True
        self.collector.start(
            self.session_run,
            "session",
            "OpenAI Agents SDK",
            inputs=inputs if self.include_data else None,
            metadata={
                "harness": "openai_agents",
                "openai_agents": version("openai-agents"),
                "trace_id": self.trace_id,
                "checkpointing": kwargs.get("session") is not None,
                "sensitive_data": self.include_data,
            },
        )
        result = None
        error = None
        try:
            with (
                nullcontext(existing)
                if existing
                else trace(
                    run_config.workflow_name,
                    trace_id=self.trace_id,
                    group_id=run_config.group_id,
                    metadata=run_config.trace_metadata,
                    tracing=run_config.tracing,
                )
            ):
                if get_current_trace() is None or get_current_trace().trace_id != self.trace_id:
                    raise ValueError("Agents SDK tracing is globally disabled; enable it before running Wabber")
                result = await Runner.run(
                    agent,
                    inputs,
                    run_config=replace(run_config, trace_id=self.trace_id),
                    hooks=self,
                    **kwargs,
                )
        except BaseException as exc:
            error = exc
            raise
        finally:
            for span in reversed(list(self.collector.spans.values())):
                if span["end_ns"] is None and span["id"] != self.session_run:
                    self._end_call(span["id"], error=error or "SDK span did not finish")
            interrupted = bool(result and result.interruptions)
            self.collector.end(
                self.session_run,
                result.final_output if result and self.include_data else None,
                error,
                status="interrupted" if interrupted else None,
            )
            self.collector.harness_state["in_progress"] = False
            active_adapter.reset(token)
            self.running = False
            self.collector.save()
        yield result
