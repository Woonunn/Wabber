"""LangGraph 1.0.5 callbacks; routing comes from executed branches, never wall time."""

from __future__ import annotations

import platform
from collections import defaultdict
from collections.abc import AsyncIterator
from importlib.metadata import version
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.config import get_config
from langgraph.errors import GraphInterrupt, ParentCommand
from langgraph.types import Command, Send

from atsuite.tracing.context import active_adapter
from atsuite.tracing.core import TraceCollector, TracingAdapter, json_value


class LangGraphTracingAdapter(TracingAdapter, BaseCallbackHandler):
    """Trace a graph request, including its checkpoint continuations and nested graphs."""

    run_inline = True
    raise_error = True
    harness = "langgraph"

    def __init__(self, collector: TraceCollector):
        super().__init__(collector)
        self.names = {}
        self.tasks = {rid: span["metadata"] for rid, span in collector.spans.items() if span["kind"] == "workflow"}
        self.routes = defaultdict(
            list, {rid: [tuple(route) for route in meta.get("routes", [])] for rid, meta in self.tasks.items()}
        )
        self.graphs = {}
        self.subgraph_runs = {}
        self.branches = {}
        self.running = False
        self.session_run = None
        self.checkpoint_before = None

    def current_span_id(self) -> str | None:
        """Resolve graph nodes and native LangChain tools, including executor threads."""
        try:
            return self._owner(get_config().get("callbacks").parent_run_id)
        except (RuntimeError, AttributeError):
            return None

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        name=None,
        metadata=None,
        **kwargs,
    ):
        rid = str(run_id)
        self.parents[rid] = str(parent_run_id) if parent_run_id else None
        self.names[rid] = name
        meta = metadata or {}
        if parent_run_id is None:
            self.collector.start(
                rid,
                "session",
                name or "LangGraph",
                inputs=inputs,
                metadata={
                    "harness": self.harness,
                    "langgraph": version("langgraph"),
                    "langchain_core": version("langchain-core"),
                    "python": platform.python_version(),
                    "checkpoint_before": self.checkpoint_before,
                    "checkpointing": self.checkpoint_before is not None,
                },
            )
            self.session_run = rid
        elif name == meta.get("langgraph_node") and any(t.startswith("graph:step:") for t in kwargs.get("tags", [])):
            owner = self._owner(parent_run_id)
            meta = dict(meta)
            meta["graph_namespace"] = meta["langgraph_checkpoint_ns"].rpartition("|")[0]
            scope = self.subgraph_runs.get(str(parent_run_id))
            if scope:
                meta["scope_dependencies"] = scope["dependencies"][:]
                scope["children"].append(rid)
            self.collector.start(rid, "workflow", name, parent_id=owner, inputs=inputs, metadata=meta)
            self.tasks[rid] = self.collector.spans[rid]["metadata"]
        else:
            owner = self._owner(parent_run_id)
            namespace = meta.get("checkpoint_ns", "")
            graph = self.graphs.get(self._graph_path(namespace))
            if owner in self.tasks and namespace != self.tasks[owner]["graph_namespace"] and name is not None:
                expected_name = ((graph.config or {}).get("run_name") or graph.name) if graph is not None else None
                if graph is None or name != expected_name:
                    self.collector.diagnostics.append(f"Unresolved subgraph definition: {namespace} ({name})")
                else:
                    self.subgraph_runs[rid] = {
                        "owner": owner,
                        "namespace": namespace,
                        "dependencies": self.frontiers[owner][:],
                        "children": [],
                    }

    def _finish_subgraph(self, rid):
        scope = self.subgraph_runs.pop(rid, None)
        if scope:
            children = scope["children"] or [
                s
                for s, meta in self.tasks.items()
                if meta["graph_namespace"] == scope["namespace"] and self.collector.spans[s]["status"] == "ok"
            ]
            owner = scope["owner"]
            self.frontiers[owner] = list(dict.fromkeys([*self.frontiers[owner], *children]))

    @staticmethod
    def _graph_path(namespace):
        return "|".join(part.split(":", 1)[0] for part in namespace.split("|") if not part.isdigit())

    def _route(self, owner, value, mapping=None):
        values = value if isinstance(value, (list, tuple)) else [value]
        for target in values:
            if isinstance(target, Send):
                self.routes[owner].append((target.node, True))
            elif isinstance(target, str):
                self.routes[owner].append(((mapping or {}).get(target, target), False))
        self.tasks[owner]["routes"] = json_value(self.routes[owner])

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        rid = str(run_id)
        owner = self._owner(self.parents.get(rid))
        if owner in self.tasks:
            namespace = self._graph_path(self.tasks[owner]["graph_namespace"])
            branches = self.branches.get(namespace, {}).get(self.tasks[owner]["langgraph_node"], {})
            name = self.names.get(rid)
            if name in branches:
                self._route(owner, outputs, branches[name])
        if isinstance(outputs, Command) and rid in self.tasks:
            self._route(rid, outputs.goto)
        error = None
        if rid in self.collector.spans and self.collector.spans[rid]["kind"] == "session":
            if isinstance(outputs, dict) and outputs.get("status") in {
                "failed",
                "error",
                "timeout",
            }:
                error = outputs.get("error") or outputs["status"]
        self.collector.end(rid, outputs, error)
        self._finish_subgraph(rid)

    def on_chain_error(self, error, *, run_id, **kwargs):
        rid = str(run_id)
        if isinstance(error, ParentCommand):
            command = error.args[0]
            if rid in self.tasks and command.graph == self.tasks[rid]["langgraph_checkpoint_ns"]:
                self._route(rid, command.goto)
            self.collector.end(rid, command)
        elif isinstance(error, GraphInterrupt):
            self.collector.end(rid, json_value(error.args[0]), status="interrupted")
        else:
            self.collector.end(rid, error=error)
        self._finish_subgraph(rid)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self.on_llm_start(serialized, messages, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self._start_call(
            run_id,
            "llm",
            kwargs.get("name") or (serialized or {}).get("name", "llm"),
            parent_run_id,
            prompts,
            kwargs.get("metadata"),
        )

    def on_llm_end(self, response, *, run_id, **kwargs):
        self._end_call(run_id, response)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end_call(run_id, error=error)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self._start_call(
            run_id,
            "tool",
            (serialized or {}).get("name", "tool"),
            parent_run_id,
            kwargs.get("inputs", input_str),
            kwargs.get("metadata"),
        )

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end_call(run_id, output)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end_call(run_id, error=error)

    def record_event(self, event: dict) -> None:
        """Bridge existing direct SDK calls that do not emit native callbacks."""
        try:
            parent = get_config().get("callbacks").parent_run_id
        except (RuntimeError, AttributeError):
            return
        kind = event["kind"]
        key = (
            self._owner(parent),
            "tool" if kind.startswith("tool") else "llm",
            event.get("key", event.get("phase")),
        )
        if kind in {"tool_start", "llm_start"}:
            rid = str(uuid4())
            self._start_call(
                rid,
                key[1],
                event.get("tool_name", event.get("phase")),
                parent,
                event.get("args", event.get("request")),
                {k: event[k] for k in ("call_id", "runtime_uid", "target_id", "provider_session_id") if k in event},
            )
            self.active[key] = rid
        elif kind in {"tool", "llm", "tool_error", "llm_error"}:
            if kind == "llm_error":
                rid = self.active.get(key)
                if rid is not None:
                    self.collector.spans[rid]["metadata"].setdefault("attempt_errors", []).append(
                        {
                            k: event[k]
                            for k in (
                                "attempt",
                                "status_code",
                                "error_type",
                                "error",
                                "duration_s",
                                "retry_delay_s",
                                "usage",
                            )
                            if k in event
                        }
                    )
                    self.collector.spans[rid]["metadata"]["attempts"] = event["attempt"]
                if event.get("retry_delay_s") is not None:
                    return
            rid = self.active.pop(key, None)
            if rid is None:
                self.collector.diagnostics.append(f"Unmatched direct SDK event: {kind}")
                return
            output = event.get("result", event.get("response"))
            error = event.get("error") if kind.endswith("error") else None
            if isinstance(output, dict) and output.get("status") in {
                "error",
                "timeout",
            }:
                error = output.get("error", output["status"])
            transport = event.get("transport", {})
            if transport.get("status", "ok") != "ok":
                error = transport.get("error") or transport["status"]
            self._end_call(rid, output, error)
            span = self.collector.spans[rid]
            span["metadata"].update(
                {
                    k: event[k]
                    for k in (
                        "usage",
                        "source",
                        "attempts",
                        "iteration",
                        "requested_model",
                        "actual_model",
                    )
                    if k in event
                }
            )
            if kind == "llm":
                span["duration_s"] = event["duration_s"]
                span["start_ns"] = span["end_ns"] - int(event["duration_s"] * 1e9)
            if "transport" in event:
                self.collector.observe_runtime(rid, event["transport"])

    def _callback_free_dependencies(self, rid, previous):
        return None

    def _resolve_dependencies(self):
        namespaces = defaultdict(list)
        for rid, meta in self.tasks.items():
            namespaces[meta["graph_namespace"]].append(rid)
        for namespace, tasks in namespaces.items():
            graph = self.graphs.get(self._graph_path(namespace))
            if graph is None:
                self.collector.diagnostics.append(f"Unresolved subgraph: {namespace}")
                continue
            by_step = defaultdict(list)
            for rid in tasks:
                by_step[self.tasks[rid]["langgraph_step"]].append(rid)
            joins = {edge: {} for edge in graph.builder.waiting_edges}
            attempts = {}
            for step, current in sorted(by_step.items()):
                previous = {}
                for source in by_step.get(step - 1, []):
                    previous[self.tasks[source]["langgraph_checkpoint_ns"]] = source
                previous = [rid for rid in previous.values() if self.collector.spans[rid]["status"] == "ok"]
                for (starts, target), pending in joins.items():
                    for source_name in starts:
                        writers = [s for s in previous if self.tasks[s]["langgraph_node"] == source_name]
                        if writers:
                            pending[source_name] = writers
                sends = []
                for source in sorted(previous, key=lambda s: tuple(self.tasks[s]["langgraph_path"][:2])):
                    sends.extend((source, target) for target, push in self.routes[source] if push)
                for rid in current:
                    meta = self.tasks[rid]
                    name = meta["langgraph_node"]
                    deps = []
                    path = meta["langgraph_path"]
                    native = self._callback_free_dependencies(rid, previous)
                    if native is not None:
                        deps.extend(native)
                    elif path[0] == "__pregel_push":
                        index = path[1]
                        if index < len(sends) and sends[index][1] == name:
                            deps.append(sends[index][0])
                        else:
                            self.collector.diagnostics.append(f"Unresolved Send parent: {rid}")
                    else:
                        for source in previous:
                            source_name = self.tasks[source]["langgraph_node"]
                            if (source_name, name) in graph.builder.edges or (name, False) in self.routes[source]:
                                deps.append(source)
                        for (starts, target), pending in joins.items():
                            if target == name and all(source_name in pending for source_name in starts):
                                deps.extend(s for writers in pending.values() for s in writers)
                                pending.clear()
                    deps.extend(meta.get("scope_dependencies", []))
                    task_id = meta["langgraph_checkpoint_ns"]
                    if task_id in attempts:
                        meta["retry_of"] = attempts[task_id]
                        deps.append(attempts[task_id])
                    attempts[task_id] = rid
                    self.collector.spans[rid]["dependencies"] = list(dict.fromkeys(deps))
                    if not deps and not (step == 1 and ("__start__", name) in graph.builder.edges):
                        self.collector.diagnostics.append(f"Unresolved workflow dependencies: {rid}")

    @staticmethod
    async def _checkpoint(snapshot, checkpointer):
        checkpoints = {}
        tasks = {}
        configs = {}

        def visit(state):
            config = state.config.get("configurable", {})
            namespace = config.get("checkpoint_ns", "")
            if config.get("checkpoint_id"):
                checkpoints[namespace] = config["checkpoint_id"]
                configs[namespace] = state.config
            for task in state.tasks:
                task_ns = "|".join(filter(None, (namespace, f"{task.name}:{task.id}")))
                tasks[task_ns] = {
                    "error": task.error,
                    "completed": task.result is not None,
                    "interrupts": json_value(task.interrupts),
                }
                if hasattr(task.state, "tasks"):
                    visit(task.state)

        visit(snapshot)
        controls = {}
        for namespace, config in configs.items():
            saved = await checkpointer.aget_tuple(config)
            controls[namespace] = (
                [
                    json_value(write)
                    for write in (saved.pending_writes or [])
                    if write[1] in {"__resume__", "__interrupt__", "__error__"}
                ]
                if saved
                else []
            )
            controls[namespace].sort(key=lambda write: (write[0], write[1]))
        return {
            "thread_id": str(snapshot.config.get("configurable", {}).get("thread_id", "")),
            "checkpoints": checkpoints,
            "tasks": tasks,
            "control_writes": controls,
        }

    async def astream(self, graph: Any, inputs: Any, config: dict | None = None, **kwargs) -> AsyncIterator[Any]:
        """Yield normal values; resume the same trace with None or Command and the same checkpoint thread."""
        if self.running:
            raise ValueError("A TracingAdapter cannot trace concurrent invocations")
        self.running = True
        try:
            config = dict(config or {})
            self.graphs = {"": graph, **dict(graph.get_subgraphs(recurse=True))}
            self.branches = {
                path: {
                    node: {branch.path.name: branch.ends for branch in branches.values()}
                    for node, branches in item.builder.branches.items()
                }
                for path, item in self.graphs.items()
            }
            self.checkpoint_before = None
            if graph.checkpointer:
                snapshot = await graph.aget_state(config, subgraphs=True)
                self.checkpoint_before = await self._checkpoint(snapshot, graph.checkpointer)
            if self.collector.spans:
                if self.checkpoint_before is None or (inputs is not None and not isinstance(inputs, Command)):
                    raise ValueError(
                        "Continue a trace only with a checkpoint and None/Command; use a new collector for new input"
                    )
                if self.collector.harness_state.get("langgraph") != self.checkpoint_before:
                    raise ValueError(
                        "Trace does not match the checkpoint/thread; load its matching saved TraceCollector"
                    )
            elif self.checkpoint_before and self.checkpoint_before["checkpoints"]:
                self.collector.diagnostics.append("Checkpoint history was not traced; workload totals are incomplete")
            callbacks = config.get("callbacks")
            if callbacks is None or isinstance(callbacks, list):
                config["callbacks"] = [*(callbacks or []), self]
            else:
                config["callbacks"] = callbacks.copy()
                config["callbacks"].add_handler(self, inherit=True)
            self.collector.harness_state["in_progress"] = True
            self.collector.save()
        except BaseException:
            self.running = False
            raise
        self.session_run = None
        interrupted = False
        token = active_adapter.set(self)
        try:
            async for state in graph.astream(inputs, config=config, stream_mode="values", **kwargs):
                interrupted = isinstance(state, dict) and "__interrupt__" in state
                yield state
        finally:
            try:
                try:
                    if graph.checkpointer:
                        snapshot = await graph.aget_state(config, subgraphs=True)
                        self.collector.harness_state["langgraph"] = await self._checkpoint(snapshot, graph.checkpointer)
                        interrupted = interrupted or bool(snapshot.next) or bool(snapshot.interrupts)
                        if self.session_run:
                            self.collector.spans[self.session_run]["metadata"]["checkpoint_after"] = (
                                self.collector.harness_state["langgraph"]
                            )
                except Exception as exc:
                    self.collector.diagnostics.append(f"Checkpoint observation failed: {type(exc).__name__}: {exc}")
                if interrupted and self.session_run and self.collector.spans[self.session_run]["status"] == "ok":
                    self.collector.spans[self.session_run]["status"] = "interrupted"
                self._resolve_dependencies()
                self.collector.harness_state["in_progress"] = False
                self.collector.save()
            finally:
                self.running = False
                active_adapter.reset(token)
