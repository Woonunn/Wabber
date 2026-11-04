"""Harness-neutral, per-request traces and ATSuite DAG export."""

from __future__ import annotations

import json
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from itertools import accumulate
from pathlib import Path
from typing import Any
from uuid import uuid4

from atsuite.tracing.resources import resource_end, resource_start
from atsuite.tracing.storage import read_text, write_json


def json_value(value: Any) -> Any:
    """Keep JSON results, including LangChain messages, without pickling objects."""
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class TraceCollector:
    """One request/session; callbacks may arrive from multiple worker threads."""

    def __init__(self, request_id: str, session_id: str, directory: Path | None = None):
        self.request_id = request_id
        self.session_id = session_id
        self.directory = Path(directory) if directory is not None else None
        self.spans: dict[str, dict] = {}
        self.diagnostics: list[str] = []
        self.harness_state: dict = {}
        self._resource_starts: dict[str, float] = {}
        self.lock = threading.RLock()

    @classmethod
    def load(cls, directory: Path) -> TraceCollector:
        """Continue a saved trace using the caller's separately persisted Harness checkpoint."""
        directory = Path(directory)
        payload = json.loads(read_text(directory / "trace.json"))
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported trace schema")
        collector = cls(payload["request_id"], payload["session_id"], directory)
        collector.spans = {span["id"]: span for span in payload["spans"]}
        if len(collector.spans) != len(payload["spans"]):
            raise ValueError("Duplicate trace span IDs")
        collector.diagnostics = payload["diagnostics"]
        collector.harness_state = payload.get("harness_state", {})
        if collector.harness_state.get("in_progress"):
            collector.diagnostics.append(
                "Previous execution did not finish saving; invocation history may be incomplete"
            )
        return collector

    def start(
        self,
        span_id: str,
        kind: str,
        name: str,
        *,
        parent_id: str | None = None,
        dependencies: list[str] | None = None,
        inputs: Any = None,
        metadata: dict | None = None,
    ) -> None:
        """Begin a session, workflow node, model call or tool call."""
        with self.lock:
            if kind == "tool":
                self._resource_starts[span_id] = resource_start()
            self.spans[span_id] = {
                "id": span_id,
                "kind": kind,
                "name": name,
                "parent_id": parent_id,
                "dependencies": list(dependencies or []),
                "start_ns": time.perf_counter_ns(),
                "end_ns": None,
                "duration_s": None,
                "status": "running",
                "input": json_value(inputs),
                "output": None,
                "error": None,
                "metadata": json_value(metadata or {}),
                "state_read_bytes": None,
                "state_write_bytes": None,
                "resources": None,
            }

    def end(
        self,
        span_id: str,
        output: Any = None,
        error: BaseException | str | None = None,
        *,
        status: str | None = None,
    ) -> None:
        """Finish a span, retaining errors and measured elapsed time."""
        with self.lock:
            span = self.spans.get(span_id)
            if span is None or span["end_ns"] is not None:
                return
            span.update(
                end_ns=time.perf_counter_ns(),
                output=json_value(output),
                error=str(error) if error is not None else None,
                status=status or ("error" if error is not None else "ok"),
            )
            span["duration_s"] = (span["end_ns"] - span["start_ns"]) / 1e9
            started = self._resource_starts.pop(span_id, None)
            if started is not None:
                observation = resource_end(started)
                span["metadata"]["harness_resource_usage"] = observation
                if span["resources"] is None:
                    span["resources"] = observation

    def observe_runtime(self, span_id: str, transport: dict) -> None:
        """Attach actual Runtime measurements; absent counters stay unknown."""
        with self.lock:
            span = self.spans[span_id]
            span["metadata"]["runtime"] = json_value(transport)
            runtime_cpu = transport.get("provider_metadata", {}).get("resource_usage")
            if isinstance(runtime_cpu, dict):
                span["metadata"]["runtime_resource_usage"] = json_value(runtime_cpu)
                span["resources"] = json_value(runtime_cpu)
            sync = transport.get("provider_metadata", {}).get("state_sync", {})
            read = [sync.get(k) for k in ("load_bytes", "branch_read_bytes")]
            write = [sync.get(k) for k in ("save_bytes", "branch_write_bytes")]
            for key, values in (
                ("state_read_bytes", read),
                ("state_write_bytes", write),
            ):
                if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in values):
                    span[key] = sum(values)
            result = transport.get("result")
            if isinstance(result, dict):
                logical = result.get("logical_state")
                if isinstance(logical, dict):
                    span["metadata"].pop("state_accesses", None)
                    span["metadata"]["logical_state"] = json_value(logical)
                    if logical.get("contract") == "datasci-tool-snapshot-v1" and logical.get("complete") is True:
                        try:
                            self.observe_state_accesses(span_id, logical.get("accesses"))
                        except ValueError:
                            span["metadata"]["logical_state"]["complete"] = False
                            span["metadata"]["logical_state"]["errors"] = ["Invalid logical state accesses"]
                worker = result.get("worker_usage") or result.get("kernel_execution", {}).get("worker_usage")
                if worker:
                    # Worker and dispatch-thread counters stay separate, never inclusive sums.
                    span["resources"] = json_value(worker)
                span["metadata"]["state_lifecycle"] = result.get("state_lifecycle") or result.get(
                    "kernel_execution", {}
                ).get("state_lifecycle")

    def observe_state_accesses(self, span_id: str, accesses: list[dict]) -> None:
        """Record exhaustive logical externalization needs, independent of transport.

        Repeated accesses must appear repeatedly. Use [] only for proven stateless
        calls; omit the observation when object sizes or access coverage are unknown.
        """
        if not isinstance(accesses, list):
            raise ValueError("Logical state accesses must be a list")
        for access in accesses:
            if (
                not isinstance(access, dict)
                or not isinstance(access.get("object_id"), str)
                or not access["object_id"]
                or access.get("operation") not in ("read", "write")
                or not isinstance(access.get("bytes"), int)
                or isinstance(access["bytes"], bool)
                or access["bytes"] < 0
            ):
                raise ValueError("Invalid logical state access")
        with self.lock:
            self.spans[span_id]["metadata"]["state_accesses"] = json_value(accesses)

    def snapshot(self) -> dict:
        """Return a standalone JSON trace with workload features and optional rho."""
        with self.lock:
            spans = json_value(list(self.spans.values()))
            diagnostics = self.diagnostics[:]
        sessions = [s for s in spans if s["kind"] == "session"]
        successful = bool(sessions) and sessions[-1]["status"] == "ok" and not self.harness_state.get("in_progress")
        tools = [s for s in spans if s["kind"] == "tool"]
        llms = [s for s in spans if s["kind"] == "llm"]
        complete = bool(spans) and all(s["end_ns"] is not None for s in spans) and not diagnostics
        complete = complete and (not sessions or sessions[-1]["status"] != "interrupted")
        complete = complete and not self.harness_state.get("in_progress")
        state_spans = tools + [s for s in sessions if s["metadata"].get("checkpointing")]
        state_complete = complete and all(
            s["state_read_bytes"] is not None and s["state_write_bytes"] is not None for s in state_spans
        )
        observed = sum((s["state_read_bytes"] or 0) + (s["state_write_bytes"] or 0) for s in state_spans)
        idle = sum(s["duration_s"] or 0 for s in llms)
        total = observed if state_complete else None
        workflow = [s for s in spans if s["kind"] == "workflow"]
        containers = {s["parent_id"] for s in workflow}
        boundaries = defaultdict(list)
        for span in workflow:
            if span["id"] in containers:
                continue
            root = span
            while root["parent_id"] in self.spans:
                root = self.spans[root["parent_id"]]
            boundaries[root["id"]].append((span["start_ns"], 1))
            if span["end_ns"] is not None:
                boundaries[root["id"]].append((span["end_ns"], -1))
        parallel_peak = max(
            (max(accumulate(delta for _, delta in sorted(events)), default=0) for events in boundaries.values()),
            default=0,
        )
        payload = {
            "schema_version": 1,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "spans": spans,
            "diagnostics": diagnostics,
            "harness_state": json_value(self.harness_state),
            "summary": {
                "complete": complete,
                "successful": successful,
                "llm_calls": len(llms),
                "llm_attempts": sum(s["metadata"].get("attempts", 1) for s in llms),
                "tool_calls": len(tools),
                "workflow_nodes": len(workflow),
                "workflow_dependency_edges": sum(len(s["dependencies"]) for s in workflow),
                "parallel_workflow_peak": parallel_peak,
                "workflow_duration_s": sum(s["duration_s"] or 0 for s in sessions),
                "execution_segments": len(sessions),
                "llm_idle_s": idle,
                "tool_elapsed_s": sum(s["duration_s"] or 0 for s in tools),
                "observed_state_transfer_bytes": observed,
                "state_measurements_complete": state_complete,
                "state_transfer_bytes": total,
                "rho_kb_s": None,
            },
        }

        # Transport counters remain available, but cannot define workload coupling.
        from atsuite.workload.profile import extract_profile

        features = extract_profile(payload)
        payload["summary"]["workload_state_bytes"] = features["metrics"]["state_bytes"]
        payload["summary"]["rho_kb_s"] = features["metrics"]["rho_kb_s"] if successful else None
        return payload

    def save(self) -> None:
        """Atomically save the canonical compressed trace; derived formats are exported on demand."""
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "trace.json", self.snapshot())


class TracingAdapter(ABC):
    """A Harness supplies causal spans to a request-local collector."""

    def __init__(self, collector: TraceCollector):
        self.collector = collector
        self.parents = {}
        self.active = {}
        self.frontiers = defaultdict(list)
        self.include_data = True

    def _owner(self, run_id):
        run_id = str(run_id) if run_id is not None else None
        while run_id is not None and run_id not in self.collector.spans:
            run_id = self.parents.get(run_id)
        return run_id

    def _start_call(self, run_id, kind, name, parent, inputs, metadata=None):
        with self.collector.lock:
            owner = self._owner(parent)
            self.collector.start(
                str(run_id),
                kind,
                name,
                parent_id=owner,
                dependencies=self.frontiers[owner][:],
                inputs=inputs,
                metadata=metadata,
            )
            self.active[str(run_id)] = owner

    def _end_call(self, run_id, output=None, error=None):
        with self.collector.lock:
            run_id = str(run_id)
            transport = self.collector.spans[run_id]["metadata"].get("runtime", {})
            if error is None and transport.get("status", "ok") != "ok":
                error = transport.get("error") or transport["status"]
            if self.include_data and "result" in transport:
                output = transport["result"]
            self.collector.end(run_id, output, error)
            if run_id in self.active:
                owner = self.active.pop(run_id)
                deps = self.collector.spans[run_id]["dependencies"]
                self.frontiers[owner] = [s for s in self.frontiers[owner] if s not in deps] + [run_id]

    @abstractmethod
    def current_span_id(self) -> str | None:
        """Resolve the framework's current native invocation in this execution context."""

    @contextmanager
    def runtime_call(self, request):
        """Capture Runtime transport and worker measurements, reusing native tool spans."""
        with self.collector.lock:
            span_id, created = self._runtime_span(request)
        span = self.collector.spans[span_id]
        try:
            yield span_id
        except BaseException as exc:
            if created:
                self._end_call(span_id, error=exc)
            raise
        else:
            if created:
                transport = span["metadata"].get("runtime", {})
                self._end_call(
                    span_id,
                    transport.get("result") if self.include_data else None,
                    transport.get("error") or None,
                )

    def _runtime_span(self, request):
        span_id = next(
            (
                s["id"]
                for s in self.collector.spans.values()
                if s["metadata"].get("call_id") == request.call_id and request.call_id
            ),
            None,
        )
        parent = self.current_span_id()
        if span_id is None and parent in self.collector.spans:
            current = self.collector.spans[parent]
            if (
                current["kind"] == "tool"
                and current["name"] == request.tool_name
                and "runtime" not in current["metadata"]
            ):
                span_id = parent
        created = span_id is None
        if created:
            span_id = request.call_id or str(uuid4())
            self._start_call(
                span_id,
                "tool",
                request.tool_name,
                parent,
                request.args if self.include_data else None,
            )
        span = self.collector.spans[span_id]
        span["metadata"].update(
            call_id=request.call_id,
            runtime_uid=request.uid,
            target_id=request.target_id,
            provider_session_id=(request.session.provider_session_id if request.session else ""),
        )
        return span_id, created

    async def ainvoke(self, agent: Any, inputs: Any, config: dict | None = None, **kwargs) -> Any:
        """Run a native agent and return its final value while recording the same trace."""
        result = None
        async for result in self.astream(agent, inputs, config, **kwargs):
            pass
        return result

    @abstractmethod
    async def astream(self, graph: Any, inputs: Any, config: dict | None = None, **kwargs) -> AsyncIterator[Any]:
        """Yield normal graph states while collecting the execution."""
        yield  # pragma: no cover


def export_atg(trace: dict, path: Path) -> None:
    """Project containment and causal edges into a DAG, retaining fork/join nodes."""
    spans = {s["id"]: s for s in trace["spans"] if s["kind"] != "session"}
    children: dict[str, list[str]] = {}
    for s in spans.values():
        children.setdefault(s["parent_id"], []).append(s["id"])
    nodes = [
        {
            "id": 0,
            "name": "start",
            "type": "start",
            "edge_to": [],
            "time": 0,
            "output": None,
        }
    ]
    entries = {}
    exits = {}
    for s in spans.values():
        entry = len(nodes)
        entries[s["id"]] = entry
        nodes.append(
            {
                "id": entry,
                "name": s["name"],
                "type": "llm" if s["kind"] == "llm" else "tool_use" if s["kind"] == "tool" else "start",
                "edge_to": [],
                "time": (s["duration_s"] or 0) * 1000 if s["kind"] in {"llm", "tool"} else 0,
                "output": s["output"],
                "span_id": s["id"],
                "status": s["status"],
            }
        )
        exits[s["id"]] = len(nodes) if children.get(s["id"]) else entry
        if children.get(s["id"]):
            nodes.append(
                {
                    "id": len(nodes),
                    "name": s["name"] + ":join",
                    "type": "start",
                    "edge_to": [],
                    "time": 0,
                    "output": None,
                }
            )

    def edge(source, target, inputs=None):
        if not any(e["id"] == target for e in nodes[source]["edge_to"]):
            nodes[source]["edge_to"].append(
                {
                    "id": target,
                    "params": {"input": inputs} if inputs is not None else {},
                    "interval": 0,
                }
            )

    for s in spans.values():
        deps = [d for d in s["dependencies"] if d in spans]
        if s["parent_id"] in spans:
            edge(entries[s["parent_id"]], entries[s["id"]])
        if deps:
            for dep in deps:
                edge(exits[dep], entries[s["id"]], s["input"])
        else:
            edge(entries.get(s["parent_id"], 0), entries[s["id"]], s["input"])
        if children.get(s["id"]):
            nested = children[s["id"]]
            consumed = {d for child in nested for d in spans[child]["dependencies"]}
            for child in nested:
                if child not in consumed:
                    edge(exits[child], exits[s["id"]])
    end = len(nodes)
    for node in nodes:
        if not node["edge_to"]:
            edge(node["id"], end)
    nodes.append(
        {
            "id": end,
            "name": "end",
            "type": "end",
            "edge_to": [],
            "time": 0,
            "output": None,
        }
    )
    Path(path).write_text(
        json.dumps(
            {
                "name": "Wabber harness trace",
                "discription": "Observed causal DAG",
                "deploy_config": "",
                "nodes": nodes,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
