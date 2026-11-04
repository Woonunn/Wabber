"""Logical resource demand and source-scoped usage from canonical causal traces."""

import copy
import math
from collections import Counter

METRICS = (
    "llm_idle_s",
    "tool_elapsed_s",
    "tool_active_s",
    "execution_active_s",
    "runtime_tool_s",
    "tool_cpu_s",
    "tool_cpu_measured_s",
    "tool_cpu_coverage",
    "state_observation_s",
    "agent_elapsed_s",
    "state_bytes",
    "state_reads",
    "state_writes",
    "state_max_object_bytes",
    "state_transfer_bytes",
    "rho_kb_s",
)


def number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def extract_profile(trace: dict) -> dict:
    """Count each logical state access, never substitute transport counters.

    metadata.state_accesses is an exhaustive list of {object_id, operation,
    bytes} for a tool/checkpoint span. An empty list explicitly means stateless.
    Missing lists or byte sizes make S_G unknown, including for resident MCP.
    D_G is the sum of measured LLM call durations (seconds), including retries.
    """
    if trace.get("schema_version") != 1:
        raise ValueError("Unsupported trace schema")
    spans = trace["spans"]
    by_id = {s["id"]: s for s in spans}
    if len(by_id) != len(spans):
        raise ValueError("Duplicate span IDs")
    sessions = [s for s in spans if s["kind"] == "session"]
    tools = [s for s in spans if s["kind"] == "tool"]
    llms = [s for s in spans if s["kind"] == "llm"]
    workflow = [s for s in spans if s["kind"] == "workflow"]
    complete = (
        bool(sessions)
        and not trace.get("diagnostics")
        and not trace.get("harness_state", {}).get("in_progress")
    )
    complete = complete and all(
        s.get("end_ns") is not None and s.get("status") in ("ok", "error")
        for s in spans
    )
    complete = complete and all(
        d in by_id for s in spans for d in s.get("dependencies", [])
    )
    status = (
        "incomplete"
        if not complete
        else "success"
        if sessions[-1]["status"] == "ok"
        else "failed"
    )
    if complete and trace.get("request_status") == "failed":
        status = "failed"

    def total(items, field):
        values = [s.get(field) for s in items]
        return sum(values) if complete and all(number(v) for v in values) else None

    metrics = {k: None for k in METRICS}
    metrics.update(
        llm_idle_s=total(llms, "duration_s"),
        tool_elapsed_s=total(tools, "duration_s"),
        agent_elapsed_s=total(sessions, "duration_s"),
    )

    def active_time(items):
        if not complete or any(
            not number(s.get("start_ns"))
            or not number(s.get("end_ns"))
            or s["end_ns"] < s["start_ns"]
            for s in items
        ):
            return None
        total, end = 0, 0
        for start, stop in sorted((s["start_ns"], s["end_ns"]) for s in items):
            total += max(0, stop - max(start, end))
            end = max(end, stop)
        return total / 1e9

    metrics["tool_active_s"] = active_time(tools)
    metrics["execution_active_s"] = active_time(tools + llms)
    runtime = [
        s.get("metadata", {}).get("runtime", {}).get("execution_elapsed_ms")
        for s in tools
    ]
    if complete and runtime and all(number(v) for v in runtime):
        metrics["runtime_tool_s"] = sum(runtime) / 1000
    overhead = [
        s.get("metadata", {}).get("logical_state", {}).get("observation_s")
        for s in tools
    ]
    if complete and all(number(v) for v in overhead):
        metrics["state_observation_s"] = sum(overhead)
    cpu = [(s.get("resources") or {}).get("cpu_seconds") for s in tools]
    measured_cpu = [v for v in cpu if number(v)]
    metrics["tool_cpu_measured_s"] = sum(measured_cpu) if measured_cpu else None
    metrics["tool_cpu_coverage"] = len(measured_cpu) / len(tools) if tools else None
    cpu_complete = complete and bool(tools) and len(measured_cpu) == len(tools) and all(
        (s.get("resources") or {}).get("cpu_complete") is True
        and (s.get("resources") or {}).get("cpu_scope") == "tool_invocation"
        for s in tools
    )
    if cpu_complete:
        metrics["tool_cpu_s"] = sum(measured_cpu)
    state_spans = tools + [
        s for s in sessions if s.get("metadata", {}).get("checkpointing")
    ]
    accesses = []
    state_complete = complete
    for span in state_spans:
        observed = span.get("metadata", {}).get("state_accesses")
        if not isinstance(observed, list):
            state_complete = False
            continue
        for access in observed:
            if (
                not isinstance(access, dict)
                or not access.get("object_id")
                or access.get("operation") not in ("read", "write")
                or not number(access.get("bytes"))
            ):
                state_complete = False
            else:
                accesses.append(access)
    if state_complete:
        metrics.update(
            state_bytes=sum(a["bytes"] for a in accesses),
            state_reads=sum(a["operation"] == "read" for a in accesses),
            state_writes=sum(a["operation"] == "write" for a in accesses),
            state_max_object_bytes=max((a["bytes"] for a in accesses), default=0),
        )
        if metrics["llm_idle_s"] is not None and metrics["llm_idle_s"] > 0:
            metrics["rho_kb_s"] = metrics["state_bytes"] / 1000 / metrics["llm_idle_s"]
    transports = [
        s.get("metadata", {})
        .get("runtime", {})
        .get("provider_metadata", {})
        .get("state_sync")
        for s in tools
    ]
    if (
        complete
        and transports
        and all(
            isinstance(t, dict)
            and all(
                number(t.get(k))
                for k in (
                    "load_bytes",
                    "save_bytes",
                    "branch_read_bytes",
                    "branch_write_bytes",
                )
            )
            for t in transports
        )
    ):
        metrics["state_transfer_bytes"] = sum(
            t[k]
            for t in transports
            for k in (
                "load_bytes",
                "save_bytes",
                "branch_read_bytes",
                "branch_write_bytes",
            )
        )
    nodes = workflow or tools + llms
    node_ids = {s["id"] for s in nodes}
    edges = [
        [d, s["id"]] for s in nodes for d in s.get("dependencies", []) if d in node_ids
    ]
    # A repeated node in a parallel wave is not a loop; count only causal revisits.
    ancestors = {}
    pending = list(nodes)
    while pending:
        ready = [
            s
            for s in pending
            if all(d in ancestors for d in s.get("dependencies", []) if d in node_ids)
        ]
        if not ready:
            raise ValueError("Cyclic workflow dependencies")
        for s in ready:
            parents = [d for d in s.get("dependencies", []) if d in node_ids]
            ancestors[s["id"]] = set(parents).union(*(ancestors[d] for d in parents))
            pending.remove(s)
    successors = {s["id"]: [] for s in nodes}
    for source, target in edges:
        successors[source].append(by_id[target]["name"])
    revisits = Counter(
        s["name"]
        for s in nodes
        if any(by_id[d]["name"] == s["name"] for d in ancestors[s["id"]])
    )
    containers = {s.get("parent_id") for s in nodes}
    boundaries = [
        (s[key], delta)
        for s in nodes
        if s["id"] not in containers
        for key, delta in (("start_ns", 1), ("end_ns", -1))
        if s.get(key) is not None and s.get("end_ns", 0) != s.get("start_ns")
    ]
    active = peak = 0
    for _, delta in sorted(boundaries):
        active += delta
        peak = max(peak, active)
    missing = {
        k: "measurement missing or invalid" for k, v in metrics.items() if v is None
    }
    for key in missing:
        if key.startswith("state_"):
            missing[key] = "logical state accesses missing or incomplete trace"
    if metrics["state_transfer_bytes"] is None:
        missing["state_transfer_bytes"] = "physical transport counters missing or incomplete trace"
    if metrics["rho_kb_s"] is None:
        missing["rho_kb_s"] = (
            "requires complete logical state accesses and positive measured LLM idle time"
        )
    return {
        "request_id": trace["request_id"],
        "session_id": trace["session_id"],
        "status": status,
        "arrival_ns": trace.get("arrival_ns"),
        "completed_ns": trace.get("completed_ns"),
        "source": trace.get("source", "trace"),
        "cpu_observation": {
            "measured_calls": len(measured_cpu), "total_calls": len(tools),
            "complete": cpu_complete,
            "scopes": dict(Counter((s.get("resources") or {}).get("cpu_scope", "legacy_unspecified")
                                   for s, value in zip(tools, cpu) if number(value))),
            "missing_calls": [s["id"] for s, value in zip(tools, cpu) if not number(value)],
            "basis": "One non-overlapping counter per call; subtotal is not provider billable CPU",
        },
        "descriptor": copy.deepcopy(trace.get("request_context")),
        "observations": {
            "tool_invocations": [
                {k: copy.deepcopy(s.get(k)) for k in
                 ("id", "name", "input", "dependencies", "start_ns", "end_ns", "duration_s", "status", "resources", "metadata")}
                for s in tools
            ],
            "llm_calls": [
                {k: copy.deepcopy(s.get(k)) for k in
                 ("id", "name", "dependencies", "start_ns", "end_ns", "duration_s", "status", "metadata")}
                for s in llms
            ],
            "state_accesses": copy.deepcopy(accesses) if state_complete else None,
            "state_transfers": copy.deepcopy(transports),
        },
        "metrics": metrics,
        "missing": missing,
        "workflow": {
            "nodes": [
                {"id": s["id"], "name": s["name"], "kind": s["kind"]} for s in nodes
            ],
            "edges": edges,
            "calls": dict(Counter(s["name"] for s in nodes)),
            "tool_calls": dict(Counter(s["name"] for s in tools)),
            "causal_revisits": dict(revisits),
            "parallel_peak": peak if complete else None,
            "routes": [
                {
                    "node": s["name"],
                    "targets": sorted(r[0] for r in s["metadata"]["routes"])
                    if "routes" in s.get("metadata", {})
                    else sorted(successors[s["id"]]),
                }
                for s in nodes
            ],
        },
        "tool_durations": {
            name: [s.get("duration_s") for s in tools if s["name"] == name]
            for name in {s["name"] for s in tools}
        },
        "runtime_tool_durations": [v / 1000 for v in runtime]
        if complete and runtime and all(number(v) for v in runtime)
        else [],
    }
