"""Observation contract shared by Analyzer, online callers and offline experiments."""

import copy
import json
from collections import Counter

from atsuite.workload.profile import number
from atsuite.workload.descriptor import validate_descriptor

BACKENDS = {
    "local": ("state-decoupled", "Local Docker FaaS validation", "USD"),
    "aws_lambda": ("state-decoupled", "AWS Lambda", "USD"),
    "aws_agentcore": ("state-coupled", "AWS AgentCore", "USD"),
    "ali_fc": ("state-decoupled", "Aliyun FC", "CNY"),
    "ali_agentrun": ("state-coupled", "Aliyun FC Session MCP", "CNY"),
}
FEATURES = (
    "state_bytes",
    "llm_idle_s",
    "rho_kb_s",
    "tool_cpu_s",
    "tool_elapsed_s",
    "state_reads",
    "state_writes",
    "state_max_object_bytes",
)
MEASUREMENTS = (
    "latency_s",
    "tool_s",
    "state_sync_s",
    "cold_start_s",
    "queue_s",
    "schedule_s",
    "harness_queue_s",
    "service_slot_s",
    "throughput_rps",
    "concurrent_sessions",
    "observed_instances",
    "billed_compute_s",
)


def validate_pool(pool: dict) -> None:
    """Require explicit session and invocation limits; never equate the two."""
    if pool.get("provider") not in BACKENDS:
        raise ValueError("Unsupported provider")
    if pool.get("mode", BACKENDS[pool["provider"]][0]) != BACKENDS[pool["provider"]][0]:
        raise ValueError("Provider/deployment mode mismatch")
    if (
        not isinstance(pool.get("environment"), str)
        or not pool["environment"]
        or not isinstance(pool.get("resources"), dict)
    ):
        raise ValueError(
            "A frozen environment ID and resource configuration are required"
        )
    json.dumps(pool["resources"], allow_nan=False)
    for key in ("cpu", "memory_mb", "disk_mb"):
        value = pool["resources"].get(key)
        if value is not None and (
            not number(value) or (value == 0 and key != "disk_mb")
        ):
            raise ValueError(f"Invalid resource: {key}")
    for key in ("instances", "sessions_per_instance", "invocations_per_instance"):
        value = pool.get(key)
        if value is not None and (
            not number(value)
            or int(value) != value
            or (value < 1 and key != "instances")
        ):
            raise ValueError(f"Invalid capacity: {key}")


def trace_measurements(trace: dict) -> dict:
    """Extract platform timing only when every Tool exposes the corresponding counter."""
    tools = [s for s in trace["spans"] if s["kind"] == "tool"]
    runtimes = [s.get("metadata", {}).get("runtime", {}) for s in tools]
    values = {
        "tool_s": [r.get("execution_elapsed_ms") for r in runtimes],
        "state_sync_s": [
            r.get("provider_metadata", {}).get("state_sync", {}).get("total_ms")
            for r in runtimes
        ],
    }
    return {
        "latency_s": (trace["completed_ns"] - trace["arrival_ns"]) / 1e9,
        **{
            k: sum(v) / 1000 if v and all(number(x) for x in v) else None
            for k, v in values.items()
        },
    }


def execution_context(pool: dict) -> dict:
    """Identify the runtime/resources that produced physical usage, excluding tariffs."""
    return copy.deepcopy({k: pool[k] for k in ("provider", "environment", "resources")})


def features(profile: dict, pool: dict, load_rps: float) -> dict:
    """Keep logical S_G/D_G distinct from provider synchronization and latency."""
    result = {key: profile.get("metrics", {}).get(key) for key in FEATURES}
    workflow = profile.get("workflow", {})
    for group in ("calls", "tool_calls", "causal_revisits"):
        for name, value in workflow.get(group, {}).items():
            result[f"{group}:{name}"] = value
    result["parallel_peak"] = workflow.get("parallel_peak")
    result["_workflow_counts_complete"] = int(
        profile.get("status") == "success"
        and all(isinstance(workflow.get(group), dict) for group in ("calls", "tool_calls", "causal_revisits"))
        and all(isinstance(workflow.get(group), list) for group in ("nodes", "edges"))
    )
    nodes = {node["id"]: node["name"] for node in workflow.get("nodes", [])}
    for (source, target), count in Counter(
        (nodes[a], nodes[b]) for a, b in workflow.get("edges", [])
    ).items():
        result[f"edge:{source}->{target}"] = count
    result["load_rps"] = load_rps
    result["instances"] = pool.get("instances")
    for key, value in result.items():
        if value is not None and not number(value):
            raise ValueError(f"Invalid feature: {key}")
    return result


def validate_cost(cost: dict | None, provider: str) -> None:
    """Actual charges require attributable, complete evidence, never an account total."""
    if cost is None:
        return
    if cost.get("currency") != BACKENDS[provider][2]:
        raise ValueError(
            "Currency must match provider; convert explicitly outside the model"
        )
    if cost.get("basis") not in {"estimated", "actual"} or not cost.get("evidence"):
        raise ValueError("Cost requires a basis and evidence references")
    if not cost.get("scope") or not isinstance(cost.get("complete"), bool):
        raise ValueError("Cost requires an explicit scope and completeness")
    if not cost.get("components") or any(
        not number(v) for v in cost["components"].values()
    ):
        raise ValueError("Invalid or empty cost components")
    if cost["basis"] == "actual" and (
        not cost["complete"] or cost.get("attribution") != "exclusive"
    ):
        raise ValueError(
            "Actual cost requires exclusive attribution and complete scope"
        )


def validate_observation(row: dict) -> None:
    """Reject malformed observations before they enter persistent statistics."""
    if "capacity_transition" in row and not isinstance(
        row["capacity_transition"], bool
    ):
        raise ValueError("Invalid capacity transition marker")
    validate_pool(row["pool"])
    profile = row["profile"]
    if profile.get("descriptor") is not None:
        validate_descriptor(profile["descriptor"])
    if "reference_id" in profile and (
        not isinstance(profile["reference_id"], str) or not profile["reference_id"]
    ):
        raise ValueError("Reference profiles require an explicit nonempty basis")
    if profile.get("execution_pool") is not None:
        validate_pool(profile["execution_pool"])
        if not profile.get("reference_id") and execution_context(
            profile["execution_pool"]
        ) != execution_context(row["pool"]):
            raise ValueError(
                "Native observation belongs to another execution environment"
            )
    for key in ("arrival_ns", "completed_ns", "available_ns", "load_rps"):
        if not number(row.get(key)):
            raise ValueError(f"Invalid {key}")
    if not row["arrival_ns"] <= row["completed_ns"] <= row["available_ns"]:
        raise ValueError("Arrival, completion and availability must be ordered")
    if (
        row.get("kind") not in {"request", "window"}
        or not isinstance(row.get("id"), str)
        or not row["id"]
    ):
        raise ValueError("Observation requires kind and stable ID")
    if row.get("status") not in {"success", "failed", "incomplete"}:
        raise ValueError("Invalid observation status")
    features(row["profile"], row["pool"], row["load_rps"])
    for key, value in row.get("measurements", {}).items():
        if key not in MEASUREMENTS or (value is not None and not number(value)):
            raise ValueError(f"Invalid measurement: {key}")
    if row["kind"] == "window" and row["completed_ns"] == row["arrival_ns"]:
        raise ValueError("Window must have positive duration")
    validate_cost(row.get("cost"), row["pool"]["provider"])


def observation_from_report(
    profile: dict,
    report: dict,
    pool: dict,
    load_rps: float,
    available_ns: int,
) -> dict:
    """Join one completed request with Analyzer output, retaining estimated pricing.

    Shared session/instance lines without invocation IDs are excluded. Import
    those once as a separate window observation with inclusive cost components.
    """
    if report["run"]["provider"] != pool["provider"]:
        raise ValueError("Report belongs to another provider")
    if report["run"]["uid"] != profile["request_id"]:
        raise ValueError("Report/request identity mismatch")
    summary = report["summary"]
    mapping = {
        "latency_s": "run_user_e2e_ms",
        "tool_s": "total_tool_exec_ms",
        "state_sync_s": "total_state_sync_overhead_ms",
        "cold_start_s": "total_initialize_time_ms",
    }
    measurements = {
        k: summary.get(v) / 1000 if number(summary.get(v)) else None
        for k, v in mapping.items()
    }
    call_ids = {r["call_id"] for r in report["invocations"]}
    lines = [
        line for line in report["costs"] if line.get("invocation_call_id") in call_ids
    ]
    components = {}
    for line in lines:
        if line["currency"] != BACKENDS[pool["provider"]][2] or not number(
            line["amount"]
        ):
            raise ValueError("Invalid Analyzer cost line")
        components[line["category"]] = (
            components.get(line["category"], 0) + line["amount"]
        )
    cost = None
    if lines and call_ids == {line["invocation_call_id"] for line in lines}:
        cost = {
            "basis": "estimated",
            "currency": BACKENDS[pool["provider"]][2],
            "scope": "analyzer_invocations",
            "complete": False,
            "components": components,
            "evidence": [report.get("report_path") or report["run"]["run_id"]],
        }
    row = {
        "kind": "request",
        "id": profile["request_id"],
        "profile": copy.deepcopy(profile),
        "status": profile["status"],
        "pool": copy.deepcopy(pool),
        "load_rps": load_rps,
        "arrival_ns": profile["arrival_ns"],
        "completed_ns": profile["completed_ns"],
        "available_ns": available_ns,
        "measurements": measurements,
        "cost": cost,
    }
    validate_observation(row)
    return row
