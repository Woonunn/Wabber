"""Join real E3 executions to native usage; leave unattributed money unknown."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from atsuite.aws.CloudWatch import AWSCloudWatch
from atsuite.tracing.storage import read_text
from experiments.exp1.runner.arrival_scheduler import stats
from experiments.exp1.runner.provider_usage import session_usage


def price_request(row: dict, config: dict, native_sessions: dict | None = None) -> dict:
    """Adapt native E3 measurements to ATSuite's existing pricing API."""
    from atsuite.analysis.api import price_metrics
    from atsuite.analysis.model import ProviderMetric, RunContext, StateObservation

    provider = row["pool"]
    kind = "datasci" if row["workload"] == "DataSciBench" else "e3"
    actual = config["native_elasticity"][provider][kind]
    usage = row["usage"]
    metrics = []
    missing = []
    if provider == "aws_agentcore":
        for s in usage.get("sessions", []):
            if s.get("vcpu_hours") is None or s.get("memory_gb_hours") is None:
                missing.append("native AgentCore usage pending")
            else:
                metrics.append(ProviderMetric(s["session_id"], provider, "agentcore_usage", fields=s))
                last = s.get("residence", {}).get("last_usage_epoch_ms")
                last_tool = row.get("last_tool_epoch_s")
                retention_s = 0 if row.get("sessions_closed") else 295
                if last is None or last_tool is None or last / 1000 < last_tool + retention_s:
                    missing.append("native AgentCore retention tail pending")
                if row.get("sessions_closed"):
                    stops = [r["closed_epoch_s"] for r in row.get("session_release_receipts", [])
                             if r.get("provider_session_id") == s["session_id"] and r.get("http_status") == 200]
                    if not stops or last is None or last / 1000 < max(stops):
                        missing.append("native AgentCore usage through successful StopRuntimeSession response pending")
    elif provider == "ali_agentrun":
        for sid in usage["native_session_ids"]:
            s = (native_sessions or {}).get(sid)
            if not s or s.get("sessionStatus") not in {"Expired", "Deleted"}:
                missing.append("native Session lifetime pending")
                continue
            duration = (
                datetime.fromisoformat(s["lastModifiedTime"].replace("Z", "+00:00"))
                - datetime.fromisoformat(s["createdTime"].replace("Z", "+00:00"))
            ).total_seconds()
            if duration < 0:
                raise ValueError("Invalid native Session lifetime")
            active_ms = number(s.get("observed_active_time_ms"))
            if active_ms is None or not usage.get("call_coverage_complete"):
                missing.append("native Session active duration pending")
                continue
            metrics.append(
                ProviderMetric(
                    sid,
                    provider,
                    "ali_session",
                    fields={
                        "session_duration_ms": duration * 1000,
                        "duration_ms": active_ms,
                        "request_count": s.get("observed_request_count", usage["tool_calls"]),
                        "cpu": actual["cpu"],
                        "memory": actual["memory_mb"],
                        "disk": actual["disk_mb"],
                    },
                )
            )
    else:
        if not usage.get("call_coverage_complete"):
            missing.append("native invocation metrics pending")
        for i, value in enumerate(usage.get("request_metrics", [])):
            metrics.append(
                ProviderMetric(
                    str(i),
                    provider,
                    "native_request",
                    fields={
                        **value,
                        "memory": actual["memory_mb"],
                        "disk": actual.get("disk_mb", 2048),
                        "cpu": actual.get("cpu"),
                    },
                )
            )
    context = RunContext(str(row.get("request_id")), provider, "", "E3", "", "", "")
    profile = row.get("profile_metrics", {})
    state = []
    if provider in {"aws_lambda", "ali_fc"} and profile.get("state_transfer_bytes", 0):
        calls = usage["tool_calls"]
        native_state = row.get("native_state", {})
        size = native_state.get("max_bytes", profile.get("state_max_object_bytes") or profile.get("state_bytes") or 0)
        state.append(
            StateObservation(
                context.uid,
                provider,
                "snapshot",
                size_gb=size / 1024**3,
                operation_count=calls * 2,
                started_at=0,
                ended_at=row.get("latency_s", 0),
                metadata={
                    "read_operations": native_state.get("reads", calls),
                    "write_operations": native_state.get("writes", calls),
                    "basis": "traced snapshot operations; state high-water size over request duration",
                },
            )
        )
    lines = [line.to_dict() for line in price_metrics(context, metrics, state)]
    if not metrics and usage["tool_calls"]:
        missing.append("no native priced metrics")
    return {
        "basis": "atsuite_metered_usage_pricing",
        "currency": "CNY" if provider.startswith("ali") else "USD",
        "total": sum(line["amount"] for line in lines) if not missing and metrics else None,
        "observed_subtotal": sum(line["amount"] for line in lines),
        "lines": lines,
        "missing": missing,
        "state_scope": "ATSuite traced-state estimate during request; post-window object retention and unobserved APIs excluded",
        "invoice": False,
    }


def number(value):
    try:
        parsed = float(value)
    except (ValueError, TypeError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def session_active_usage(events: list[dict]) -> dict:
    """Union native full HTTP intervals; streaming durationMs can stop at the first response bytes."""
    requests = {}
    for event in events:
        key = event.get("requestid")
        measured = (number(event.get("invokefunctionstarttimestamp")), number(event.get("invokefunctionlatencyms")))
        if key in requests and requests[key] != measured:
            raise ValueError("Conflicting Ali request intervals: " + str(key))
        requests[key] = measured
    complete = bool(requests) and all(start is not None and duration is not None for start, duration in requests.values())
    total = 0.0
    if complete:
        end = -math.inf
        for start, duration in sorted(requests.values()):
            right = start + duration
            total += max(0.0, right - max(start, end))
            end = max(end, right)
    return {
        "observed_request_count": len(requests),
        "observed_active_time_ms": total if complete else None,
        "active_time_basis": "union of native invokeFunctionStartTimestamp/invokeFunctionLatencyMs; includes streaming response completion and initialization; excludes overlapping invocation double counting",
    }


def released_session_usage(session_id: str, receipt: dict, tools: list[dict], index: dict) -> dict | None:
    """Recover a deleted Session's measured lifecycle from native calls and its stop receipt."""
    if receipt.get("http_status") not in {200, 202, 204, 404} or number(receipt.get("closed_epoch_s")) is None:
        return None
    owned = [event for tool in tools for event in index.get(tool.get("provider_request_id"), [])]
    containers = {event.get("instanceid") for event in owned} - {None, "", "null"}
    if len(containers) != 1:
        return None
    container = next(iter(containers))
    events = [event for batch in index.values() for event in batch if event.get("instanceid") == container]
    active = session_active_usage(events)
    if active["observed_active_time_ms"] is None:
        return None
    starts = [number(event.get("invokefunctionstarttimestamp")) for event in events]
    ends = [start + number(event.get("invokefunctionlatencyms")) for start, event in zip(starts, events)]
    stops = index.get(receipt.get("request_id"), [])
    stop_ends = [
        number(event.get("invokefunctionstarttimestamp")) + number(event.get("invokefunctionlatencyms"))
        for event in stops
        if number(event.get("invokefunctionstarttimestamp")) is not None
        and number(event.get("invokefunctionlatencyms")) is not None
    ]
    end = max([*ends, *(stop_ends or [receipt["closed_epoch_s"] * 1000])])
    return {
        "sessionId": session_id, "containerId": container, "sessionStatus": "Deleted",
        "createdTime": datetime.fromtimestamp(min(starts) / 1000).astimezone().isoformat(),
        "lastModifiedTime": datetime.fromtimestamp(end / 1000).astimezone().isoformat(),
        **active,
        "release_receipt": receipt,
        "lifecycle_basis": (
            "native initialization through native DELETE completion" if stop_ends else
            "native initialization through observed successful DELETE response; conservative client-clock end bound"
        ),
    }


def usage_index(path, provider):
    """Index native IDs once; unrelated window traffic is not experiment usage."""
    index = defaultdict(list)
    if not path.exists():
        return index
    for event in json.loads(path.read_text())["events"]:
        if provider == "aws_lambda":
            if event.get("message", "").startswith("REPORT RequestId:"):
                key = event["message"].split("RequestId:", 1)[1].strip().split()[0]
                index[key].append(event)
        elif provider == "aws_agentcore":
            payload = json.loads(event["message"])
            key = payload.get("attributes", {}).get("session.id")
            if key:
                index[key].append(event)
        else:
            row = json.loads(event["_col0"]) if "_col0" in event else event
            row = {k.lower(): v for k, v in row.items()}
            if row.get("operation") == "InvokeFunction" and row.get("requestid") not in (None, "", "null"):
                index[row["requestid"]].append(row)
    return index


def measured_usage(provider, tools, index):
    native_ids = {t.get("provider_request_id") for t in tools}
    native_ids.discard(None)
    native_ids.discard("")
    session_ids = {t.get("provider_metadata", {}).get("session_id") for t in tools}
    session_ids.discard(None)
    session_ids.discard("")
    result = {
        "tool_calls": len(tools),
        "native_request_ids": sorted(native_ids),
        "native_session_ids": sorted(session_ids),
        "actual_cost": None,
        "billing_coverage_complete": False,
    }
    if provider == "aws_agentcore":
        sessions = [session_usage(index.get(sid, []), sid) for sid in sorted(session_ids)]
        for session in sessions:
            session.pop("raw_events")
        result["sessions"] = sessions
        for field in ("vcpu_hours", "memory_gb_hours"):
            values = [s[field] for s in sessions]
            result[field] = sum(values) if values and all(v is not None for v in values) else None
        return result
    matched = [key for key in sorted(native_ids) if key in index]
    result.update(matched_calls=len(matched), call_coverage_complete=len(matched) == len(tools))
    metrics = []
    for key in matched:
        if provider == "aws_lambda":
            unique = {row["message"]: row for row in index[key]}
            if len(unique) != 1:
                raise ValueError("Conflicting Lambda REPORT records: " + key)
            metrics.append(AWSCloudWatch._parse_lambda_logs(None, list(unique.values())))
        else:
            unique = {json.dumps(row, sort_keys=True): row for row in index[key]}
            if len(unique) != 1:
                raise ValueError("Conflicting Ali request records: " + key)
            row = next(iter(unique.values()))
            cold = str(row.get("iscoldstart")).lower()
            metrics.append(
                {
                    "duration_ms": number(row.get("durationms")),
                    "invoke_latency_ms": number(row.get("invokefunctionlatencyms")),
                    "invoke_start_epoch_ms": number(row.get("invokefunctionstarttimestamp")),
                    "memory_used_mb": number(row.get("memoryusagemb")),
                    "instance_id": row.get("instanceid"),
                    "is_cold_start": cold == "true" if cold in ("true", "false") else None,
                }
            )
    result["request_metrics"] = metrics
    durations = [m["duration_ms"] for m in metrics if m["duration_ms"] is not None]
    result["observed_duration_ms"] = sum(durations) if durations else None
    if provider == "aws_lambda":
        values = [
            m["billed_duration_ms"] / 1000 * m["memory_limit_mb"] / 1024
            for m in metrics
            if m["billed_duration_ms"] is not None and m["memory_limit_mb"] is not None
        ]
        result["observed_billed_gb_s"] = sum(values) if values else None
        result["billed_gb_s"] = (
            sum(values) if tools and result["call_coverage_complete"] and len(values) == len(tools) else None
        )
    return result


def reconcile(run: Path, usage: Path, output: Path, config: dict | None = None, native_sessions: dict | None = None):
    """Write measured latency/usage matrices without converting telemetry into invoices."""
    output.mkdir(parents=True, exist_ok=False)
    preflight = (run / "attempts.json").exists()
    rows = json.loads((run / "attempts.json").read_text()) if preflight else []
    if not preflight:
        paths = [run / "requests.jsonl"] if (run / "requests.jsonl").exists() else sorted(run.glob("*/requests.jsonl"))
        for path in paths:
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
    indexes = {}
    measurements = []
    owners = {}
    for row in rows:
        provider = row["pool"]
        if provider is None:
            measurements.append(
                {
                    **row,
                    "usage": {"tool_calls": 0},
                    "priced_cost": {
                        "total": None,
                        "currency": None,
                        "missing": ["request rejected before Provider selection"],
                    },
                }
            )
            continue
        kind = "datasci" if row["workload"] == "DataSciBench" else "e3"
        key = (kind, provider)
        if key not in indexes:
            indexes[key] = usage_index(usage / kind / (provider + "-usage.json"), provider)
        events = [json.loads(line) for line in read_text(Path(row["directory"]) / "events.jsonl").splitlines()]
        tools = [e["transport"] for e in events if e["kind"] == "tool"]
        sessions = set()
        for tool in tools:
            metadata = tool.get("provider_metadata", {})
            if metadata.get("runtime_provider") != provider:
                raise ValueError("Tool executed outside its selected provider: " + row["directory"])
            if metadata.get("session_id"):
                sessions.add(metadata["session_id"])
        if len(sessions) > 1:
            raise ValueError("Request changed native Session: " + row["directory"])
        for session in sessions:
            identity = (provider, session)
            if identity in owners and owners[identity] != row["directory"]:
                raise ValueError("Different requests shared a native Session")
            owners[identity] = row["directory"]
        syncs = [t.get("provider_metadata", {}).get("state_sync", {}) for t in tools]
        request_sessions = dict(native_sessions or {})
        releases = [receipt for event in events if event["kind"] == "tool_sessions_closed"
                    for receipt in event.get("receipts", [])]
        if provider == "ali_agentrun":
            for receipt in releases:
                sid = receipt.get("provider_session_id")
                if sid not in sessions or request_sessions.get(sid, {}).get("sessionStatus") == "Active":
                    continue
                if receipt.get("http_status") == 404 and sid in request_sessions:
                    continue
                measured = released_session_usage(sid, receipt, tools, indexes[key])
                if measured:
                    request_sessions[sid] = measured
        measurements.append(
            {
                **{
                    k: row.get(k) for k in ("pool", "workload", "task", "task_id", "request_id", "success", "latency_s")
                },
                "profile_metrics": row.get("profile", {}).get("metrics", {}),
                "sessions_closed": any(e["kind"] == "tool_sessions_closed" for e in events),
                "session_release_receipts": releases,
                "native_session_lifecycle": {sid: request_sessions[sid] for sid in sessions if sid in request_sessions},
                "last_tool_epoch_s": max(
                    [datetime.fromisoformat(e["utc"]).timestamp() for e in events if e["kind"] == "tool"] or [0]
                ),
                "native_state": {
                    "max_bytes": max([s.get(k, 0) or 0 for s in syncs for k in ("load_bytes", "save_bytes")] or [0]),
                    "reads": sum(bool(s.get("load_bytes")) for s in syncs),
                    "writes": sum(bool(s.get("save_bytes")) for s in syncs),
                },
                "usage": measured_usage(provider, tools, indexes[key]),
            }
        )
        if config is not None:
            measurements[-1]["priced_cost"] = price_request(measurements[-1], config, request_sessions)
    with (output / "request-usage.jsonl").open("w") as stream:
        for row in measurements:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    matrices = {}
    for level in ("workload", "task"):
        groups = defaultdict(list)
        for row in measurements:
            groups[(row[level], row["pool"])].append(row)
        matrix = []
        for (label, provider), batch in sorted(groups.items(), key=lambda pair: (pair[0][0], str(pair[0][1]))):
            latency = stats([r["latency_s"] for r in batch])
            successful = stats([r["latency_s"] for r in batch if r["success"]])
            matrix.append(
                {
                    level: label,
                    "provider": provider,
                    "requests": len(batch),
                    "successes": sum(r["success"] for r in batch),
                    **{f"latency_{k}_s": v for k, v in latency.items() if k != "count"},
                    **{f"successful_latency_{k}_s": v for k, v in successful.items() if k != "count"},
                    "actual_total_cost": None,
                    "currency": batch[0]["priced_cost"]["currency"] if config is not None else None,
                    "cost_status": "atsuite_native_usage_pricing" if config is not None else "native_usage_only",
                    "metered_total_cost": sum(r["priced_cost"]["total"] for r in batch)
                    if config is not None and all(r["priced_cost"]["total"] is not None for r in batch)
                    else None,
                }
            )
        matrices[level] = matrix
        with (output / (level + "-matrix.csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
            writer.writeheader()
            writer.writerows(matrix)
    report = {
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requests": len(measurements),
        "successes": sum(r["success"] for r in measurements),
        "usage_snapshot": str(usage),
        "matrices": matrices,
        "best_fixed_provider": None,
        "cost_crossover": "unknown",
        "routing_checks": "each Tool matches selected provider; one native Session per request; Sessions isolated",
        "basis": "real executions and ID-joined native usage; account bills are unallocated; missing cost is not zero",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("run", "usage", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sessions", type=Path)
    args = parser.parse_args()
    sessions = {}
    if args.sessions:
        for value in json.loads(args.sessions.read_text())["functions"].values():
            for page in value["pages"]:
                sessions.update({s["sessionId"]: s for s in page.get("sessions", [])})
        requests = defaultdict(list)
        for path in args.usage.glob("*/ali_agentrun-usage.json"):
            for request_id, events in usage_index(path, "ali_agentrun").items():
                for event in events:
                    requests[event.get("instanceid")].append(event)
        for session in sessions.values():
            if session.get("containerId") in requests:
                session.update(session_active_usage(requests[session["containerId"]]))
    report = reconcile(
        args.run, args.usage, args.output, json.loads(args.config.read_text()) if args.config else None, sessions
    )
    print(json.dumps({k: v for k, v in report.items() if k != "matrices"}))


if __name__ == "__main__":
    main()
