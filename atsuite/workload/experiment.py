"""Replay DataSciBench observations: python -m atsuite.workload.experiment ROOT --output DIR."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from atsuite.tracing.storage import read_text, trace_path
from atsuite.workload import WorkloadModel, extract_profile
from atsuite.workload.profile import METRICS


def load_capture(directory: Path) -> dict:
    """Read canonical traces or conservatively adapt archived Capture ATGs.

    Legacy event clocks define a simulated arrival/completion schedule, not proof
    of historical ingestion time. Result files are used only at completion events.
    """
    events = [
        json.loads(line)
        for line in read_text(directory / "events.jsonl").splitlines()
        if line
    ]
    result = json.loads(read_text(directory / "result.json"))
    if trace_path(directory / "trace.json").exists():
        trace = json.loads(read_text(directory / "trace.json"))
    else:
        atg = json.loads(read_text(directory / "trace.atg.json"))
        nodes = [n for n in atg["nodes"] if n["type"] in ("llm", "tool_use")]
        incoming = {n["id"]: [] for n in nodes}
        for node in atg["nodes"]:
            for edge in node["edge_to"]:
                if node["id"] in incoming and edge["id"] in incoming:
                    incoming[edge["id"]].append(str(node["id"]))
        calls = [e for e in events if e["kind"] in ("llm", "tool")]
        if len(calls) != len(nodes):
            raise ValueError(f"Capture ATG/event count mismatch: {directory}")
        spans = []
        for node, event in zip(nodes, calls):
            expected_kind = "llm" if node["type"] == "llm" else "tool"
            if event["kind"] != expected_kind or node["name"] != event.get(
                "tool_name", event.get("phase", "llm")
            ):
                raise ValueError("Capture ATG/event order mismatch")
            end = event["monotonic_ns"]
            payload = event.get("result") or {}
            logical = payload.get("logical_state")
            metadata = {"logical_state": logical} if isinstance(logical, dict) else {}
            if (
                isinstance(logical, dict)
                and logical.get("complete") is True
                and isinstance(logical.get("accesses"), list)
            ):
                metadata["state_accesses"] = logical["accesses"]
            spans.append(
                {
                    "id": str(node["id"]),
                    "kind": expected_kind,
                    "name": node["name"],
                    "parent_id": "session",
                    "dependencies": incoming[node["id"]],
                    "start_ns": end - int(event["duration_s"] * 1e9),
                    "end_ns": end,
                    "duration_s": event["duration_s"],
                    "status": "error" if payload.get("status") == "error" else "ok",
                    "metadata": metadata,
                    "resources": payload.get("worker_usage")
                    or payload.get("kernel_execution", {}).get("worker_usage"),
                }
            )
        start, end = events[0]["monotonic_ns"], events[-1]["monotonic_ns"]
        spans.append(
            {
                "id": "session",
                "kind": "session",
                "name": "DataSciBench",
                "dependencies": [],
                "start_ns": start,
                "end_ns": end,
                "duration_s": (end - start) / 1e9,
                "status": "ok" if result.get("status") == "complete" else "error",
                "metadata": {},
            }
        )
        trace = {
            "schema_version": 1,
            "request_id": events[0]["request_id"],
            "session_id": events[0]["session_id"],
            "spans": spans,
            "diagnostics": []
            if result.get("status") in ("complete", "failed")
            else ["Incomplete result"],
        }
    trace.update(
        source="capture",
        arrival_ns=int(datetime.fromisoformat(events[0]["utc"]).timestamp() * 1e9),
        completed_ns=int(datetime.fromisoformat(events[-1]["utc"]).timestamp() * 1e9),
        request_status="success" if result.get("status") == "complete" else "failed",
    )
    return trace


def evaluate(traces: list[dict], output: Path) -> dict:
    """Replay arrivals before completions at ties, scoring only held-out requests."""
    output.mkdir(parents=True, exist_ok=True)
    model = WorkloadModel()
    predictions, rows = {}, []
    events = []
    for i, trace in enumerate(traces):
        if trace["completed_ns"] < trace["arrival_ns"]:
            raise ValueError("Completion precedes arrival")
        events.extend([(trace["arrival_ns"], 0, i), (trace["completed_ns"], 1, i)])
    for timestamp, kind, index in sorted(events):
        trace = traces[index]
        if kind == 0:
            predictions[index] = model.begin(
                trace["request_id"], trace["session_id"], timestamp, descriptor=trace.get("request_context")
            )
        else:
            profile = extract_profile(trace)
            prediction = predictions[index]
            errors = {
                k: abs(prediction["metrics"][k]["p50"] - profile["metrics"][k])
                if profile["status"] == "success"
                and prediction["metrics"][k]["p50"] is not None
                and profile["metrics"][k] is not None
                else None
                for k in METRICS
            }
            model.update(profile, timestamp)
            rows.append(
                {"prediction": prediction, "actual": profile, "absolute_error": errors}
            )
    final = model.predict(max((t["completed_ns"] for t in traces), default=0) + 1)
    report = {
        "schedule": "event-time simulation; strict pre-arrival availability",
        "requests": rows,
        "model": final,
    }
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    model.save(output / "model.json")
    plot(report, output)
    return report


def plot(report: dict, output: Path) -> None:
    """Export empirical CDFs, prediction errors and quantile evolution as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("llm_idle_s", "tool_elapsed_s", "state_bytes", "rho_kb_s")
    figure, axes = plt.subplots(3, 4, figsize=(16, 10))
    rows = report["requests"]
    for column, key in enumerate(keys):
        ax = axes[0, column]
        values = report["model"]["metrics"][key]["samples"]
        if values:
            ax.step(
                values,
                [(i + 1) / len(values) for i in range(len(values))],
                where="post",
            )
        else:
            ax.text(0.5, 0.5, "No valid observations", ha="center")
        ax.set_title(key + " empirical CDF")
        for row in rows:
            error = row["absolute_error"][key]
            if error is not None:
                axes[1, column].scatter(
                    row["prediction"]["sample_count"], error, color="tab:blue"
                )
        axes[1, column].set(
            xlabel="Prior successful samples", ylabel="Absolute median error"
        )
        for quantile in ("p10", "p50", "p90"):
            points = [
                (
                    r["prediction"]["sample_count"],
                    r["prediction"]["metrics"][key][quantile],
                )
                for r in rows
                if r["prediction"]["metrics"][key][quantile] is not None
            ]
            if points:
                axes[2, column].plot(*zip(*points), marker=".", label=quantile)
        axes[2, column].set(
            xlabel="Prior successful samples", ylabel="Predicted quantile"
        )
        if axes[2, column].lines:
            axes[2, column].legend()
    figure.tight_layout()
    figure.savefig(output / "distributions.png", dpi=150)
    plt.close(figure)
    workflow = report["model"]["workflow"]
    names = list(workflow["calls"])
    figure, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.35)))
    ax.barh(names, [workflow["calls"][name]["mean"] for name in names])
    ax.set_xlabel("Mean node calls per successful request")
    figure.tight_layout()
    figure.savefig(output / "workflow.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = sorted(
        {
            p.parent
            for pattern in ("events.jsonl", "events.jsonl.gz")
            for p in args.root.rglob(pattern)
        }
    )
    traces = [load_capture(p) for p in directories]
    if not traces:
        parser.error("No captures found")
    report = evaluate(traces, args.output)
    print(
        json.dumps(
            {
                "requests": len(traces),
                "successful_samples": report["model"]["sample_count"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
