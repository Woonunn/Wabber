"""Seal the declared ten-minute E3 corpus using an unscaled contiguous Copilot slice."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections import Counter
from decimal import Decimal
from pathlib import Path

from atsuite.tracing.storage import read_text
from experiments.e3.manifest import WORKLOADS, seal
from experiments.exp1.runner.arrival_scheduler import ms_ns

ROOT = Path(__file__).resolve().parents[2]


def arrivals(source: Path, offset_ms: str, duration_ms: str) -> list[dict]:
    """Keep original IDs, timestamp precision, bursts and idle gaps within a half-open slice."""
    start, duration = ms_ns(offset_ms), ms_ns(duration_ms)
    if start < 0 or duration <= 0:
        raise ValueError("Invalid arrival slice")
    result = []
    with source.open() as stream:
        for row in csv.DictReader(stream):
            elapsed = ms_ns(row["relative_arrival_ms"])
            if start <= elapsed < start + duration:
                result.append({**row, "relative_arrival_ms": str(Decimal(elapsed - start) / Decimal(1000000))})
    if not result or len({r["arrival_id"] for r in result}) != len(result):
        raise ValueError("Missing or duplicated original arrivals")
    if [ms_ns(r["relative_arrival_ms"]) for r in result] != sorted(ms_ns(r["relative_arrival_ms"]) for r in result):
        raise ValueError("Original arrival source is not ordered")
    return result


def build(plan_path: Path, receipts: list[Path], output: Path) -> dict:
    """Refuse missing declared captures or overlap with the E3 profiling task IDs."""
    plan = json.loads(plan_path.read_text())
    source = ROOT / plan["source"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != plan["source_sha256"]:
        raise ValueError("Copilot source changed after task declaration")
    if set(plan["formal"]) != set(WORKLOADS) or set(plan["composition"]) != set(WORKLOADS):
        raise ValueError("Declare all four workloads")
    captured = {}
    evidence_files = []
    for path in receipts:
        for row in json.loads(path.read_text()):
            key = (row["workload"], row["task_id"])
            if key in captured:
                raise ValueError("Duplicate task Capture receipt")
            captured[key] = row
            for attempt in row["attempts"]:
                evidence_files.extend(p for p in Path(attempt["directory"]).rglob("*") if p.is_file())
    tasks = []
    for workload in WORKLOADS:
        ids = plan["formal"][workload]
        if not ids or len(set(ids)) != len(ids) or set(ids) & set(plan["profile"][workload]):
            raise ValueError("Formal task IDs must be unique and separate from E3 profiling")
        for task_id in ids:
            if workload == "DataSciBench":
                manifest = ROOT / "results/stage8/reproducible-mix/manifest.json"
                declared = next(t for t in json.loads(manifest.read_text())["tasks"] if t["task_id"] == task_id)
                capture = ROOT / declared["replay_reference"]["directory"]
                inputs = [
                    str(p.resolve())
                    for p in (ROOT / "benchmarks/DataSciBench/data" / task_id).rglob("*")
                    if p.is_file()
                ]
            else:
                row = captured.get((workload, task_id))
                if not row or not row["capture"]:
                    raise ValueError("Declared task lacks successful live Capture: " + task_id)
                manifest, capture, inputs = Path(row["manifest"]), Path(row["capture"]), []
            tasks.append(
                {
                    "id": workload.lower() + "-" + task_id,
                    "workload": workload,
                    "task_id": task_id,
                    "manifest": str(manifest.resolve()),
                    "capture": str(capture.resolve()),
                    "inputs": inputs,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = output.parent / (output.name + "-environment.json")
    if environment.exists():
        raise FileExistsError(environment)
    environment.write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "packages": {
                    p: importlib.metadata.version(p) for p in ("langgraph", "numpy", "pandas", "openai", "pydantic")
                },
                "scope": "local capture environment and sealed source; new tasks require rebuilt cloud images",
            },
            indent=2,
        )
        + "\n"
    )
    agents = {}
    for task in tasks:
        start = next(
            json.loads(line)
            for line in read_text(Path(task["capture"]) / "events.jsonl").splitlines()
            if json.loads(line)["kind"] == "run_start"
        )
        config = {k: v for k, v in start["config"].items() if k != "tool"}
        if task["workload"] == "DataSciBench" and task["workload"] in agents:
            if config["model"]["seed"] != agents[task["workload"]]["model"]["seed"]:
                raise ValueError("DataSci Replay requires a consistent original execution seed")
        agents.setdefault(task["workload"], config)
    agent_path = output.parent / (output.name + "-agents.json")
    agent_path.write_text(json.dumps(agents, indent=2) + "\n")
    original = arrivals(source, str(plan["source_offset_ms"]), str(plan["duration_ms"]))
    if original[0]["original_timestamp_utc"] != plan["source_start_utc"]:
        raise ValueError("Declared Copilot window start differs")
    runtime = plan_path.parent / "runtime/E3Tools"
    if not runtime.is_dir():
        runtime = ROOT / "results/e3-ten-minute/runtime/E3Tools"
    runtime_files = sorted(p for p in runtime.rglob("*") if p.is_file())
    if not runtime_files:
        raise ValueError("Prepare the immutable expanded runtime before sealing")
    # The official evaluator checks Git HEAD/diff, and DataSci loads independent compressed upstream tests.
    for directory in (ROOT / "results/e3-travelplanner-upstream/.git", ROOT / "benchmarks/DataSciBench/tasks/mix"):
        evidence_files.extend(p for p in directory.rglob("*") if p.is_file())
    evidence_files.extend(p for p in (ROOT / "benchmarks/DataSciBench/nodes/AgentTools").iterdir() if p.is_file())
    evidence_files.extend(
        [
            ROOT / "benchmarks/DataSciBench/tasks/evaluation_alignment.json",
            ROOT / "experiments/exp1/profile/requirements.mix.txt",
        ]
    )
    upstream = ROOT / "results/e3-travelplanner-upstream"
    evidence_files.extend(
        upstream / p
        for p in subprocess.check_output(
            ["git", "-C", str(upstream), "ls-files", "evaluation", "utils", "tools"], text=True
        ).splitlines()
    )
    evidence_files.extend(p for p in (ROOT / "dockerfiles").rglob("*") if p.is_file())
    evidence_files.extend((ROOT / "benchmarks/ClaudeCodeReview/data").glob("*.py"))
    value = seal(
        tasks,
        original,
        [{"start_ms": 0, "weights": plan["composition"]}],
        plan["seed"],
        plan["duration_ms"],
        output,
        artifacts=[plan_path, source, environment, agent_path, *receipts, *runtime_files, *evidence_files],
    )
    counts = Counter(r["workload"] for r in value["requests"])
    if set(counts) != set(WORKLOADS) or {r["task"] for r in value["requests"]} != {t["id"] for t in tasks}:
        raise ValueError("Declared seed does not cover every task; retain the incomplete artifact for inspection")
    lookup = {t["id"]: t for t in tasks}
    with (output / "requests.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "arrival_id",
                "original_timestamp_utc",
                "relative_arrival_ms",
                "workload",
                "task",
                "task_id",
                "capture",
            ],
        )
        writer.writeheader()
        for request in value["requests"]:
            writer.writerow({k: request.get(k, lookup[request["task"]].get(k)) for k in writer.fieldnames})
    summary = {
        "duration_ms": plan["duration_ms"],
        "requests": len(original),
        "workload_requests": dict(counts),
        "distinct_tasks": len(tasks),
        "task_occurrences": dict(Counter(r["task"] for r in value["requests"])),
        "seed": plan["seed"],
        "first_arrival_ms": original[0]["relative_arrival_ms"],
        "last_arrival_ms": original[-1]["relative_arrival_ms"],
        "source_sha256": plan["source_sha256"],
        "timing": "original contiguous Copilot slice; no scaling, thinning or synthetic timestamps",
        "deployment": "capture and local replay verified separately; expanded cloud images not deployed",
        "manifest_sha256": json.loads((output / "seal.json").read_text())["manifest_sha256"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--receipts", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.plan, args.receipts, args.output)))


if __name__ == "__main__":
    main()
