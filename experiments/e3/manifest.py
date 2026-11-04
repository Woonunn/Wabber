"""Seal workload assignments before profiling; all strategies consume these bytes."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

WORKLOADS = ("DataSciBench", "TravelPlanner", "ClaudeCodeReview", "ScientificComputation")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(
    tasks: list[dict],
    arrivals: list[dict],
    phases: list[dict],
    seed: int,
    duration_ms: float,
    output: Path,
    artifacts: list[Path] | None = None,
    assignments: list[dict] | None = None,
) -> dict:
    """Freeze task inputs, source timing, captures and predetermined composition phases."""
    if not arrivals or not tasks or type(seed) is not int:
        raise ValueError("Tasks, real arrivals and integer seed are required")
    ids = [t["id"] for t in tasks]
    if assignments is not None:
        if len(tasks) != len(arrivals) or len(assignments) != len(arrivals):
            raise ValueError("Independent requests need one task and assignment per arrival")
        if len({str(Path(t["capture"]).resolve()) for t in tasks}) != len(tasks):
            raise ValueError("Independent requests cannot share a Capture directory")
        if {a["task"] for a in assignments} != set(ids):
            raise ValueError("Every independent request must execute exactly once")
        lookup = {t["id"]: t for t in tasks}
        for arrival, assignment in zip(arrivals, assignments):
            if (
                assignment["arrival_id"] != arrival["arrival_id"]
                or assignment["workload"] != lookup[assignment["task"]]["workload"]
            ):
                raise ValueError("Independent assignment does not match its arrival or workload")
    for value in ids + [a["arrival_id"] for a in arrivals]:
        if not isinstance(value, str) or value in {".", ".."} or Path(value).name != value or not value:
            raise ValueError("Request and task IDs must be safe path components")
    if len(set(ids)) != len(ids) or {t["workload"] for t in tasks} != set(WORKLOADS):
        raise ValueError("Unique tasks covering all four workloads required")
    from experiments.exp1.runner.arrival_scheduler import ms_ns

    offsets = [ms_ns(a["relative_arrival_ms"]) for a in arrivals]
    if (
        offsets != sorted(offsets)
        or len({a["arrival_id"] for a in arrivals}) != len(arrivals)
        or offsets[-1] >= ms_ns(duration_ms)
    ):
        raise ValueError("Invalid original arrival time axis")
    if not phases or phases[0]["start_ms"] != 0:
        raise ValueError("Composition must start at zero")
    starts = [ms_ns(p["start_ms"]) for p in phases]
    if starts != sorted(set(starts)) or starts[-1] >= ms_ns(duration_ms):
        raise ValueError("Invalid composition phase boundaries")
    for phase in phases:
        weights = phase["weights"]
        if (
            set(weights) != set(WORKLOADS)
            or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights.values())
            or sum(weights.values()) <= 0
        ):
            raise ValueError("Invalid predeclared workload weights")
    files = {str(Path(p).resolve()): digest(Path(p)) for p in artifacts or ()}
    for task in tasks:
        task_manifest = json.loads(Path(task["manifest"]).read_text())
        required = [Path(task["manifest"]), *map(Path, task["inputs"]), *map(Path, task_manifest.get("assets", []))]
        declared = next(t for t in task_manifest["tasks"] if t["task_id"] == task["task_id"])
        capture = Path(task["capture"])
        required += sorted(p for p in capture.rglob("*") if p.is_file())
        if task.get("agent_config"):
            required.append(Path(task["agent_config"]))
        if not required or not capture.is_dir():
            raise ValueError("Successful live Capture required")
        from atsuite.tracing.storage import read_text

        if json.loads(read_text(capture / "result.json")).get("status") != "complete":
            raise ValueError("Unsuccessful Capture")
        events = [json.loads(line) for line in read_text(capture / "events.jsonl").splitlines()]
        capture_starts = [e for e in events if e["kind"] == "run_start"]
        prompt_hash = hashlib.sha256(declared["prompt"].encode()).hexdigest()
        if (
            len(capture_starts) != 1
            or capture_starts[0].get("task_id") != task["task_id"]
            or capture_starts[0].get("prompt_sha256") != prompt_hash
        ):
            raise ValueError("Capture does not match the sealed original task and prompt")
        if not any(e.get("kind") == "llm" and e.get("source") == "live_official_api" for e in events):
            from experiments.exp1.runner.e1b_references import seal_reference

            source = Path(capture_starts[0].get("source_capture", ""))
            seal_reference(capture, source, task["task_id"], capture_starts[0]["config"]["tool"]["provider"])
            source_events = [json.loads(line) for line in read_text(source / "events.jsonl").splitlines()]
            if not any(e.get("kind") == "llm" and e.get("source") == "live_official_api" for e in source_events):
                raise ValueError("Calibrated reference requires its original live Capture")
            required += sorted(p for p in source.rglob("*") if p.is_file())
        for path in required:
            key = str(path.resolve())
            if key not in files:
                files[key] = digest(path)
    root = Path(__file__).resolve().parents[2]
    for workload in WORKLOADS:
        for directory in ("nodes", "langgraph_agent"):
            for source in (root / "benchmarks" / workload / directory).rglob("*.py"):
                files[str(source)] = digest(source)
    for source in (
        root / "atsuite/runtime.py",
        root / "experiments/exp1/capture/capture_agent_traces.py",
        root / "experiments/exp1/runner/arrival_scheduler.py",
    ):
        files[str(source)] = digest(source)
    for directory in (
        "atsuite",
        "atsuite/routing",
        "atsuite/workload",
        "atsuite/provider_model",
        "atsuite/tracing",
        "atsuite_sdk",
        "experiments/e3",
        "experiments/exp1",
        "benchmarks/DataSciBench/langgraph_agent",
    ):
        for source in (root / directory).rglob("*.py"):
            files[str(source)] = digest(source)
    for source in (
        root / "uv.lock",
        root / "pyproject.toml",
        root / "experiments/e3/travel_source.lock.json",
        root / "experiments/e3/evaluation.requirements.lock",
    ):
        files[str(source)] = digest(source)
    rng = random.Random(seed)
    by_workload = {w: [t["id"] for t in tasks if t["workload"] == w] for w in WORKLOADS}
    requests = []
    for index, arrival in enumerate(arrivals):
        phase = max(p for p in range(len(phases)) if starts[p] <= ms_ns(arrival["relative_arrival_ms"]))
        if assignments is None:
            workload = rng.choices(WORKLOADS, [phases[phase]["weights"][w] for w in WORKLOADS])[0]
            task_id = rng.choice(by_workload[workload])
        else:
            workload, task_id = assignments[index]["workload"], assignments[index]["task"]
        requests.append({**arrival, "workload": workload, "task": task_id, "phase": phase})
    value = {
        "schema": 1,
        "seed": seed,
        "duration_ms": duration_ms,
        "tasks": tasks,
        "phases": phases,
        "requests": requests,
        "files": files,
        "source_arrivals_sha256": hashlib.sha256(json.dumps(arrivals, sort_keys=True).encode()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=False)
    path = output / "manifest.json"
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    (output / "seal.json").write_text(json.dumps({"manifest_sha256": digest(path)}) + "\n")
    return value


def load(directory: Path) -> dict:
    """Refuse mutated prompts, input datasets, fixtures or capture evidence."""
    path = directory / "manifest.json"
    if digest(path) != json.loads((directory / "seal.json").read_text())["manifest_sha256"]:
        raise ValueError("E3 manifest changed")
    value = json.loads(path.read_text())
    if any(digest(Path(p)) != checksum for p, checksum in value["files"].items()):
        raise ValueError("E3 input or Capture changed")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    source = Path(spec.pop("arrivals_file"))
    value = seal(arrivals=json.loads(source.read_text()), output=args.output, **spec)
    print(json.dumps({"requests": len(value["requests"]), "files": len(value["files"])}))


if __name__ == "__main__":
    main()
