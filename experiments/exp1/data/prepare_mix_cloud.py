"""Adapt a sealed Mix sequence to the existing cloud replay manifest without changing arrivals."""

import argparse
import json
from pathlib import Path

from atsuite.tracing.storage import read_text
from benchmarks.DataSciBench.langgraph_agent.trace_recorder import export_atg
from experiments.exp1.data.prepare_e1b import PROVIDERS, load, sha, write_csv
from experiments.exp1.runner.mix import ROOT, input_records, verify_capture, write


def prepare(sequence: Path, output: Path) -> dict:
    """Reuse the approved inputs and frozen decisions; retain the original request assignments."""
    sequence, output = sequence.resolve(), output.resolve()
    plan = json.loads((sequence / "manifest.json").read_text())
    if sha(sequence / "manifest.json") != json.loads((sequence / "seal.json").read_text())["manifest_sha256"]:
        raise ValueError("Sequence manifest changed")
    if sha(sequence / "arrivals.json") != plan["arrivals_sha256"]:
        raise ValueError("Arrival assignments changed")
    source = Path(plan["mix_manifest"])
    if sha(source) != plan["mix_manifest_sha256"]:
        raise ValueError("Mix manifest changed")
    mix = json.loads(source.read_text())
    if (mix.get("reference_verified_count", 0) < 100 or not mix["formal_experiment_ready"]
            or len({t["prompt_sha256"] for t in mix["tasks"]}) < 100):
        raise ValueError("Cloud Mix requires 100 validated prompts and environment references")
    tasks = {}
    for task in mix["tasks"]:
        directory = ROOT / "benchmarks/DataSciBench/data" / task["task_id"]
        files = input_records(directory)
        if files != task["readiness"]["inputs"]["files"]:
            raise ValueError("Prepared input changed")
        reference = task["replay_reference"]
        capture = Path(reference["directory"])
        if verify_capture(task, capture, require_live=False) != reference:
            raise ValueError("Environment reference changed")
        tasks[task["task_id"]] = {
            "capture": str(capture.resolve()), "capture_files": reference["hashes"],
            "dataset_files": [{**f, "path": str((directory / f["name"]).relative_to(ROOT))} for f in files],
            "prompt_sha256": task["prompt_sha256"],
        }
    rows = json.loads((sequence / "arrivals.json").read_text())
    for row in rows:
        row.update(scenario=plan["composition"], window_role="mix", stage=row["segment"],
                   sequence_seed=plan["seed"], task_seed=0, prompt_id=row["task_id"],
                   trace_id=Path(tasks[row["task_id"]]["capture"]).name,
                   trace_sha256=tasks[row["task_id"]]["capture_files"]["events.jsonl"])
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "requests.csv", rows)
    first = Path(next(iter(tasks.values()))["capture"])
    export_atg([json.loads(line) for line in read_text(first / "events.jsonl").splitlines()], output / "trace.atg.json")
    manifest = {"experiment": "E2-B", "task_manifest": str(source.resolve()), "tasks": tasks,
                "request_table_sha256": sha(output / "requests.csv"), "requests": len(rows),
                "sources": {str(p): sha(p) for p in (source, sequence / "manifest.json", sequence / "arrivals.json")},
                "groups": [{"provider_order": PROVIDERS}],
                "config": {"seed": plan["seed"], "window_roles": ["mix"],
                           "scenarios": {plan["composition"]: []}, "providers": PROVIDERS,
                           "window_duration_ms": plan["duration_ms"], "queue_capacity": 1000, "slo_ms": 120000},
                "scope": "Identical sealed Mix arrivals; fixed independent provider runs; no PLAN/DAG routing features"}
    write(output / "manifest.json", manifest)
    load(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.sequence, args.output)
    print(json.dumps({"requests": result["requests"], "tasks": len(result["tasks"])}))
