"""Audit all catalog prompts and import pinned BCB inputs/tests without executing on the host."""

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha(data: bytes) -> str:
    """Hash exact bytes for provenance and sealing."""
    return hashlib.sha256(data).hexdigest()


def audit(catalog: dict, upstream: bytes) -> tuple[dict, dict]:
    """Reconcile every candidate with the upstream input and full test suite."""
    rows = {r["Task ID"].replace("BigCodeBench/", "bcb"): r
            for r in csv.DictReader(io.StringIO(upstream.decode()))}
    cases, records = {}, []
    seen = set()
    for task in catalog["tasks"]:
        if sha(task["prompt"].encode()) != task["prompt_sha256"]:
            raise ValueError("Catalog prompt checksum mismatch")
        if task["task_id"] in seen:
            raise ValueError("Duplicate catalog task")
        seen.add(task["task_id"])
        row = {"task_id": task["task_id"], "prompt_sha256": task["prompt_sha256"],
               "task_type": task["task_type"], "status": "needs_execution_adapter",
               "reasons": ["Input preparation and independent validator must be verified"]}
        if task["task_type"] == "bcb":
            original = rows[task["task_id"]]
            if original["Prompt"] != task["prompt"]:
                raise ValueError("Upstream evaluation prompt differs: " + task["task_id"])
            case = {"prompt_sha256": task["prompt_sha256"], "data": original["Data"],
                    "testcase": original["Test"], "reference": original["Ground Truth Function"]}
            imports = set()
            for key in ("data", "testcase", "reference"):
                tree = ast.parse(case[key])
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.update(n.name.split(".")[0] for n in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.add(node.module.split(".")[0])
            row.update(status="pending_sandbox_preflight", reasons=[], imports=sorted(imports),
                       hashes={k: sha(case[k].encode()) for k in ("data", "testcase", "reference")})
            cases[task["task_id"]] = case
        records.append(row)
    provenance = {**catalog["source"], "evaluation_path": "metric/bcb_all_data.csv",
                  "evaluation_sha256": sha(upstream)}
    return ({"schema_version": 1, "source": provenance, "cases": cases},
            {"schema_version": 1, "source": provenance, "candidate_count": len(records),
             "counts": dict(Counter(r["status"] for r in records)), "tasks": records,
             "agent_verified_count": 0, "formal_experiment_ready": False})


def main() -> None:
    """Write the review and reproducible evaluation bundle from a pinned git blob."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads((ROOT / "benchmarks/DataSciBench/tasks/catalog.json").read_text())
    raw = subprocess.check_output(["git", "-C", str(args.repo), "show",
                                   catalog["source"]["commit_sha"] + ":metric/bcb_all_data.csv"])
    bundle, review = audit(catalog, raw)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "bcb_cases.json.gz").write_bytes(gzip.compress(
        json.dumps(bundle, ensure_ascii=False, sort_keys=True).encode(), mtime=0))
    runtime = {**bundle, "cases": {key: {k: case[k] for k in ("prompt_sha256", "testcase")}
                                   for key, case in bundle["cases"].items()}}
    (args.output / "runtime_checks.json.gz").write_bytes(gzip.compress(
        json.dumps(runtime, sort_keys=True).encode(), mtime=0))
    (args.output / "audit.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidates": review["candidate_count"], "bcb": len(bundle["cases"])}))


if __name__ == "__main__":
    main()
