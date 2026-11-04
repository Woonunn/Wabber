"""Prepare reproducible numerical and review tasks from existing ATSuite tools."""

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from benchmarks.ClaudeCodeReview.data.materialize_fixture_repo import materialize


def definitions(modules: list[str]) -> list[dict]:
    """Read existing Tool schemas; do not duplicate tool signatures in prompts."""
    from atsuite_sdk.abstract import registry

    before = set(registry.functions)
    for module in modules:
        loaded = importlib.import_module(module)
        # Imports may already be cached when preparing multiple workloads.
        before -= {n for n, t in registry.functions.items() if t.func.__module__ == loaded.__name__}
    return [
        {"name": name, "description": tool.docstring, "parameters": tool.parameters_schema}
        for name, tool in sorted(registry.functions.items())
        if name not in before
    ]


def prepare_scientific(output: Path) -> dict:
    """Create six declared numerical problems; large matrices travel as immutable input artifacts."""
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    tools = definitions(["benchmarks.ScientificComputation.nodes.tensor_workspace.implementation"])
    from atsuite_sdk.abstract import registry

    tool_access = {d["name"]: registry.functions[d["name"]].access for d in tools}
    tasks, assets = [], []
    for size, rhs_count in [(16, 1), (32, 4), (96, 8)]:
        for seed in [1001, 1002]:
            rng = np.random.default_rng(seed)
            design = rng.normal(size=(size, size))
            matrix = design.T @ design + size * np.eye(size)
            rhs = rng.normal(size=(size, rhs_count))
            task_id = f"normal-{size}-{rhs_count}-seed-{seed}"
            path = inputs / (task_id + ".json")
            path.write_text(json.dumps({"A": matrix.tolist(), "B": rhs.tolist()}) + "\n")
            assets.append(str(path.resolve()))
            prompt = (
                f"Solve the regularized normal equations A X = B for immutable problem {task_id}: "
                f"A has shape {size} by {size}, B has {rhs_count} response vectors. "
                "Load A and B from the problem inputs, retain and reuse the factorization or inverse across "
                "response vectors, and store the complete solution tensor. Return its stored tensor name."
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "report_values": False,
                    "readiness": {
                        "inputs": {
                            "files": [
                                {
                                    "name": path.name,
                                    "bytes": path.stat().st_size,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                }
                            ]
                        }
                    },
                    "tool_definitions": tools,
                    "tool_access": tool_access,
                    "verification": {"matrix": matrix.tolist(), "rhs": rhs.tolist()},
                }
            )
    (output / "ScientificComputation.json").write_text(json.dumps({"tasks": tasks, "assets": assets}, indent=2) + "\n")
    return {"scientific_tasks": len(tasks), "inputs": str(inputs)}


def prepare(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    scicom = definitions(["benchmarks.ScientificComputation.nodes.tensor_workspace.implementation"])
    scientific = []
    for size, rhs_count in [(8, 1), (32, 4), (96, 8)]:
        rng = np.random.default_rng(size)
        design = rng.normal(size=(size, size))
        matrix = design.T @ design + size * np.eye(size)
        rhs = rng.normal(size=(size, rhs_count))
        prompt = (
            f"Solve the regularized normal equations A X = B for {rhs_count} response vectors. "
            "Create A and B in the tensor workspace, reuse A inverse across response vectors, "
            "store the solution tensor and report its numeric values. "
            + json.dumps({"A": matrix.tolist(), "B": rhs.tolist()})
        )
        scientific.append(
            {
                "task_id": f"normal-{size}-{rhs_count}",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "readiness": {"inputs": {"files": []}},
                "tool_definitions": scicom,
                "verification": {"matrix": matrix.tolist(), "rhs": rhs.tolist()},
            }
        )
    (output / "ScientificComputation.json").write_text(json.dumps({"tasks": scientific}, indent=2) + "\n")
    fixture = materialize(output / "review-data")
    previous_data = os.environ.get("ATSUITE_DATA_PATH")
    os.environ["ATSUITE_DATA_PATH"] = str((output / "review-data").resolve())
    node_root = Path(__file__).resolve().parents[2] / "benchmarks/ClaudeCodeReview/nodes"
    sys.path.insert(0, str(node_root / "repo_git"))
    try:
        review_defs = definitions(
            [f"benchmarks.ClaudeCodeReview.nodes.{node}.implementation" for node in ("repo_git", "repo_files")]
        )
    finally:
        sys.path.pop(0)
        if previous_data is None:
            os.environ.pop("ATSUITE_DATA_PATH", None)
        else:
            os.environ["ATSUITE_DATA_PATH"] = previous_data
    reviews = []
    for check in fixture["checks"]:
        prompt = f"Review PR {check['pr_id']}. Inspect its diff and repository files, verify the behavior against the base and head refs, and report introduced bugs."
        reviews.append(
            {
                "task_id": f"pr-{check['pr_id']}",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "readiness": {"inputs": {"files": []}},
                "tool_definitions": review_defs,
                "verification": {"defects": [check]},
            }
        )
    (output / "ClaudeCodeReview.json").write_text(
        json.dumps(
            {
                "tasks": reviews,
                "assets": [str(p.resolve()) for p in (output / "review-data").rglob("*") if p.is_file()],
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "scientific_tasks": len(scientific),
        "review_tasks": len(reviews),
        "review_fixture": str(output / "review-data"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scientific-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_scientific(args.output) if args.scientific_only else prepare(args.output)))


if __name__ == "__main__":
    main()
