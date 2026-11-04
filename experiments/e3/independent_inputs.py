"""Derive meaningful request variations without changing the original business validators."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.e3.manifest import digest

PREFERENCES = (
    "Explain the result for a technical colleague, including assumptions and limitations.",
    "Make the final explanation concise and distinguish evidence from interpretation.",
    "Write the explanation for a user unfamiliar with the implementation details.",
    "Organize the explanation around the requested deliverables and their correctness.",
)
CONTEXTS = (
    "I need to understand any edge cases that could affect this result.",
    "Please explain how the supplied evidence supports the conclusion.",
    "Please make assumptions about missing information explicit.",
    "Keep the required output formats and all original constraints intact.",
)


def derive(parent: dict, document: dict, original: dict, ordinal: int, folder: Path) -> tuple[dict, list[str]]:
    """Create a fresh request while preserving original contracts and source provenance."""
    task = copy.deepcopy(original)
    assets = list(document.get("assets", []))
    workload = parent["workload"]
    if workload == "TravelPlanner":
        source = next(Path(p) for p in assets if Path(p).name == "source-rows.json")
        lock = json.loads(Path(__file__).with_name("travel_source.lock.json").read_text())
        if digest(source) != lock["source_rows_sha256"]:
            raise ValueError("Official TravelPlanner validation snapshot changed")
        excluded = {0, 1, 2, 60, 61, 62, 120, 121, 122}
        candidates = sorted(
            (
                e
                for e in json.loads(source.read_text())["rows"]
                if e["row_idx"] not in excluded and e["row"]["level"] == task["source"]["level"]
            ),
            key=lambda e: e["row_idx"],
        )
        lane = (task["source"]["row"] % 60) - 1
        if lane not in (0, 1):
            raise ValueError("Expected one of the six retained official TravelPlanner tasks")
        selected = candidates[2 * (ordinal - 1) + lane]
        query = folder / "official-query.json"
        query.write_text(json.dumps(selected["row"], ensure_ascii=False) + "\n")
        task["task_id"] = f"validation-{selected['row_idx']}"
        task["prompt"] = selected["row"]["query"]
        task["source"]["row"] = selected["row_idx"]
        command = task["verification"]["command"]
        command[command.index("--query") + 1] = str(query.resolve())
        assets.append(str(query.resolve()))
    elif workload == "ScientificComputation":
        if "problem" in task:
            size, responses = task["problem"]["size"], task["problem"]["responses"]
        else:
            size = len(task["verification"]["matrix"])
            responses = len(task["verification"]["rhs"][0])
        size += ordinal if size < 128 else 8 * (ordinal - 4)
        responses += ordinal % 3
        seed = 21000 + sum(parent["task_id"].encode()) * 31 + ordinal
        rng = np.random.default_rng(seed)
        design = rng.normal(size=(size, size)) / np.sqrt(size)
        matrix = design + 2 * np.eye(size) if "problem" in task else design.T @ design + 2 * np.eye(size)
        rhs = rng.normal(size=(size, responses))
        problem = folder.name
        path = folder / (problem + ".npz")
        np.savez(path, A=matrix, B=rhs)
        task["task_id"] = problem
        task["report_values"] = False
        task["readiness"]["inputs"]["files"] = [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest(path),
                "rows": size,
                "columns": size,
                "features": responses,
            }
        ]
        task["verification"] = {"input_npz": str(path.resolve()), "basis": "independent solve and residual"}
        task["problem"] = {
            "size": size,
            "responses": responses,
            "seed": seed,
            "synthetic_input": True,
            "source_family": parent["task_id"],
        }
        factor = "singular-value decomposition" if original.get("problem") else "factorization or inverse"
        task["prompt"] = (
            f"Solve A X = B for input problem {problem}. A is {size} by {size}; B has {responses} response vectors. "
            f"Load the supplied tensors, retain the {factor}, and reuse it across the response vectors. "
            "Store the solution tensor and report its name. Use summary_only=true for large matrix operations."
        )
        assets = [str(path.resolve())]
    else:
        if ordinal > len(PREFERENCES) * len(CONTEXTS):
            raise ValueError("Insufficient predeclared request variations; do not generate superficial IDs")
        task["prompt"] += (
            "\n\n" + PREFERENCES[(ordinal - 1) % len(PREFERENCES)] + " " + CONTEXTS[(ordinal - 1) // len(PREFERENCES)]
        )
        task["request_derivation"] = {
            "basis": "curated communication and evidence preferences; original inputs and business requirements preserved",
            "source_task": original["task_id"],
            "variant": ordinal,
            "observed_user_prompt": False,
        }
    task["prompt_sha256"] = hashlib.sha256(task["prompt"].encode()).hexdigest()
    return task, assets
