"""Prepare additional real compute/state coverage without modifying the original E3 corpus."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.e3.prepare import definitions


def prepare_scientific(output: Path, sizes: list[int]) -> dict:
    """Create dense full-rank inverse problems with reusable spectral factors and response batches."""
    if (
        not sizes
        or len(sizes) != len(set(sizes))
        or any(n < 2 or n > 4096 for n in sizes)
    ):
        raise ValueError("Declare unique matrix sizes in [2, 4096]")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    names = {
        "scicom_load_tensor",
        "scicom_svd_decompose",
        "scicom_solve_from_svd",
        "scicom_view_tensor",
    }
    tools = [
        d
        for d in definitions(
            ["benchmarks.ScientificComputation.nodes.tensor_workspace.implementation"]
        )
        if d["name"] in names
    ]
    from atsuite_sdk.abstract import registry

    access = {d["name"]: registry.functions[d["name"]].access for d in tools}
    tasks, assets = [], []
    for size in sizes:
        for seed in (1103, 1104):
            rng = np.random.default_rng(seed)
            matrix = rng.normal(size=(size, size)) / np.sqrt(size) + 2 * np.eye(size)
            rhs = rng.normal(size=(size, 16))
            task_id = f"spectral-{size}-16-seed-{seed}"
            path = inputs / (task_id + ".npz")
            np.savez(path, A=matrix, B=rhs)
            prompt = (
                f"Solve the dense inverse problem A X = B in immutable problem {task_id}. "
                f"A is {size} by {size}, and B contains 16 response vectors. "
                "The response batches share the same forward operator. Compute and retain its singular-value "
                "decomposition in the tensor workspace, then reuse the stored factors to solve for all responses. "
                "Keep the factors for subsequent analysis. Use summary_only=true for the decomposition; "
                "do not print factor arrays. Store the solution and return its tensor name."
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
                                    "sha256": hashlib.sha256(
                                        path.read_bytes()
                                    ).hexdigest(),
                                }
                            ]
                        }
                    },
                    "tool_definitions": tools,
                "tool_access": access,
                    "verification": {
                        "input_npz": str(path.resolve()),
                        "basis": "independent LU solve and original-system residual; no SVD recomputation",
                    },
                    "problem": {
                        "kind": "dense full-rank multiple-response inverse problem",
                        "size": size,
                        "responses": 16,
                        "seed": seed,
                        "synthetic_input": True,
                        "scope": "new scientific problem, not an upstream benchmark task; no idle padding",
                    },
                }
            )
            assets.append(str(path.resolve()))
    manifest = {"tasks": tasks, "assets": assets}
    (output / "ScientificComputation.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2560, 3072])
    args = parser.parse_args()
    manifest = prepare_scientific(args.output, args.sizes)
    print(
        json.dumps(
            {
                "tasks": len(manifest["tasks"]),
                "input_bytes": sum(Path(p).stat().st_size for p in manifest["assets"]),
            }
        )
    )


if __name__ == "__main__":
    main()
