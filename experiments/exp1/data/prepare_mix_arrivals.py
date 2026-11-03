"""Seal E2 arrival assignments from validated Mix tasks and original Copilot windows."""

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import random

from atsuite.workload import describe
from atsuite.workload.descriptor import family_key
from experiments.exp1.data.prepare_e1b import allocate
from experiments.exp1.data.prepare_mix import sha
from experiments.exp1.runner.mix import write
from experiments.exp1.runner.validate_arrivals import load_span

TOOLS = ["prepare_workspace", "execute_analysis", "merge_artifacts", "validate_result"]


def assign(tasks: list[dict], windows: list[tuple], seed: int, composition: str) -> list[dict]:
    """Assign original prompts before execution; repetition and family novelty are separate."""
    if type(seed) is not int or composition not in {"stable", "shift"} or not tasks:
        raise ValueError("Invalid mixture")
    descriptors = {t["task_id"]: describe(prompt=t["prompt"], files=t["readiness"]["inputs"]["files"],
                                          tools=TOOLS) for t in tasks}
    families = {key: family_key(value) for key, value in descriptors.items()}
    family_order = sorted(set(families.values()))
    first_group = set(family_order[:max(1, len(family_order) // 2)])
    offset = 0
    rows, seen_prompts, seen_families = [], set(), set()
    by_id = {t["task_id"]: t for t in tasks}
    for segment, (arrivals, duration_ms) in enumerate(windows):
        weights = dict.fromkeys(by_id, 1.0)
        if composition == "shift":
            favor_first = segment < len(windows) // 2
            weights = {key: 4.0 if (family in first_group) == favor_first else 1.0
                       for key, family in families.items()}
        assigned = allocate(weights, len(arrivals), random.Random(f"{seed}:{segment}"))
        for arrival, key in zip(arrivals, assigned):
            task, family = by_id[key], families[key]
            rows.append({**vars(arrival), "arrival_id": f"s{segment}-{arrival.arrival_id}",
                         "relative_arrival_ms": str(Decimal(offset) + Decimal(arrival.relative_arrival_ms)),
                         "segment": segment, "task_id": key, "prompt_sha256": task["prompt_sha256"],
                         "family": family, "first_prompt": task["prompt_sha256"] not in seen_prompts,
                         "first_family": family not in seen_families})
            seen_prompts.add(task["prompt_sha256"])
            seen_families.add(family)
        offset += duration_ms
    return rows


def prepare(mix: Path, traffic: Path, output: Path, *, roles: list[str], seed: int,
            composition: str, duration_ms: int = 600000, experiment: str = "E2-B") -> dict:
    """Require 100 independently verified prompts; freeze assignments and provenance."""
    if experiment not in {"E2-A", "E2-B"}:
        raise ValueError("Unknown experiment protocol")
    manifest = json.loads((mix / "manifest.json").read_text())
    seal = json.loads((mix / "seal.json").read_text())
    if sha((mix / "manifest.json").read_bytes()) != seal["manifest_sha256"]:
        raise ValueError("Mix seal mismatch")
    hashes = {t["prompt_sha256"] for t in manifest["tasks"]}
    if (not manifest["formal_experiment_ready"] or len(hashes) < 100
            or any(not t.get("agent_validation") for t in manifest["tasks"])):
        raise ValueError("Formal E2 requires at least 100 distinct Agent-validated prompts")
    windows, provenance = [], []
    for role in roles:
        arrivals, source = load_span(traffic, role, 0, duration_ms)
        windows.append((arrivals, duration_ms))
        provenance.append(source)
    rows = assign(manifest["tasks"], windows, seed, composition)
    if not rows:
        raise ValueError("No arrivals")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "arrivals.json", rows)
    result = {"experiment": experiment, "composition": composition, "seed": seed,
              "mix_manifest": str((mix / "manifest.json").resolve()),
              "mix_manifest_sha256": seal["manifest_sha256"], "sources": provenance,
              "arrivals_sha256": sha((output / "arrivals.json").read_bytes()),
              "duration_ms": len(roles) * duration_ms, "requests": len(rows),
              "distinct_prompts": len({r["prompt_sha256"] for r in rows}),
              "task_counts": dict(Counter(r["task_id"] for r in rows)),
              "initial_history": "Empty target-family workload/provider histories for E2-A; never import capture profiles",
              "traffic_transform": "Original intervals within windows; explicitly concatenated windows, including repeated windows",
              "routing_features": "Original prompt, prepared input metadata and supported tools only; no task_id or completed DAG"}
    write(output / "manifest.json", result)
    write(output / "seal.json", {"manifest_sha256": sha((output / "manifest.json").read_bytes())})
    return result


def main() -> None:
    """Create a stable or changing long-run request sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--traffic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", default=["typical"] * 6)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--composition", choices=["stable", "shift"], default="stable")
    parser.add_argument("--experiment", choices=["E2-A", "E2-B"], default="E2-B")
    args = parser.parse_args()
    result = prepare(args.mix, args.traffic, args.output, roles=args.roles, seed=args.seed,
                     composition=args.composition, experiment=args.experiment)
    print(json.dumps({k: v for k, v in result.items() if k not in {"sources", "task_counts"}}))


if __name__ == "__main__":
    main()
