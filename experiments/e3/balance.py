"""Reassign tasks within a sealed arrival stream to balance measured deployment penalties."""

import argparse
import bisect
import csv
import json
import random
from collections import Counter
from decimal import Decimal
from pathlib import Path

from experiments.e3.manifest import digest, load

TOLERANCE_CNY = Decimal("0.00000001")


def balanced_counts(pairs: list[dict], original: dict[str, int]) -> dict[str, int]:
    """Keep every task and workload quota while balancing FC/MCP excess costs."""
    ids = [p["task"] for p in pairs]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(original):
        raise ValueError("Exactly one measured pair per original task is required")
    if any(type(n) is not int or n < 1 for n in original.values()):
        raise ValueError("Every original task needs a positive integer request count")
    costs = [
        (Decimal(str(p["fc_cost_cny"])), Decimal(str(p["mcp_cost_cny"]))) for p in pairs
    ]
    if any(not c.is_finite() or c < 0 for row in costs for c in row):
        raise ValueError("Finite nonnegative measured CNY costs are required")
    deltas = [fc - mcp for fc, mcp in costs]
    if not any(d > 0 for d in deltas) or not any(d < 0 for d in deltas):
        raise ValueError("Both deployment cost preferences are required")
    scale = 10 ** max(0, max(-d.as_tuple().exponent for d in deltas))
    units = [int(d * scale) for d in deltas]
    counts = [original[t] for t in ids]
    moves = [(0, 0, 0)] + [
        (units[j] - units[i], i, j)
        for i, left in enumerate(pairs)
        for j, right in enumerate(pairs)
        if i != j and left["workload"] == right["workload"]
    ]
    two_moves = sorted(
        (a[0] + b[0], a[1:], b[1:]) for i, a in enumerate(moves) for b in moves[i:]
    )
    values = [m[0] for m in two_moves]
    # shortcut: bounded four-transfer refinement; use an integer solver if a future matrix misses tolerance.
    for _ in range(3):
        net = sum(n * d for n, d in zip(counts, units))
        if Decimal(abs(net)) / scale <= TOLERANCE_CNY:
            return dict(zip(ids, counts))
        best, change = abs(net), None
        for first in two_moves:
            position = bisect.bisect_left(values, -net - first[0])
            for step, index in [(-1, position - 1), (1, position)]:
                while 0 <= index < len(two_moves):
                    second = two_moves[index]
                    error = abs(net + first[0] + second[0])
                    if error >= best:
                        break
                    candidate = Counter()
                    for source, target in [*first[1:], *second[1:]]:
                        candidate[source] -= 1
                        candidate[target] += 1
                    if all(counts[i] + n >= 1 for i, n in candidate.items()):
                        best, change = error, candidate
                        break
                    index += step
        if change is None:
            break
        for i, n in change.items():
            counts[i] += n
    if Decimal(abs(sum(n * d for n, d in zip(counts, units)))) / scale > TOLERANCE_CNY:
        raise ValueError(
            "Four-transfer refinement did not reach the declared cost-balance tolerance"
        )
    return dict(zip(ids, counts))


def build(sequence: Path, matrix: Path, output: Path) -> dict:
    """Derive a separate sealed trace without modifying tasks, captures or arrival timing."""
    if output.exists():
        raise FileExistsError(output)
    source = load(sequence)
    pairs = json.loads(matrix.read_text(), parse_float=Decimal)["pairs"]
    tasks = {t["id"]: t for t in source["tasks"]}
    if set(tasks) != {p["task"] for p in pairs} or any(
        tasks[p["task"]]["workload"] != p["workload"]
        or tasks[p["task"]]["task_id"] != p["task_id"]
        for p in pairs
    ):
        raise ValueError("Cost matrix does not match the sealed workload tasks")
    before = Counter(r["task"] for r in source["requests"])
    counts = balanced_counts(pairs, before)
    requests = [dict(r) for r in source["requests"]]
    rng = random.Random(source["seed"])
    for workload in dict.fromkeys(t["workload"] for t in tasks.values()):
        slots, replacements = [], []
        for task, definition in tasks.items():
            if definition["workload"] != workload:
                continue
            surplus = before[task] - counts[task]
            if surplus > 0:
                slots.extend(
                    rng.sample(
                        [i for i, r in enumerate(requests) if r["task"] == task],
                        surplus,
                    )
                )
            elif surplus < 0:
                replacements.extend([task] * -surplus)
        rng.shuffle(slots)
        rng.shuffle(replacements)
        if len(slots) != len(replacements):
            raise ValueError("Workload quotas changed during allocation")
        for index, task in zip(slots, replacements):
            requests[index]["task"] = task
    assert Counter(r["task"] for r in requests) == counts
    assert all(
        {k: v for k, v in a.items() if k != "task"}
        == {k: v for k, v in b.items() if k != "task"}
        for a, b in zip(source["requests"], requests)
    )
    totals = {
        "fixed_fc_cny": sum(
            counts[p["task"]] * Decimal(str(p["fc_cost_cny"])) for p in pairs
        ),
        "fixed_mcp_cny": sum(
            counts[p["task"]] * Decimal(str(p["mcp_cost_cny"])) for p in pairs
        ),
        "fc_excess_on_mcp_cheaper_tasks_cny": sum(
            counts[p["task"]]
            * max(
                Decimal(str(p["fc_cost_cny"])) - Decimal(str(p["mcp_cost_cny"])),
                Decimal(0),
            )
            for p in pairs
        ),
        "mcp_excess_on_fc_cheaper_tasks_cny": sum(
            counts[p["task"]]
            * max(
                Decimal(str(p["mcp_cost_cny"])) - Decimal(str(p["fc_cost_cny"])),
                Decimal(0),
            )
            for p in pairs
        ),
    }
    gap = abs(totals["fixed_fc_cny"] - totals["fixed_mcp_cny"])
    assert gap <= TOLERANCE_CNY
    report = {
        "scope": "Profile-conditioned balanced experiment, designed using measured Ali FC/Session MCP costs; estimates are not a cloud replay or a held-out result. AWS costs are not balanced by this allocation.",
        "source_sequence": str(sequence.resolve()),
        "source_manifest_sha256": digest(sequence / "manifest.json"),
        "cost_matrix": str(matrix.resolve()),
        "cost_matrix_sha256": digest(matrix),
        "seed": source["seed"],
        "requests": len(requests),
        "duration_ms": source["duration_ms"],
        "distinct_tasks": len(tasks),
        "workload_requests": dict(Counter(r["workload"] for r in requests)),
        "requests_by_cheaper_mode": {
            "fc": sum(
                counts[p["task"]]
                for p in pairs
                if Decimal(str(p["fc_cost_cny"])) < Decimal(str(p["mcp_cost_cny"]))
            ),
            "mcp": sum(
                counts[p["task"]]
                for p in pairs
                if Decimal(str(p["mcp_cost_cny"])) < Decimal(str(p["fc_cost_cny"]))
            ),
        },
        "estimated_costs": {k: format(v, "f") for k, v in totals.items()},
        "absolute_balance_gap_cny": format(gap, "f"),
        "tolerance_cny": str(TOLERANCE_CNY),
        "task_counts": [
            {
                "task": p["task"],
                "task_id": p["task_id"],
                "workload": p["workload"],
                "before": before[p["task"]],
                "after": counts[p["task"]],
            }
            for p in pairs
        ],
        "changed_requests": [
            {
                "arrival_id": a["arrival_id"],
                "old_task": a["task"],
                "new_task": b["task"],
            }
            for a, b in zip(source["requests"], requests)
            if a["task"] != b["task"]
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    allocation = output / "balance.json"
    allocation.write_text(json.dumps(report, indent=2) + "\n")
    files = {
        **source["files"],
        **{
            str(p.resolve()): digest(p)
            for p in [
                sequence / "manifest.json",
                sequence / "seal.json",
                matrix,
                allocation,
                Path(__file__),
            ]
        },
    }
    value = {
        **source,
        "requests": requests,
        "files": files,
        "assignment": {
            "method": "measured_cost_balance_within_workload",
            "report": str(allocation.resolve()),
        },
    }
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    (output / "seal.json").write_text(
        json.dumps({"manifest_sha256": digest(manifest)}) + "\n"
    )
    with (output / "requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(requests[0]))
        writer.writeheader()
        writer.writerows(requests)
    summary = {
        k: report[k]
        for k in [
            "scope",
            "requests",
            "duration_ms",
            "distinct_tasks",
            "workload_requests",
            "requests_by_cheaper_mode",
            "estimated_costs",
            "absolute_balance_gap_cny",
        ]
    }
    summary.update(
        changed_requests=len(report["changed_requests"]),
        manifest_sha256=digest(manifest),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.sequence, args.matrix, args.output), indent=2))


if __name__ == "__main__":
    main()
