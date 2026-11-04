"""Measured E3 matrices and two-arm ATSuite usage-price comparison."""

from collections import Counter, defaultdict

from atsuite.workload.profile import number
from experiments.exp1.runner.arrival_scheduler import stats


def matrix(rows: list[dict]) -> dict:
    """Aggregate paired completed attempts; missing price evidence stays unknown."""
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (row["workload"], row["task"], row["pool"], row["repeat"])
        if identity in seen or not number(row["latency_s"]) or type(row["success"]) is not bool:
            raise ValueError("Duplicate or malformed profiling attempt")
        seen.add(identity)
        if row.get("actual_cost") is not None and (
            not number(row["actual_cost"]) or not row.get("billing_evidence") or not row.get("currency")
        ):
            raise ValueError("Actual costs require billing evidence")
        groups[row["workload"], row["pool"]].append(row)
    result = {}
    for (workload, pool), batch in groups.items():
        priced = all(r.get("actual_cost") is not None for r in batch)
        currencies = {r.get("currency") for r in batch}
        result.setdefault(workload, {})[pool] = {
            "attempts": len(batch),
            "success_rate": sum(r["success"] for r in batch) / len(batch),
            "latency_s": stats([r["latency_s"] for r in batch]),
            "mean_actual_cost": sum(r["actual_cost"] for r in batch) / len(batch)
            if priced and len(currencies) == 1
            else None,
            "currency": next(iter(currencies)) if len(currencies) == 1 else None,
            "paired_tasks": sorted({r["task"] for r in batch}),
        }
    return result


def compare(runs: list[dict]) -> dict:
    """Compare exactly the fixed and exploration arms using ATSuite-priced usage."""
    if len(runs) != 2 or {r["strategy"] for r in runs} != {"fixed", "exploration"}:
        raise ValueError("Exactly fixed and exploration runs are required")
    fixed = next(r for r in runs if r["strategy"] == "fixed")
    routed = next(r for r in runs if r["strategy"] == "exploration")
    if any(routed[k] != fixed[k] for k in ("manifest_sha256", "native_elasticity")):
        raise ValueError("Unmatched inputs or native elasticity")
    ids = {r["request_id"] for r in fixed["requests"]}
    if not ids or any(len(r["requests"]) != len(ids) or {q["request_id"] for q in r["requests"]} != ids for r in runs):
        raise ValueError("Unmatched or duplicate logical requests")
    reports = []
    for run in runs:
        priced = run.get("pricing", {})
        if priced.get("basis") != "atsuite_metered_usage_pricing":
            raise ValueError("E3 requires ATSuite pricing of native measurements")
        by_workload = defaultdict(Counter)
        for row in run["requests"]:
            by_workload[row["workload"]][row.get("pool") or "rejected"] += 1
        reports.append(
            {
                "strategy": run["strategy"],
                "pricing": priced,
                "success_rate": sum(r["success"] for r in run["requests"]) / len(ids),
                "latency_s": stats([r["latency_s"] for r in run["requests"]]),
                "workload_routes": dict(by_workload),
            }
        )
    baseline = fixed.get("pricing", {}).get("total_cny")
    online = routed.get("pricing", {}).get("total_cny")
    complete = all(r.get("pricing", {}).get("complete") for r in runs)
    return {
        "runs": reports,
        "fixed_provider": fixed.get("fixed_provider"),
        "savings_vs_fixed": (baseline - online) / baseline if complete and baseline and online is not None else None,
        "scope": "ATSuite native usage pricing, with explicit traced-state scope; not a vendor invoice",
    }
