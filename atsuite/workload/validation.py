"""Causal fixed-trace validation; never present replay estimates as online routing gains."""

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

from atsuite.provider_model import ProviderModel
from atsuite.routing import OnlineRouter
from atsuite.workload import WorkloadModel
from atsuite.workload.descriptor import compatible
from atsuite.workload.model import distribution
from atsuite.workload.profile import METRICS, number
from atsuite.workload.retrieval import ContextBackend, ContextRetriever


def behavior_distance(left: dict, right: dict) -> float | None:
    """Observed joint demand distance for offline supervision, never an arrival feature."""
    parts = [
        abs(math.log1p(left["metrics"][k]) - math.log1p(right["metrics"][k]))
        for k in (
            "tool_elapsed_s",
            "state_bytes",
            "state_reads",
            "state_writes",
            "state_max_object_bytes",
            "llm_idle_s",
        )
        if number(left["metrics"].get(k)) and number(right["metrics"].get(k))
    ]
    for key in ("tool_calls", "causal_revisits"):
        a, b = left["workflow"].get(key, {}), right["workflow"].get(key, {})
        parts.extend(
            abs(math.log1p(a.get(n, 0)) - math.log1p(b.get(n, 0)))
            for n in set(a) | set(b)
        )

    def transitions(p):
        nodes = {n["id"]: n["name"] for n in p["workflow"].get("nodes", [])}
        return Counter((nodes[a], nodes[b]) for a, b in p["workflow"].get("edges", []))

    a, b = transitions(left), transitions(right)
    parts.extend(abs(math.log1p(a[n]) - math.log1p(b[n])) for n in a.keys() | b.keys())
    x, y = left["workflow"].get("parallel_peak"), right["workflow"].get("parallel_peak")
    if number(x) and number(y):
        parts.append(abs(math.log1p(x) - math.log1p(y)))
    return sum(parts) / len(parts) if parts else None


def supervision_pairs(records: list[dict]) -> list[dict]:
    """Export causal compatible context pairs labeled by realized execution distance."""
    pairs = []
    for row in records:
        query = row["profile"]
        if query["status"] != "success":
            continue
        for historical in records:
            p = historical["profile"]
            if (
                p["status"] == "success"
                and historical["available_ns"] < query["arrival_ns"]
                and compatible(query.get("descriptor"), p.get("descriptor"))
            ):
                pairs.append(
                    {
                        "query_id": query["request_id"],
                        "history_id": p["request_id"],
                        "query": query["descriptor"],
                        "history": p["descriptor"],
                        "behavior_distance": behavior_distance(query, p),
                        "label_available_ns": row["available_ns"],
                    }
                )
    return pairs


def evaluate(
    records: list[dict], config: dict, *, backend_factory=None, frozen_baseline=None
) -> dict:
    """Replay arrivals before feedback at ties; score only genuine matched outcomes.

    Each record is a ProviderModel request row plus optional outcomes keyed by pool.
    Outcomes contain cost, latency_s, status in config currency, from matched runs.
    All variants see the same fixed history; this is not policy-dependent execution.
    """
    records = copy.deepcopy(records)
    ids = [r["profile"]["request_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(
            "Replay needs one record per logical request, with matched outcomes attached"
        )
    for r in records:
        if (
            r["kind"] != "request"
            or not number(r["profile"].get("arrival_ns"))
            or not number(r.get("available_ns"))
            or not number(r["profile"].get("completed_ns"))
            or r["available_ns"] < r["profile"]["completed_ns"]
            or r["profile"]["completed_ns"] < r["profile"]["arrival_ns"]
        ):
            raise ValueError(
                "Replay requires causal arrival/completion/availability times"
            )
    held = set(config.get("held_tasks", []))
    methods = {}
    for method in ("context", "embedding", "oracle_demand"):
        backend = (
            backend_factory() if backend_factory and method != "oracle_demand" else None
        )
        workload = WorkloadModel(
            retriever=ContextRetriever(backend, rerank=method != "embedding")
        )
        provider = ProviderModel(tariffs=config["tariffs"])
        router = OnlineRouter(
            workload,
            provider,
            config["pools"],
            dict.fromkeys(config["pools"], 10**9),
            default_pool=config["default_pool"],
            execution_bounds=config["execution_bounds"],
            latency_weight=config.get("latency_weight", 0.001),
            fx=config.get("fx"),
            enable_context_routing=True,
        )
        events = [(r["profile"]["arrival_ns"], 0, i) for i, r in enumerate(records)]
        events += [(r["available_ns"], 1, i) for i, r in enumerate(records)]
        predictions = {}
        for stamp, event, index in sorted(events):
            row = records[index]
            profile = row["profile"]
            rid = profile["request_id"]
            descriptor = profile.get("descriptor")
            if event:
                if row.get("task_id") not in held:
                    workload.update(profile, stamp)
                    provider.update({k: v for k, v in row.items() if k != "outcomes"})
                continue
            if held and row.get("task_id") not in held:
                continue
            if method == "oracle_demand":
                # Deliberate future-information upper reference, isolated from all learned histories.
                prediction = {
                    "joint_profiles": [profile],
                    "sample_count": 1,
                    "low_confidence": False,
                    "retrieval": {"cost": 0, "currency": None, "latency_s": 0},
                    "descriptor": descriptor,
                    "metrics": {
                        k: distribution([profile["metrics"][k]], 1)
                        if number(profile["metrics"].get(k))
                        else distribution([], 1)
                        for k in METRICS
                    },
                }
                original = workload.predict
                workload.predict = lambda *a, prediction=prediction, **kw: prediction
                decision = router.decide(
                    rid,
                    as_of_ns=stamp,
                    descriptor=descriptor,
                    shadow=True,
                    load_rps=row["load_rps"],
                )
                workload.predict = original
                workload.pending.pop(rid, None)
            else:
                decision = router.decide(
                    rid,
                    as_of_ns=stamp,
                    descriptor=descriptor,
                    shadow=True,
                    load_rps=row["load_rps"],
                )
                prediction = decision["workload"]
            errors, coverage, width = {}, {}, {}
            for key in METRICS:
                actual, estimate = (
                    profile["metrics"].get(key),
                    prediction["metrics"][key]["p50"],
                )
                errors[key] = (
                    abs(actual - estimate)
                    if number(actual) and number(estimate)
                    else None
                )
                interval = (
                    prediction.get("uncertainty", {}).get("intervals", {}).get(key, {})
                )
                lo, hi = interval.get("lower"), interval.get("upper")
                coverage[key] = (
                    lo <= actual <= hi
                    if number(actual) and number(lo) and number(hi)
                    else None
                )
                width[key] = hi - lo if number(lo) and number(hi) else None
            predictions[rid] = {
                "selected": decision["selected"],
                "sample_count": prediction["sample_count"],
                "cold_start": prediction["sample_count"] == 0,
                "absolute_error": errors,
                "interval_covered": coverage,
                "interval_width": width,
                "overhead": decision["prediction_overhead"],
                "joint_behavior_error": distribution(
                    [
                        d
                        for p in prediction["joint_profiles"]
                        if (d := behavior_distance(profile, p)) is not None
                    ],
                    prediction["sample_count"],
                ),
            }
        methods[method] = {
            "predictions": predictions,
            "retrieval_available": backend is not None,
            "cold_start_rate": sum(p["cold_start"] for p in predictions.values())
            / len(predictions)
            if predictions
            else None,
            "interval_width": {
                k: distribution(
                    [
                        p["interval_width"][k]
                        for p in predictions.values()
                        if number(p["interval_width"][k])
                    ],
                    len(predictions),
                )
                for k in METRICS
            },
            "behavior_error": {
                k: distribution(
                    [
                        p["absolute_error"][k]
                        for p in predictions.values()
                        if number(p["absolute_error"][k])
                    ],
                    len(predictions),
                )
                for k in METRICS
            },
            "uncertainty_coverage": {
                k: distribution(
                    [
                        int(p["interval_covered"][k])
                        for p in predictions.values()
                        if p["interval_covered"][k] is not None
                    ],
                    len(predictions),
                )
                for k in METRICS
            },
        }
    if frozen_baseline is not None:
        methods["frozen_legacy"] = {"predictions": copy.deepcopy(frozen_baseline)}
    weight = config.get("latency_weight", 0.001)

    def score(outcome):
        return outcome["cost"] + weight * outcome["latency_s"]

    scored = [
        r
        for r in records
        if r["profile"]["request_id"] in methods["context"]["predictions"]
        and set(r.get("outcomes", {})) == set(config["pools"])
        and all(
            number(o.get("cost"))
            and number(o.get("latency_s"))
            and o.get("status") in {"success", "failed"}
            for o in r["outcomes"].values()
        )
    ]
    fixed = (
        min(
            config["pools"],
            key=lambda k: (
                sum(r["outcomes"][k]["status"] != "success" for r in scored),
                sum(score(r["outcomes"][k]) for r in scored),
            ),
        )
        if scored
        else None
    )
    for method in methods.values():
        realized = []
        for row in scored:
            p = method["predictions"].get(row["profile"]["request_id"], {})
            selected = p.get("selected")
            if selected not in row["outcomes"]:
                continue
            overhead = p.get(
                "overhead", {"cost": None, "total_latency_s": None, "currency": None}
            )
            if overhead["cost"] and overhead.get("currency") != config["currency"]:
                continue
            if not number(overhead["cost"]) or not number(overhead["total_latency_s"]):
                continue
            out = row["outcomes"][selected]
            cost = out["cost"] + overhead["cost"]
            latency = out["latency_s"] + overhead["total_latency_s"]
            successful = [
                o for o in row["outcomes"].values() if o["status"] == "success"
            ]
            best = min((score(o) for o in successful), default=None)
            realized.append(
                {
                    "request_id": row["profile"]["request_id"],
                    "cost": cost,
                    "latency_s": latency,
                    "failed": out["status"] != "success",
                    "routing_loss": cost + weight * latency - best
                    if out["status"] == "success" and best is not None
                    else None,
                }
            )
        method["matched_replay"] = {
            "scored_requests": len(realized),
            "requests": realized,
            "cost": sum(r["cost"] for r in realized) if realized else None,
            "latency_s": distribution(
                [r["latency_s"] for r in realized], len(realized)
            ),
            "failures": sum(r["failed"] for r in realized),
            "routing_loss": distribution(
                [r["routing_loss"] for r in realized if number(r["routing_loss"])],
                len(realized),
            ),
        }
    return {
        "scope": "fixed-trace historical replay; not real online routing",
        "methods": methods,
        "currency": config["currency"],
        "matched_requests": len(scored),
        "best_fixed_pool": fixed,
        "best_fixed_objective": sum(score(r["outcomes"][fixed]) for r in scored)
        if fixed
        else None,
        "best_fixed_failures": sum(
            r["outcomes"][fixed]["status"] != "success" for r in scored
        )
        if fixed
        else None,
        "matched_oracle_objective": sum(
            min(score(o) for o in r["outcomes"].values() if o["status"] == "success")
            for r in scored
        )
        if scored
        and all(
            any(o["status"] == "success" for o in r["outcomes"].values())
            for r in scored
        )
        else None,
        "online_routing_results": None,
        "legacy_baseline": "frozen results supplied"
        if frozen_baseline is not None
        else "not supplied; use archived commit results",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervision", type=Path)
    args = parser.parse_args()
    records, config = (
        json.loads(args.records.read_text()),
        json.loads(args.config.read_text()),
    )
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    backend = (
        (lambda: ContextBackend(config["retrieval"]))
        if config.get("retrieval")
        else None
    )
    report = evaluate(
        records, config, backend_factory=backend, frozen_baseline=baseline
    )
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    if args.supervision:
        args.supervision.write_text(
            json.dumps(
                supervision_pairs(records),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
