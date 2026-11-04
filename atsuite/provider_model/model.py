"""Provider-specific empirical response surfaces with causal observation cutoffs."""

import copy
import json
import math
import threading
import time
from collections import Counter
from pathlib import Path

from atsuite.provider_model.analytical import predict_cost, validate_tariff
from atsuite.provider_model.observation import (
    BACKENDS,
    MEASUREMENTS,
    execution_context,
    features,
    trace_measurements,
    validate_observation,
    validate_pool,
)
from atsuite.tracing.storage import read_text, write_json
from atsuite.workload.descriptor import compatibility_key, compatible
from atsuite.workload.model import distribution
from atsuite.workload.profile import number


def _distribution(values: list[float], total: int) -> dict:
    result = distribution(values, total)
    values = result["samples"]
    position = (len(values) - 1) * 0.95
    lo = int(position)
    result["p95"] = (
        (values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)) if values else None
    )
    return result


def _compatible(left: dict, right: dict) -> bool:
    return all(
        left.get(k) == right.get(k)
        for k in (
            "provider",
            "environment",
            "resources",
            "sessions_per_instance",
            "invocations_per_instance",
        )
    )


def _nearest(rows: list[dict], query: dict, neighbors: int) -> tuple[list[dict], dict]:
    query = dict(query)
    vectors = [dict(r["_features"]) for r in rows]
    keys = sorted(k for k in set(query).union(*(set(v) for v in vectors)) if not k.startswith("_"))
    for vector in [query, *vectors]:
        if vector.get("_workflow_counts_complete"):
            for key in keys:
                if key.startswith(("calls:", "tool_calls:", "causal_revisits:", "edge:")):
                    vector.setdefault(key, 0)
    ranges, missing, outside = {}, [], []
    for key in keys:
        values = [v[key] for v in vectors if number(v.get(key))]
        ranges[key] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "coverage": len(values) / len(rows) if rows else 0,
        }
        if not number(query.get(key)) or not values:
            missing.append(key)
        elif query[key] < min(values) or query[key] > max(values):
            outside.append(key)
    active = [k for k in keys if k not in missing]

    def distance(vector):
        parts = []
        for key in active:
            if not number(vector.get(key)):
                parts.append(1.0)
                continue
            extent = ranges[key]
            scale = max(1.0, math.log1p(extent["max"]) - math.log1p(extent["min"]))
            parts.append(((math.log1p(vector[key]) - math.log1p(query[key])) / scale) ** 2)
        return sum(parts) / len(parts) if parts else 0

    ranked = sorted(
        zip(rows, vectors),
        key=lambda pair: (distance(pair[1]), -pair[0]["completed_ns"]),
    )
    return [row for row, _ in ranked[:neighbors]], {
        "feature_ranges": ranges,
        "missing_features": missing,
        "extrapolated_features": outside,
        "nearest_distance": distance(ranked[0][1]) if ranked else None,
    }


def target_confidence(coverage, unique_count, *, required, minimum, marginal=False) -> dict:
    """Missing optional regressors do not invalidate observed target evidence.

    Retain sample, required-feature coverage and observed-support checks. An
    unmeasured optional feature is disclosed, never assigned a fabricated value.
    """
    missing = sorted(
        {k for c in coverage for k in c["missing_features"] if k in required}
        | {k for c in coverage for k, v in c["feature_ranges"].items() if k in required and v["coverage"] < 1}
    )
    outside = sorted({k for c in coverage for k in c["extrapolated_features"]})
    optional = sorted(
        {k for c in coverage for k in c["missing_features"] if k not in required}
        | {k for c in coverage for k, v in c["feature_ranges"].items() if k not in required and v["coverage"] < 1}
    )
    reasons = []
    if not coverage or unique_count < minimum:
        reasons.append("sample_insufficient")
    if missing:
        reasons.append("metric_missing")
    if outside:
        reasons.append("distribution_outside_support")
    if marginal:
        reasons.append("joint_profile_missing")
    return {
        "reliable": not reasons,
        "reasons": reasons,
        "required_metrics": sorted(required),
        "missing_metrics": missing,
        "optional_missing": optional,
        "outside_support": outside,
        "unique_samples": unique_count,
        "minimum_samples": minimum,
    }


class ProviderModel:
    """Exact local samples, independent provider pools, no cloud side effects.

    A row is an immutable revision. Later provider evidence can enrich a request
    without becoming visible to predictions made before that evidence arrived.
    """

    def __init__(self, neighbors: int = 20, *, tariffs: dict | None = None, minimum_samples: int = 20) -> None:
        if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 1:
            raise ValueError("neighbors must be a positive integer")
        self.neighbors = neighbors
        if type(minimum_samples) is not int or minimum_samples < 1:
            raise ValueError("minimum_samples must be a positive integer")
        self.minimum_samples = minimum_samples
        self.tariffs = copy.deepcopy(tariffs or {})
        for provider, card in self.tariffs.items():
            validate_tariff(card, provider)
        # shortcut: keep exact revision history; bound retention when online volume warrants it.
        self.samples = {}
        self.lock = threading.RLock()

    def update(self, observation: dict) -> None:
        """Ingest a request/window once, or append a later evidence revision."""
        validate_observation(observation)
        row = copy.deepcopy(observation)
        if not row["profile"].get("reference_id"):
            row["profile"]["execution_pool"] = execution_context(row["pool"])
        row["_features"] = features(row["profile"], row["pool"], row["load_rps"])
        key = (
            row["pool"]["provider"],
            row["pool"]["environment"],
            row["kind"],
            row["id"],
        )
        with self.lock:
            history = self.samples.setdefault(key, [])
            if history:
                for field in (
                    "task_id",
                    "prompt_id",
                    "logical_id",
                    "capacity_transition",
                ):
                    if field in history[-1]:
                        value = history[-1][field]
                        if row.get(field, value) != value:
                            raise ValueError(f"Revision changes request context: {field}")
                        row[field] = value
            if row in history:
                return
            if history:
                previous = history[-1]
                if row["available_ns"] <= previous["available_ns"]:
                    raise ValueError("Conflicting or out-of-order observation revision")
                for field in (
                    "pool",
                    "profile",
                    "arrival_ns",
                    "completed_ns",
                    "load_rps",
                    "status",
                ):
                    if row[field] != previous[field]:
                        raise ValueError(f"Revision changes request identity/context: {field}")
            history.append(row)

    def observe(
        self,
        profile: dict,
        report: dict,
        pool: dict,
        load_rps: float,
        available_ns: int | None = None,
    ) -> None:
        """Completion hook for an Analyzer report and its associated workload profile."""
        from atsuite.provider_model.observation import observation_from_report

        self.update(
            observation_from_report(
                profile,
                report,
                pool,
                load_rps,
                time.time_ns() if available_ns is None else available_ns,
            )
        )

    def complete(
        self,
        trace: dict,
        pool: dict,
        load_rps: float,
        *,
        reference_profile: dict | None = None,
        available_ns: int | None = None,
    ) -> dict:
        """Record completion immediately; delayed Analyzer data can enrich it later."""
        from atsuite.workload.profile import extract_profile

        profile = extract_profile(trace)
        profile["execution_pool"] = execution_context(pool)
        if reference_profile is not None and not reference_profile.get("reference_id"):
            raise ValueError("An explicit reference profile requires its shared reference_id")
        row = {
            "kind": "request",
            "id": profile["request_id"],
            "profile": reference_profile if reference_profile is not None else profile,
            "status": profile["status"],
            "pool": pool,
            "load_rps": load_rps,
            "arrival_ns": trace["arrival_ns"],
            "completed_ns": trace["completed_ns"],
            "available_ns": time.time_ns() if available_ns is None else available_ns,
            "measurements": trace_measurements(trace),
            "cost": None,
        }
        self.update(row)
        return copy.deepcopy(row)

    def predict(
        self,
        workload: dict | list[dict],
        pools: list[dict],
        load_rps: float,
        *,
        horizon_s: float = 60,
        as_of_ns: int | None = None,
    ) -> dict:
        """Predict each pool independently, mixing raw samples rather than percentiles.

        Accepts a profile, joint profiles, or WorkloadModel.predict() output.
        Marginal-only inputs use their means and explicitly report the loss of
        joint information. Pool window costs are inclusive, never added to
        request costs. All output costs are forecasts, including invoice-trained ones.
        """
        as_of_ns = time.time_ns() if as_of_ns is None else as_of_ns
        if not number(as_of_ns) or not number(load_rps) or not number(horizon_s) or horizon_s == 0:
            raise ValueError("Invalid prediction time, load or horizon")
        marginal = False
        if isinstance(workload, list):
            profiles = workload
        elif "joint_profiles" in workload:
            profiles = workload["joint_profiles"]
        elif "sample_count" in workload:
            marginal = True
            profiles = [
                {
                    "metrics": {k: v.get("mean") for k, v in workload["metrics"].items()},
                    "workflow": {},
                }
            ]
        else:
            profiles = [workload]
        if len({compatibility_key(p.get("descriptor")) for p in profiles}) > 1:
            raise ValueError("Joint predictions must share one tool/Harness identity")
        with self.lock:
            visible = []
            for history in self.samples.values():
                eligible = [r for r in history if r["available_ns"] < as_of_ns]
                if eligible:
                    visible.append(eligible[-1])
        results = []
        for pool in pools:
            validate_pool(pool)
            candidates = [r for r in visible if _compatible(r["pool"], pool) and not r.get("capacity_transition")]
            descriptors = [p.get("descriptor") for p in profiles if p.get("descriptor") is not None]
            candidates = [
                r for r in candidates
                if any(compatible(d, r["profile"].get("descriptor")) for d in descriptors)
            ]
            requests = [r for r in candidates if r["kind"] == "request"]
            successes = [r for r in requests if r["status"] == "success"]
            cost_profiles = profiles
            queries = [features(p, pool, load_rps) for p in cost_profiles]
            # Identical resource queries keep their mixture weight without repeating neighbor searches.
            query_keys = [json.dumps(q, sort_keys=True) for q in queries]
            unique_queries = dict(zip(query_keys, queries))
            neighbors = {k: _nearest(successes, q, self.neighbors) for k, q in unique_queries.items()}
            selected, coverage = [], []
            for key in query_keys:
                rows, info = neighbors[key]
                selected.extend(rows)
                coverage.append(info)
            metrics = {
                key: _distribution(
                    [r["measurements"][key] for r in selected if number(r.get("measurements", {}).get(key))],
                    len(selected),
                )
                for key in MEASUREMENTS
            }
            capacity = self._capacity(pool, load_rps, metrics["service_slot_s"])
            windows = [r for r in candidates if r["kind"] == "window"]
            window_neighbors = {k: _nearest(windows, q, self.neighbors)[0] for k, q in unique_queries.items()}
            window_samples = [r for key in query_keys for r in window_neighbors[key]]
            capacity["observed_goodput_rps"] = _distribution(
                [
                    r["measurements"]["throughput_rps"]
                    for r in window_samples
                    if number(r["measurements"].get("throughput_rps"))
                ],
                len(window_samples),
            )
            card = pool.get("billing", self.tariffs.get(pool["provider"]))
            costs = []
            # Each scope is separate: partial subtotals never masquerade as a full bill.
            groups = (
                []
                if card
                else sorted(
                    {
                        (
                            r["kind"],
                            r["cost"]["scope"],
                            r["cost"]["basis"],
                            r["cost"]["complete"],
                        )
                        for r in candidates
                        if r.get("cost")
                    }
                )
            )
            for kind, scope, basis, complete in groups:
                source = [
                    r
                    for r in candidates
                    if r["kind"] == kind
                    and r.get("cost")
                    and (r["cost"]["scope"], r["cost"]["basis"], r["cost"]["complete"]) == (scope, basis, complete)
                ]
                cost_neighbors = {k: _nearest(source, q, self.neighbors) for k, q in unique_queries.items()}
                chosen, cost_coverage = [], []
                for key in query_keys:
                    rows, info = cost_neighbors[key]
                    chosen.extend(rows)
                    cost_coverage.append(info)
                values = []
                components = {}
                for row in chosen:
                    scale = horizon_s / ((row["completed_ns"] - row["arrival_ns"]) / 1e9) if kind == "window" else 1
                    values.append(sum(row["cost"]["components"].values()) * scale)
                    for category, value in row["cost"]["components"].items():
                        components.setdefault(category, []).append(value * scale)
                costs.append(
                    {
                        "kind": kind,
                        "scope": scope,
                        "currency": BACKENDS[pool["provider"]][2],
                        "basis": "predicted",
                        "training_basis": basis,
                        "complete": complete,
                        "subtotal": _distribution(values, len(chosen)),
                        "total": _distribution(values, len(chosen)) if complete else None,
                        "components": {k: _distribution(v, len(chosen)) for k, v in components.items()},
                        "coverage": cost_coverage,
                        "source_count": len(source),
                        "note": "Inclusive pool window; stationary rate scaled to horizon; do not add request costs"
                        if kind == "window"
                        else "Per attempted request within the named scope",
                    }
                )
            analytical = (
                predict_cost(
                    cost_profiles,
                    pool,
                    card,
                    sorted(successes, key=lambda r: r["completed_ns"])[-self.neighbors :],
                )
                if card
                else None
            )
            if analytical and analytical["complete"] and analytical["calibrated"]:
                # Mix complete latency samples; never add component percentiles.
                # Native Session durations already include their observed waiting time.
                queue = (
                    [0.0]
                    if all(s["timing_basis"] == "target_native" for s in analytical["resource_samples"])
                    else metrics["queue_s"]["samples"] or [0.0]
                )
                latency = [v + q for v in analytical["latency"]["samples"] for q in queue]
                metrics["latency_s"] = _distribution(latency, len(latency))
            unique_count = len({r["id"] for r in selected})
            minimum = 3 if analytical else self.minimum_samples
            latency_confidence = target_confidence(
                coverage,
                unique_count,
                required={"tool_elapsed_s", "llm_idle_s", "parallel_peak", "load_rps"},
                minimum=minimum,
                marginal=marginal,
            )
            cost_required = {"tool_elapsed_s", "parallel_peak", "load_rps"}
            if card and card["cpu_basis"] == "active" and card["rates"]["cpu_s"]:
                cost_required.add("tool_cpu_s")
            cost_confidence = target_confidence(
                coverage,
                unique_count,
                required=cost_required,
                minimum=minimum,
                marginal=marginal,
            )
            cost_confidence["missing_cost_evidence"] = analytical["missing"] if analytical else []
            if (
                analytical and (not analytical["complete"] or not analytical["calibrated"])
                or not analytical and not any(c["complete"] and c["total"]["n"] >= minimum for c in costs)
            ):
                cost_confidence["reasons"].append("cost_evidence_insufficient")
            cost_confidence["reliable"] = not cost_confidence["reasons"]
            latency_evidence = len({r["id"] for r in selected if number(r["measurements"].get("latency_s"))})
            latency_confidence["measured_target_samples"] = latency_evidence
            if latency_evidence < minimum:
                latency_confidence["reasons"].append("latency_evidence_insufficient")
                latency_confidence["reliable"] = False
            results.append(
                {
                    "pool": copy.deepcopy(pool),
                    "mode": BACKENDS[pool["provider"]][0],
                    "label": BACKENDS[pool["provider"]][1],
                    "latency": metrics.pop("latency_s"),
                    "measurements": metrics,
                    "capacity": capacity,
                    "cost": [analytical] if analytical else costs,
                    "historical_cost": costs,
                    "analytical_cost": analytical,
                    "status_counts": dict(Counter(r["status"] for r in requests)),
                    "sample_count": len(successes),
                    "selected_unique_count": len({r["id"] for r in selected}),
                    "coverage": coverage,
                    "confidence": {"latency": latency_confidence, "cost": cost_confidence},
                    "low_confidence": not (latency_confidence["reliable"] and cost_confidence["reliable"]),
                    "latency_scope": "successful requests, observed queue/cold-start mixture; no additive p95",
                    "cost_missing": not analytical["complete"] if analytical else not costs,
                }
            )
        return {
            "as_of_ns": as_of_ns,
            "load_rps": load_rps,
            "horizon_s": horizon_s,
            "workload_basis": "marginal_means" if marginal else "joint_profiles",
            "workload_samples": len(profiles),
            "pools": results,
        }

    @staticmethod
    def _capacity(pool: dict, load: float, service: dict) -> dict:
        coupled = BACKENDS[pool["provider"]][0] == "state-coupled"
        concurrency = pool.get("sessions_per_instance" if coupled else "invocations_per_instance")
        instances = pool.get("instances")
        slots = instances * concurrency if instances is not None and concurrency is not None else None
        mean = service["mean"]
        rate = slots / mean if slots is not None and mean is not None and mean > 0 else None
        if slots == 0:
            rate = 0
        utilization = load / rate if rate and rate > 0 else (0 if load == 0 and rate == 0 else None)
        return {
            "slots": slots,
            "service_slot_s": service,
            "max_service_rps": rate,
            "effective_rps": min(load, rate) if rate is not None else None,
            "utilization": utilization,
            "insufficient": load >= rate if rate is not None and load > 0 else False if rate is not None else None,
            "basis": "slot-work conservation bound, not a queueing SLA",
            "missing": [
                k
                for k, v in {
                    "instances": instances,
                    "slot_concurrency": concurrency,
                    "measured_slot_occupancy": mean,
                }.items()
                if v is None
            ],
            "note": "Coupled occupancy must include session residence/tail; RPC concurrency is not session concurrency",
        }

    def save(self, path: Path) -> None:
        """Atomically persist revision history using existing compressed trace storage."""
        with self.lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            write_json(
                Path(path),
                {
                    "schema_version": 1,
                    "neighbors": self.neighbors,
                    "minimum_samples": self.minimum_samples,
                    "tariffs": self.tariffs,
                    "observations": [r for history in self.samples.values() for r in history],
                },
            )

    @classmethod
    def load(cls, path: Path) -> "ProviderModel":
        """Restore a snapshot with its original evidence availability times."""
        payload = json.loads(read_text(Path(path)))
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported provider model schema")
        model = cls(
            payload["neighbors"], tariffs=payload.get("tariffs"), minimum_samples=payload.get("minimum_samples", 20)
        )
        for row in payload["observations"]:
            model.update(row)
        return model
