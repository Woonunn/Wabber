"""Causal cost/latency routing with bounded fallback and whole-session admission."""

import copy
import json
import math
import threading
import time
from collections import Counter
from pathlib import Path

from atsuite.provider_model.observation import (
    BACKENDS,
    execution_context,
    trace_measurements,
    validate_pool,
)
from atsuite.routing.exploration import Exploration, model_revision
from atsuite.routing.objective import pareto, validate_weights
from atsuite.workload.descriptor import context_key, validate_descriptor
from atsuite.workload.profile import number


def risk_estimate(rows: list[dict], slo_s: float) -> dict:
    """Wilson 95% upper bound over unique attempts, including failures/unknowns.

    This interval assumes exchangeable Bernoulli attempts, not a queueing SLA.
    """
    n = len(rows)
    bad = sum(
        r["status"] != "success"
        or not number(r["measurements"].get("latency_s"))
        or r["measurements"]["latency_s"] > slo_s
        for r in rows
    )
    if not n:
        return {"n": 0, "violations": 0, "probability": None, "upper": 1.0}
    p, z = bad / n, 1.96
    upper = (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    return {"n": n, "violations": bad, "probability": p, "upper": upper}


class OnlineRouter:
    """One owner per pool; limits are measured session admission limits, not RPC slots.

    Callers must include external occupancy by reducing limits before construction.
    Unknown limits reject admission. No migration or automatic retry is performed.
    """

    def __init__(
        self,
        workload,
        provider,
        pools: dict,
        limits: dict,
        *,
        cost_weight=1.0,
        latency_weight=0.001,
        default_pool: str | None = None,
        execution_bounds: dict | None = None,
        budget: float | None = None,
        fx: dict | None = None,
        log_path: Path | None = None,
        exploration: dict | None = None,
        enable_context_routing: bool = False,
    ):
        if type(enable_context_routing) is not bool:
            raise ValueError("Invalid context routing rollout flag")
        self.enable_context_routing = enable_context_routing
        self.prediction_costs = Counter()
        validate_weights(cost_weight, latency_weight)
        bounds = copy.deepcopy(execution_bounds or {})
        if default_pool is not None and (default_pool not in pools or default_pool not in bounds):
            raise ValueError("Default pool requires validated execution bounds")
        for key, value in bounds.items():
            if (
                key not in pools
                or not value.get("evidence")
                or not value.get("currency")
                or any(
                    not number(value.get(k)) or value[k] == 0
                    for k in (
                        ("timeout_s", "max_cost") if budget is not None or "max_cost" in value else ("timeout_s",)
                    )
                )
                or not isinstance(value.get("tools"), list)
                or any(not isinstance(t, str) for t in value["tools"])
            ):
                raise ValueError("Bounds require validation evidence, supported tools, timeout and cost cap")
        if budget is not None and (not number(budget) or set(bounds) != set(pools)):
            raise ValueError("A budget requires execution cost bounds for every pool")
        if fx is not None and (
            not fx.get("date")
            or not fx.get("source")
            or fx.get("rates", {}).get(fx.get("currency")) != 1
            or any(not number(v) or v == 0 for v in fx.get("rates", {}).values())
        ):
            raise ValueError("Invalid dated currency conversion reference")
        currencies = {b["currency"] for b in bounds.values()}
        if budget is not None and len(currencies) > 1 and fx is None:
            raise ValueError("Budget caps need one currency or an explicit FX reference")
        if fx and not currencies <= fx["rates"].keys():
            raise ValueError("Missing budget currency conversion")
        for pool in pools.values():
            validate_pool(pool)
        identities = {
            json.dumps({k: v for k, v in p.items() if k != "billing"}, sort_keys=True) for p in pools.values()
        }
        if len(identities) != len(pools):
            raise ValueError("Duplicate execution pools")
        for key, limit in limits.items():
            if key not in pools or (
                limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0)
            ):
                raise ValueError("Invalid session admission limit")
        self.workload, self.provider = workload, provider
        self.pools, self.limits = copy.deepcopy(pools), dict(limits)
        self.cost_weight, self.latency_weight = cost_weight, latency_weight
        self.default_pool, self.execution_bounds = default_pool, bounds
        self.budget, self.fx, self.spent = budget, copy.deepcopy(fx), 0.0
        self.exploration = Exploration(exploration) if exploration is not None else None
        if (
            self.exploration
            and "budget" in self.exploration.config
            and (budget is None or self.exploration.config["budget"] > budget)
        ):
            raise ValueError("Exploration requires a bounded total budget covering its budget")
        self.exploration_revision = model_revision(provider, pools)
        self.exploration_spent = 0.0
        self.capacity_reserved = {}
        self.feedback = {}
        self.log_path = Path(log_path) if log_path else None
        # shortcut: retain decisions in memory; bound retention for longer-running studies.
        self.active, self.decisions, self.completed = {}, {}, set()
        self.retained_sessions = {}
        self.external_occupancy = {}
        self.session_retention_s = {}
        self.lock = threading.RLock()

    def _log(self, event):
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")

    def set_capacity(self, key: str, pool: dict, limit: int | None) -> None:
        """Publish read-back capacity atomically without rebinding active sessions."""
        validate_pool(pool)
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
            raise ValueError("Invalid session admission limit")
        with self.lock:
            old = self.pools[key]
            if any(old[k] != pool[k] for k in ("provider", "environment")):
                raise ValueError("Capacity updates cannot replace an execution pool")
            if old != pool:
                for session in self.active.values():
                    if session["selected"] == key:
                        session["capacity_transition"] = True
            self.pools[key], self.limits[key] = copy.deepcopy(pool), limit

    def observe_occupancy(self, key: str, occupied: int) -> None:
        """Observe native external Sessions without changing platform instance counts."""
        if key not in self.pools or type(occupied) is not int or occupied < 0:
            raise ValueError("Invalid external Session occupancy")
        with self.lock:
            self.external_occupancy[key] = occupied

    def decide(
        self,
        request_id: str,
        task: str | None = None,
        load_rps: float = 0.0,
        *,
        as_of_ns=None,
        shadow=False,
        pool_load_rps: dict | None = None,
        allowed_pools: set | None = None,
        exploration_pools: set | None = None,
        fallback_pool: str | None = None,
        arrival_ns: int | None = None,
        descriptor: dict | None = None,
    ) -> dict:
        """Freeze arrival features; reserve capacity and a conservative execution cost cap."""
        if fallback_pool is not None and (
            fallback_pool not in self.execution_bounds
            or (allowed_pools is not None and fallback_pool not in allowed_pools)
        ):
            raise ValueError("Policy fallback requires validated bounds and an allowed pool")
        fallback = self.default_pool if fallback_pool is None else fallback_pool
        now = time.time_ns() if as_of_ns is None else as_of_ns
        arrival = now if arrival_ns is None else arrival_ns
        if (
            not isinstance(request_id, str)
            or not request_id
            or (task is not None and (not isinstance(task, str) or not task))
            or not isinstance(shadow, bool)
            or not number(load_rps)
            or not number(now)
            or not number(arrival)
            or arrival > now
        ):
            raise ValueError("Invalid routing input")
        if descriptor is not None:
            validate_descriptor(descriptor)
        rates = dict(pool_load_rps) if pool_load_rps is not None else dict.fromkeys(self.pools, load_rps)
        if set(rates) != set(self.pools) or any(not number(v) for v in rates.values()):
            raise ValueError("Every pool needs a finite nonnegative load")
        if allowed_pools is not None and not set(allowed_pools) <= self.pools.keys():
            raise ValueError("Unknown routing policy pool")
        if exploration_pools is not None and not set(exploration_pools) <= self.pools.keys():
            raise ValueError("Unknown exploration pool")
        with self.lock:
            if request_id in self.decisions:
                if context_key(descriptor) != context_key(self.decisions[request_id]["descriptor"]):
                    raise ValueError("Request ID reused with a different context")
                return copy.deepcopy(self.decisions[request_id])
            with self.workload.lock:
                if request_id in self.workload.samples:
                    raise ValueError("Request ID already exists in workload history")
            with self.provider.lock:
                if any(key[3] == request_id for key in self.provider.samples):
                    raise ValueError("Request ID already exists in provider history")
            revision = model_revision(self.provider, self.pools)
            if self.exploration and revision != self.exploration_revision and not shadow:
                self.exploration.config["epoch_ns"] = max(now, self.exploration.config["epoch_ns"])
                self.exploration_revision = revision
                self._log(
                    {
                        "event": "exploration_epoch",
                        "observed_ns": now,
                        "model_revision": revision,
                    }
                )
            prediction_start = time.perf_counter()
            workload = self.workload.predict(now, descriptor=descriptor)
            overhead = workload["retrieval"]
            self.prediction_costs[overhead["currency"] or "local"] += overhead["cost"]
            inference_budget_charge = overhead["cost"]
            if inference_budget_charge and self.fx:
                conversion = self.fx["rates"].get(overhead["currency"])
                inference_budget_charge = inference_budget_charge * conversion if conversion is not None else None
            elif inference_budget_charge and self.budget is not None:
                currencies = {b["currency"] for b in self.execution_bounds.values()}
                if currencies != {overhead["currency"]}:
                    inference_budget_charge = None
            if not shadow and inference_budget_charge is not None:
                self.spent += inference_budget_charge
            profiles = workload["joint_profiles"]
            predictions = [
                self.provider.predict(profiles, [pool], rates[key], as_of_ns=now)["pools"][0]
                for key, pool in self.pools.items()
            ]
            candidates = []
            reserved = sum(
                d.get("reserved_cost", 0) for rid, d in self.active.items() if rid not in self.completed
            ) + sum(self.capacity_reserved.values())
            for (key, pool), pred in zip(self.pools.items(), predictions):
                limit = self.limits.get(key)
                self.retained_sessions = {rid: lease for rid, lease in self.retained_sessions.items() if lease[1] > now}
                occupied = (
                    sum(v["selected"] == key for v in self.active.values())
                    + sum(lease[0] == key for lease in self.retained_sessions.values())
                    + self.external_occupancy.get(key, 0)
                )
                flags = []
                if self.budget is not None and (inference_budget_charge is None or not overhead.get("cost_complete", True)):
                    flags.append("prediction_cost_unknown")
                if allowed_pools is not None and key not in allowed_pools and key not in (exploration_pools or set()):
                    flags.append("excluded_by_control_policy")
                if limit is None:
                    flags.append("capacity_unknown")
                elif occupied >= limit:
                    flags.append("capacity_exhausted")
                bound = self.execution_bounds.get(key)
                if descriptor and (not bound or not set(descriptor["tools"]) <= set(bound["tools"])):
                    flags.append("unsupported_tool_environment")
                if (
                    descriptor
                    and descriptor["hints"].get("memory_mb", 0)
                    and (
                        not number(pool["resources"].get("memory_mb"))
                        or descriptor["hints"]["memory_mb"] > pool["resources"]["memory_mb"]
                    )
                ):
                    flags.append("insufficient_memory")
                cap = bound.get("max_cost", 0) if bound else 0
                if self.fx and bound:
                    cap *= self.fx["rates"][bound["currency"]]
                if self.budget is not None and self.spent + reserved + cap > self.budget:
                    flags.append("budget_exhausted")
                if bound and (now - arrival) / 1e9 + time.perf_counter() - prediction_start >= bound["timeout_s"]:
                    flags.append("execution_deadline_elapsed")
                cost = pred.get("decision_cost", pred.get("analytical_cost"))
                if cost and not cost["complete"]:
                    cost = None
                amount = cost["total"]["mean"] if cost else None
                currency = cost["currency"] if cost else None
                if cost and self.fx:
                    conversion = self.fx["rates"].get(currency)
                    amount = amount * conversion if conversion is not None else None
                    currency = self.fx["currency"] if conversion is not None else currency
                inference_cost = overhead["cost"]
                inference_currency = overhead["currency"]
                if amount is not None and inference_cost:
                    if self.fx:
                        conversion = self.fx["rates"].get(inference_currency)
                        inference_cost = inference_cost * conversion if conversion is not None else None
                    elif currency != inference_currency:
                        inference_cost = None
                    amount = amount + inference_cost if inference_cost is not None else None
                latency = pred["latency"]["mean"]
                # Occupancy is a snapshot, not a fabricated queue measurement.
                if latency is not None:
                    latency += (now - arrival) / 1e9 + time.perf_counter() - prediction_start
                candidates.append(
                    {
                        "key": key,
                        "prediction": pred,
                        "flags": flags,
                        "prediction_flags": [
                            name
                            for name, missing in (
                                ("workload_history_insufficient", workload["low_confidence"]),
                                ("latency_prediction_unreliable", not pred["confidence"]["latency"]["reliable"]),
                                ("cost_prediction_unreliable", not pred["confidence"]["cost"]["reliable"]),
                                ("cost_prediction_missing", amount is None),
                                ("latency_prediction_missing", latency is None),
                            )
                            if missing
                        ],
                        "prediction_diagnostics": pred["confidence"],
                        "available_sessions": None if limit is None else max(0, limit - occupied),
                        "cost": cost,
                        "currency": currency,
                        "objective_cost": amount,
                        "objective_latency": latency,
                        "reserved_cost": cap,
                        "execution_bounds": bound,
                        "evidence": self.exploration.evidence(self.provider, pool, descriptor, now)
                        if self.exploration
                        else None,
                    }
                )
            feasible = [
                c for c in candidates if not c["flags"] and (allowed_pools is None or c["key"] in allowed_pools)
            ]
            priced = [
                c
                for c in feasible
                if c["objective_cost"] is not None and (self.latency_weight == 0 or c["objective_latency"] is not None)
            ]
            comparable = bool(priced) and len({c["currency"] for c in priced}) == 1
            # Uncalibrated demand or platform forecasts use the bounded default.
            ranked = [
                c for c in priced
                if c["prediction_diagnostics"]["cost"]["reliable"]
                and (self.latency_weight == 0 or c["prediction_diagnostics"]["latency"]["reliable"])
            ] if comparable and not workload["low_confidence"] and (shadow or self.enable_context_routing) else []
            frontier = pareto(ranked)
            selected = (
                min(
                    ranked,
                    key=lambda c: (
                        self.cost_weight * c["objective_cost"] + self.latency_weight * (c["objective_latency"] or 0),
                        c["key"],
                    ),
                )
                if ranked
                else next((c for c in feasible if c["key"] == fallback), None)
            )
            exploration = {
                "strategy": "none",
                "explored": False,
                "probabilities": {c["key"]: float(c is selected) for c in candidates},
                "selection_probability": 1.0 if selected else 0.0,
            }
            if self.exploration and (shadow or self.enable_context_routing):
                exploration_reserved = sum(
                    d["reserved_cost"]
                    for rid, d in self.active.items()
                    if rid not in self.completed and d["exploration"]["explored"]
                )
                remaining = (
                    max(
                        0,
                        self.exploration.config["budget"]
                        - self.exploration_spent
                        - exploration_reserved
                        - sum(self.capacity_reserved.values()),
                    )
                    if "budget" in self.exploration.config
                    else None
                )
                selected, exploration = self.exploration.choose(
                    request_id,
                    descriptor,
                    candidates,
                    selected,
                    remaining,
                    cost_weight=self.cost_weight,
                    latency_weight=self.latency_weight,
                )
            result = {
                "exploration": exploration,
                "model_revision": revision,
                "request_id": request_id,
                "task_id": task,
                "descriptor": copy.deepcopy(descriptor),
                "arrival_ns": arrival,
                "decision_ns": now,
                "load_rps": load_rps,
                "pool_load_rps": rates,
                "load_basis": "per_pool_snapshot" if pool_load_rps is not None else "offered_load_per_candidate",
                "shadow": shadow,
                "context_routing_enabled": self.enable_context_routing,
                "selected": selected["key"] if selected else None,
                "pool_snapshot": copy.deepcopy(self.pools),
                "mode": selected["prediction"]["mode"] if selected else None,
                "reason": "provider_exploration"
                if exploration["explored"]
                else "cost_latency_objective"
                if selected and ranked
                else "validated_default"
                if selected
                else "no_feasible_pool",
                "cost_comparable": comparable,
                "ranking": "weighted_cost_latency",
                "objective": {
                    "cost_weight": self.cost_weight,
                    "latency_weight": self.latency_weight,
                    "cost_unit": next(iter({c["currency"] for c in priced}), None) if comparable else None,
                    "latency_unit": "seconds",
                },
                "pareto": [c["key"] for c in frontier],
                "reserved_cost": selected["reserved_cost"] if selected else 0,
                "execution_bounds": copy.deepcopy(selected["execution_bounds"]) if selected else None,
                "workload": workload,
                "provider_workload_profiles": len(profiles),
                "candidates": candidates,
            }
            result["prediction_overhead"] = {**overhead, "total_latency_s": time.perf_counter() - prediction_start}
            self.workload.record_prediction(request_id, workload)
            self._log({"event": "decision", **result})
            self.decisions[request_id] = copy.deepcopy(result)
            if selected and not shadow:
                self.active[request_id] = copy.deepcopy(result)
            return result

    def complete(
        self,
        request_id: str,
        trace: dict,
        *,
        available_ns=None,
        actual_pool=None,
        actual_cost=None,
        cost_basis="attributable_execution_cost",
    ) -> dict:
        """Feedback goes only to the bound pool; failed traces also release admission."""
        from atsuite.workload.profile import extract_profile

        now = time.time_ns() if available_ns is None else available_ns
        with self.lock:
            decision = self.decisions[request_id]
            if request_id in self.completed:
                raise ValueError("Request already completed")
            if decision["shadow"]:
                if actual_pool not in self.pools:
                    raise ValueError("Shadow completion requires the actual execution pool")
                selected = actual_pool
            else:
                selected = self.active[request_id]["selected"]
                if actual_pool is not None and actual_pool != selected:
                    raise ValueError("Session migration is forbidden")
            if cost_basis not in {
                "attributable_execution_cost",
                "simulated_usage_estimate",
                "synthetic_usage_estimate",
            }:
                raise ValueError("Unknown execution cost basis")
            if actual_cost is not None and not number(actual_cost):
                raise ValueError("Invalid attributable cost")
            profile = extract_profile(trace)
            profile["execution_pool"] = execution_context(decision["pool_snapshot"][selected])
            profile["descriptor"] = copy.deepcopy(decision["descriptor"])
            if profile["request_id"] != request_id or trace["arrival_ns"] != decision["arrival_ns"]:
                raise ValueError("Completion identity/arrival differs from admission")
            row = {
                "kind": "request",
                "id": request_id,
                "task_id": decision["task_id"],
                "profile": profile,
                "pool": copy.deepcopy(decision["pool_snapshot"][selected]),
                "capacity_transition": self.active.get(request_id, {}).get("capacity_transition", False),
                "load_rps": decision["pool_load_rps"][selected],
                "status": profile["status"],
                "arrival_ns": trace["arrival_ns"],
                "completed_ns": trace["completed_ns"],
                "available_ns": now,
                "cost": {
                    "basis": "actual" if cost_basis == "attributable_execution_cost" else "estimated",
                    "currency": BACKENDS[decision["pool_snapshot"][selected]["provider"]][2],
                    "scope": "request_execution", "complete": True,
                    "components": {"execution": actual_cost},
                    "attribution": "exclusive",
                    "evidence": [f"router.complete:{request_id}:{cost_basis}"],
                } if actual_cost is not None else None,
                "measurements": trace_measurements(trace),
            }
            self.workload.update(profile, now, descriptor=decision["descriptor"])
            self.provider.update(row)
            capped = "max_cost" in (decision.get("execution_bounds") or {})
            charge = actual_cost if actual_cost is not None else decision["reserved_cost"] if capped else None
            accounting = {
                "amount": charge,
                "actual_cost": actual_cost if cost_basis == "attributable_execution_cost" else None,
                "basis": cost_basis if actual_cost is not None else "execution_cap" if capped else "unpriced",
                "selected": selected,
                "available_ns": now,
                "shadow": decision["shadow"],
                "explored": decision["exploration"]["explored"],
                "selection_probability": decision["exploration"]["selection_probability"],
                "cap_exceeded": capped and charge is not None and charge > decision["reserved_cost"],
                "latency_s": row["measurements"].get("latency_s"),
                "status": row["status"],
                "realized_counterfactual_gain": None,
            }
            self.feedback[request_id] = accounting
            if not decision["shadow"]:
                self.spent += charge or 0
                if accounting["explored"]:
                    self.exploration_spent += charge or 0
            self._log(
                {
                    "event": "completion",
                    "request_id": request_id,
                    "selected": selected,
                    "observation": row,
                    "accounting": accounting,
                }
            )
            self.active.pop(request_id, None)
            if not decision["shadow"] and not trace.get("sessions_closed") and self.session_retention_s.get(selected, 0):
                self.retained_sessions[request_id] = (selected, now + int(self.session_retention_s[selected] * 1e9))
            self.completed.add(request_id)
            return row

    def exploration_report(self) -> dict:
        """Summarize observed choices and accounting without inventing counterfactual gains."""
        with self.lock:
            decisions = [d for d in self.decisions.values() if not d["shadow"] and d["selected"]]
            counts = Counter(d["selected"] for d in decisions)
            feedback = [v for v in self.feedback.values() if not v["shadow"] and v["explored"]]
            return {
                "admitted": len(decisions),
                "selection_counts": {k: counts[k] for k in self.pools},
                "selection_proportions": {k: counts[k] / len(decisions) if decisions else 0 for k in self.pools},
                "exploration_attempts": sum(d["exploration"]["explored"] for d in decisions),
                "exploration_completed": len(feedback),
                "exploration_successes": sum(v["status"] == "success" for v in feedback),
                "exploration_accounted_cost": self.exploration_spent,
                "exploration_execution_cost_by_basis": {
                    basis: sum(v["amount"] for v in feedback if v["basis"] == basis and v["amount"] is not None)
                    for basis in sorted({v["basis"] for v in feedback})
                },
                "capacity_reserved": dict(self.capacity_reserved),
                "budget_spent_or_capped": self.spent,
                "prediction_cost_by_currency": dict(self.prediction_costs),
                "total_infrastructure_actual_cost": None,
                "realized_counterfactual_gain": None,
                "scope": "Decision/feedback evidence; capacity caps are not bills; benefit requires matched policy runs",
            }
