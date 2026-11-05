"""Contextual epsilon-greedy; all evidence is visible before the decision."""

import hashlib
import json
import math
import random

from atsuite.provider_model.model import _compatible
from atsuite.workload.descriptor import compatible, family_key, resource_distance
from atsuite.workload.profile import number


class Exploration:
    """A small reproducible policy; accounting and admission belong to OnlineRouter."""

    def __init__(self, config: dict):
        allowed = {
            "strategy",
            "epsilon",
            "budget",
            "seed",
            "min_samples",
            "window_s",
            "epoch_ns",
            "optimism",
        }
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError("Unknown exploration configuration")
        self.config = {
            "strategy": "contextual",
            "epsilon": 0.1,
            "seed": 0,
            "min_samples": 3,
            "window_s": 3600,
            "epoch_ns": 0,
            "optimism": 0.5,
            **config,
        }
        c = self.config
        if (
            c["strategy"] not in {"contextual", "random"}
            or any(not number(c.get(k)) for k in ("epsilon", "window_s", "epoch_ns", "optimism"))
            or ("budget" in c and not number(c["budget"]))
            or c["epsilon"] > 1
            or c["optimism"] > 1
            or c["window_s"] == 0
            or type(c["min_samples"]) is not int
            or c["min_samples"] < 1
            or type(c["seed"]) is not int
        ):
            raise ValueError("Invalid bounded exploration policy")

    def evidence(self, provider, pool, descriptor, now):
        """Share completed attempts across compatible settings and nearby input scales."""
        rows = []
        has_cost_feedback = False
        cutoff = max(self.config["epoch_ns"], now - self.config["window_s"] * 1e9)
        with provider.lock:
            for history in provider.samples.values():
                visible = [r for r in history if r["available_ns"] < now]
                if not visible:
                    continue
                row = visible[-1]
                d = row["profile"].get("descriptor")
                if (
                    row["kind"] == "request"
                    and cutoff <= row["completed_ns"] < now
                    and _compatible(row["pool"], pool)
                    and not row.get("capacity_transition")
                    and descriptor is not None
                    and d is not None
                    and compatible(descriptor, d)
                    and (descriptor.get("model") or {}).get("name")
                    and (descriptor.get("model") or {}).get("name") == (d.get("model") or {}).get("name")
                ):
                    charge = row.get("cost") or {}
                    has_cost_feedback |= bool(charge.get("complete") and charge.get("scope") in {
                        "request_execution", "matched_native_usage",
                    })
                    if resource_distance(descriptor, d) <= 0.5:
                        rows.append(row)
        return {
            "attempts": len(rows),
            "successes": sum(r["status"] == "success" for r in rows),
            "last_available_ns": max((r["available_ns"] for r in rows), default=None),
            "has_cost_feedback": has_cost_feedback,
        }

    def opportunities(self, candidates, incumbent, *, cost_weight, latency_weight):
        """Use an optimistic analytical objective, never unobserved realized outcomes."""

        def score(c):
            return cost_weight * c["objective_cost"] + latency_weight * (c["objective_latency"] or 0)

        def priced(c):
            return c["objective_cost"] is not None and (latency_weight == 0 or c["objective_latency"] is not None)

        base = (
            score(incumbent)
            if incumbent is not None and priced(incumbent)
            else None
        )
        result = []
        for c in candidates:
            if (incumbent and c["key"] == incumbent["key"]) or c["flags"] or not c["execution_bounds"]:
                continue
            evidence = c["evidence"]
            n, ok = evidence["attempts"], evidence["successes"]
            # An observed incumbent can serve traffic while a sparse arm's probe returns its reward.
            if (incumbent is not None and (incumbent.get("cost") or {}).get("basis") == "predicted"
                    and n < self.config["min_samples"] and evidence.get("pending_feedback", 0)):
                continue
            if c["objective_latency"] is not None and c["objective_latency"] >= c["execution_bounds"]["timeout_s"]:
                continue
            if base is None or not priced(c):
                if n < self.config["min_samples"]:
                    result.append(
                        {
                            "key": c["key"],
                            "weight": 1 / math.sqrt(1 + n),
                            "reason": "bounded_cold_start",
                            "optimistic_objective": None,
                            "incumbent_objective": base,
                            "evidence": evidence,
                        }
                    )
                continue
            if c["currency"] != incumbent["currency"]:
                continue
            uncertainty = self.config["optimism"] / math.sqrt(1 + max(n, self.config["min_samples"]))
            optimistic = score(c) * (1 - uncertainty)
            # Failed attempts consume evidence and budget; repeatedly failing pools are not promising.
            promising = optimistic <= base and (n < self.config["min_samples"] or ok / n >= 0.5)
            if self.config["strategy"] == "random" or promising:
                result.append(
                    {
                        "key": c["key"],
                        "weight": 1 if self.config["strategy"] == "random" else 1 / math.sqrt(1 + n),
                        "reason": "random_baseline"
                        if self.config["strategy"] == "random"
                        else "uncertain_analytical_opportunity",
                        "optimistic_objective": optimistic,
                        "incumbent_objective": base,
                        "evidence": evidence,
                    }
                )
        if incumbent is None and result:
            pending = {c["key"]: c["evidence"].get("pending_feedback", 0) for c in candidates}
            unprobed = [o for o in result if pending[o["key"]] == 0]
            if unprobed:
                return unprobed
            for opportunity in result:
                opportunity["weight"] /= math.sqrt(1 + pending[opportunity["key"]])
        return result

    def choose(
        self,
        request_id,
        descriptor,
        candidates,
        incumbent,
        remaining,
        *,
        cost_weight,
        latency_weight,
    ):
        """Return a draw and its full propensity distribution, conditional on feasible pools."""
        eligible = [c for c in candidates if remaining is None or c["reserved_cost"] <= remaining]
        opportunities = self.opportunities(eligible, incumbent, cost_weight=cost_weight, latency_weight=latency_weight)
        attempts = sum(c["evidence"]["attempts"] for c in candidates)
        epsilon = self.config["epsilon"] / math.sqrt(1 + attempts / self.config["min_samples"])
        if not opportunities:
            epsilon = 0.0
        elif incumbent is None and epsilon > 0:
            epsilon = 1.0
        probabilities = {c["key"]: 0.0 for c in candidates}
        if incumbent:
            probabilities[incumbent["key"]] = 1 - epsilon
        total = sum(o["weight"] for o in opportunities)
        for o in opportunities:
            probabilities[o["key"]] = epsilon * o["weight"] / total
        seed = hashlib.sha256(f"{self.config['seed']}:{self.config['epoch_ns']}:{request_id}".encode()).digest()
        draw, cumulative, selected = random.Random(seed).random(), 0.0, incumbent
        for c in candidates:
            cumulative += probabilities[c["key"]]
            if draw < cumulative:
                selected = c
                break
        explored = bool(selected and (incumbent is None or selected["key"] != incumbent["key"]))
        return selected, {
            "strategy": self.config["strategy"],
            "config": dict(self.config),
            "family": family_key(descriptor),
            "epsilon": epsilon,
            "draw": draw,
            "explored": explored,
            "incumbent": incumbent["key"] if incumbent else None,
            "probabilities": probabilities,
            "selection_probability": probabilities[selected["key"]] if selected else 0,
            "opportunities": opportunities,
            "remaining_budget": remaining,
            "budget_excluded": [
                c["key"]
                for c in candidates
                if remaining is not None and not c["flags"] and c["reserved_cost"] > remaining
            ],
            "reason": next(
                (o["reason"] for o in opportunities if selected and o["key"] == selected["key"]),
                "exploration_disabled"
                if self.config["epsilon"] == 0
                else "exploration_not_selected"
                if opportunities
                else "exploration_budget_exhausted"
                if any(remaining is not None and not c["flags"] and c["reserved_cost"] > remaining for c in candidates)
                else "no_exploration_opportunity",
            ),
            "predicted_objective_gain": next(
                (
                    o["incumbent_objective"]
                    - (cost_weight * selected["objective_cost"] + latency_weight * (selected["objective_latency"] or 0))
                    for o in opportunities
                    if explored
                    and o["key"] == selected["key"]
                    and o["incumbent_objective"] is not None
                    and selected["objective_cost"] is not None
                    and (latency_weight == 0 or selected["objective_latency"] is not None)
                ),
                None,
            ),
            "realized_counterfactual_gain": None,
        }


def model_revision(provider, pools):
    """Record the exact price/environment configuration used by the decision."""
    return hashlib.sha256(
        json.dumps({"tariffs": provider.tariffs, "pools": pools}, sort_keys=True).encode()
    ).hexdigest()
