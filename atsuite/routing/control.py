"""Finite joint routing/capacity search with causal cost/latency and drain protection."""

import copy
import itertools
import math
import time
from collections import Counter, deque

from atsuite.provider_model.observation import validate_pool
from atsuite.routing.capacity import count, session_retention_s
from atsuite.provider_model.analytical import pool_cost
from atsuite.routing.objective import pareto
from atsuite.workload.descriptor import family_key
from atsuite.routing.router import OnlineRouter
from atsuite.workload.profile import number


def candidate_plans(levels: dict, weights: list[dict], *, maximum=256) -> list[dict]:
    """Enumerate an explicit, bounded grid; never silently truncate the search."""
    size = math.prod(len(v) for v in levels.values()) * len(weights)
    if not 0 < size <= maximum:
        raise ValueError("Candidate grid is empty or exceeds the search bound")
    keys = list(levels)
    return [
        {"targets": dict(zip(keys, targets)), "weights": dict(weight)}
        for targets in itertools.product(*levels.values())
        for weight in weights
    ]


class JointController:
    """Single-owner controller, usable by the existing LangGraph routing hook.

    Levels map each pool and target to a measured pool configuration, Session
    limit, scaling-delay upper bound and complete disjoint infrastructure rates.
    A None target denotes a platform-managed pool. Plans specify traffic shares;
    current requests use only a successfully read-back, capacity-checked level.
    """

    def __init__(
        self,
        router: OnlineRouter,
        adapters: dict,
        levels: dict,
        plans: list[dict],
        fx: dict,
        *,
        period_s=10.0,
        horizon_s=60.0,
        cooldown_s=60.0,
        observation_s=30.0,
        target_utilization=0.7,
        policy="joint",
        capacity_exploration: dict | None = None,
    ):
        keys = set(router.pools)
        if (
            keys != set(adapters)
            or keys != set(levels)
            or not plans
            or len(plans) > 256
        ):
            raise ValueError(
                "Every execution pool needs an adapter and bounded candidates"
            )
        if policy not in {"joint", "independent", "routing", "static"}:
            raise ValueError("Unknown control policy")
        if any(
            not number(x) or x == 0 for x in (period_s, horizon_s, observation_s)
        ) or not number(cooldown_s):
            raise ValueError("Invalid control interval")
        if not number(target_utilization) or not 0 < target_utilization <= 1:
            raise ValueError("Invalid utilization target")
        if (
            not fx.get("date")
            or not fx.get("source")
            or fx.get("rates", {}).get(fx.get("currency")) != 1
            or any(not number(v) or v == 0 for v in fx.get("rates", {}).values())
        ):
            raise ValueError("A dated positive FX reference is required")
        self.router, self.adapters = router, adapters
        self.levels, self.plans, self.fx = (
            copy.deepcopy(levels),
            copy.deepcopy(plans),
            copy.deepcopy(fx),
        )
        for key, options in self.levels.items():
            if not options:
                raise ValueError("Empty capacity levels")
            for target, level in options.items():
                count(target, optional=True)
                count(level["session_limit"])
                validate_pool(level["pool"])
                session_retention_s(level["pool"])
                if level["pool"].get("instances") is not None:
                    raise ValueError(
                        "Serverless controls do not establish a fixed physical instance count"
                    )
                if (
                    not level.get("evidence")
                    or not number(level["scale_delay_s"])
                    or level["pool"]["provider"] != router.pools[key]["provider"]
                    or level["pool"]["environment"] != router.pools[key]["environment"]
                ):
                    raise ValueError(
                        "Capacity level needs matching deployment and calibration evidence"
                    )
                expected = {"control": adapters[key].control, "target": target}
                if level["pool"]["resources"].get("capacity_control") != expected:
                    raise ValueError(
                        "Pool identity must include the capacity control and target"
                    )
                if (target is None) != (adapters[key].control == "managed"):
                    raise ValueError(
                        "Managed capacity cannot have an invented instance target"
                    )
            if len({str(v["pool"]) for v in options.values()}) != len(options):
                raise ValueError(
                    "Capacity levels must have distinct calibration identities"
                )
        for plan in self.plans:
            if set(plan["targets"]) != keys or set(plan["weights"]) != keys:
                raise ValueError("Incomplete joint plan")
            if any(not number(w) for w in plan["weights"].values()) or not math.isclose(
                sum(plan["weights"].values()), 1
            ):
                raise ValueError("Routing weights must sum to one")
            if any(
                target not in self.levels[key]
                for key, target in plan["targets"].items()
            ):
                raise ValueError("Candidate exceeds calibrated capacity bounds")
        self.period_s, self.horizon_s, self.cooldown_s = period_s, horizon_s, cooldown_s
        self.observation_s, self.target_utilization, self.policy = (
            observation_s,
            target_utilization,
            policy,
        )
        self.snapshots, self.pending, self.draining, self.last_change = {}, {}, {}, {}
        self.arrivals, self.routed = deque(), deque()
        # shortcut: deduplicate for one experiment; bound retention before operating as a long-lived service.
        self.offered_ids = set()
        self.offered_families = {}
        self.last_tick, self.plan, self.initial = None, None, None
        self.deficit = Counter()
        self.release_holds = set()
        self.retained_until = {}
        self.last_report = None
        self.descriptors = {}
        self.probe = None
        self.probed_until = {}
        self.capacity_exploration = copy.deepcopy(capacity_exploration)
        if capacity_exploration is not None:
            c = capacity_exploration
            if (
                not router.exploration
                or policy != "joint"
                or set(c) != {"lease_s", "max_cost", "evidence"}
                or not number(c["lease_s"])
                or c["lease_s"] < period_s
                or not c["evidence"]
                or not isinstance(c["max_cost"], dict)
                or not set(c["max_cost"]) <= keys
                or any(not number(v) or v == 0 for v in c["max_cost"].values())
            ):
                raise ValueError(
                    "Exploration capacity needs an audited lease and full lifecycle cost caps"
                )

    @property
    def pools(self):
        return self.router.pools

    @property
    def active(self):
        return self.router.active

    @property
    def decisions(self):
        return self.router.decisions

    def _log(self, event, **data):
        self.router._log({"event": event, **data})

    def _occupied(self, key):
        return sum(d["selected"] == key for d in self.active.values())

    def hold_release(self, request_id: str) -> None:
        """Retain occupancy when remote Session teardown was not confirmed."""
        with self.router.lock:
            self.release_holds.add(request_id)
            self._log("session_release_unconfirmed", request_id=request_id)

    def complete(self, request_id: str, trace: dict, **kwargs) -> dict:
        """Feed back exactly once; uncertain remote cleanup keeps its reservation."""
        with self.router.lock:
            bound = copy.deepcopy(self.active.get(request_id))
            result = self.router.complete(request_id, trace, **kwargs)
            if request_id in self.release_holds and bound:
                self.active[request_id] = bound
            elif bound and not trace.get("sessions_closed"):
                retention = session_retention_s(result["pool"])
                if retention:
                    self.active[request_id] = bound
                    self.retained_until[request_id] = result["available_ns"] + int(
                        retention * 1e9
                    )
                    self._log(
                        "session_retained",
                        request_id=request_id,
                        until_ns=self.retained_until[request_id],
                        basis="configured_maximum_lifetime_after_http_close",
                    )
            return result

    def expire_retained(self, now_ns: int) -> None:
        """Release local holds only after the conservative configured lifetime bound."""
        with self.router.lock:
            for rid, deadline in list(self.retained_until.items()):
                if now_ns >= deadline:
                    self.retained_until.pop(rid)
                    self.active.pop(rid, None)
                    self._log(
                        "session_retention_elapsed",
                        request_id=rid,
                        observed_ns=now_ns,
                        physical_release_observed=False,
                    )

    def _refresh(self, now):
        self.expire_retained(now)
        for key, adapter in self.adapters.items():
            try:
                state = adapter.read(now)
                for field, value in getattr(adapter, "configuration", {}).items():
                    expected = self.pools[key]["resources"].get(field)
                    if field in self.pools[key]["resources"] and expected != value:
                        raise ValueError(
                            "Live deployment configuration drift: " + field
                        )
                self.snapshots[key] = state
                self._log(
                    "capacity_observation",
                    pool=key,
                    **state.to_dict(),
                    active_sessions=sum(
                        d["selected"] == key and rid not in self.router.completed
                        for rid, d in self.active.items()
                    ),
                    occupied_sessions=self._occupied(key),
                    session_capacity=self.levels[key]
                    .get(state.effective, {})
                    .get("session_limit")
                    if state.status == "READY"
                    else None,
                    retained_sessions=sum(
                        self.active[rid]["selected"] == key
                        for rid in self.retained_until
                    ),
                )
                target = state.effective
                known = state.status == "READY" and target in self.levels[key]
                if adapter.control != "managed" and target is None:
                    known = False
                if known:
                    level = self.levels[key][target]
                    limit = 0 if key in self.draining else level["session_limit"]
                    self.router.set_capacity(key, level["pool"], limit)
                    if key in self.pending and self.pending[key]["target"] == target:
                        request = self.pending.pop(key)
                        self.last_change[key] = now
                        self._log(
                            "scaling_effective",
                            pool=key,
                            target=target,
                            observed_ns=now,
                            requested_ns=request["requested_ns"],
                            observed_delay_s=(now - request["requested_ns"]) / 1e9,
                        )
                else:
                    self.router.set_capacity(key, self.pools[key], None)
            except Exception as exc:  # noqa: BLE001 - any failed SDK read closes admission.
                self.snapshots.pop(key, None)
                self.router.set_capacity(key, self.pools[key], None)
                for session in self.active.values():
                    if session["selected"] == key:
                        session["capacity_transition"] = True
                self._log(
                    "capacity_error",
                    pool=key,
                    observed_ns=now,
                    error=type(exc).__name__,
                )

    def _submit(self, key, target, now):
        state = self.snapshots[key]
        if target == state.target or self.adapters[key].control == "managed":
            return
        if (
            key in self.pending
            or now - self.last_change.get(key, -(10**30)) < self.cooldown_s * 1e9
        ):
            return
        if state.target is None:
            return
        if target < state.target and self._occupied(key):
            self.draining[key] = target
            self.router.set_capacity(key, self.pools[key], 0)
            self._log(
                "scaling_drain",
                pool=key,
                target=target,
                observed_ns=now,
                active_sessions=self._occupied(key),
            )
            return
        self._log("scaling_requested", pool=key, target=target, requested_ns=now)
        # On a timeout the mutation may have succeeded: reconcile, never blind retry.
        self.pending[key] = {
            "target": target,
            "from_target": state.target,
            "requested_ns": now,
        }
        self.last_change[key] = now
        for session in self.active.values():
            if session["selected"] == key:
                session["capacity_transition"] = True
        self.router.set_capacity(key, self.pools[key], None)
        try:
            self.adapters[key].apply(target, active_sessions=self._occupied(key))
        except Exception as exc:  # noqa: BLE001 - ambiguous SDK writes require reconciliation.
            self._log(
                "scaling_error",
                pool=key,
                target=target,
                observed_ns=now,
                error=type(exc).__name__,
                reconciliation_required=True,
            )
        self.draining.pop(key, None)

    def _evaluate(self, plan, task_rates, now, queued):
        pools = {k: self.levels[k][v]["pool"] for k, v in plan["targets"].items()}
        limits = {
            k: self.levels[k][v]["session_limit"] for k, v in plan["targets"].items()
        }
        offered = sum(task_rates.values())
        rates = {k: offered * w for k, w in plan["weights"].items()}
        flags, duration, predictions, latencies = [], {}, {}, []
        for family, rate in task_rates.items():
            if rate == 0:
                continue
            workload = self.router.workload.predict(
                now, descriptor=self.descriptors.get(family)
            )
            for key, pool in pools.items():
                if not plan["weights"][key]:
                    continue
                pred = self.router.provider.predict(
                    workload["joint_profiles"], [pool], rates[key], as_of_ns=now
                )["pools"][0]
                predictions[(key, family)] = pred
                if workload["low_confidence"] or pred["low_confidence"]:
                    flags.append(f"{key}:unreliable_prediction")
                seconds = pred["latency"]["p95"]
                mean = pred["latency"]["mean"]
                if seconds is None:
                    bound = self.router.execution_bounds.get(key)
                    seconds = bound["timeout_s"] if bound else None
                    mean = seconds
                if seconds is None:
                    flags.append(f"{key}:unknown_session_duration")
                else:
                    duration[key] = max(duration.get(key, 0), seconds)
                    latencies.append(rate / offered * plan["weights"][key] * mean)
        costs, demand = {}, {}
        for key, target in plan["targets"].items():
            level, state = self.levels[key][target], self.snapshots.get(key)
            if state is None or state.target not in self.levels[key]:
                flags.append(f"{key}:unknown_current_capacity_cost")
            if key in self.pending and self.pending[key]["target"] != target:
                flags.append(f"{key}:scaling_in_progress")
            occupied = self._occupied(key)
            demand[key] = math.ceil(
                rates[key]
                * (duration.get(key, 0) + session_retention_s(level["pool"]))
                / self.target_utilization
                + queued * plan["weights"][key]
            )
            if demand[key] > limits[key]:
                flags.append(f"{key}:forecast_capacity_exceeded")
            changed = state is None or target != state.target
            if offered and plan["weights"][key]:
                wait = (
                    queued
                    * plan["weights"][key]
                    * duration.get(key, 0)
                    / max(1, limits[key])
                )
                if changed:
                    wait += min(level["scale_delay_s"], self.horizon_s)
                latencies.append(plan["weights"][key] * wait)
            if changed and self.adapters[key].control != "managed":
                if level["scale_delay_s"] >= self.horizon_s:
                    flags.append(f"{key}:scaling_outside_horizon")
                if now - self.last_change.get(key, -(10**30)) < self.cooldown_s * 1e9:
                    flags.append(f"{key}:cooldown")
            try:
                card = level["pool"].get(
                    "billing",
                    self.router.provider.tariffs.get(level["pool"]["provider"]),
                )
                if card is None:
                    raise ValueError(
                        "An analytical tariff is required for capacity optimization"
                    )
                costs[key] = pool_cost(
                    card,
                    requests=0,
                    marginal=None,
                    horizon_s=self.horizon_s,
                    retained_s=occupied * self.horizon_s,
                    memory_mb=level["pool"]["resources"].get("memory_mb", 0),
                    changes=int(changed),
                )
                for family, rate in task_rates.items():
                    count_requests = rate * plan["weights"][key] * self.horizon_s
                    if not count_requests:
                        continue
                    marginal = predictions[(key, family)].get("analytical_cost")
                    increment = pool_cost(
                        card, requests=count_requests, marginal=marginal, horizon_s=0
                    )
                    for component, amount in increment["components"].items():
                        costs[key]["components"][component] += amount
                    costs[key]["total"] += increment["total"]
                old_target = (
                    self.pending[key]["from_target"]
                    if key in self.pending
                    else state.target
                    if state
                    else None
                )
                if (changed or key in self.pending) and old_target in self.levels[key]:
                    old = self.levels[key][old_target]["pool"]
                    old_card = old.get(
                        "billing", self.router.provider.tariffs.get(old["provider"])
                    )
                    overlap_s = (
                        self.horizon_s
                        if occupied or key in self.pending
                        else level["scale_delay_s"]
                    )
                    overlap = pool_cost(
                        old_card,
                        requests=0,
                        marginal=None,
                        horizon_s=0,
                        overlap_s=overlap_s,
                    )
                    costs[key]["components"]["scaling"] += overlap["total"]
                    costs[key]["total"] += overlap["total"]
                conversion = self.fx["rates"][card["currency"]]
                if not number(conversion) or conversion == 0:
                    raise ValueError("Invalid currency conversion")
                costs[key]["components"] = {
                    k: v * conversion for k, v in costs[key]["components"].items()
                }
                costs[key]["total"] *= conversion
                costs[key]["currency"] = self.fx["currency"]
            except (ValueError, KeyError):
                costs.pop(key, None)
                flags.append(f"{key}:incomplete_cost")
        return {
            "plan": plan,
            "flags": sorted(set(flags)),
            "predicted_latency_s": sum(latencies) if latencies else 0.0,
            "demand": demand,
            "costs": costs,
            "predicted_total": sum(c["total"] for c in costs.values())
            if len(costs) == len(pools)
            else None,
        }

    def _score(self, candidate):
        return (
            self.router.cost_weight * candidate["predicted_total"] / self.horizon_s
            + self.router.latency_weight * candidate["predicted_latency_s"]
        )

    def tick(
        self, task_rates: dict, *, now_ns=None, queued=0, utilization=None, apply=True
    ) -> dict:
        """Poll capacity, search candidates, submit changes; never count targets as ready."""
        now = time.time_ns() if now_ns is None else now_ns
        count(now)
        count(queued)
        if not task_rates or any(
            not isinstance(t, str) or not t or not number(r)
            for t, r in task_rates.items()
        ):
            raise ValueError("A causal task arrival-rate snapshot is required")
        utilization = utilization or {}
        if not set(utilization) <= self.pools.keys() or any(
            not number(v) or v > 1 for v in utilization.values()
        ):
            raise ValueError("Invalid utilization snapshot")
        with self.router.lock:
            if self.last_tick is not None and now < self.last_tick:
                raise ValueError("Control clock moved backwards")
            self._refresh(now)
            if len(self.snapshots) != len(self.pools):
                self.plan = None
                return {"reason": "capacity_readback_failed", "selected": None}
            if self.initial is None:
                self.initial = {k: s.target for k, s in self.snapshots.items()}
            utilization = {
                **{
                    key: min(1.0, self._occupied(key) / self.router.limits[key])
                    for key in self.pools
                    if self.router.limits.get(key)
                },
                **utilization,
            }
            if (
                self.last_tick is not None
                and now - self.last_tick < self.period_s * 1e9
            ):
                for key, target in list(self.draining.items()):
                    if apply and not self._occupied(key):
                        self._submit(key, target, now)
                return copy.deepcopy(self.last_report)
            self.last_tick = now
            candidates = [
                self._evaluate(p, task_rates, now, queued) for p in self.plans
            ]
            if self.policy == "joint":
                feasible = [c for c in candidates if not c["flags"]]
                selected = (
                    min(feasible, key=lambda c: self._score(c)) if feasible else None
                )
            else:
                # Baselines keep the first declared routing split; routing-only freezes capacity.
                feasible = [
                    c
                    for c in candidates
                    if not [f for f in c["flags"] if not f.endswith(":incomplete_cost")]
                ]
                if self.policy in {"static", "routing"}:
                    feasible = [
                        c for c in feasible if c["plan"]["targets"] == self.initial
                    ]
                if self.policy in {"static", "independent"}:
                    feasible = [
                        c
                        for c in feasible
                        if c["plan"]["weights"] == self.plans[0]["weights"]
                    ]
                if self.policy == "independent":
                    for c in feasible:
                        for key, value in utilization.items():
                            state = self.snapshots[key]
                            if value > self.target_utilization and self.levels[key][
                                c["plan"]["targets"][key]
                            ]["session_limit"] <= self.levels[key].get(
                                state.target, {}
                            ).get("session_limit", 0):
                                c["flags"].append(f"{key}:high_utilization")
                    feasible = [
                        c
                        for c in feasible
                        if not any(f.endswith(":high_utilization") for f in c["flags"])
                    ]
                if self.policy == "routing":
                    comparable = all(c["predicted_total"] is not None for c in feasible)
                    selected = (
                        min(
                            feasible,
                            key=lambda c: (
                                self._score(c)
                                if comparable
                                else c["predicted_latency_s"]
                            ),
                        )
                        if feasible
                        else None
                    )
                else:
                    selected = (
                        min(
                            feasible,
                            key=lambda c: sum(
                                self.levels[k][t]["session_limit"]
                                for k, t in c["plan"]["targets"].items()
                            ),
                        )
                        if feasible
                        else None
                    )
            self.plan = self._probe_plan(
                selected["plan"] if selected else None, task_rates, now, apply=apply
            )
            for key, target in list(self.draining.items()):
                if self.plan is None or self.plan["targets"][key] != target:
                    self.draining.pop(key)
                    state = self.snapshots[key]
                    if state.status == "READY" and state.effective in self.levels[key]:
                        level = self.levels[key][state.effective]
                        self.router.set_capacity(
                            key, level["pool"], level["session_limit"]
                        )
                    self._log(
                        "scaling_drain_cancelled",
                        pool=key,
                        target=target,
                        observed_ns=now,
                    )
            self.last_report = {
                "observed_ns": now,
                "policy": self.policy,
                "task_rates": task_rates,
                "queued": queued,
                "utilization": utilization,
                "selected": self.plan,
                "exploration_capacity": copy.deepcopy(self.probe),
                "exploration_capacity_reserved": dict(self.router.capacity_reserved),
                "candidates": candidates,
                "reason": "selected" if selected else "no_trusted_feasible_plan",
                "cost_basis": "predicted",
                "objective": {
                    "cost_weight": self.router.cost_weight,
                    "latency_weight": self.router.latency_weight,
                    "cost_unit": self.fx["currency"] + "/second",
                    "latency_unit": "seconds/request",
                },
                "pareto": [
                    c["plan"]
                    for c in pareto(
                        [c for c in candidates if not c["flags"]],
                        "predicted_total",
                        "predicted_latency_s",
                    )
                ],
            }
            self._log("control_decision", **self.last_report)
            if self.plan and apply:
                for key, target in self.plan["targets"].items():
                    self._submit(key, target, now)
            return copy.deepcopy(self.last_report)

    def _probe_plan(self, plan, task_rates, now, *, apply):
        """Lease one promising idle pool; retain its cap until restoration is read back."""
        if not self.capacity_exploration:
            return plan
        if plan is None:
            if not self.probe:
                default = self.router.default_pool
                if default is None or any(s.target not in self.levels[k] for k, s in self.snapshots.items()):
                    return None
                plan = {"targets": {k: s.target for k, s in self.snapshots.items()},
                        "weights": {k: float(k == default) for k in self.pools}}
            else:
                plan = {
                    "targets": {k: s.target for k, s in self.snapshots.items()},
                    "weights": self.probe["restore_weights"],
                }
        plan = copy.deepcopy(plan)
        router = self.router
        if self.probe:
            p = self.probe
            state = self.snapshots.get(p["key"])
            if now >= p["expires_ns"]:
                if (
                    state
                    and state.status == "READY"
                    and state.effective == p["restore_target"]
                    and p["key"] not in self.pending
                    and not self._occupied(p["key"])
                ):
                    if apply:
                        cap = router.capacity_reserved.pop(p["key"])
                        router.spent += cap
                        router.exploration_spent += cap
                        self._log(
                            "exploration_capacity_settled",
                            **p,
                            amount=cap,
                            basis="lifecycle_cap_not_actual_bill",
                            observed_ns=now,
                        )
                        self.probe = None
                else:
                    plan["targets"][p["key"]] = p["restore_target"]
                    plan["weights"][p["key"]] = 0
                    total = sum(plan["weights"].values())
                    if total:
                        plan["weights"] = {
                            k: w / total for k, w in plan["weights"].items()
                        }
                    else:
                        plan["weights"] = p["restore_weights"]
                    return plan
            else:
                plan["targets"][p["key"]] = p["target"]
                return plan
        if not apply or self.probe or not any(task_rates.values()):
            return plan
        family = max(task_rates, key=task_rates.get)
        descriptor = self.descriptors.get(family)
        if descriptor is None:
            return plan
        options = {}
        for key, cap in self.capacity_exploration["max_cost"].items():
            if (
                plan["weights"][key]
                or self.levels[key][plan["targets"][key]]["session_limit"] > 0
                or now - self.last_change.get(key, -(10**30)) < self.cooldown_s * 1e9
                or key in self.pending
                or key in self.draining
                or self.probed_until.get(key, 0) > now
                or self._occupied(key)
            ):
                continue
            levels = [
                v
                for v in self.levels[key].values()
                if v["session_limit"] > 0
                and v["scale_delay_s"] < self.capacity_exploration["lease_s"]
            ]
            if levels:
                options[key] = min(levels, key=lambda v: v["session_limit"])
        if not options:
            return plan
        pools = {
            k: options[k]["pool"] if k in options else self.levels[k][t]["pool"]
            for k, t in plan["targets"].items()
        }
        # Preview uses proposed capacity only to price opportunities; live admission still needs readback.
        preview = OnlineRouter(
            router.workload,
            router.provider,
            pools,
            {
                k: options[k]["session_limit"] if k in options else router.limits.get(k)
                for k in pools
            },
            cost_weight=router.cost_weight,
            latency_weight=router.latency_weight,
            default_pool=router.default_pool,
            execution_bounds=router.execution_bounds,
            budget=router.budget,
            fx=router.fx,
            exploration=router.exploration.config,
        )
        preview.spent = router.spent + sum(
            d["reserved_cost"]
            for rid, d in router.active.items()
            if rid not in router.completed
        )
        preview.exploration_spent = router.exploration_spent + sum(
            d["reserved_cost"]
            for rid, d in router.active.items()
            if rid not in router.completed and d["exploration"]["explored"]
        )
        decision = preview.decide(
            f"capacity:{now}",
            descriptor=descriptor,
            as_of_ns=now,
            shadow=True,
            allowed_pools=set(options) | {k for k, w in plan["weights"].items() if w},
        )
        key = decision["selected"]
        if key not in options or not decision["exploration"]["explored"]:
            return plan
        cap = self.capacity_exploration["max_cost"][key]
        execution = decision["reserved_cost"]
        if (
            preview.spent + cap + execution > router.budget
            or preview.exploration_spent + cap + execution
            > router.exploration.config["budget"]
        ):
            return plan
        self.probe = {
            "key": key,
            "target": options[key]["target"],
            "restore_target": self.snapshots[key].target,
            "restore_weights": dict(plan["weights"]),
            "started_ns": now,
            "expires_ns": now + int(self.capacity_exploration["lease_s"] * 1e9),
        }
        self.probed_until[key] = now + int(router.exploration.config["window_s"] * 1e9)
        router.capacity_reserved[key] = cap
        plan["targets"][key] = options[key]["target"]
        self._log(
            "exploration_capacity_reserved",
            **self.probe,
            amount=cap,
            evidence=self.capacity_exploration["evidence"],
            decision=decision,
        )
        return plan

    def observe_arrival(
        self, request_id: str, task: str | None, now_ns: int, *, descriptor=None
    ) -> None:
        """Count offered traffic before harness admission, including queue rejection."""
        count(now_ns)
        if not request_id:
            raise ValueError("Arrival identity is required")
        family = family_key(descriptor)
        if descriptor is not None:
            self.descriptors[family] = copy.deepcopy(descriptor)
        with self.router.lock:
            if descriptor is not None or request_id not in self.offered_families:
                self.offered_families[request_id] = family
            if request_id not in self.offered_ids:
                self.arrivals.append((now_ns, request_id))
                self.offered_ids.add(request_id)

    def decide(
        self,
        request_id: str,
        task: str | None = None,
        load_rps: float = 0.0,
        *,
        as_of_ns=None,
        shadow=False,
        pool_load_rps=None,
        arrival_ns=None,
        descriptor=None,
    ) -> dict:
        """Apply the current split; default execution still requires fresh capacity readback."""
        now = time.time_ns() if as_of_ns is None else as_of_ns
        with self.router.lock:
            if request_id in self.decisions:
                return copy.deepcopy(self.decisions[request_id])
            self.observe_arrival(
                request_id,
                task,
                now if arrival_ns is None else arrival_ns,
                descriptor=descriptor,
            )
            cutoff = now - self.observation_s * 1e9
            while self.arrivals and self.arrivals[0][0] < cutoff:
                self.arrivals.popleft()
            while self.routed and self.routed[0][0] < cutoff:
                self.routed.popleft()
            counts = Counter(k for _, k in self.routed)
            rates = {
                k: max(
                    counts[k] / self.observation_s,
                    load_rps * (self.plan["weights"][k] if self.plan else 1),
                )
                for k in self.pools
            }
            if pool_load_rps is not None:
                if set(pool_load_rps) != set(rates) or any(
                    not number(r) for r in pool_load_rps.values()
                ):
                    raise ValueError("Invalid pool load snapshot")
                rates = {k: max(rates[k], pool_load_rps[k]) for k in rates}
            allowed = set()
            if (
                self.plan
                and self.last_tick is not None
                and now - self.last_tick <= self.period_s * 2e9
            ):
                for key, weight in self.plan["weights"].items():
                    self.deficit[key] = self.deficit[key] + weight if weight else 0
                key = max(
                    (k for k, w in self.plan["weights"].items() if w),
                    key=lambda k: (self.deficit[k], k),
                )
                self.deficit[key] -= 1
                state = self.snapshots.get(key)
                if (
                    self.plan["weights"][key]
                    and key not in self.draining
                    and key not in self.pending
                    and state is not None
                    and state.effective == self.plan["targets"][key]
                ):
                    allowed.add(key)
            if (
                self.policy == "routing"
                and self.plan
                and self.last_tick is not None
                and now - self.last_tick <= self.period_s * 2e9
            ):
                allowed = {
                    k
                    for k, state in self.snapshots.items()
                    if state.status == "READY"
                    and now - state.observed_ns <= self.period_s * 2e9
                    and k not in self.draining
                    and k not in self.pending
                    and state.effective == self.plan["targets"][k]
                }
            if self.router.default_pool:
                key = self.router.default_pool
                state = self.snapshots.get(key)
                if (
                    state is not None
                    and state.status == "READY"
                    and now - state.observed_ns <= self.period_s * 2e9
                    and key not in self.draining
                    and key not in self.pending
                ):
                    workload = self.router.workload.predict(now, descriptor=descriptor)
                    predictions = self.router.provider.predict(
                        workload["joint_profiles"],
                        [self.pools[k] for k in allowed],
                        load_rps,
                        as_of_ns=now,
                    )["pools"]
                    if (
                        not allowed
                        or workload["low_confidence"]
                        or not any(
                            not p["low_confidence"]
                            and p.get("analytical_cost")
                            and p["analytical_cost"]["complete"]
                            for p in predictions
                        )
                    ):
                        allowed = allowed | {key} if self.policy == "routing" else {key}
            exploration_allowed = set()
            if self.router.exploration:
                for key, state in self.snapshots.items():
                    if (
                        state.status == "READY"
                        and now - state.observed_ns <= self.period_s * 2e9
                        and key not in self.draining
                        and key not in self.pending
                        and self.router.limits.get(key)
                        and (
                            not self.probe
                            or key != self.probe["key"]
                            or now < self.probe["expires_ns"]
                        )
                    ):
                        exploration_allowed.add(key)
            decision = self.router.decide(
                request_id,
                task,
                load_rps,
                as_of_ns=now,
                shadow=shadow,
                pool_load_rps=rates,
                allowed_pools=allowed,
                exploration_pools=exploration_allowed,
                arrival_ns=arrival_ns,
                descriptor=descriptor,
            )
            if decision["selected"] and not shadow:
                self.routed.append((now, decision["selected"]))
            return decision

    def observed_rates(self, *, now_ns=None) -> dict:
        """Use offered arrivals, including rejected requests, in the trailing window."""
        now = time.time_ns() if now_ns is None else now_ns
        with self.router.lock:
            return {
                t: n / self.observation_s
                for t, n in Counter(
                    self.offered_families[rid]
                    for stamp, rid in self.arrivals
                    if now - self.observation_s * 1e9 <= stamp <= now
                ).items()
            }
