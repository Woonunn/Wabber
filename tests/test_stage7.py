"""Semantic matching, causal exploration and lifecycle accounting without cloud RPC."""

import copy
import tempfile
import unittest
from pathlib import Path

from atsuite.provider_model import ProviderModel
from atsuite.routing import OnlineRouter
from atsuite.workload import WorkloadModel
from atsuite.workload.descriptor import (
    family_key,
)
from tests.stage6_fixtures import bounds, tariff
from tests.test_provider_model import observation, pool, profile
from tests.test_workload import trace
from tests.workload_fixtures import describe


def router_fixture(
    *,
    strategy="contextual",
    epsilon=1,
    budget=10,
    exploration_budget=3,
    promising_rate=0.0022,
):
    descriptor = describe(prompt="Calculate mean and variance", tools=["python"])
    workload, provider = WorkloadModel(), ProviderModel()
    pools = {}
    # Exploration tests need an optimistic alternative, not an already-cheaper excluded incumbent.
    for name, rate in (
        ("known", 0.002),
        ("promising", promising_rate),
        ("expensive", 0.1),
    ):
        pools[name] = {**pool(), "environment": name, "billing": tariff()}
        pools[name]["billing"]["rates"]["memory_gb_s"] = rate
    for i in range(3):
        p = profile()
        p["request_id"] = str(i)
        workload.update(p, 500, descriptor=descriptor)
        row = observation(i)
        row["pool"] = pools["known"]
        row["profile"]["descriptor"] = descriptor
        provider.update(row)
    workload.errors = [{"request_id": "calibration:" + str(i), "available_ns": 499,
                        "descriptor": descriptor, "revision": workload.retriever.revision,
                        "metrics": {k: 0.1 for k in ("tool_elapsed_s", "state_bytes", "llm_idle_s")}}
                       for i in range(10)]
    router = OnlineRouter(
        workload,
        provider,
        pools,
        dict.fromkeys(pools, 3),
        default_pool="known",
        enable_context_routing=True,
        execution_bounds={k: bounds() for k in pools},
        budget=budget,
        exploration={
            "strategy": strategy,
            "epsilon": epsilon,
            "budget": exploration_budget,
            "seed": 17,
        },
    )
    return router, descriptor


class ExplorationTests(unittest.TestCase):
    def test_promising_only_probabilities_and_idempotence(self):
        router, d = router_fixture()
        result = router.decide("query", descriptor=d, as_of_ns=1000)
        e = result["exploration"]
        self.assertEqual([o["key"] for o in e["opportunities"]], ["promising"])
        self.assertAlmostEqual(sum(e["probabilities"].values()), 1)
        self.assertEqual(e["probabilities"]["expensive"], 0)
        self.assertEqual(result, router.decide("query", descriptor=d, as_of_ns=2000))
        self.assertIn(result["selected"], {"known", "promising"})

    def test_budget_concurrency_tool_memory_deadline_and_shadow(self):
        router, d = router_fixture(exploration_budget=1)
        decisions = [
            router.decide(f"q{i}", descriptor=d, as_of_ns=1000) for i in range(8)
        ]
        self.assertEqual(sum(r["exploration"]["explored"] for r in decisions), 1)
        router, d = router_fixture()
        router.limits["promising"] = 0
        self.assertFalse(
            router.decide("none", descriptor=d, as_of_ns=1000)["exploration"][
                "opportunities"
            ]
        )
        for field, val in (("tools", ["untrusted"]), ("hints", {"memory_mb": 999999})):
            invalid = describe(prompt=d["prompt"], tools=val) if field == "tools" else {**d, field: val}
            self.assertIsNone(
                router.decide(field, descriptor=invalid, as_of_ns=1000)["selected"]
            )
        self.assertIsNone(
            router.decide("late", descriptor=d, as_of_ns=10**12, arrival_ns=0)[
                "selected"
            ]
        )
        router, d = router_fixture()
        before = copy.deepcopy(router.active)
        router.decide("shadow", descriptor=d, as_of_ns=1000, shadow=True)
        self.assertEqual(before, router.active)
        self.assertEqual(router.exploration_spent, 0)

    def test_feedback_failed_attempt_and_strict_future_cutoff(self):
        router, d = router_fixture()
        decision = next(
            r
            for i in range(20)
            if (r := router.decide(f"q{i}", descriptor=d, as_of_ns=1000))[
                "exploration"
            ]["explored"]
        )
        rid = decision["request_id"]
        before = router.exploration.evidence(
            router.provider, router.pools["promising"], d, 2000
        )
        t = trace(rid, arrival=1000, end=2000)
        t["request_status"] = "failed"
        router.complete(rid, t, available_ns=2000, actual_cost=0.2)
        self.assertEqual(router.exploration_spent, 0.2)
        self.assertEqual(
            router.exploration.evidence(
                router.provider, router.pools["promising"], d, 2000
            ),
            before,
        )
        after = router.exploration.evidence(
            router.provider, router.pools["promising"], d, 2001
        )
        self.assertEqual(after["attempts"], 1)
        self.assertEqual(after["successes"], 0)
        self.assertFalse(router.active.get(rid))
        self.assertEqual(router.feedback[rid]["basis"], "attributable_execution_cost")
        with self.assertRaises(ValueError):
            router.complete(rid, t, available_ns=2001)

    def test_evidence_decays_and_environment_drift_resets(self):
        router, d = router_fixture()
        evidence = router.exploration.evidence(
            router.provider, router.pools["known"], d, 1000
        )
        self.assertEqual(evidence["attempts"], 3)
        future = router.exploration.evidence(
            router.provider, router.pools["known"], d, 4000 * 10**9
        )
        self.assertEqual(future["attempts"], 0)
        changed = copy.deepcopy(router.pools["known"])
        changed["resources"]["memory_mb"] *= 2
        self.assertEqual(
            router.exploration.evidence(router.provider, changed, d, 1000)["attempts"],
            0,
        )
        router.exploration.config["epoch_ns"] = 900
        self.assertEqual(
            router.exploration.evidence(
                router.provider, router.pools["known"], d, 1000
            )["attempts"],
            0,
        )

    def test_random_baseline_and_disabled_policy(self):
        router, d = router_fixture(strategy="random")
        result = router.decide("q", descriptor=d, as_of_ns=1000)
        self.assertEqual(len(result["exploration"]["opportunities"]), 2)
        router.exploration = None
        result = router.decide("q2", descriptor=d, as_of_ns=1000)
        self.assertEqual(result["selected"], "known")
        self.assertEqual(result["exploration"]["selection_probability"], 1)

    def test_joint_can_explore_ready_pool_outside_traffic_split(self):
        from tests.test_joint_control import controller

        control, _ = controller()
        router, d = router_fixture()
        control.router = router
        from atsuite.routing import Capacity

        control.snapshots = {
            k: Capacity(1000, "reserved_concurrency", 1, 1, status="READY")
            for k in router.pools
        }
        control.plan = {
            "targets": dict.fromkeys(router.pools, 1),
            "weights": {"known": 1, "promising": 0, "expensive": 0},
        }
        control.last_tick = 1000
        results = [
            control.decide(f"q{i}", descriptor=d, as_of_ns=1001) for i in range(8)
        ]
        self.assertTrue(any(r["selected"] == "promising" for r in results))
        self.assertFalse(any(r["selected"] == "expensive" for r in results))

    def test_learned_pool_can_be_exploited_under_fixed_capacity(self):
        from atsuite.routing import Capacity
        from tests.test_joint_control import controller

        control, _ = controller(policy="routing")
        router, d = router_fixture(epsilon=0, promising_rate=0.001)
        control.router = router
        for i in range(3):
            row = observation(i)
            row.update(id=f"learned-{i}", pool=router.pools["promising"], load_rps=0)
            row["profile"]["descriptor"] = d
            router.provider.update(row)
        control.snapshots = {
            k: Capacity(1000, "reserved_concurrency", 1, 1, status="READY")
            for k in router.pools
        }
        control.plan = {
            "targets": dict.fromkeys(router.pools, 1),
            "weights": {"known": 1, "promising": 0, "expensive": 0},
        }
        control.last_tick = 1000
        result = control.decide(
            "exploit-learned", descriptor=d, load_rps=0.2, as_of_ns=1001
        )
        self.assertEqual(result["selected"], "promising")
        self.assertFalse(result["exploration"]["explored"])

    def test_price_change_reopens_evidence_epoch_without_ignoring_budget(self):
        router, d = router_fixture()
        before = router.decide("before", descriptor=d, as_of_ns=1000, shadow=True)
        router.pools["promising"]["billing"]["rates"]["memory_gb_s"] /= 2
        after = router.decide("after", descriptor=d, as_of_ns=1001)
        self.assertNotEqual(before["model_revision"], after["model_revision"])
        self.assertEqual(after["exploration"]["config"]["epoch_ns"], 1001)
        self.assertTrue(
            all(c["evidence"]["attempts"] == 0 for c in after["candidates"])
        )

    def test_dominated_pool_stops_receiving_contextual_probability(self):
        router, d = router_fixture(promising_rate=0.001)
        for i in range(20):
            row = observation(i)
            row.update(id=f"evidence-{i}", pool=router.pools["promising"])
            row["profile"]["descriptor"] = d
            router.provider.update(row)
        result = router.decide("converged", descriptor=d, load_rps=0.5, as_of_ns=3000)
        self.assertEqual(result["selected"], "promising")
        self.assertEqual(
            result["exploration"]["probabilities"],
            {"known": 0, "promising": 1, "expensive": 0},
        )


class CapacityExplorationTests(unittest.TestCase):
    def test_lease_reserves_before_scaling_and_settles_only_after_readback(self):
        from atsuite.routing import JointController, candidate_plans
        from tests.test_joint_control import FX, DelayedAdapter

        router, descriptor = router_fixture()
        levels, adapters = {}, {}
        for key, pool_config in router.pools.items():
            levels[key] = {}
            for target in (0, 1):
                config = copy.deepcopy(pool_config)
                config["instances"] = None
                config["resources"]["capacity_control"] = {
                    "control": "reserved_concurrency",
                    "target": target,
                }
                levels[key][target] = {
                    "target": target,
                    "pool": config,
                    "session_limit": target * 3,
                    "scale_delay_s": 0.1,
                    "evidence": ["synthetic readback"],
                }
            initial = int(key == "known")
            adapters[key] = DelayedAdapter(initial)
            router.set_capacity(key, levels[key][initial]["pool"], initial * 3)
        for history in router.provider.samples.values():
            history[0]["pool"] = levels["known"][1]["pool"]
        plan = {
            "targets": {"known": 1, "promising": 0, "expensive": 0},
            "weights": {"known": 1, "promising": 0, "expensive": 0},
        }
        control = JointController(
            router,
            adapters,
            levels,
            candidate_plans(levels, [plan["weights"]]),
            FX,
            period_s=1,
            cooldown_s=0,
            capacity_exploration={
                "lease_s": 5,
                "max_cost": {"promising": 0.25},
                "evidence": ["full lifecycle cap"],
            },
        )
        family = family_key(descriptor)
        control.descriptors[family] = descriptor
        for now in range(1000, 1020):
            control._refresh(now)
            proposed = control._probe_plan(plan, {family: 0.1}, now, apply=True)
            if control.probe:
                break
        self.assertIsNotNone(control.probe)
        self.assertEqual(router.capacity_reserved, {"promising": 0.25})
        self.assertEqual(adapters["promising"].calls, [])
        control._submit("promising", 1, now)
        self.assertIsNone(router.limits["promising"])
        adapters["promising"].effective = 1
        control._refresh(now + 1)
        self.assertEqual(router.limits["promising"], 3)
        deadline = control.probe["expires_ns"]
        restore = control._probe_plan(proposed, {family: 0}, deadline, apply=True)
        self.assertEqual(restore["targets"]["promising"], 0)
        control._submit("promising", 0, deadline)
        control._probe_plan(restore, {family: 0}, deadline + 1, apply=True)
        self.assertEqual(router.exploration_spent, 0)
        self.assertIn("promising", router.capacity_reserved)
        adapters["promising"].effective = 0
        control._refresh(deadline + 2)
        control._probe_plan(None, {family: 0}, deadline + 3, apply=True)
        self.assertEqual(router.capacity_reserved, {})
        self.assertEqual(router.exploration_spent, 0.25)
        self.assertIsNone(control.probe)


class ExperimentTests(unittest.TestCase):
    def test_entire_family_holdout_and_no_future_outcome_training(self):
        from atsuite.routing.generalization import evaluate

        rows = []
        descriptors = {
            "stat": describe(prompt="Calculate mean"),
            "train-a": describe(prompt="Train classifier"),
            "train-b": describe(prompt="Fit regression"),
        }
        for i, name in enumerate(("stat", "stat", "stat", "train-a", "train-b")):
            r = observation(i)
            r["task_id"] = name
            rows.append(r)
        before = evaluate(
            rows, descriptors, {"aws_lambda": tariff()}, held_family="training", family_labels={"stat": "statistics", "train-a": "training", "train-b": "training"}
        )
        self.assertEqual(before["held_tasks"], ["train-a", "train-b"])
        self.assertTrue(
            all(
                p["training_count"] == 3 and p["unknown_request"]
                for p in before["predictions"]
            )
        )
        for row in rows[3:]:
            row["profile"]["metrics"]["tool_elapsed_s"] = 1000000
        after = evaluate(
            rows, descriptors, {"aws_lambda": tariff()}, held_family="training", family_labels={"stat": "statistics", "train-a": "training", "train-b": "training"}
        )
        self.assertEqual(
            [p["cost"] for p in before["predictions"]],
            [p["cost"] for p in after["predictions"]],
        )

    def test_strategy_configs_are_independent_and_panel_is_real(self):
        from atsuite.routing.experiment import STRATEGIES, strategy_config
        from experiments.exp1.runner.stage7 import prompt_panel

        config = {
            "pools": {
                k: {"levels": [{"pool": pool(k)}]}
                for k in ("aws_lambda", "aws_agentcore", "ali_fc", "ali_agentrun")
            },
            "routing": {"exploration": {"budget": 1}},
            "control": {"capacity_exploration": {"lease_s": 10}},
        }
        original = copy.deepcopy(config)
        for strategy in STRATEGIES:
            variant, policy = strategy_config(config, strategy)
            self.assertEqual(variant["strategy"], strategy)
            if strategy.startswith("fixed"):
                self.assertEqual(policy, "static")
                self.assertEqual(sum(variant["weights"][0].values()), 1)
                self.assertNotIn("exploration", variant["routing"])
            if strategy != "full_wabber":
                self.assertNotIn("capacity_exploration", variant["control"])
        self.assertEqual(config, original)
        panel = prompt_panel()
        self.assertGreaterEqual(len(panel["cases"]), 10)
        self.assertTrue(panel["source"]["commit_sha"])
        self.assertTrue(
            any(
                c["descriptor"]["prompt"] == c["prompt"]
                for c in panel["cases"]
            )
        )

    def test_accounting_requires_all_components_resources_and_full_window(self):
        from experiments.exp1.runner.stage7 import COMPONENTS, collect, write
        from tests.test_joint_control import FX

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(
                root / "run.json",
                {
                    "policy": "joint",
                    "workload_sha256": "w",
                    "config_sha256": "c",
                    "started_ns": 1,
                    "finished_ns": 10,
                    "cost_resources": [{"provider": "aws_lambda", "resource": "r"}],
                },
            )
            (root / "control.jsonl").write_text("")
            charges = [
                {
                    "provider": "aws_lambda",
                    "resource": "r",
                    "charge_id": k,
                    "component": k,
                    "start_ns": 1,
                    "end_ns": 10,
                    "amount": 1,
                    "currency": "USD",
                    "basis": "actual",
                    "attribution": "exclusive",
                    "evidence": ["fixture"],
                    "complete": True,
                }
                for k in COMPONENTS
            ]
            config = {"fx": FX, "pools": {"aws_lambda": {}}}
            self.assertEqual(collect(root, config, charges)["cost"]["total"], 5)
            self.assertIsNone(collect(root, config, charges[:-1])["cost"]["total"])
            for c in charges:
                c["end_ns"] = 9
            self.assertIsNone(collect(root, config, charges)["cost"]["total"])

    def test_comparison_does_not_invent_missing_cost_or_compare_different_arrivals(
        self,
    ):
        from experiments.exp1.runner.stage7 import compare

        baseline = {
            "strategy": "cost_aware",
            "comparison_sha256": "common",
            "deployment_sha256": "deployment",
            "workload_sha256": "w",
            "exploration_attempts": 0,
            "selection_proportions": {"a": 1},
            "metrics": {"p95_s": 10, "success_rate": 1},
            "cost": {
                "complete": True,
                "currency": "USD",
                "basis": "actual",
                "total": 2,
            },
        }
        candidate = copy.deepcopy(baseline)
        candidate["strategy"] = "full_wabber"
        candidate["cost"]["total"] = 1
        self.assertEqual(
            compare([baseline, candidate])["comparisons"][1][
                "cost_saving_vs_no_exploration"
            ],
            1,
        )
        candidate["cost"]["complete"] = False
        self.assertIsNone(
            compare([baseline, candidate])["comparisons"][1][
                "cost_saving_vs_no_exploration"
            ]
        )
        candidate["workload_sha256"] = "other"
        with self.assertRaises(ValueError):
            compare([baseline, candidate])


if __name__ == "__main__":
    unittest.main()
