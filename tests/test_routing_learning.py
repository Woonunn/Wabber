"""Cold-start learning from completed requests, without semantic or cloud API calls."""

import copy
import unittest

from atsuite.provider_model import ProviderModel
from atsuite.routing import OnlineRouter
from atsuite.routing.exploration import Exploration
from atsuite.workload import WorkloadModel
from atsuite.workload.descriptor import compatible, context_key
from tests.stage6_fixtures import bounds, tariff
from tests.test_provider_model import cost, observation, pool, profile
from tests.test_stage7 import router_fixture
from tests.test_workload import trace
from tests.workload_fixtures import calibrate, describe


class RoutingLearningTests(unittest.TestCase):
    def test_model_revision_is_removed_and_old_snapshots_do_not_gate_confidence(self):
        descriptor = describe(
            model={"name": "model", "revision": "ignored", "temperature": 0}
        )
        self.assertNotIn("revision", descriptor["model"])
        legacy = copy.deepcopy(descriptor)
        legacy["model"]["revision"] = "old-metadata"
        self.assertEqual(context_key(legacy), context_key(descriptor))
        model = WorkloadModel()
        for i in range(3):
            p = profile()
            p["request_id"] = str(i)
            model.update(p, 100, descriptor=descriptor)
        calibrate(model, legacy)
        prediction = model.predict(101, descriptor=descriptor)
        self.assertFalse(prediction["low_confidence"])
        self.assertEqual(prediction["missing_context"], [])

    def test_exploration_shares_prompts_and_decays_but_keeps_isolation(self):
        router, descriptor = router_fixture()
        changed = describe(prompt="Compute the sample average", tools=["python"])
        self.assertNotEqual(context_key(changed), context_key(descriptor))
        self.assertEqual(
            router.exploration.evidence(
                router.provider, router.pools["known"], changed, 1000
            )["attempts"],
            3,
        )
        decision = router.decide("paraphrase", descriptor=changed, as_of_ns=1000)
        self.assertLess(decision["exploration"]["epsilon"], 1)
        for field, value in [("model", {"name": "other"})]:
            isolated = {**changed, field: value}
            self.assertEqual(
                router.exploration.evidence(
                    router.provider, router.pools["known"], isolated, 1000
                )["attempts"],
                0,
            )
        self.assertEqual(
            router.exploration.evidence(
                router.provider,
                router.pools["known"],
                {**changed, "control": {"max_iterations": 999}},
                1000,
            )["attempts"],
            3,
        )
        huge = describe(
            prompt=changed["prompt"],
            tools=["python"],
            files=[{"name": "large.csv", "bytes": 10**12}],
        )
        self.assertEqual(
            router.exploration.evidence(
                router.provider, router.pools["known"], huge, 1000
            )["attempts"],
            0,
        )
        self.assertEqual(
            router.exploration.evidence(
                router.provider, router.pools["known"], changed, 10
            )["attempts"],
            0,
        )

    def test_maximum_optimism_does_not_turn_any_cold_candidate_into_zero_cost(self):
        policy = Exploration({"optimism": 1})
        candidates = [
            {
                "key": key,
                "objective_cost": amount,
                "objective_latency": None,
                "execution_bounds": {"timeout_s": 900},
                "flags": [],
                "currency": "USD",
                "evidence": {"attempts": 0, "successes": 0},
            }
            for key, amount in [("cheap", 1), ("expensive", 100)]
        ]
        self.assertEqual(
            policy.opportunities(
                candidates, candidates[0], cost_weight=1, latency_weight=0
            ),
            [],
        )

    def test_sparse_probe_waits_for_pending_feedback_when_an_incumbent_exists(self):
        policy = Exploration({"min_samples": 3})
        candidates = [
            {
                "key": key,
                "objective_cost": 1,
                "cost": {"basis": "predicted"},
                "objective_latency": None,
                "execution_bounds": {"timeout_s": 900},
                "flags": [],
                "currency": "USD",
                "evidence": {"attempts": 0, "successes": 0, "pending_feedback": 0},
            }
            for key in ["established", "new-arm"]
        ]
        def opportunities(incumbent):
            return policy.opportunities(candidates, incumbent, cost_weight=1, latency_weight=0)

        self.assertEqual([o["key"] for o in opportunities(candidates[0])], ["new-arm"])
        candidates[1]["evidence"]["pending_feedback"] = 1
        self.assertEqual(opportunities(candidates[0]), [])
        self.assertTrue(opportunities(None))
        self.assertEqual(candidates[1]["evidence"]["attempts"], 0)
        candidates[0]["cost"]["basis"] = "analytical"
        self.assertEqual([o["key"] for o in opportunities(candidates[0])], ["new-arm"])
        candidates[0]["cost"]["basis"] = "predicted"
        candidates[1]["evidence"].update(pending_feedback=0, attempts=1, successes=1)
        self.assertEqual([o["key"] for o in opportunities(candidates[0])], ["new-arm"])
        candidates[1]["evidence"].update(pending_feedback=1, attempts=3, successes=3)
        self.assertEqual([o["key"] for o in opportunities(candidates[0])], ["new-arm"])

    def test_failed_zero_cost_is_retained_with_its_original_basis(self):
        router, descriptor = router_fixture(epsilon=0)
        decision = router.decide("failed-free", descriptor=descriptor, as_of_ns=1000)
        completed = trace("failed-free", arrival=1000, end=2000)
        completed["request_status"] = "failed"
        row = router.complete(
            "failed-free",
            completed,
            available_ns=2000,
            actual_cost=0,
            cost_basis="synthetic_usage_estimate",
        )
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["cost"]["basis"], "estimated")
        self.assertEqual(row["cost"]["components"], {"execution": 0})
        stored = [
            history[-1]
            for key, history in router.provider.samples.items()
            if key[3] == "failed-free"
        ]
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["pool"], router.pools[decision["selected"]])
        self.assertEqual(stored[0]["cost"], row["cost"])

    def test_native_costs_survive_tariffs_and_exclude_future_partial_and_pool_totals(
        self,
    ):
        model = ProviderModel(minimum_samples=3)
        target = {**pool(), "billing": tariff()}
        for i in range(3):
            row = observation(i)
            row["pool"] = target
            row["cost"] = {**cost(i + 1, complete=True), "scope": "request_execution"}
            model.update(row)
        query = profile()
        query["metrics"]["state_max_object_bytes"] = None
        before = model.predict(query, [target], 0.5, as_of_ns=10)["pools"][0]
        self.assertEqual(before["decision_cost"]["basis"], "analytical")
        self.assertFalse(before["confidence"]["cost"]["reliable"])
        prediction = model.predict(query, [target], 0.5, as_of_ns=1000)["pools"][0]
        self.assertFalse(prediction["analytical_cost"]["complete"])
        self.assertEqual(prediction["decision_cost"]["total"]["mean"], 2)
        self.assertTrue(prediction["confidence"]["cost"]["reliable"])
        self.assertTrue(prediction["historical_cost"])
        for scope, complete in [("request_execution", False), ("pool_total", True)]:
            excluded = ProviderModel(minimum_samples=3)
            for i in range(3):
                row = observation(i)
                row["pool"] = target
                row["cost"] = {**cost(0, complete=complete), "scope": scope}
                excluded.update(row)
            p = excluded.predict(query, [target], 0.5, as_of_ns=1000)["pools"][0]
            self.assertNotEqual(p["decision_cost"]["basis"], "predicted")

    def test_distinct_requests_exit_cold_start_and_route_using_completed_costs(self):
        class RecordedBehavior:
            revision = "test-similar-executions"

            def select(self, descriptor, profiles):
                samples = [
                    p for p in profiles if compatible(descriptor, p["descriptor"])
                ][-20:]
                return samples, {
                    "revision": self.revision,
                    "cost": 0,
                    "currency": None,
                    "latency_s": 0,
                    "match_basis": "fixture",
                    "cost_complete": True,
                }

        workload = WorkloadModel(retriever=RecordedBehavior())
        pools = {
            key: {**pool(), "environment": key, "billing": tariff()}
            for key in ["default", "learned"]
        }
        router = OnlineRouter(
            workload,
            ProviderModel(minimum_samples=3),
            pools,
            dict.fromkeys(pools, 10),
            default_pool="default",
            latency_weight=0,
            enable_context_routing=True,
            execution_bounds={key: bounds() for key in pools},
        )
        decisions = []
        for i in range(16):
            stamp = 10000 * (i + 1)
            descriptor = describe(prompt=f"Compute average for input {i}")
            fallback = "learned" if 3 <= i < 6 else "default"
            decision = router.decide(
                str(i), descriptor=descriptor, as_of_ns=stamp, fallback_pool=fallback
            )
            decisions.append(decision)
            t = trace(str(i), arrival=stamp, end=stamp + 1000)
            next(s for s in t["spans"] if s["kind"] == "tool")["metadata"][
                "state_accesses"
            ] = []
            amount = 0.2 if decision["selected"] == "learned" else 1
            row = router.complete(
                str(i), t, available_ns=stamp + 1000, actual_cost=amount
            )
            self.assertEqual(row["cost"]["components"]["execution"], amount)
        self.assertEqual(decisions[-1]["reason"], "cost_latency_objective")
        self.assertEqual(decisions[-1]["selected"], "learned")
        self.assertFalse(decisions[-1]["workload"]["low_confidence"])
        self.assertTrue(
            all(
                c["prediction"]["decision_cost"]["basis"] == "predicted"
                for c in decisions[-1]["candidates"]
            )
        )
