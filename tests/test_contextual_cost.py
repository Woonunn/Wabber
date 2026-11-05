"""Contextual charge learning on unseen inputs; fixtures do not test language models."""

import copy
import math
import unittest

from atsuite.provider_model import ProviderModel
from atsuite.routing import OnlineRouter
from atsuite.routing.exploration import Exploration
from atsuite.workload import WorkloadModel
from atsuite.workload.descriptor import compatible, resource_distance
from tests.stage6_fixtures import bounds, tariff
from tests.test_provider_model import cost, observation, pool
from tests.test_workload import trace
from tests.workload_fixtures import describe


class DeclaredScaleRetriever:
    revision = "independent-numeric-context-fixture"

    def select(self, descriptor, profiles):
        eligible = [p for p in profiles if compatible(descriptor, p["descriptor"])]
        eligible.sort(
            key=lambda p: abs(
                math.log(p["descriptor"]["input_bytes"] / descriptor["input_bytes"])
            )
        )
        return [
            p for p in eligible if resource_distance(descriptor, p["descriptor"]) <= 0.5
        ], {
            "revision": self.revision,
            "cost": 0,
            "currency": None,
            "latency_s": 0,
            "match_basis": "numeric-test-context",
            "cost_complete": True,
            "context_neighbors": [p["request_id"] for p in eligible],
        }


class ContextualCostTests(unittest.TestCase):
    def test_comparable_paid_input_scale_is_not_diluted_by_smaller_inputs(self):
        provider = ProviderModel(minimum_samples=3)
        for i, (size, amount) in enumerate([(100, .01), (200, .02), (10**6, 1)]):
            row = observation(i)
            row["profile"]["request_id"] = row["id"]
            row["profile"]["descriptor"] = describe(files=[{"name": "data.bin", "bytes": size}])
            row["cost"] = {**cost(amount, complete=True), "scope": "request_execution"}
            provider.update(row)
        workload = {
            "joint_profiles": [],
            "descriptor": describe(files=[{"name": "new.bin", "bytes": 10**6}]),
            "retrieval": {"context_neighbors": ["0", "1", "2"]},
        }
        prediction = provider.predict(workload, [pool()], .5, as_of_ns=1000)["pools"][0]
        self.assertEqual(prediction["decision_cost"]["training_request_ids"], ["2"])
        self.assertEqual(prediction["decision_cost"]["total"]["mean"], 1)
        self.assertFalse(prediction["confidence"]["cost"]["reliable"])
        self.assertIn("sample_insufficient", prediction["confidence"]["cost"]["reasons"])
        self.assertNotIn("input_scale_outside_support", prediction["confidence"]["cost"]["reasons"])

    def test_paid_failed_or_partial_attempts_supply_cost_without_resource_scenarios(self):
        for status in ["failed", "incomplete"]:
            with self.subTest(status=status):
                workload, provider = WorkloadModel(), ProviderModel(minimum_samples=3)
                descriptor = describe(prompt="A paid attempt whose result is unavailable")
                row = observation(0)
                row["status"] = row["profile"]["status"] = status
                row["profile"]["request_id"] = row["id"]
                row["profile"]["descriptor"] = descriptor
                row["cost"] = {**cost(.25, complete=True), "scope": "request_execution"}
                workload.update(row["profile"], row["available_ns"], descriptor=descriptor)
                provider.update(row)

                before = workload.predict(row["available_ns"], descriptor=descriptor)
                self.assertEqual(before["retrieval"].get("context_neighbors", []), [])
                now = row["available_ns"] + 1
                prediction = workload.predict(now, descriptor=descriptor)
                self.assertEqual(prediction["joint_profiles"], [])
                self.assertEqual(prediction["eligible_success_count"], 0)
                self.assertEqual(prediction["retrieval"]["context_neighbors"], [row["id"]])
                result = provider.predict(prediction, [row["pool"]], .5, as_of_ns=now)["pools"][0]
                self.assertEqual(result["decision_cost"]["total"]["mean"], .25)
                self.assertEqual(result["decision_cost"]["training_request_ids"], [row["id"]])
                self.assertFalse(result["confidence"]["cost"]["reliable"])
                self.assertEqual(result["latency"]["n"], 0)

    def fixture(self, identity):
        workload = WorkloadModel(retriever=DeclaredScaleRetriever())
        provider = ProviderModel(minimum_samples=3)
        pools = {
            name: {
                **pool(),
                "environment": f"new-{identity}-{name}",
                "billing": tariff(),
            }
            for name in ["east", "west"]
        }
        for i, (name, scale) in enumerate(
            (name, scale) for name in pools for scale in [1, 2, 4, 8]
        ):
            descriptor = describe(
                prompt=f"Prior unseen operation {identity}-{i}",
                tools=[f"new-tool-{identity}"],
                files=[{"name": f"old-{i}.csv", "bytes": scale * 100}],
            )
            row = observation(i)
            row["pool"] = pools[name]
            row["profile"]["request_id"] = row["id"]
            row["profile"]["descriptor"] = descriptor
            row["profile"]["metrics"].update(
                state_bytes=None, state_max_object_bytes=None
            )
            amount = scale * 0.01 if name == "east" else 0.04
            row["cost"] = {**cost(amount, complete=True), "scope": "request_execution"}
            provider.update(row)
            workload.update(row["profile"], row["available_ns"], descriptor=descriptor)
        router = OnlineRouter(
            workload,
            provider,
            pools,
            dict.fromkeys(pools, 10),
            default_pool="east",
            latency_weight=0,
            enable_context_routing=True,
            execution_bounds={
                name: {**bounds(), "tools": [f"new-tool-{identity}"]} for name in pools
            },
        )
        return router

    def test_new_tools_and_unseen_sizes_learn_from_prior_context_charges(self):
        for identity in ["alpha", "beta", "gamma", "delta"]:
            router = self.fixture(identity)
            for scale, expected in [(3, "east"), (6, "west")]:
                with self.subTest(identity=identity, scale=scale):
                    descriptor = describe(
                        prompt=f"A completely new request {identity}-{scale}",
                        tools=[f"new-tool-{identity}"],
                        files=[{"name": "new-held-out.csv", "bytes": scale * 100}],
                    )
                    decision = router.decide(
                        f"new-{scale}",
                        descriptor=descriptor,
                        as_of_ns=10000,
                        shadow=True,
                    )
                    self.assertEqual(decision["selected"], expected)
                    self.assertEqual(decision["reason"], "cost_latency_objective")
                    self.assertTrue(decision["workload"]["low_confidence"])
                    for candidate in decision["candidates"]:
                        prediction = candidate["prediction"]["decision_cost"]
                        self.assertTrue(prediction["contextual"])
                        self.assertEqual(len(prediction["training_request_ids"]), 3)
                        self.assertNotIn(
                            f"new-{scale}", prediction["training_request_ids"]
                        )

    def test_unseen_scale_keeps_low_confidence_cost_values_without_resource_extrapolation(
        self,
    ):
        router = self.fixture("new-scale")
        descriptor = describe(
            prompt="A new input size",
            tools=["new-tool-new-scale"],
            files=[{"name": "large.csv", "bytes": 10**10}],
        )
        decision = router.decide(
            "new-scale", descriptor=descriptor, as_of_ns=10000, shadow=True
        )
        self.assertEqual(decision["selected"], "west")
        self.assertEqual(decision["reason"], "contextual_bandit_value")
        self.assertEqual(decision["workload"]["joint_profiles"], [])
        for candidate in decision["candidates"]:
            confidence = candidate["prediction_diagnostics"]["cost"]
            self.assertFalse(confidence["reliable"])
            self.assertIn("input_scale_outside_support", confidence["reasons"])
            self.assertEqual(candidate["prediction"]["latency"]["n"], 0)

    def test_outside_context_prices_inform_actions_without_claiming_scale_support(self):
        router = self.fixture("cold-prior")
        router.exploration = Exploration({"epsilon": .35, "seed": 9})
        descriptor = describe(prompt="A previously unseen input range", tools=["new-tool-cold-prior"],
                              files=[{"name": "large.csv", "bytes": 10**10}])
        decision = router.decide("cold-prior", descriptor=descriptor, as_of_ns=10000, shadow=True)
        self.assertEqual(decision["exploration"]["incumbent"], "west")
        self.assertGreaterEqual(decision["exploration"]["probabilities"]["west"], .65)
        for candidate in decision["candidates"]:
            self.assertEqual(candidate["evidence"]["attempts"], 0)
            self.assertGreater(candidate["prediction"]["decision_cost"]["total"]["n"], 0)
            self.assertFalse(candidate["prediction_diagnostics"]["cost"]["reliable"])
            self.assertIn("input_scale_outside_support", candidate["prediction_diagnostics"]["cost"]["reasons"])

    def test_unknown_model_or_future_fees_do_not_create_confident_prices(self):
        router = self.fixture("isolated")
        descriptor = describe(
            prompt="New",
            tools=["new-tool-isolated"],
            files=[{"name": "new.csv", "bytes": 300}],
        )
        decision = router.decide(
            "early", descriptor=descriptor, as_of_ns=10, shadow=True
        )
        self.assertEqual(decision["reason"], "validated_default")
        self.assertTrue(
            all(
                not c["prediction"]["confidence"]["cost"]["reliable"]
                for c in decision["candidates"]
            )
        )
        changed = copy.deepcopy(descriptor)
        changed["model"]["name"] = "unseen-planner"
        decision = router.decide(
            "unknown-model", descriptor=changed, as_of_ns=10000, shadow=True
        )
        self.assertEqual(decision["reason"], "validated_default")
        self.assertTrue(
            all(
                not c["prediction"]["confidence"]["cost"]["reliable"]
                for c in decision["candidates"]
            )
        )

    def test_seed_and_retry_budgets_remain_context_features_not_empty_price_groups(
        self,
    ):
        router = self.fixture("sampling")
        descriptor = describe(
            prompt="New request with a fresh seed",
            tools=["new-tool-sampling"],
            files=[{"name": "new.csv", "bytes": 600}],
            model={"name": "synthetic-model", "seed": 12345, "retries": 3},
        )
        decision = router.decide(
            "fresh-seed", descriptor=descriptor, as_of_ns=10000, shadow=True
        )
        self.assertEqual(decision["reason"], "cost_latency_objective")
        self.assertEqual(decision["selected"], "west")

    def test_one_observed_arm_value_is_not_reported_as_a_matched_price_comparison(self):
        router = self.fixture("comparison")
        router.provider.samples = {
            key: rows
            for key, rows in router.provider.samples.items()
            if rows[-1]["pool"]["environment"].endswith("west")
        }
        descriptor = describe(
            prompt="Unknown default price",
            tools=["new-tool-comparison"],
            files=[{"name": "new.csv", "bytes": 600}],
        )
        decision = router.decide(
            "one-priced-arm", descriptor=descriptor, as_of_ns=10000, shadow=True
        )
        self.assertEqual(decision["selected"], "west")
        self.assertEqual(decision["reason"], "contextual_bandit_value")
        self.assertFalse(decision["cost_comparable"])

    def test_first_arm_reward_updates_value_without_claiming_high_confidence_and_recovers(
        self,
    ):
        router = self.fixture("sparse")
        for history in router.provider.samples.values():
            row = history[-1]
            if row["profile"]["descriptor"]["input_bytes"] != 800:
                row["cost"] = None
            elif row["pool"]["environment"].endswith("east"):
                row["cost"]["components"] = {"compute": 0.001}
        descriptor = describe(
            prompt="A fresh input",
            tools=["new-tool-sparse"],
            files=[{"name": "new.csv", "bytes": 600}],
        )
        first = router.decide("first", descriptor=descriptor, as_of_ns=10000)
        self.assertEqual(first["reason"], "contextual_bandit_value")
        self.assertEqual(first["selected"], "east")
        self.assertTrue(
            all(
                not c["prediction_diagnostics"]["cost"]["reliable"]
                for c in first["candidates"]
            )
        )
        router.complete(
            "first",
            trace("first", arrival=10000, end=11000),
            available_ns=11000,
            actual_cost=0.08,
        )
        second = router.decide("second", descriptor=descriptor, as_of_ns=12000)
        self.assertEqual(second["selected"], "west")
        self.assertEqual(second["reason"], "contextual_bandit_value")

    def test_exploration_counts_share_completed_nearby_context_despite_new_settings(
        self,
    ):
        router = self.fixture("support")
        router.exploration = Exploration({"epsilon": 0.35, "seed": 9, "optimism": 1})
        descriptor = describe(
            prompt="Fresh random seed",
            tools=["new-tool-support"],
            files=[{"name": "new.csv", "bytes": 600}],
            model={"name": "synthetic-model", "seed": 12345},
        )
        decision = router.decide(
            "support", descriptor=descriptor, as_of_ns=10000, shadow=True
        )
        self.assertEqual(
            [c["evidence"]["attempts"] for c in decision["candidates"]], [3, 3]
        )
        self.assertAlmostEqual(decision["exploration"]["epsilon"], 0.35 / math.sqrt(3))

    def test_limited_rerank_budget_does_not_erase_observed_exploration_experience(self):
        router = self.fixture("budget")
        router.exploration = Exploration({"epsilon": 0.35, "seed": 9})
        select = router.workload.retriever.select

        def limited(descriptor, profiles):
            selected, evidence = select(descriptor, profiles)
            return selected[:2], {
                **evidence,
                "context_neighbors": evidence["context_neighbors"][:2],
            }

        router.workload.retriever.select = limited
        descriptor = describe(
            prompt="A new task",
            tools=["new-tool-budget"],
            files=[{"name": "new.csv", "bytes": 600}],
            model={"name": "synthetic-model", "seed": 987},
        )
        decision = router.decide(
            "limited", descriptor=descriptor, as_of_ns=10000, shadow=True
        )
        self.assertEqual(
            [c["evidence"]["attempts"] for c in decision["candidates"]], [3, 3]
        )
        self.assertEqual(
            [
                len(c["prediction"]["decision_cost"]["training_request_ids"])
                for c in decision["candidates"]
            ],
            [1, 1],
        )

    def test_cold_start_uses_existing_symmetric_warmup_and_respects_disabled_exploration(
        self,
    ):
        for epsilon in [0.35, 0]:
            pools = {
                name: {**pool(), "environment": name} for name in ["first", "second"]
            }
            router = OnlineRouter(
                WorkloadModel(),
                ProviderModel(),
                pools,
                dict.fromkeys(pools, 2),
                default_pool="first",
                latency_weight=0,
                enable_context_routing=True,
                execution_bounds={name: bounds() for name in pools},
                exploration={"epsilon": epsilon, "seed": 9},
            )
            decision = router.decide(
                "new", descriptor=describe(), as_of_ns=10000, shadow=True
            )
            if epsilon:
                self.assertIsNone(decision["exploration"]["incumbent"])
                self.assertEqual(
                    decision["exploration"]["probabilities"],
                    {"first": 0.5, "second": 0.5},
                )
            else:
                self.assertEqual(decision["selected"], "first")
                self.assertEqual(decision["reason"], "validated_default")

    def test_pending_probes_diversify_cold_sampling_without_inventing_observed_rewards(
        self,
    ):
        pools = {name: {**pool(), "environment": name} for name in ["first", "second"]}
        workload, provider = WorkloadModel(), ProviderModel()
        router = OnlineRouter(
            workload,
            provider,
            pools,
            dict.fromkeys(pools, 4),
            default_pool="first",
            latency_weight=0,
            enable_context_routing=True,
            execution_bounds={name: bounds() for name in pools},
            exploration={"epsilon": 0.35, "seed": 9},
        )
        first = router.decide(
            "one", descriptor=describe(prompt="First request"), as_of_ns=10000
        )
        second = router.decide(
            "two", descriptor=describe(prompt="A similar new request"), as_of_ns=11000
        )
        self.assertNotEqual(first["selected"], second["selected"])
        self.assertTrue(
            all(
                c["evidence"]["attempts"] == c["evidence"]["successes"] == 0
                for c in second["candidates"]
            )
        )
        self.assertEqual(
            sum(c["evidence"]["pending_feedback"] for c in second["candidates"]), 1
        )
        self.assertEqual(workload.samples, {})
        self.assertEqual(provider.samples, {})

    def test_first_cost_feedback_ends_cross_scale_probe_sharing(self):
        pools = {name: {**pool(), "environment": name} for name in ["first", "second"]}
        workload, provider = WorkloadModel(), ProviderModel()
        router = OnlineRouter(
            workload, provider, pools, dict.fromkeys(pools, 4),
            default_pool="first", latency_weight=0, enable_context_routing=True,
            execution_bounds={name: bounds() for name in pools},
            exploration={"epsilon": 0.35, "seed": 9},
        )
        small = describe(prompt="First input", files=[{"name": "small.bin", "bytes": 1}])
        large = describe(prompt="New input scale", files=[{"name": "large.bin", "bytes": 10**8}])
        self.assertGreater(resource_distance(small, large), .5)
        first = router.decide("small-first", descriptor=small, as_of_ns=10000)
        second = router.decide("large-first", descriptor=large, as_of_ns=11000)
        self.assertNotEqual(first["selected"], second["selected"])
        pending = next(c["evidence"] for c in second["candidates"] if c["key"] == first["selected"])
        self.assertFalse(pending["has_cost_feedback"])
        self.assertEqual(pending["attempts"], 0)
        self.assertEqual(pending["pending_feedback"], 1)

        router.complete(
            "small-first", trace("small-first", arrival=10000, end=20000),
            available_ns=20000, actual_cost=.02, cost_basis="synthetic_usage_estimate",
        )
        router.exploration.config["epsilon"] = 0
        repeated = router.decide("small-next", descriptor=small, as_of_ns=21000)
        self.assertEqual(repeated["selected"], first["selected"])
        after = router.decide("large-next", descriptor=large, as_of_ns=22000)
        pending = next(c["evidence"] for c in after["candidates"] if c["key"] == first["selected"])
        self.assertTrue(pending["has_cost_feedback"])
        self.assertEqual(pending["attempts"], 0)
        self.assertEqual(pending["pending_feedback"], 0)
