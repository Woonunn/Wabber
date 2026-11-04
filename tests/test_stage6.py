"""Generalization, billing identities, causal feedback and bounded admission."""

import copy
import tempfile
import unittest
from pathlib import Path

from atsuite.provider_model import ProviderModel
from atsuite.provider_model.analytical import (
    calibration,
    pool_cost,
    predict_cost,
    price_profile,
)
from atsuite.routing import OnlineRouter
from atsuite.routing.objective import pareto
from atsuite.workload import WorkloadModel
from tests.workload_fixtures import describe, stable_prediction, calibrate
from tests.stage6_fixtures import bounds, tariff
from tests.test_provider_model import observation, pool, profile
from tests.test_workload import trace


class Stage6Tests(unittest.TestCase):
    def test_controller_unknown_family_uses_fresh_default_without_a_plan(self):
        from tests.test_joint_control import controller

        control, adapter = controller()
        control.router.workload = WorkloadModel()
        control.router.provider = ProviderModel()
        report = control.tick({"new-family": 0.5}, now_ns=1000)
        self.assertIsNone(report["selected"])
        descriptor = describe(prompt="An unobserved workflow", tools=["python"])
        decision = control.decide(
            "new-family-request", descriptor=descriptor, as_of_ns=1001
        )
        self.assertEqual(decision["selected"], "aws")
        adapter.fail = True
        control.tick({"new-family": 0.5}, now_ns=2000)
        self.assertIsNone(
            control.decide("read-failed", descriptor=descriptor, as_of_ns=2001)[
                "selected"
            ]
        )

    def test_holdout_evaluator_does_not_train_on_test_outcomes(self):
        from atsuite.routing.generalization import evaluate

        rows = [observation(i) for i in range(6)]
        for i, row in enumerate(rows):
            row["task_id"] = "train" if i < 3 else "held"
        descriptors = {
            t: describe(prompt="CSV analysis", tools=["python"])
            for t in ("train", "held")
        }
        cards = {"aws_lambda": tariff()}
        before = evaluate(rows, descriptors, cards, held_task="held")
        changed = copy.deepcopy(rows)
        for row in changed[3:]:
            row["profile"]["metrics"]["tool_elapsed_s"] = 10000
            row["measurements"]["latency_s"] = 10000
        after = evaluate(changed, descriptors, cards, held_task="held")
        self.assertEqual(
            [p["cost"] for p in before["predictions"]],
            [p["cost"] for p in after["predictions"]],
        )
        self.assertEqual(
            [p["latency"] for p in before["predictions"]],
            [p["latency"] for p in after["predictions"]],
        )
        self.assertTrue(all(p["training_count"] == 3 for p in after["predictions"]))
        first = evaluate(
            rows, descriptors, cards, held_task="held", prediction_stride=2
        )
        for i, row in enumerate(changed[3:]):
            row.update(completed_ns=700 - i * 10, available_ns=700 - i * 10)
        second = evaluate(
            changed, descriptors, cards, held_task="held", prediction_stride=2
        )
        self.assertEqual(
            [p["request_id"] for p in first["predictions"]],
            [p["request_id"] for p in second["predictions"]],
        )

    def test_descriptor_family_global_prior_and_future_invariance(self):
        small = describe(
            prompt="Summarize CSV",
            files=[{"name": "data.csv", "bytes": 1000}],
            tools=["python"],
        )
        large = describe(
            prompt="Train model",
            files=[{"name": "image.png", "bytes": 10**9}],
            tools=["gpu"],
        )
        model = WorkloadModel(initial_profiles=[profile()])
        self.assertEqual(
            model.predict(1, descriptor=small)["match_basis"], "cold_start"
        )
        for i, descriptor in enumerate((small, large)):
            p = profile(state=1000 if i == 0 else 10**9)
            p["request_id"] = str(i)
            model.update(p, 100 + i, descriptor=descriptor)
        predicted = model.predict(102, descriptor=small)
        self.assertEqual(predicted["match_basis"], "exact_context")
        self.assertEqual(predicted["metrics"]["state_bytes"]["mean"], 1000)
        future = profile(state=10**12)
        future["request_id"] = "future"
        model.update(future, 200, descriptor=small)
        self.assertEqual(stable_prediction(predicted), stable_prediction(model.predict(102, descriptor=small)))
        self.assertEqual(
            model.predict(102, descriptor=describe(tools=["other"]))["match_basis"],
            "cold_start",
        )
        with self.assertRaises(ValueError):
            model.predict(102, descriptor={**small, "task_id": "leak"})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "workload.json.gz"
            model.save(path)
            self.assertEqual(
                stable_prediction(predicted), stable_prediction(WorkloadModel.load(path).predict(102, descriptor=small))
            )

    def test_unseen_prompt_without_task_id_reserves_budget_and_capacity(self):
        router = OnlineRouter(
            WorkloadModel(),
            ProviderModel(),
            {"safe": pool()},
            {"safe": 1},
            default_pool="safe",
            execution_bounds={"safe": bounds()},
            budget=1,
        )
        decision = router.decide(
            "new",
            descriptor=describe(prompt="Never seen before", tools=["python"]),
            as_of_ns=100,
        )
        self.assertEqual(decision["selected"], "safe")
        self.assertIsNone(decision["task_id"])
        self.assertEqual(decision["reason"], "validated_default")
        self.assertIsNone(router.decide("second", as_of_ns=101)["selected"])
        router.complete("new", trace("new", arrival=100, end=200), available_ns=200)
        self.assertIsNone(router.decide("over-budget", as_of_ns=201)["selected"])
        self.assertIn(
            "budget_exhausted",
            router.decisions["over-budget"]["candidates"][0]["flags"],
        )
        self.assertEqual(
            router.workload.predict(201, descriptor=decision["descriptor"])["joint_profiles"][0]["descriptor"],
            decision["descriptor"],
        )

    def test_default_still_checks_tools_deadline_and_validation(self):
        with self.assertRaisesRegex(ValueError, "Duplicate execution pools"):
            OnlineRouter(
                WorkloadModel(),
                ProviderModel(),
                {"a": pool(), "b": {**pool(), "billing": tariff()}},
                {"a": 1, "b": 1},
            )
        with self.assertRaises(ValueError):
            OnlineRouter(
                WorkloadModel(),
                ProviderModel(),
                {"safe": pool()},
                {"safe": 1},
                default_pool="safe",
            )
        router = OnlineRouter(
            WorkloadModel(),
            ProviderModel(),
            {"safe": pool()},
            {"safe": 1},
            default_pool="safe",
            execution_bounds={"safe": bounds()},
        )
        bad = router.decide(
            "untrusted", descriptor=describe(tools=["shell"]), as_of_ns=10
        )
        self.assertIsNone(bad["selected"])
        late = router.decide("late", as_of_ns=901_000_000_000, arrival_ns=1)
        self.assertIn("execution_deadline_elapsed", late["candidates"][0]["flags"])
        slow = router.decide("slow", as_of_ns=121_000_000_000, arrival_ns=1)
        self.assertEqual(slow["selected"], "safe")

    def test_faas_rounds_each_call_and_mcp_retains_idle_memory(self):
        p = profile()
        p["metrics"].update(
            tool_elapsed_s=0.0022,
            tool_active_s=0.0022,
            execution_active_s=10.0022,
            tool_cpu_s=0.001,
            llm_idle_s=10,
            state_max_object_bytes=1000,
        )
        p["workflow"]["tool_calls"] = {"python": 2}
        p["tool_durations"] = {"python": [0.0011, 0.0011]}
        card = tariff()
        card["rates"].update(memory_gb_s=1, read=2, write=3, invocation=4)
        result = price_profile(p, pool(), card, calibration(card, []))
        self.assertAlmostEqual(result["usage"]["billed_s"], 0.004)
        self.assertAlmostEqual(result["components"]["execution"], 0.008 + 8)
        self.assertEqual(result["components"]["storage"], 10)
        resident = tariff("aws_agentcore")
        resident["rates"].update(memory_gb_s=1, read=2, write=3)
        resident["session_tail_s"] = 20
        result = price_profile(
            p, pool("aws_agentcore"), resident, calibration(resident, [])
        )
        self.assertEqual(result["usage"]["read"], 0)
        self.assertEqual(result["usage"]["transfer_bytes"], 0)
        self.assertAlmostEqual(
            result["components"]["execution"] + result["components"]["retention"],
            2 * 30.003 + 0.0000004,
        )

    def test_pool_floor_and_shared_charges_are_counted_once(self):
        card = tariff(floor=2)
        card["rates"]["shared_per_s"] = 3
        card["rates"]["scale_event"] = 4
        card["compute_in_pool_floor"] = True
        marginal = predict_cost([profile()], pool(), card, [])
        self.assertAlmostEqual(marginal["components"]["execution"]["mean"], 0.0000002)
        one = pool_cost(
            card, requests=1, marginal=marginal, horizon_s=10, overlap_s=5, changes=1
        )
        ten = pool_cost(
            card, requests=10, marginal=marginal, horizon_s=10, overlap_s=5, changes=1
        )
        self.assertAlmostEqual(
            ten["total"] - one["total"], 9 * marginal["total"]["mean"]
        )
        self.assertEqual(one["components"]["shared"], 30)
        self.assertEqual(one["components"]["scaling"], 14)

    def test_missing_state_keeps_compute_subtotal_separate(self):
        p = profile()
        p["metrics"]["state_max_object_bytes"] = None
        result = predict_cost([p], pool(), tariff(), [])
        self.assertFalse(result["complete"])
        self.assertTrue(result["execution_complete"])
        self.assertAlmostEqual(
            sum(v["mean"] for v in result["execution_details"].values()),
            result["components"]["execution"]["mean"],
        )

    def test_no_same_task_cost_and_late_calibration(self):
        card = tariff()
        model = ProviderModel(tariffs={"aws_lambda": card})
        p = profile()
        before = model.predict(p, [pool()], 0.5, as_of_ns=100)["pools"][0][
            "analytical_cost"
        ]
        row = observation()
        row.update(available_ns=100)
        row["measurements"]["tool_s"] = 4
        row["measurements"]["state_sync_s"] = 0
        model.update(row)
        self.assertEqual(
            before,
            model.predict(p, [pool()], 0.5, as_of_ns=100)["pools"][0][
                "analytical_cost"
            ],
        )
        after = model.predict(p, [pool()], 0.5, as_of_ns=101)["pools"][0][
            "analytical_cost"
        ]
        self.assertGreater(after["total"]["mean"], before["total"]["mean"])
        self.assertEqual(after["calibration"]["tool_scale"], 2)
        inclusive = copy.deepcopy(row)
        inclusive["measurements"].update(tool_s=4, state_sync_s=2)
        self.assertEqual(calibration(card, [inclusive])["tool_scale"], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provider.json.gz"
            model.save(path)
            self.assertEqual(ProviderModel.load(path).tariffs, model.tariffs)

    def test_pareto_and_configured_tradeoff_change_selection(self):
        workload, provider = WorkloadModel(), ProviderModel()
        configs = {"cheap": pool(), "fast": pool()}
        for key, scale, rate in (("cheap", 5, 0.00001), ("fast", 1, 0.01)):
            configs[key]["environment"] = key
            configs[key]["billing"] = tariff()
            configs[key]["billing"]["rates"]["memory_gb_s"] = rate
            configs[key]["billing"]["calibration"] = {
                "tool_scale": scale,
                "reference_id": "test-reference",
                "evidence": ["fixture"],
            }
            for i in range(3):
                row = observation(i)
                row.update(id=key + str(i), pool=configs[key])
                provider.update(row)
                p = copy.deepcopy(row["profile"])
                p["request_id"] = key + str(i)
                workload.update(p, 1000)
        calibrate(workload, describe())
        results = []
        for weight in (0, 1):
            router = OnlineRouter(
                workload,
                provider,
                configs,
                {"cheap": 3, "fast": 3},
                latency_weight=weight,
                execution_bounds={k: bounds() for k in configs},
            )
            result = router.decide("query", load_rps=0.5, as_of_ns=2000, shadow=True, descriptor=describe())
            self.assertEqual(set(result["pareto"]), {"cheap", "fast"})
            results.append(result["selected"])
        self.assertEqual(results, ["cheap", "fast"])
        self.assertEqual(
            len(
                pareto(
                    [
                        {"objective_cost": 1, "objective_latency": 2},
                        {"objective_cost": 2, "objective_latency": 3},
                    ]
                )
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
