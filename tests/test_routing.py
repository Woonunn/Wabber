"""Causal admission checks; synthetic fixtures are not cloud evidence."""

import copy
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from atsuite.provider_model import ProviderModel
from atsuite.routing import OnlineRouter
from atsuite.routing.evaluate import evaluate
from atsuite.routing.router import risk_estimate
from atsuite.workload import WorkloadModel
from tests.test_provider_model import observation, pool
from tests.test_workload import trace
from tests.stage6_fixtures import tariff, bounds


class RoutingTests(unittest.TestCase):
    def router(self, limit=1, count=100):
        model = ProviderModel()
        for i in range(count):
            row = observation(i)
            row["task_id"] = "task"
            model.update(row)
        return OnlineRouter(
            WorkloadModel(),
            model,
            {"aws": pool()},
            {"aws": limit},
            default_pool="aws",
            execution_bounds={"aws": bounds()},
        )

    def test_unknown_ood_and_capacity_fail_closed(self):
        for limit, task, load in [
            (None, "task", 0.5),
            (0, "task", 0.5),
        ]:
            router = self.router(limit)
            self.assertIsNone(
                router.decide("r", task, load, as_of_ns=20000)["selected"]
            )

    def test_pool_load_snapshot_overrides_offered_load(self):
        router = self.router()
        result = router.decide(
            "pool-load", "task", 100, as_of_ns=20000, pool_load_rps={"aws": 0.5}
        )
        self.assertEqual(result["selected"], "aws")
        self.assertEqual(result["load_basis"], "per_pool_snapshot")
        row = router.complete(
            "pool-load",
            trace("pool-load", arrival=20000, end=30000),
            available_ns=30000,
        )
        self.assertEqual(row["load_rps"], 0.5)

    def test_failures_and_unknown_latency_are_violations(self):
        rows = [observation(i) for i in range(60)]
        rows[0]["status"] = "failed"
        rows[1]["measurements"]["latency_s"] = None
        rows[2]["measurements"]["latency_s"] = 121
        self.assertEqual(risk_estimate(rows, 120)["violations"], 3)
        self.assertGreater(risk_estimate(rows, 120)["upper"], 0.05)
        self.assertGreater(risk_estimate(rows[:20], 120)["upper"], 0.05)

    def test_cutoff_no_revision_multiplication_and_shadow(self):
        router = self.router()
        self.assertEqual(
            router.decide("early", "task", 0.5, as_of_ns=10, shadow=True)["selected"],
            "aws",
        )
        shadow = router.decide("s", "task", 0.5, as_of_ns=20000, shadow=True)
        self.assertEqual(shadow["selected"], "aws")
        self.assertFalse(router.active)
        row = observation(0)
        row.update(task_id="task", available_ns=21000)
        row["measurements"]["latency_s"] = 999
        router.provider.update(row)
        self.assertEqual(
            router.decide("x", "task", 0.5, as_of_ns=21000, shadow=True)["candidates"][
                0
            ]["prediction"]["sample_count"],
            100,
        )
        self.assertEqual(
            router.decide("y", "task", 0.5, as_of_ns=21001, shadow=True)["candidates"][
                0
            ]["prediction"]["sample_count"],
            100,
        )

    def test_atomic_session_binding_and_feedback(self):
        router = self.router()
        with ThreadPoolExecutor(max_workers=8) as executor:
            decisions = list(
                executor.map(
                    lambda i: router.decide(
                        "live-" + str(i), "task", 0.5, as_of_ns=20000
                    ),
                    range(8),
                )
            )
        selected = [d for d in decisions if d["selected"]]
        self.assertEqual(len(selected), 1)
        rid = selected[0]["request_id"]
        self.assertEqual(
            router.decide(rid, "different", 100, as_of_ns=30000), selected[0]
        )
        completed = trace(rid, arrival=20000, end=30000)
        completed["request_status"] = "failed"
        row = router.complete(rid, completed, available_ns=30000)
        self.assertEqual(row["status"], "failed")
        row["pool"]["environment"] = "tampered"
        self.assertEqual(router.pools["aws"]["environment"], "frozen-test-v1")
        self.assertFalse(router.active)
        self.assertIn(rid, router.workload.samples)
        next_decision = router.decide("next", "task", 0.5, as_of_ns=40000)
        self.assertEqual(next_decision["selected"], "aws")
        self.assertEqual(next_decision["candidates"][0]["available_sessions"], 1)

    def test_logging_and_defensive_copies(self):
        router = self.router()
        with tempfile.TemporaryDirectory() as tmp:
            router.log_path = Path(tmp) / "events.jsonl"
            result = router.decide("r", "task", 0.5, as_of_ns=20000)
            result["selected"] = "tampered"
            self.assertEqual(router.active["r"]["selected"], "aws")
            self.assertIn("validated_default", router.log_path.read_text())

    def test_cost_ranking_requires_comparable_complete_forecasts(self):
        from tests.test_provider_model import cost

        provider = ProviderModel(
            tariffs={p: tariff(p) for p in ("aws_lambda", "aws_agentcore")}
        )
        provider.tariffs["aws_lambda"]["rates"]["memory_gb_s"] = 0.001
        workload = WorkloadModel()
        for backend, amount in [("aws_lambda", 5), ("aws_agentcore", 1)]:
            for i in range(100):
                row = observation(i, provider=backend)
                row.update(task_id="task", cost=cost(amount=amount, complete=True))
                provider.update(row)
        router = OnlineRouter(
            workload,
            provider,
            {"lambda": pool(), "agent": pool("aws_agentcore")},
            {"lambda": 1, "agent": 1},
            default_pool="agent",
            execution_bounds={"agent": bounds()},
        )
        # A prior profile supplies joint workload input; empty history cannot price it.
        workload.update(observation()["profile"], 10000)
        result = router.decide("priced", "task", 0.5, as_of_ns=20000, shadow=True)
        self.assertTrue(result["cost_comparable"])
        self.assertEqual(result["selected"], "agent")
        self.assertNotIn("slo_s", result)

    def test_late_measurements_preserve_task_identity(self):
        router = self.router()
        revision = observation(0)
        revision["available_ns"] = 20000
        revision["measurements"]["queue_s"] = 0.1
        router.provider.update(revision)
        router.provider.update(revision)
        result = router.decide("later", "task", 0.5, as_of_ns=30000, shadow=True)
        self.assertEqual(result["candidates"][0]["prediction"]["sample_count"], 100)
        revision.update(available_ns=25000, task_id="different")
        with self.assertRaisesRegex(ValueError, "request context"):
            router.provider.update(revision)

    def test_task_holdout_and_future_outcome_invariance(self):
        rows = [observation(i) for i in range(10)]
        for r in rows:
            r["task_id"] = "held"
        report = evaluate(rows, held_task="held")
        self.assertTrue(
            all(r["p50"] is None and r["risk"]["n"] == 0 for r in report["predictions"])
        )
        before = evaluate(rows)["predictions"]
        changed = copy.deepcopy(rows)
        changed[-1]["measurements"]["latency_s"] = 10000
        after = evaluate(changed)["predictions"]
        self.assertEqual([r["p50"] for r in before], [r["p50"] for r in after])


class CaptureRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_binds_target_and_shadow_preserves_target(self):
        from contextlib import nullcontext
        from unittest.mock import AsyncMock, Mock, patch

        from experiments.exp1.capture import capture_agent_traces as capture

        class Adapter:
            def __init__(self, collector):
                self.collector = collector

            async def astream(self, *args, **kwargs):
                yield {"status": "complete", "iteration": 0}

        for shadow, close_fails in ((False, False), (True, False), (False, True)):
            router = RoutingTests().router()
            router.hold_release = Mock()
            for history in router.provider.samples.values():
                history[0]["task_id"] = "csv_excel_34"
            collector = Mock()
            collector.snapshot.return_value = trace("online")
            tools = Mock(renderings=False, aclose=AsyncMock(), assert_consumed=Mock())
            if close_fails:
                tools.aclose.side_effect = RuntimeError("close failed")
            original = {"provider": "aws_lambda", "execution_scope": "original"}
            selected = {"provider": "aws_lambda", "execution_scope": "selected"}
            config = {
                "tool": original,
                "model": {"seed": 0},
                "max_iterations": 1,
                "max_parallel": 2,
            }
            target = original if shadow else selected
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(capture.uuid, "uuid4", return_value=Mock(hex="online")),
                patch.object(capture, "TraceCollector", return_value=collector),
                patch.object(capture, "LangGraphTracingAdapter", Adapter),
                patch.object(capture, "TraceRecorder"),
                patch.object(capture, "ToolClient", return_value=tools) as client,
                patch.object(
                    capture, "ReplayModel", return_value=Mock(assert_consumed=Mock())
                ),
                patch.object(capture, "build_graph"),
                self.assertRaisesRegex(RuntimeError, "close failed")
                if close_fails
                else nullcontext(),
            ):
                await capture.run(
                    "csv_excel_34",
                    config,
                    Path(directory),
                    capture=Path(directory),
                    router=router,
                    routing_context={
                        "targets": {"aws": {"tool": target}},
                        "load_rps": 0.5,
                        "shadow": shadow,
                        "actual_pool": "aws",
                    },
                )
            self.assertEqual(client.call_args.args[0], target)
            self.assertFalse(router.active)
            self.assertIn("online", router.completed)
            self.assertEqual(config["tool"], original)
            if close_fails:
                router.hold_release.assert_called_once_with("online")
                self.assertEqual(
                    router.workload.samples["online"]["profile"]["status"], "failed"
                )

    async def test_baseline_diagnostics_do_not_claim_mixed_latency(self):
        from atsuite.routing.shadow import baseline_choices, decision_quality

        router = RoutingTests().router()
        decision = router.decide("shadow", "task", 0.5, as_of_ns=20000, shadow=True)
        choices = baseline_choices(decision["candidates"], 0, None, rho_threshold=1000)
        self.assertEqual(choices["round_robin"], "aws")
        self.assertIsNone(choices["rho_rule"])
        report = decision_quality(
            [decision], {"shadow": {"aws": observation()}}, rho_threshold=1000
        )
        self.assertEqual(report["paired_best_fraction"]["cost_latency"], 1)
        self.assertIn("not mixed-routing", report["scope"])


class ConfidenceRankingTests(unittest.TestCase):
    def router(self):
        from unittest.mock import Mock
        from atsuite.workload import describe

        pools = {
            name: {**pool(), "environment": name} for name in ["cheap", "experienced"]
        }
        model = ProviderModel()

        def prediction(profiles, targets, load, **kwargs):
            cheap = targets[0]["environment"] == "cheap"
            return {
                "pools": [
                    {
                        "mode": "state-decoupled",
                        "analytical_cost": {
                            "complete": True,
                            "currency": "USD",
                            "total": {"mean": 0.001 if cheap else 0.1},
                        },
                        "latency": {"mean": 5},
                        "confidence": {
                            "cost": {"reliable": not cheap},
                            "latency": {"reliable": not cheap},
                        },
                    }
                ]
            }

        model.predict = Mock(side_effect=prediction)
        workload = Mock()
        workload.predict.return_value = {"joint_profiles": [], "low_confidence": True}
        execution = bounds()
        execution.pop("max_cost")
        router = OnlineRouter(
            workload,
            model,
            pools,
            dict.fromkeys(pools, 1),
            default_pool="experienced",
            execution_bounds={name: copy.deepcopy(execution) for name in pools},
        )
        return router, describe(prompt="Compute mean", tools=["python"])

    def test_uncertain_cheaper_pool_stays_in_normal_ranking_at_low_and_high_load(self):
        for load in [0.5, 100]:
            router, descriptor = self.router()
            decision = router.decide(
                "new", load_rps=load, descriptor=descriptor, as_of_ns=10000, shadow=True
            )
            self.assertEqual(decision["selected"], "cheap")
            self.assertEqual(decision["reason"], "cost_latency_objective")
            cheap = next(c for c in decision["candidates"] if c["key"] == "cheap")
            self.assertIn("cost_prediction_unreliable", cheap["prediction_flags"])
            self.assertIn("workload_history_insufficient", cheap["prediction_flags"])

    def test_cost_only_ranking_does_not_require_an_unused_latency_prediction(self):
        router, descriptor = self.router()
        router.latency_weight = 0
        original = router.provider.predict.side_effect

        def predict(*args, **kwargs):
            result = original(*args, **kwargs)
            if args[1][0]["environment"] == "cheap":
                result["pools"][0]["latency"]["mean"] = None
            return result

        router.provider.predict.side_effect = predict
        decision = router.decide("cost-only", descriptor=descriptor, as_of_ns=10000)
        self.assertEqual(decision["selected"], "cheap")
        cheap = next(c for c in decision["candidates"] if c["key"] == "cheap")
        self.assertIsNone(cheap["objective_latency"])
        self.assertIn("latency_prediction_missing", cheap["prediction_flags"])

    def test_uncertain_cheaper_pool_still_obeys_real_capacity_and_tool_constraints(
        self,
    ):
        for restriction in ["full", "unknown", "tools"]:
            router, descriptor = self.router()
            if restriction == "tools":
                router.execution_bounds["cheap"]["tools"] = []
            else:
                router.limits["cheap"] = 0 if restriction == "full" else None
            decision = router.decide(
                "blocked", load_rps=100, descriptor=descriptor, as_of_ns=10000
            )
            self.assertEqual(decision["selected"], "experienced")
            cheap = next(c for c in decision["candidates"] if c["key"] == "cheap")
            self.assertTrue(cheap["flags"])
