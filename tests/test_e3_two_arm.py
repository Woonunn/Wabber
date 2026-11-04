"""Two-arm routing, budget-free admission, and shared native-usage pricing."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from atsuite.analysis.api import price_metrics
from atsuite.analysis.model import ProviderMetric, RunContext
from atsuite.routing import OnlineRouter
from experiments.e3.report import compare
from experiments.e3.run import STRATEGIES
from tests.test_stage7 import router_fixture
from tests.test_workload import trace


class TwoArmTests(unittest.TestCase):
    def router(self):
        old, descriptor = router_fixture(epsilon=0)
        bounds = copy.deepcopy(old.execution_bounds)
        for b in bounds.values():
            b.pop("max_cost")
        router = OnlineRouter(
            old.workload,
            old.provider,
            old.pools,
            {"known": 1, "promising": 0, "expensive": 0},
            default_pool="known",
            enable_context_routing=True,
            execution_bounds=bounds,
            exploration={"epsilon": 0, "seed": 9},
        )
        return router, descriptor

    def test_budget_free_exploration_has_no_implicit_financial_cap(self):
        router, descriptor = self.router()
        row = router.decide("new-task", load_rps=0.5, as_of_ns=10000, descriptor=descriptor)
        self.assertEqual(row["selected"], "known")
        self.assertIsNone(row["exploration"]["remaining_budget"])
        self.assertNotIn("budget", router.exploration.config)
        router.complete("new-task", trace("new-task", arrival=10000, end=20000), available_ns=20000)
        self.assertIsNone(router.feedback["new-task"]["amount"])

    def test_native_occupancy_and_retention_prevent_early_slot_reuse(self):
        router, descriptor = self.router()
        router.session_retention_s = {"known": 300}
        router.observe_occupancy("known", 1)
        self.assertIsNone(router.decide("occupied", load_rps=0.5, as_of_ns=10000, descriptor=descriptor)["selected"])
        router.observe_occupancy("known", 0)
        self.assertEqual(
            router.decide("first", load_rps=0.5, as_of_ns=10001, descriptor=descriptor)["selected"], "known"
        )
        router.complete("first", trace("first", arrival=10001, end=20000), available_ns=20000)
        self.assertIsNone(router.decide("retained", load_rps=0.5, as_of_ns=30000, descriptor=descriptor)["selected"])
        self.assertEqual(
            router.decide("expired", load_rps=0.5, as_of_ns=301_000_000_000, descriptor=descriptor)["selected"], "known"
        )

    def test_session_pricing_uses_native_lifetime_and_preserves_request_count(self):
        context = RunContext("one", "ali_agentrun", "", "E3", "", "", "")
        fields = {
            "session_duration_ms": 300000,
            "duration_ms": 1,
            "request_count": 4,
            "cpu": 2,
            "memory": 3072,
            "disk": 20480,
        }

        def price(f):
            return {
                r.category: r.amount
                for r in price_metrics(context, [ProviderMetric("s", "ali_agentrun", "ali_session", fields=f)], [])
            }

        one, two = price(fields), price({**fields, "session_duration_ms": 600000})
        self.assertEqual(one["request"], two["request"])
        self.assertEqual(one["cpu"], two["cpu"])
        self.assertAlmostEqual(one["cpu"], 2 * 0.001 * 0.000088)
        self.assertAlmostEqual(one["memory"], 3 * 300 * 0.15 * 0.000088)
        self.assertAlmostEqual(one["disk"], 10 * 300 * 0.05 * 0.000088)
        for key in ["memory", "disk"]:
            self.assertAlmostEqual(two[key], 2 * one[key])
        active = price({**fields, "duration_ms": 2})
        self.assertAlmostEqual(active["cpu"], 2 * one["cpu"])
        self.assertEqual(active["memory"], one["memory"])
        self.assertNotIn("cpu", price({**fields, "duration_ms": 0}))
        for disk in (512, 2048, 10240):
            with self.subTest(disk=disk):
                self.assertNotIn("disk", price({**fields, "disk": disk}))
        for duration in (None, -1, float("nan"), float("inf"), 300001):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                price({**fields, "duration_ms": duration})
        self.assertEqual(
            price_metrics(context, [ProviderMetric("bad", "ali_agentrun", "sls", fields={"duration_ms": 1})], []), []
        )

    def test_confirmed_session_release_does_not_reserve_retention_capacity(self):
        router, descriptor = self.router()
        router.session_retention_s = {"known": 300}
        self.assertEqual(router.decide("first", load_rps=0.5, as_of_ns=10000, descriptor=descriptor)["selected"], "known")
        completed = {**trace("first", arrival=10000, end=20000), "sessions_closed": True}
        router.complete("first", completed, available_ns=20000)
        self.assertNotIn("first", router.retained_sessions)
        self.assertEqual(router.decide("next", load_rps=0.5, as_of_ns=30000, descriptor=descriptor)["selected"], "known")

    def test_faas_pricing_uses_invocation_time_and_free_disk_quota(self):
        context = RunContext("one", "ali_fc", "", "E3", "", "", "")
        fields = {"duration_ms": 1000, "cpu": 2, "memory": 3072, "disk": 10240}
        costs = {
            r.category: r.amount
            for r in price_metrics(context, [ProviderMetric("call", "ali_fc", "native_request", fields=fields)], [])
        }
        self.assertAlmostEqual(costs["cpu"], 2 * 0.000088)
        self.assertAlmostEqual(costs["memory"], 3 * 0.15 * 0.000088)
        self.assertNotIn("disk", costs)

    def test_only_two_arms_and_matched_requests_are_comparable(self):
        self.assertEqual(STRATEGIES, ("fixed", "exploration"))
        request = {"request_id": "one", "workload": "TravelPlanner", "pool": "ali_fc", "success": True, "latency_s": 1}
        shared = {"manifest_sha256": "same", "native_elasticity": {}, "requests": [request]}
        fixed = {
            **shared,
            "strategy": "fixed",
            "fixed_provider": "ali_fc",
            "pricing": {"basis": "atsuite_metered_usage_pricing", "complete": True, "total_cny": 2},
        }
        routed = {
            **shared,
            "strategy": "exploration",
            "pricing": {"basis": "atsuite_metered_usage_pricing", "complete": True, "total_cny": 1},
        }
        self.assertEqual(compare([fixed, routed])["savings_vs_fixed"], 0.5)
        with self.assertRaises(ValueError):
            compare([fixed, {**routed, "manifest_sha256": "different"}])
        with self.assertRaises(ValueError):
            compare([fixed, {**routed, "strategy": "online"}])


class DriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_driver_keeps_scheduler_in_its_own_directory(self):
        from unittest.mock import patch

        from experiments.e3.run import execute

        class Oracle:
            def __init__(self, *args):
                pass

            async def start(self):
                return {}

            async def validate(self, *args):
                return {}

            async def close(self):
                pass

        async def measured(task, config, provider, output, **kw):
            return {"request_id": kw["request_id"], "success": True, "latency_s": 0.001, "pool": provider}

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "task.json"
            manifest.write_text('{"tasks":[{"task_id":"one"}]}')
            sequence = root / "sequence"
            sequence.mkdir()
            (sequence / "seal.json").write_text('{"manifest_sha256":"test-only"}')
            plan = {
                "duration_ms": 1,
                "tasks": [
                    {
                        "id": "one",
                        "task_id": "one",
                        "workload": "TravelPlanner",
                        "manifest": str(manifest),
                        "capture": "none",
                    }
                ],
                "requests": [
                    {
                        "arrival_id": "one",
                        "relative_arrival_ms": "0",
                        "original_timestamp_utc": "2026-01-01T00:00:00Z",
                        "source_partition": "test",
                        "source_type": "test",
                        "window_id": "test",
                        "task": "one",
                        "workload": "TravelPlanner",
                    }
                ],
            }
            config = {
                "fixed_provider": "ali_fc",
                "targets": {"TravelPlanner": {}},
                "load_rps": 1,
                "concurrency": 1,
                "native_elasticity": {},
            }
            with (
                patch("experiments.e3.run.load", return_value=plan),
                patch("experiments.e3.run.OfficialEvaluator", Oracle),
                patch("experiments.e3.run.measure", side_effect=measured),
            ):
                result = await execute(sequence, config, "fixed", root / "output")
            self.assertEqual(result["successes"], 1)
            self.assertTrue((root / "output/schedule/actual_arrivals.csv").exists())
