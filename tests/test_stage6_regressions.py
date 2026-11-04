"""Numerical regressions for reference provenance, feedback, concurrency and billing."""

import copy
import tempfile
import unittest
from pathlib import Path

from atsuite.provider_model import ProviderModel
from atsuite.provider_model.analytical import calibration, predict_cost, price_profile
from atsuite.provider_model.observation import execution_context
from atsuite.workload import WorkloadModel, extract_profile
from tests.workload_fixtures import describe, stable_prediction
from tests.stage6_fixtures import tariff
from tests.test_provider_model import observation, pool, profile
from tests.test_workload import trace


def native_profile(config, seconds):
    p = profile()
    p.pop("reference_id")
    p["execution_pool"] = execution_context(config)
    p["descriptor"] = describe(prompt="CSV summary", tools=["python"])
    p["metrics"].update(
        tool_elapsed_s=seconds,
        runtime_tool_s=seconds,
        tool_active_s=seconds,
        execution_active_s=seconds + 2,
        agent_elapsed_s=seconds + 2,
    )
    p["tool_durations"] = {"python": [seconds]}
    p["runtime_tool_durations"] = [seconds]
    return p


class Stage6RegressionTests(unittest.TestCase):
    def test_controller_does_not_scale_from_uncalibrated_forecasts(self):
        from tests.test_joint_control import controller

        control, adapter = controller()
        for level in control.levels["aws"].values():
            level["pool"]["billing"].pop("calibration")
        report = control.tick({"task": 0.5}, now_ns=20000)
        self.assertIsNone(report["selected"])
        self.assertTrue(
            all("aws:unreliable_prediction" in c["flags"] for c in report["candidates"])
        )
        self.assertFalse(adapter.calls)
        self.assertEqual(
            control.decide("bounded-default", as_of_ns=20001)["selected"], "aws"
        )

    def test_native_times_are_not_cross_platform_speed_references(self):
        card = tariff()
        card.pop("calibration")
        model = ProviderModel(tariffs={"aws_lambda": card})
        configs = []
        for name, seconds in (("slow", 10), ("fast", 1)):
            config = pool()
            config["environment"] = name
            configs.append(config)
            for i in range(3):
                row = observation(i)
                row.update(pool=config, profile=native_profile(config, seconds))
                row["measurements"].update(tool_s=seconds, state_sync_s=0)
                model.update(row)
        query = native_profile(configs[0], 10)
        result = model.predict(query, configs, 0.5, as_of_ns=1000)["pools"]
        self.assertEqual([r["latency"]["mean"] for r in result], [12, 2])
        self.assertEqual([r["low_confidence"] for r in result], [False, True])
        self.assertFalse(result[1]["analytical_cost"]["calibrated"])
        future = observation(3)
        future.update(
            pool=configs[1], profile=native_profile(configs[1], 0.1), available_ns=2000
        )
        model.update(future)
        self.assertEqual(
            model.predict(query, configs, 0.5, as_of_ns=1000)["pools"], result
        )
        query.pop("descriptor")
        unsupported = model.predict(query, [configs[1]], 0.5, as_of_ns=1000)["pools"][0]
        self.assertTrue(unsupported["low_confidence"])
        self.assertFalse(unsupported["analytical_cost"]["calibrated"])

    def test_common_reference_calibration_does_not_mix_bases(self):
        card = tariff()
        card.pop("calibration")
        rows = [observation(i) for i in range(3)]
        for row in rows:
            row["measurements"].update(tool_s=1, state_sync_s=0)
        fitted = calibration(card, rows, "test-reference")
        self.assertEqual(fitted["tool_scale"], 0.5)
        self.assertTrue(fitted["calibrated"])
        other = calibration(card, rows, "unrelated-environment")
        self.assertFalse(other["calibrated"])
        self.assertEqual(other["sample_count"], 0)
        invalid = copy.deepcopy(rows)
        for row in invalid:
            row["measurements"]["state_sync_s"] = 100
        self.assertEqual(
            calibration(card, invalid, "test-reference")["sample_count"], 0
        )
        for row in rows:
            row["profile"].pop("reference_id")
        self.assertFalse(calibration(card, rows, "test-reference")["calibrated"])

    def test_recent_same_descriptor_feedback_replaces_old_neighbors_causally(self):
        model = WorkloadModel()
        descriptor = describe(
            prompt="Calculate the mean with the same shape", tools=["python"]
        )
        for i in range(40):
            p = profile()
            p["request_id"] = str(i)
            p["metrics"]["tool_elapsed_s"] = 1 if i < 20 else 100
            model.update(p, 100 + i, descriptor=descriptor)
        self.assertEqual(
            model.predict(120, descriptor=descriptor)["metrics"]["tool_elapsed_s"][
                "mean"
            ],
            1,
        )
        self.assertEqual(
            model.predict(121, descriptor=descriptor)["metrics"]["tool_elapsed_s"][
                "mean"
            ],
            5.95,
        )
        latest = model.predict(140, descriptor=descriptor)
        self.assertEqual(latest["metrics"]["tool_elapsed_s"]["mean"], 100)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "workload.json"
            model.save(path)
            self.assertEqual(
                stable_prediction(WorkloadModel.load(path).predict(140, descriptor=descriptor)), stable_prediction(latest)
            )

    def test_three_successes_do_not_hide_missing_calibration_or_extrapolation(self):
        card = tariff()
        card.pop("calibration")
        model = ProviderModel(tariffs={"aws_lambda": card})
        for i in range(3):
            model.update(observation(i))
        result = model.predict(profile(), [pool()], 0.5, as_of_ns=1000)["pools"][0]
        self.assertTrue(result["low_confidence"])
        self.assertEqual(result["analytical_cost"]["calibration"]["sample_count"], 0)
        for i in range(3):
            row = observation(i)
            row["available_ns"] = 500 + i
            row["measurements"].update(tool_s=2, state_sync_s=0)
            model.update(row)
        reliable = model.predict(profile(), [pool()], 0.5, as_of_ns=1000)["pools"][0]
        self.assertFalse(reliable["low_confidence"])
        query = profile()
        query["metrics"]["tool_elapsed_s"] = 100000
        outside = model.predict(query, [pool()], 1000, as_of_ns=1000)["pools"][0]
        self.assertTrue(outside["low_confidence"])
        self.assertIn("load_rps", outside["coverage"][0]["extrapolated_features"])

    def test_transport_counters_are_scoped_to_the_measured_environment(self):
        card = tariff()
        fitted = calibration(card, [])
        fitted["sync_s_per_byte"] = 0.001
        p = native_profile(pool("aws_agentcore"), 2)
        p["metrics"]["state_transfer_bytes"] = 0
        faas = price_profile(p, pool(), card, fitted)
        self.assertGreater(faas["usage"]["transfer_bytes"], 0)
        self.assertFalse(faas["calibrated"])
        p["execution_pool"] = execution_context(pool())
        local = price_profile(p, pool(), card, fitted)
        self.assertEqual(local["usage"]["transfer_bytes"], 0)
        other = pool()
        other["environment"] = "different-snapshot-format"
        self.assertGreater(
            price_profile(p, other, card, fitted)["usage"]["transfer_bytes"], 0
        )

    def test_parallel_work_and_resident_wall_time_are_distinct(self):
        results = []
        for parallel in (True, False):
            t = trace()
            session = next(s for s in t["spans"] if s["kind"] == "session")
            llm = next(s for s in t["spans"] if s["kind"] == "llm")
            template = next(s for s in t["spans"] if s["kind"] == "tool")
            tools = []
            for i in range(4):
                s = copy.deepcopy(template)
                start = 2 if parallel else 2 + i * 10
                s.update(
                    id=f"tool-{i}",
                    start_ns=start * 10**9,
                    end_ns=(start + 10) * 10**9,
                    duration_s=10,
                )
                tools.append(s)
            wall = 12 if parallel else 42
            session.update(end_ns=wall * 10**9, duration_s=wall)
            t["spans"] = [session, llm, *tools]
            p = extract_profile(t)
            p["reference_id"] = "test-reference"
            self.assertEqual(p["metrics"]["tool_elapsed_s"], 40)
            self.assertEqual(p["metrics"]["tool_active_s"], 10 if parallel else 40)
            card = tariff("aws_agentcore")
            card["rates"]["memory_gb_s"] = 1
            r = price_profile(p, pool("aws_agentcore"), card, calibration(card, []))
            self.assertEqual(r["latency_s"], wall)
            self.assertAlmostEqual(
                r["components"]["execution"] + r["components"]["retention"],
                2 * wall + 4 * card["rates"]["invocation"],
            )
            results.append(r)
        self.assertEqual([r["session_s"] for r in results], [12, 42])

    def test_completed_usage_is_not_accelerated_again(self):
        card = tariff()
        card["calibration"]["tool_scale"] = 0.2
        p = native_profile(pool(), 2)
        measured = predict_cost([p], pool(), card, [], measured=True)
        self.assertEqual(measured["resource_samples"][0]["usage"]["billed_s"], 2)
        self.assertEqual(measured["basis"], "measured_usage_estimate")
        card["calibration"]["tool_scale"] = 20
        other = predict_cost([p], pool(), card, [], measured=True)
        self.assertEqual(measured["total"], other["total"])
        foreign = pool()
        foreign["environment"] = "foreign"
        with self.assertRaisesRegex(ValueError, "actual execution environment"):
            predict_cost([p], foreign, card, [], measured=True)


if __name__ == "__main__":
    unittest.main()
