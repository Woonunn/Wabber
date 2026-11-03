"""Partial observations and target-specific confidence; no cloud calls."""

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
import threading
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from atsuite.provider_model.model import ProviderModel, _nearest
from atsuite.provider_model.observation import features
from atsuite.tracing import TraceCollector
from atsuite.workload import extract_profile
from atsuite_sdk.abstract import UnifiedAgentTool
from atsuite_sdk.resources import get_resource_usage, observe_call
from tests.stage6_fixtures import tariff
from tests.test_provider_model import observation
from tests.test_workload import trace
from tests.phase0_environment import LocalToolEnvironment


class CPUObservationTests(unittest.TestCase):
    def test_real_local_transports_preserve_framework_usage_and_state(self):
        from atsuite.runtime import FunctionRuntimeAdapter, InvocationRequest, MCPRuntimeAdapter

        with tempfile.TemporaryDirectory() as directory:
            for family, adapter_type in (("function", FunctionRuntimeAdapter), ("mcp", MCPRuntimeAdapter)):
                server = LocalToolEnvironment(Path(directory) / family, family, "cpu-test")
                adapter = adapter_type("test")
                adapter.connect({"targets": {"datasci": server.url}})
                try:
                    for code in ("retained = 41\nprint(retained)", "print(retained + 1)"):
                        response = adapter.invoke(InvocationRequest(
                            "datasci", "datasci_run_code", {"code_str": code}, "cpu-test", timeout=30))
                        self.assertEqual(response.status, "ok", response.error)
                        usage = response.provider_metadata["resource_usage"]
                        self.assertGreaterEqual(usage["cpu_seconds"], 0)
                        self.assertFalse(usage["cpu_complete"])
                        self.assertEqual(usage["cpu_interval"], "call_delta")
                    self.assertEqual(response.result, "42\n")
                finally:
                    adapter.cleanup_run("cpu-test")
                    server.close()

    def test_legacy_partial_cpu_is_not_zero_filled_or_a_complete_total(self):
        value = trace()
        tool = next(s for s in value["spans"] if s["kind"] == "tool")
        tool["resources"] = {"cpu_seconds": 0.5}
        other = copy.deepcopy(tool)
        other.update(id="prepare", resources=None)
        value["spans"].append(other)
        profile = extract_profile(value)
        self.assertEqual(profile["metrics"]["tool_cpu_measured_s"], 0.5)
        self.assertEqual(profile["metrics"]["tool_cpu_coverage"], 0.5)
        self.assertIsNone(profile["metrics"]["tool_cpu_s"])
        self.assertFalse(profile["cpu_observation"]["complete"])
        self.assertEqual(profile["cpu_observation"]["missing_calls"], ["prepare"])
        other["resources"] = {"cpu_seconds": 0.25, "cpu_scope": "runtime_calling_thread", "cpu_complete": False}
        profile = extract_profile(value)
        self.assertEqual(profile["metrics"]["tool_cpu_coverage"], 1)
        self.assertIsNone(profile["metrics"]["tool_cpu_s"])

    def test_runtime_collects_without_business_instrumentation_and_on_failure(self):
        def tool(value: int):
            if value < 0:
                raise ValueError("failure")
            return value + 1

        runtime = Mock()
        runtime.execution.side_effect = lambda *args: nullcontext()
        with patch("atsuite_sdk.state.get_state_runtime", return_value=runtime):
            wrapped = UnifiedAgentTool(tool)
            with patch("atsuite_sdk.resources.time.thread_time", side_effect=[1, 1.25, 10, 10.5]):
                self.assertEqual(wrapped.execute_from_dict({"value": 2}), 3)
                self.assertEqual(get_resource_usage()["cpu_seconds"], 0.25)
                with self.assertRaises(ValueError):
                    wrapped.execute_from_dict({"value": -1})
                usage = get_resource_usage()
                self.assertEqual(usage["cpu_seconds"], 0.5)
                self.assertFalse(usage["cpu_complete"])
                self.assertFalse(usage["includes_children"])

    def test_concurrent_calls_keep_thread_ownership_and_nested_calls_do_not_double_count(self):
        barrier = threading.Barrier(2)

        def run():
            with observe_call():
                barrier.wait(timeout=5)
                sum(i * i for i in range(10000))
            return get_resource_usage()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = list(executor.map(lambda _: run(), range(2)))
        self.assertNotEqual(first["thread_id"], second["thread_id"])
        self.assertTrue(all(r["cpu_seconds"] >= 0 for r in (first, second)))
        with patch("atsuite_sdk.resources.time.thread_time", side_effect=[1, 2, 3, 5]):
            with observe_call():
                with observe_call():
                    pass
                inner = get_resource_usage()["cpu_seconds"]
            outer = get_resource_usage()["cpu_seconds"]
        self.assertEqual((inner, outer), (1, 3))
        with patch("atsuite_sdk.resources.time.thread_time", side_effect=[1, 2, 3, 5]):
            with observe_call():
                context = copy_context()
                def nested_thread():
                    with observe_call():
                        pass
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(context.run, nested_thread).result()
            self.assertEqual(get_resource_usage()["cpu_seconds"], 4)

    def test_unsupported_meter_is_unknown_and_worker_and_thread_are_not_added(self):
        with patch("atsuite_sdk.resources.time.thread_time", side_effect=OSError):
            with observe_call():
                pass
            self.assertIsNone(get_resource_usage()["cpu_seconds"])
        with patch("atsuite_sdk.resources.time.thread_time", side_effect=[1, 2, OSError(), 4]):
            with observe_call():
                with observe_call():
                    pass
            self.assertIsNone(get_resource_usage()["cpu_seconds"])
        collector = TraceCollector("r", "s")
        collector.start("tool", "tool", "generic")
        collector.observe_runtime("tool", {
            "provider_metadata": {"resource_usage": {"cpu_seconds": 2, "cpu_complete": False}},
            "result": {"worker_usage": {"cpu_seconds": 3, "cpu_complete": False}},
        })
        self.assertEqual(collector.spans["tool"]["resources"]["cpu_seconds"], 3)
        self.assertEqual(collector.spans["tool"]["metadata"]["runtime_resource_usage"]["cpu_seconds"], 2)


class ConfidenceTests(unittest.TestCase):
    def predict(self, provider="aws_lambda", count=5, load=0.5, missing=None, calibrated=True):
        card = tariff(provider)
        if not calibrated:
            card["calibration"] = {}
        card["cpu_basis"] = "active" if provider == "aws_agentcore" else "allocated"
        card["rates"]["cpu_s"] = 0.001 if provider != "aws_lambda" else 0
        model = ProviderModel(tariffs={provider: card})
        for index in range(count):
            row = observation(index, provider)
            row["pool"]["resources"].update(cpu=2, disk_mb=1024)
            row["profile"]["metrics"]["tool_cpu_s"] = None
            if missing:
                row["profile"]["metrics"][missing] = None
            model.update(row)
        return model.predict(row["profile"], [row["pool"]], load, as_of_ns=10000)["pools"][0]

    def test_allocated_cpu_platforms_can_route_but_agentcore_cost_stays_unknown(self):
        for provider in ("aws_lambda", "ali_fc"):
            result = self.predict(provider)
            self.assertTrue(result["confidence"]["latency"]["reliable"])
            self.assertTrue(result["confidence"]["cost"]["reliable"])
            self.assertIn("tool_cpu_s", result["confidence"]["latency"]["optional_missing"])
        result = self.predict("aws_agentcore")
        self.assertTrue(result["confidence"]["latency"]["reliable"])
        self.assertFalse(result["confidence"]["cost"]["reliable"])
        self.assertIn("tool_cpu_s", result["confidence"]["cost"]["missing_cost_evidence"])

    def test_small_samples_missing_required_metrics_and_ood_still_reject(self):
        for kwargs, reason in (({"count": 2}, "sample_insufficient"),
                               ({"load": 100}, "distribution_outside_support"),
                               ({"missing": "llm_idle_s"}, "metric_missing")):
            result = self.predict(**kwargs)
            self.assertFalse(result["confidence"]["latency"]["reliable"])
            self.assertIn(reason, result["confidence"]["latency"]["reasons"])
        result = self.predict(missing="state_max_object_bytes")
        self.assertTrue(result["confidence"]["latency"]["reliable"])
        self.assertFalse(result["confidence"]["cost"]["reliable"])

    def test_uncalibrated_cost_does_not_replace_observed_latency(self):
        result = self.predict(calibrated=False)
        self.assertEqual(result["latency"]["mean"], 2)
        self.assertTrue(result["confidence"]["latency"]["reliable"])
        self.assertFalse(result["confidence"]["cost"]["reliable"])

    def test_absent_workflow_events_are_zero_only_for_complete_observations(self):
        row = observation()
        row["_features"] = features(row["profile"], row["pool"], 0.5)
        query = dict(row["_features"], **{"calls:optional": 1})
        _, coverage = _nearest([row], query, 1)
        self.assertEqual(coverage["feature_ranges"]["calls:optional"]["min"], 0)
        self.assertIn("calls:optional", coverage["extrapolated_features"])
        row["_features"]["_workflow_counts_complete"] = 0
        _, coverage = _nearest([row], query, 1)
        self.assertIn("calls:optional", coverage["missing_features"])


if __name__ == "__main__":
    unittest.main()
