"""Offline contracts for provider modeling; synthetic rows are not cloud evidence."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from atsuite.analysis.model import ProviderMetric, RunContext
from atsuite.analysis.pricing import create_pricing_policy
from atsuite.provider_model import BACKENDS, ProviderModel, observation_from_report
from atsuite.provider_model.e1a import _estimate
from atsuite.provider_model.experiment import evaluate
from atsuite.workload import WorkloadModel, extract_profile
from tests.test_workload import trace
from tests.workload_fixtures import describe


def profile(state=8000, idle=2):
    result = extract_profile(trace())
    result["reference_id"] = "test-reference"
    result["descriptor"] = describe()
    result["metrics"].update(
        state_bytes=state,
        llm_idle_s=idle,
        execution_active_s=2 + idle,
        rho_kb_s=state / 1000 / idle,
    )
    return result


def pool(provider="aws_lambda", instances=2):
    return {
        "provider": provider,
        "environment": "frozen-test-v1",
        "resources": {"memory_mb": 2048},
        "instances": instances,
        "sessions_per_instance": 1,
        "invocations_per_instance": 10,
    }


def observation(i=0, provider="aws_lambda", latency=2, state=8000):
    return {
        "kind": "request",
        "id": str(i),
        "pool": pool(provider),
        "profile": profile(state),
        "load_rps": 0.5,
        "arrival_ns": i * 100 + 1,
        "completed_ns": i * 100 + 10,
        "available_ns": i * 100 + 10,
        "status": "success",
        "measurements": {"latency_s": latency, "service_slot_s": 4},
        "cost": None,
    }


def cost(amount=3, basis="estimated", complete=False):
    return {
        "components": {"compute": amount},
        "currency": "USD",
        "scope": "pool_total",
        "basis": basis,
        "complete": complete,
        "evidence": ["test-fixture"],
        "attribution": "exclusive",
    }


class ProviderModelTests(unittest.TestCase):
    def test_four_independent_backends_and_missing_pool(self):
        model = ProviderModel()
        for i, provider in enumerate(BACKENDS):
            model.update(observation(i, provider, i + 1))
        result = model.predict(
            profile(), [pool(p) for p in BACKENDS], 0.5, as_of_ns=1000
        )
        self.assertEqual(
            [p["latency"]["p95"] for p in result["pools"]], [1, 2, 3, 4, 5]
        )
        self.assertEqual(result["pools"][-1]["label"], "Aliyun FC Session MCP")
        query = pool()
        query["resources"]["memory_mb"] = 4096
        empty = model.predict(profile(), [query], 0.5)["pools"][0]
        self.assertIsNone(empty["latency"]["p95"])
        self.assertTrue(empty["cost_missing"])

    def test_joint_mixture_quantile_not_average_p95(self):
        model = ProviderModel(neighbors=1)
        model.update(observation(0, latency=1, state=1000))
        model.update(observation(1, latency=100, state=1000000))
        mixture = model.predict([profile(1000), profile(1000000)], [pool()], 0.5)[
            "pools"
        ][0]["latency"]
        self.assertEqual(mixture["samples"], [1, 100])
        self.assertAlmostEqual(mixture["p95"], 95.05)
        self.assertNotEqual(mixture["p95"], 50.5)

    def test_same_rho_different_raw_state_and_idle(self):
        model = ProviderModel(neighbors=1)
        first = observation(0, latency=1, state=1000)
        first["profile"] = profile(1000, 1)
        second = observation(1, latency=10, state=10000)
        second["profile"] = profile(10000, 10)
        model.update(first)
        model.update(second)
        self.assertEqual(
            model.predict(profile(10000, 10), [pool()], 0.5)["pools"][0]["latency"][
                "p50"
            ],
            10,
        )

    def test_load_capacity_features_and_extrapolation(self):
        model = ProviderModel(neighbors=1)
        low = observation(0, latency=1)
        high = observation(1, latency=9)
        high["load_rps"] = 10
        model.update(low)
        model.update(high)
        self.assertEqual(
            model.predict(profile(), [pool()], 10)["pools"][0]["latency"]["p50"], 9
        )
        result = model.predict(profile(), [pool(instances=4)], 20)["pools"][0]
        self.assertIn("load_rps", result["coverage"][0]["extrapolated_features"])
        self.assertIn("instances", result["coverage"][0]["extrapolated_features"])
        self.assertTrue(result["low_confidence"])
        self.assertTrue(result["capacity"]["insufficient"])

    def test_session_capacity_is_not_rpc_concurrency(self):
        model = ProviderModel()
        model.update(observation(provider="ali_agentrun"))
        capacity = model.predict(profile(), [pool("ali_agentrun")], 0.6)["pools"][0][
            "capacity"
        ]
        self.assertEqual(capacity["slots"], 2)
        self.assertEqual(capacity["max_service_rps"], 0.5)
        self.assertTrue(capacity["insufficient"])
        query = pool("ali_agentrun")
        query["instances"] = None
        self.assertIsNone(
            model.predict(profile(), [query], 0.5)["pools"][0]["capacity"][
                "insufficient"
            ]
        )
        query["instances"] = 0
        self.assertEqual(
            model.predict(profile(), [query], 0.5)["pools"][0]["capacity"][
                "effective_rps"
            ],
            0,
        )

    def test_causal_enrichment_dedup_and_persistence(self):
        model = ProviderModel()
        row = observation()
        model.update(row)
        model.update(row)
        self.assertEqual(
            model.predict(profile(), [pool()], 0.5, as_of_ns=10)["pools"][0][
                "sample_count"
            ],
            0,
        )
        enriched = copy.deepcopy(row)
        enriched.update(available_ns=50, cost=cost())
        model.update(enriched)
        self.assertTrue(
            model.predict(profile(), [pool()], 0.5, as_of_ns=50)["pools"][0][
                "cost_missing"
            ]
        )
        self.assertFalse(
            model.predict(profile(), [pool()], 0.5, as_of_ns=51)["pools"][0][
                "cost_missing"
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = ProviderModel.load(path)
            self.assertEqual(
                model.predict(profile(), [pool()], 0.5, as_of_ns=51),
                restored.predict(profile(), [pool()], 0.5, as_of_ns=51),
            )
        row["measurements"]["latency_s"] = 999
        self.assertEqual(
            model.predict(profile(), [pool()], 0.5)["pools"][0]["latency"]["p50"], 2
        )
        with self.assertRaises(ValueError):
            model.update(row)

    def test_shared_window_charged_once_and_partial_never_total(self):
        model = ProviderModel()
        request = observation()
        request["cost"] = cost(3)
        model.update(request)
        window = observation(1)
        window.update(
            kind="window",
            id="window",
            arrival_ns=0,
            completed_ns=10 * 10**9,
            available_ns=10 * 10**9,
            cost=cost(10, complete=True),
        )
        window["cost"]["components"] = {
            "compute": 2,
            "retention": 5,
            "idle": 2,
            "state": 1,
        }
        model.update(window)
        model.update(window)
        result = model.predict(profile(), [pool()], 0.5, horizon_s=20)["pools"][0]
        costs = {c["kind"]: c for c in result["cost"]}
        self.assertIsNone(costs["request"]["total"])
        self.assertEqual(costs["window"]["total"]["p50"], 20)
        self.assertEqual(costs["window"]["source_count"], 1)
        self.assertEqual(costs["window"]["components"]["idle"]["p50"], 4)

    def test_unknown_failed_and_invalid_observations(self):
        model = ProviderModel()
        row = observation()
        row["status"] = "failed"
        model.update(row)
        result = model.predict(profile(), [pool()], 0.5)["pools"][0]
        self.assertEqual(result["status_counts"], {"failed": 1})
        self.assertIsNone(result["latency"]["p95"])
        for key, value in [("available_ns", 1), ("load_rps", float("nan"))]:
            row = observation(1)
            row[key] = value
            with self.assertRaises(ValueError):
                model.update(row)
        row = observation(1)
        row["cost"] = cost(basis="actual")
        with self.assertRaises(ValueError):
            model.update(row)
        row["cost"].update(complete=True, attribution="account_daily")
        with self.assertRaises(ValueError):
            model.update(row)
        row["cost"].update(attribution="exclusive", currency="CNY")
        with self.assertRaises(ValueError):
            model.update(row)
        wrong = pool()
        wrong["mode"] = "state-coupled"
        with self.assertRaises(ValueError):
            model.predict(profile(), [wrong], 1)

    def test_workload_model_joint_profiles_are_copies(self):
        workload = WorkloadModel()
        workload.observe(trace(), 20)
        predicted = workload.predict(21, descriptor=describe())
        self.assertEqual(len(predicted["joint_profiles"]), 1)
        predicted["joint_profiles"][0]["metrics"]["state_bytes"] = 0
        self.assertEqual(
            workload.predict(21, descriptor=describe())["joint_profiles"][0]["metrics"]["state_bytes"], 8000
        )
        result = ProviderModel().predict(workload.predict(21, descriptor=describe()), [pool()], 0.5)
        self.assertEqual(result["workload_basis"], "joint_profiles")
        self.assertEqual(result["workload_samples"], 1)

    def test_completion_hook_and_report_identity(self):
        model = ProviderModel()
        model.complete(trace(), pool(), 0.5, available_ns=20)
        self.assertEqual(
            model.predict(profile(), [pool()], 0.5, as_of_ns=21)["pools"][0][
                "sample_count"
            ],
            1,
        )
        report = {"run": {"provider": "ali_fc", "uid": "r"}}
        with self.assertRaises(ValueError):
            observation_from_report(profile(), report, pool(), 0.5, 20)

    def test_lambda_billed_duration_does_not_bill_init_twice(self):
        context = RunContext(
            uid="r",
            provider="aws_lambda",
            observability_provider="",
            benchmark="",
            trace="",
            family="",
            config_path="",
        )
        metric = ProviderMetric(
            metric_id="m",
            provider="aws_lambda",
            source="cloudwatch",
            fields={
                "billed_duration_ms": 1000,
                "init_duration_ms": 200,
                "memory": 1024,
            },
        )
        lines = create_pricing_policy("aws_lambda").price(
            context, metrics=[metric], state=[]
        )
        self.assertAlmostEqual(
            next(l.amount for l in lines if l.category == "memory"), 0.0000166667
        )
        metric.fields = {"duration_ms": 800, "init_duration_ms": 200, "memory": 1024}
        lines = create_pricing_policy("aws_lambda").price(
            context, metrics=[metric], state=[]
        )
        self.assertAlmostEqual(
            next(l.amount for l in lines if l.category == "memory"), 0.0000166667
        )

    def test_session_http_duration_is_not_tool_cpu_cost(self):
        metric = {
            "metric_id": "m",
            "provider": "ali_agentrun",
            "source": "sls",
            "fields": {"duration_ms": 999999},
            "evidence_refs": ["sls:window"],
        }
        result = _estimate("ali_agentrun", [metric], pool("ali_agentrun"), "fixture")
        self.assertEqual(set(result["components"]), {"request"})
        self.assertFalse(result["complete"])
        self.assertEqual(result["scope"], "fc_mcp_tool_requests")

    def test_temporal_validation_does_not_read_future_profiles(self):
        rows = [observation(i, latency=2 + i) for i in range(5)]
        rows[0]["available_ns"] = 250
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("atsuite.provider_model.experiment.plot"),
        ):
            report = evaluate(rows, Path(directory), training_fraction=0.6)
        requests = sorted(report["requests"], key=lambda r: r["arrival_ns"])
        self.assertIsNone(requests[0]["predicted_p50"])
        self.assertIsNone(requests[1]["predicted_p50"])
        self.assertEqual(requests[2]["sample_count"], 1)
        self.assertEqual(
            report["providers"]["aws_lambda"]["actual_cost_observations"], 0
        )


class ProfilingTests(unittest.TestCase):
    def test_e1a_import_matches_calls_and_preserves_delayed_evidence(self):
        from atsuite.provider_model.e1a import load_e1a

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            capture.mkdir()
            (capture / "events.jsonl").write_text("{}\n")
            digest = hashlib.sha256(b"{}\n").hexdigest()
            run_dir = root / "run"
            request_dir = run_dir / "requests" / "request"
            request_dir.mkdir(parents=True)
            (request_dir / "events.jsonl").write_text(
                json.dumps({"kind": "tool", "transport": {"request_id": "call"}}) + "\n"
            )
            run = {
                "provider": "aws_lambda",
                "duration_ms": 10000,
                "started_epoch": 0,
                "source_files": {
                    "benchmarks/DataSciBench/langgraph_agent/graph.py": "g",
                    "benchmarks/DataSciBench/langgraph_agent/trace_replayer.py": "r",
                },
                "deployment": {
                    "image": {
                        "source_sha256": {"/app/implementation.py": "tool"},
                        "capture_events_sha256": digest,
                        "id": "image",
                        "architecture": "x86_64",
                    },
                    "actual": {"memory_mb": 1024, "instance_concurrency": 1},
                },
            }
            (run_dir / "run.json").write_text(json.dumps(run))
            record = {
                "arrival_id": "request",
                "task_id": "fixture-task",
                "trace_id": str(capture),
                "data_version": "data",
                "status": "succeeded",
                "request_arrival_utc": "1970-01-01T00:00:01+00:00",
                "request_end_utc": "1970-01-01T00:00:02+00:00",
                "end_to_end_ms": 1000,
            }
            (run_dir / "request_metrics.jsonl").write_text(json.dumps(record) + "\n")
            metric = {
                "metric_id": "m",
                "provider": "aws_lambda",
                "source": "cloudwatch",
                "invocation_call_id": "call",
                "evidence_refs": ["cloudwatch:1"],
                "fields": {
                    "billed_duration_ms": 1000,
                    "duration_ms": 1000,
                    "memory_limit_mb": 1024,
                    "is_cold_start": "false",
                },
            }
            (run_dir / "provider_evidence-1.json").write_text(
                json.dumps({"collected_epoch": 30, "metrics": [metric]})
            )
            with patch("atsuite.provider_model.e1a.load_capture", return_value=trace()):
                rows, provenance = load_e1a(root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["task_id"], "fixture-task")
            self.assertEqual(rows[1]["task_id"], "fixture-task")
            self.assertEqual(rows[0]["available_ns"], 2 * 10**9)
            self.assertIsNone(rows[0]["cost"])
            self.assertEqual(rows[1]["available_ns"], 30 * 10**9)
            self.assertEqual(rows[1]["cost"]["basis"], "estimated")
            self.assertIsNone(rows[1]["pool"]["instances"])
            self.assertIn(str(request_dir / "events.jsonl"), provenance["sources"])
            model = ProviderModel()
            for row in rows:
                model.update(row)
            predicted = model.predict(
                profile(), [rows[0]["pool"]], 0.1, as_of_ns=30 * 10**9
            )
            self.assertTrue(predicted["pools"][0]["cost_missing"])
            (capture / "events.jsonl").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_e1a(root)

    def test_plan_and_complete_cost_crossings(self):
        from atsuite.provider_model import crossover_intervals, profiling_plan

        plan = profiling_plan()
        self.assertEqual(len(plan["cases"]), 9)
        self.assertEqual(set(plan["providers"]), set(BACKENDS) - {"local"})
        model = ProviderModel(neighbors=1)
        for provider, amounts in [("aws_lambda", (1, 5)), ("aws_agentcore", (3, 3))]:
            for i, amount in enumerate(amounts):
                row = observation(i, provider)
                row.update(
                    kind="window", load_rps=i * 2, cost=cost(amount, complete=True)
                )
                model.update(row)
        sweep = [
            {
                "x": load,
                "prediction": model.predict(
                    profile(), [pool("aws_lambda"), pool("aws_agentcore")], load
                ),
            }
            for load in (0, 2)
        ]
        crossings = crossover_intervals(sweep)
        self.assertEqual(len(crossings), 1)
        self.assertEqual(crossings[0]["interval"], [0, 2])
        for point in sweep:
            for result in point["prediction"]["pools"]:
                result["cost"][0]["complete"] = False
        self.assertEqual(crossover_intervals(sweep), [])

    def test_each_backend_gets_a_temporal_holdout(self):
        rows = [
            observation(i + offset, provider)
            for offset, provider in [(0, "aws_lambda"), (100, "ali_fc")]
            for i in range(5)
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("atsuite.provider_model.experiment.plot"),
        ):
            report = evaluate(rows, Path(directory), training_fraction=0.6)
        self.assertEqual(report["providers"]["aws_lambda"]["held_out_requests"], 2)
        self.assertEqual(report["providers"]["ali_fc"]["held_out_requests"], 2)


class OnlineHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_freezes_prediction_and_updates_success_or_failure(self):
        from atsuite.tracing.storage import read_text
        from experiments.exp1.capture import capture_agent_traces as capture

        for failed in (False, True):

            class Adapter:
                def __init__(self, collector):
                    self.collector = collector

                async def astream(self, *args, fail=failed, **kwargs):
                    if fail:
                        raise RuntimeError("test graph failure")
                    yield {"status": "complete", "iteration": 0}

            model = ProviderModel()
            collector = Mock()
            collector.snapshot.return_value = trace("online")
            tools = Mock(renderings=False, aclose=AsyncMock(), assert_consumed=Mock())
            config = {
                "tool": {"provider": "aws_lambda", "execution_scope": "test"},
                "model": {"seed": 0},
                "max_iterations": 1,
                "max_parallel": 2,
            }
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(capture, "TraceCollector", return_value=collector),
                patch.object(capture, "LangGraphTracingAdapter", Adapter),
                patch.object(capture, "TraceRecorder"),
                patch.object(capture, "ToolClient", return_value=tools),
                patch.object(
                    capture, "ReplayModel", return_value=Mock(assert_consumed=Mock())
                ),
                patch.object(capture, "build_graph"),
            ):
                output = Path(directory)
                operation = capture.run(
                    "csv_excel_34",
                    config,
                    output,
                    capture=output,
                    provider_model=model,
                    provider_context={"pool": pool(), "load_rps": 0.5},
                )
                if failed:
                    with self.assertRaisesRegex(RuntimeError, "test graph failure"):
                        await operation
                else:
                    await operation
                prediction = json.loads(read_text(output / "provider_prediction.json"))
                self.assertEqual(prediction["pools"][0]["sample_count"], 0)
                observed = json.loads(read_text(output / "provider_observation.json"))
                self.assertEqual(observed["status"], "failed" if failed else "success")
                self.assertEqual(len(model.samples), 1)
                tools.aclose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
