"""Native usage evidence must not silently become zero cost or a complete invoice."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.e3.evidence import main, measured_usage, price_request, reconcile


class EvidenceTests(unittest.TestCase):
    def test_failed_attempts_do_not_lower_reported_successful_latency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            rows = []
            for i, (success, latency) in enumerate([(True, 10), (False, 1)]):
                directory = run / str(i)
                directory.mkdir()
                (directory / "events.jsonl").write_text("")
                rows.append(
                    {
                        "pool": "ali_agentrun",
                        "workload": "TravelPlanner",
                        "task": "task",
                        "task_id": "task",
                        "request_id": str(i),
                        "success": success,
                        "latency_s": latency,
                        "directory": str(directory),
                    }
                )
            (run / "attempts.json").write_text(json.dumps(rows))
            report = reconcile(run, root / "missing-usage", root / "out")
            cell = report["matrices"]["workload"][0]
            self.assertEqual(cell["successes"], 1)
            self.assertEqual(cell["latency_mean_s"], 5.5)
            self.assertEqual(cell["successful_latency_mean_s"], 10)
            self.assertIsNone(report["best_fixed_provider"])

    def test_lambda_joins_native_ids_and_rejects_incomplete_totals(self):
        report = {"message": "REPORT RequestId: matching\tDuration: 5 ms\tBilled Duration: 10 ms\tMemory Size: 2048 MB"}
        tools = [{"provider_request_id": "matching"}, {"provider_request_id": "missing"}]
        row = measured_usage("aws_lambda", tools, {"matching": [report, report]})
        self.assertEqual(row["matched_calls"], 1)
        self.assertEqual(row["observed_billed_gb_s"], 0.02)
        self.assertIsNone(row["billed_gb_s"])
        self.assertIsNone(row["actual_cost"])
        empty = measured_usage("aws_lambda", tools, {})
        self.assertIsNone(empty["observed_billed_gb_s"])

    def test_agentcore_missing_session_is_unknown_and_duplicate_samples_count_once(self):
        event = {
            "message": json.dumps(
                {
                    "event_timestamp": 1000,
                    "attributes": {"session.id": "session", "time_elapsed_seconds": 1},
                    "metrics": {"agent.runtime.vcpu.hours.used": 0.1, "agent.runtime.memory.gb_hours.used": 0.2},
                }
            )
        }
        tools = [{"provider_metadata": {"session_id": "session"}}]
        row = measured_usage("aws_agentcore", tools, {"session": [event, event]})
        self.assertEqual(row["vcpu_hours"], 0.1)
        self.assertFalse(row["billing_coverage_complete"])
        self.assertIsNone(measured_usage("aws_agentcore", tools, {})["vcpu_hours"])

    def test_ali_duration_is_not_session_residence_or_cost(self):
        row = measured_usage(
            "ali_agentrun",
            [{"provider_request_id": "call"}],
            {"call": [{"durationms": "2.5", "memoryusagemb": "null", "instanceid": "instance"}]},
        )
        self.assertEqual(row["observed_duration_ms"], 2.5)
        self.assertIsNone(row["request_metrics"][0]["memory_used_mb"])
        self.assertIsNone(row["request_metrics"][0]["is_cold_start"])
        self.assertIsNone(row["actual_cost"])
        self.assertFalse(row["billing_coverage_complete"])

    def test_session_pricing_collects_active_duration_once_including_initialize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            usage_dir = root / "usage" / "e3"
            usage_dir.mkdir(parents=True)
            events = [
                {"operation": "InvokeFunction", "requestId": rid, "instanceId": "instance", "durationMs": ms,
                 "invokeFunctionStartTimestamp": start, "invokeFunctionLatencyMs": ms}
                for rid, start, ms in (("initialize", 0, 5), ("call", 5, 20), ("call", 5, 20))
            ]
            (usage_dir / "ali_agentrun-usage.json").write_text(json.dumps({"events": events}))
            session_file = root / "sessions.json"
            session = {
                "sessionId": "session",
                "containerId": "instance",
                "sessionStatus": "Expired",
                "createdTime": "2026-01-01T00:00:00Z",
                "lastModifiedTime": "2026-01-01T00:05:00Z",
            }
            session_file.write_text(json.dumps({"functions": {"server": {"pages": [{"sessions": [session]}]}}}))
            argv = [
                "evidence", "--run", str(root), "--usage", str(root / "usage"), "--output", str(root / "out"),
                "--sessions", str(session_file),
            ]
            with patch("sys.argv", argv), patch("experiments.e3.evidence.reconcile", return_value={}) as collect:
                main()
            sessions = collect.call_args.args[4]
            self.assertEqual(sessions["session"]["observed_request_count"], 2)
            self.assertEqual(sessions["session"]["observed_active_time_ms"], 25)
            row = {
                "pool": "ali_agentrun",
                "workload": "TravelPlanner",
                "usage": {"native_session_ids": ["session"], "tool_calls": 1, "call_coverage_complete": True},
            }
            config = {
                "native_elasticity": {"ali_agentrun": {"e3": {"cpu": 2, "memory_mb": 3072, "disk_mb": 10240}}}
            }
            priced = price_request(row, config, sessions)
            self.assertIsNotNone(priced["total"])
            costs = {line["category"]: line["amount"] for line in priced["lines"]}
            self.assertAlmostEqual(costs["cpu"], 2 * 0.025 * 0.000088)
            self.assertAlmostEqual(costs["request"], 2 * 0.0075 * 0.000088)
            self.assertAlmostEqual(costs["memory"], 3 * 300 * 0.15 * 0.000088)
            self.assertNotIn("disk", costs)
            sessions["session"]["sessionStatus"] = "Deleted"
            self.assertEqual(price_request(row, config, sessions)["total"], priced["total"])
            row["usage"]["call_coverage_complete"] = False
            self.assertIsNone(price_request(row, config, sessions)["total"])
            row["usage"]["call_coverage_complete"] = True
            sessions["session"].pop("observed_active_time_ms")
            missing = price_request(row, config, sessions)
            self.assertIsNone(missing["total"])
            self.assertIn("native Session active duration pending", missing["missing"])
            for duration in (None, 21):
                with self.subTest(duration=duration):
                    events[-1]["invokeFunctionLatencyMs"] = duration
                    (usage_dir / "ali_agentrun-usage.json").write_text(json.dumps({"events": events}))
                    with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "Conflicting Ali request"):
                        main()
            events = events[:2]
            events[-1]["invokeFunctionLatencyMs"] = None
            (usage_dir / "ali_agentrun-usage.json").write_text(json.dumps({"events": events}))
            with patch("sys.argv", argv), patch("experiments.e3.evidence.reconcile", return_value={}) as collect:
                main()
            incomplete = collect.call_args.args[4]
            self.assertIsNone(incomplete["session"]["observed_active_time_ms"])
            self.assertIsNone(price_request(row, config, incomplete)["total"])

    def test_stopped_agentcore_session_does_not_require_a_300_second_tail(self):
        row = {
            "pool": "aws_agentcore", "workload": "TravelPlanner", "last_tool_epoch_s": 1000,
            "usage": {"tool_calls": 1, "sessions": [{
                "session_id": "native-session", "vcpu_hours": 0.001, "memory_gb_hours": 0.002,
                "residence": {"last_usage_epoch_ms": 1000000},
            }]},
        }
        config = {"native_elasticity": {"aws_agentcore": {"e3": {}}}}
        self.assertIsNone(price_request(row, config)["total"])
        row["sessions_closed"] = True
        self.assertIsNone(price_request(row, config)["total"])
        row["session_release_receipts"] = [{"provider_session_id": "native-session", "http_status": 200, "closed_epoch_s": 1001}]
        self.assertIsNone(price_request(row, config)["total"])
        row["usage"]["sessions"][0]["residence"]["last_usage_epoch_ms"] = 1001000
        self.assertIsNotNone(price_request(row, config)["total"])


if __name__ == "__main__":
    unittest.main()
