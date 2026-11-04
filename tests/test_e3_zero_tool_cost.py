"""Zero Tool calls prove zero FC usage, but still incur an initialized MCP Session."""

import json
import tempfile
import unittest
from pathlib import Path

from experiments.e3.evidence import price_request, reconcile


class ZeroToolCostTests(unittest.TestCase):
    def test_zero_fc_requires_a_closed_trace_with_no_attempts(self):
        config = {"native_elasticity": {"ali_fc": {"e3": {"cpu": 2, "memory_mb": 4096}}}}
        row = {"pool": "ali_fc", "workload": "TravelPlanner", "sessions_closed": True,
               "tool_attempts": 0, "usage": {"tool_calls": 0, "call_coverage_complete": True}}
        self.assertEqual(price_request(row, config)["total"], 0)
        for changed in ({"tool_attempts": 1}, {"sessions_closed": False}):
            with self.subTest(changed=changed):
                priced = price_request({**row, **changed}, config)
                self.assertIsNone(priced["total"])
                self.assertTrue(priced["missing"])
        row.pop("tool_attempts")
        self.assertIsNone(price_request(row, config)["total"])

    def test_init_only_mcp_is_joined_and_priced_without_fabricating_tool_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            request = run / "request"
            request.mkdir(parents=True)
            usage = root / "usage" / "e3"
            usage.mkdir(parents=True)
            events = [
                {"kind": "tool_session_open", "provider_session_id": "session",
                 "metadata": {"initialize_request_id": "initialize"}},
                {"kind": "tool_sessions_closed", "receipts": [{"provider_session_id": "session",
                 "request_id": "stop", "http_status": 204, "closed_epoch_s": 5}]},
            ]
            (request / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            (run / "attempts.json").write_text(json.dumps([{
                "pool": "ali_agentrun", "workload": "TravelPlanner", "task": "task", "task_id": "task",
                "request_id": "arrival", "success": False, "latency_s": 4, "directory": str(request),
            }]))
            native = [
                {"operation": "InvokeFunction", "requestid": key, "instanceid": "container",
                 "invokefunctionstarttimestamp": start, "invokefunctionlatencyms": duration, "durationms": duration}
                for key, start, duration in [("initialize", 1000, 10), ("notification", 1020, 2), ("stop", 4000, 1)]
            ]
            native_path = usage / "ali_agentrun-usage.json"
            native_path.write_text(json.dumps({"events": native}))
            config = {"native_elasticity": {"ali_agentrun": {"e3": {
                "cpu": 2, "memory_mb": 4096, "disk_mb": 10240,
            }}}}
            reconcile(run, root / "usage", root / "priced", config)
            row = json.loads((root / "priced/request-usage.jsonl").read_text())
            self.assertEqual(row["usage"]["tool_calls"], 0)
            self.assertEqual(row["usage"]["initialize_request_ids"], ["initialize"])
            self.assertEqual(row["usage"]["native_session_ids"], ["session"])
            self.assertGreater(row["priced_cost"]["total"], 0)
            self.assertEqual(row["native_session_lifecycle"]["session"]["observed_request_count"], 3)
            native_path.write_text(json.dumps({"events": []}))
            reconcile(run, root / "usage", root / "missing", config)
            missing = json.loads((root / "missing/request-usage.jsonl").read_text())
            self.assertIsNone(missing["priced_cost"]["total"])
            self.assertTrue(missing["priced_cost"]["missing"])


if __name__ == "__main__":
    unittest.main()
