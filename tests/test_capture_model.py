"""An unsuccessful task is still a complete recording of that execution."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from atsuite.tracing.storage import read_text, write_json
from experiments.e3.capture_model import _capture, outcome, select_requests
from experiments.exp1.capture.capture_agent_traces import run


class CaptureOutcomeTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_marker_prevents_new_tool_or_model_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            support = root / "local-capture"
            support.mkdir()
            (support / "tool_service.py").write_text("# fixture")
            (root / "plan.json").write_text("{}")
            output = root / "output"
            output.mkdir()
            (output / "STOP_AFTER_ACTIVE").touch()
            plan = {"tasks": [{"id": "one", "workload": "DataSciBench"}], "requests": []}
            with patch("experiments.e3.capture_model.command", AsyncMock(return_value="fixture-image")), patch(
                "experiments.e3.capture_model.run", AsyncMock(side_effect=AssertionError("model must not run"))
            ):
                result = await _capture(plan, {}, root, output, {}, 2)
            self.assertEqual(result["summary"]["expected"], 1)
            self.assertEqual(result["summary"]["recorded"], 0)
            self.assertFalse((output / "requests").exists())

    def test_frozen_selection_keeps_only_selected_requests_and_does_not_mutate_source(self):
        plan = {"tasks": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
                "requests": [{"task": "a", "arrival_id": "original-a"},
                             {"task": "b", "arrival_id": "original-b"}, {"task": "c"}]}
        selected = select_requests(plan, ["b", "a"])
        self.assertEqual(selected["tasks"], [{"id": "b"}, {"id": "a"}])
        self.assertEqual(selected["requests"], plan["requests"][:2])
        self.assertEqual(len(plan["tasks"]), 3)
        for invalid in ([], ["a", "a"], ["unknown"]):
            with self.assertRaisesRegex(ValueError, "unique known"):
                select_requests(plan, invalid)

    async def test_real_graph_failure_is_retained_and_not_a_capture_failure(self):
        class Tools:
            def __init__(self, config, recorder, *args):
                self.recorder = recorder

            async def aclose(self):
                self.recorder.emit("tool_sessions_closed")

        class Model:
            def __init__(self, recorder, config):
                self.recorder = recorder

            async def decide(self, phase, iteration, messages):
                self.recorder.emit("llm_start", phase=phase, iteration=iteration, request={})
                decision = {"steps": []}
                self.recorder.emit("llm", phase=phase, iteration=iteration, request={}, response={},
                                   decision=decision, response_id="test-response", duration_s=0,
                                   usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
                return decision

            async def aclose(self):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "task.json"
            manifest.write_text(json.dumps({"tasks": [{"task_id": "one", "prompt": "analyze",
                "prompt_sha256": "test", "readiness": {"inputs": {"files": []}}, "output_files": []}]}))
            config = {"tool": {"execution_scope": "unit test"}, "model": {"seed": 0},
                      "max_iterations": 1, "max_parallel": 1}
            directory = root / "trace"
            with patch("experiments.exp1.capture.capture_agent_traces.ToolClient", Tools), patch(
                "experiments.exp1.capture.capture_agent_traces.create_model_client", Model
            ):
                with self.assertRaisesRegex(RuntimeError, "Agent did not complete"):
                    await run("one", config, directory, manifest=manifest)
            result = outcome(directory)
            self.assertEqual(result["capture_status"], "complete")
            self.assertEqual(result["task_status"], "failed")
            self.assertEqual(result["failure_origin"], "task")
            self.assertEqual(result["usage"]["total_tokens"], 3)
            state = json.loads(read_text(directory / "result.json"))
            self.assertEqual(state["execution_plan"], [])
            trace = json.loads(read_text(directory / "trace.json"))
            trace["spans"][0]["end_ns"] = None
            write_json(directory / "trace.json", trace)
            self.assertEqual(outcome(directory)["capture_status"], "incomplete")

    async def test_model_output_and_api_errors_are_not_business_success(self):
        for response, expected in (({"usage": {}}, "model_output"), (None, "model_api")):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                events = [{"kind": "run_start"},
                          {"kind": "llm_error", "error_type": "ValueError", "response": response},
                          {"kind": "run_error", "error_type": "ValueError", "error": "invalid response"},
                          {"kind": "tool_sessions_closed"}]
                for i, event in enumerate(events):
                    event.update(seq=i, request_id="test")
                (directory / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
                write_json(directory / "result.json", {"status": "failed"})
                write_json(directory / "trace.json", {"request_id": "test", "completed_ns": 1,
                           "sessions_closed": True, "spans": [{"end_ns": 1}]})
                result = outcome(directory)
                self.assertEqual(result["capture_status"], "complete")
                self.assertEqual(result["failure_origin"], expected)
                self.assertEqual(result["task_status"], "failed")
