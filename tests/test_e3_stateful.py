import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.ScientificComputation.langgraph_agent.graph import validate
from benchmarks.ScientificComputation.nodes.tensor_workspace.implementation import (
    scicom_load_tensor,
    scicom_solve_from_svd,
    scicom_svd_decompose,
    scicom_view_tensor,
)
from experiments.e3.stateful import prepare_scientific
from benchmarks.ClaudeCodeReview.langgraph_agent.graph import (
    validate as validate_review,
)


class StatefulScientificTests(unittest.TestCase):
    def test_npz_inputs_reused_factors_and_independent_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "problems"
            manifest = prepare_scientific(output, [8])
            task = manifest["tasks"][0]
            with patch.dict(
                "os.environ", {"ATSUITE_SCIENTIFIC_PROBLEMS": str(output / "inputs")}
            ):
                for tensor in ("A", "B"):
                    result = json.loads(
                        scicom_load_tensor(task["task_id"], tensor, "heavy-" + tensor)
                    )
                    self.assertFalse(result["isError"])
                result = json.loads(
                    scicom_svd_decompose(
                        "heavy-A", "heavy-u", "heavy-s", "heavy-vt", summary_only=True
                    )
                )
                self.assertEqual(result["result"]["u_shape"], [8, 8])
                self.assertNotIn("u", result["result"])
                with patch(
                    "numpy.linalg.svd", side_effect=AssertionError("must reuse factors")
                ):
                    solved = json.loads(
                        scicom_solve_from_svd(
                            "heavy-u", "heavy-s", "heavy-vt", "heavy-B", "heavy-x"
                        )
                    )
                self.assertFalse(solved["isError"])

            class Tools:
                async def call(self, name, args, key):
                    return scicom_view_tensor(**args)

            self.assertTrue(
                asyncio.run(validate(task, {"solution_tensor": "heavy-x"}, Tools()))[
                    "ok"
                ]
            )
            self.assertFalse(
                asyncio.run(validate(task, {"solution_tensor": "heavy-B"}, Tools()))[
                    "ok"
                ]
            )
            with np.load(
                task["verification"]["input_npz"], allow_pickle=False
            ) as inputs:
                self.assertEqual(inputs["B"].shape, (8, 16))

    def test_summary_requires_stored_factors(self):
        result = json.loads(scicom_svd_decompose("unused", summary_only=True))
        self.assertTrue(result["isError"])

    def test_invalid_sizes_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            for sizes in ([], [2, 2], [0], [4097]):
                with self.assertRaises(ValueError):
                    prepare_scientific(output, sizes)
                self.assertFalse(output.exists())


class ReviewEvidenceTests(unittest.TestCase):
    def test_findings_require_real_passing_base_and_failing_head_receipts(self):
        task = {
            "pr_id": 401,
            "base_ref": "main",
            "head_ref": "pr/401",
            "verification": {
                "defects": [{"file": "algebra.py", "start": 12, "end": 12}],
                "required_test_suites": ["algebra"],
                "base_revision": "base",
                "head_revision": "head",
            },
        }
        answer = {
            "verdict": "changes_requested",
            "findings": [
                {"file": "algebra.py", "line": 12, "body": "Lost coefficient"}
            ],
        }
        runs = {
            "401:main:algebra": {
                "exit_code": 0,
                "tests": 10,
                "skipped": 1,
                "revision": "base",
            },
            "401:pr/401:algebra": {
                "exit_code": 1,
                "failed_tests": ["test_factor"],
                "revision": "head",
            },
        }

        class Tools:
            async def call(self, name, args, key):
                return runs

        def check():
            return asyncio.run(validate_review(task, answer, Tools()))["ok"]

        self.assertTrue(check())
        runs["401:pr/401:algebra"]["exit_code"] = 0
        self.assertFalse(check())
        runs["401:pr/401:algebra"]["exit_code"] = 1
        runs["401:main:algebra"]["revision"] = "unrelated-commit"
        self.assertFalse(check())
        runs.clear()
        self.assertFalse(check())


class NativeDecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_finish_is_required_only_when_thinking_is_disabled(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from benchmarks.DataSciBench.langgraph_agent.model_client import DeepSeekClient

        response = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(
                content=None, tool_calls=[SimpleNamespace(function=SimpleNamespace(
                    name="submit_result", arguments='{"answer":{}}'))]))],
            model="deepseek-flash", id="fixture", model_dump=lambda **kwargs: {},
        )
        api = MagicMock()
        api.api_key = "fixture"
        api.chat.completions.create = AsyncMock(return_value=response)
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture"}), patch(
            "benchmarks.DataSciBench.langgraph_agent.model_client.AsyncOpenAI", return_value=api
        ):
            for thinking in ("disabled", "enabled"):
                model = DeepSeekClient(MagicMock(), {"seed": 0, "temperature": 0, "top_p": 1,
                    "max_tokens": 1000, "retries": 0, "thinking": thinking})
                model.tool_definitions = [{"name": "inspect", "parameters": {"type": "object"}}]
                self.assertEqual(await model.decide("ACT", 0, []), {"answer": {}})
                request = api.chat.completions.create.call_args.kwargs
                self.assertEqual(request.get("tool_choice"), "required" if thinking == "disabled" else None)


class DeploymentLimitTests(unittest.TestCase):
    def test_invalid_concurrency_fails_before_build_or_cloud_access(self):
        from experiments.exp1.runner.deploy_e1a import configure

        for value in (0, -1, True, 1.5, "8"):
            with self.assertRaises(ValueError):
                configure("ali_fc", Path("unused"), concurrency_limit=value)


class StateSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_access_metadata_does_not_parallelize_state_writes(self):
        from experiments.e3.agent import build_graph

        seen, waves = [], []

        class Model:
            async def decide(self, phase, iteration, messages):
                if iteration == 0:
                    return {"calls": [{"tool": "write", "args": {}}, {"tool": "read", "args": {}}]}
                return {"answer": {}}

        class Tools:
            async def call(self, name, args, key, wave=None):
                waves.append(wave)
                if name == "write":
                    await asyncio.sleep(0)
                    seen.append("written")
                else:
                    seen.append("read-after-write" if seen else "stale-read")
                return len(seen)

        async def verify(*args):
            return {"ok": True}

        graph = build_graph(Model(), Tools(), {"tool_definitions": [{"name": "write"}, {"name": "read"}]}, "", verify)
        result = await graph.ainvoke({"user_prompt": "state", "iteration": 0})
        self.assertEqual(result["status"], "complete")
        self.assertEqual(seen, ["written", "read-after-write"])
        self.assertEqual(waves, [None, None])


class StreamingMeteringTests(unittest.TestCase):
    def test_full_native_duration_and_overlap_are_accounted_once(self):
        from experiments.e3.evidence import session_active_usage

        first = {"requestid": "svd", "durationms": "3.22", "invokefunctionstarttimestamp": "1000", "invokefunctionlatencyms": "17588.37"}
        overlap = {"requestid": "parallel", "durationms": "2", "invokefunctionstarttimestamp": "2000", "invokefunctionlatencyms": "5000"}
        result = session_active_usage([first, overlap, first])
        self.assertEqual(result["observed_request_count"], 2)
        self.assertAlmostEqual(result["observed_active_time_ms"], 17588.37)
        self.assertIsNone(session_active_usage([{k:v for k,v in first.items() if k != "invokefunctionlatencyms"}])["observed_active_time_ms"])
        with self.assertRaises(ValueError):
            session_active_usage([first, {**first, "invokefunctionlatencyms": "10"}])

    def test_deleted_session_uses_native_init_and_delete_not_an_idle_tail(self):
        from experiments.e3.evidence import released_session_usage

        def event(key, start, duration):
            return {"requestid": key, "instanceid": "container", "durationms": "3",
                    "invokefunctionstarttimestamp": start, "invokefunctionlatencyms": duration}

        index = {"init": [event("init", 1000, 1000)], "work": [event("work", 5000, 10000)],
                 "stop": [event("stop", 20000, 5)]}
        receipt = {"provider_session_id": "session", "http_status": 204, "request_id": "stop", "closed_epoch_s": 20.2}
        result = released_session_usage("session", receipt, [{"provider_request_id": "work"}], index)
        from datetime import datetime

        self.assertEqual(result["sessionStatus"], "Deleted")
        lifetime = (datetime.fromisoformat(result["lastModifiedTime"]) - datetime.fromisoformat(result["createdTime"])).total_seconds()
        self.assertAlmostEqual(lifetime, 19.005)
        self.assertAlmostEqual(result["observed_active_time_ms"], 11005)
        self.assertIn("native DELETE", result["lifecycle_basis"])
        self.assertIsNone(released_session_usage("session", {**receipt, "http_status": 500}, [{"provider_request_id": "work"}], index))


if __name__ == "__main__":
    unittest.main()
