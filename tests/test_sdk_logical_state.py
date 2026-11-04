"""Explicit zero and nonzero SDK state observations in resident and restored calls."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atsuite.tracing import TraceCollector
from atsuite.workload import extract_profile
from atsuite_sdk.abstract import UnifiedAgentTool
from atsuite_sdk.state import StateRuntime


class SDKLogicalStateTests(unittest.TestCase):
    def profile(self, logical):
        collector = TraceCollector("request", "session")
        collector.start("session", "session", "agent")
        collector.start("tool", "tool", "test")
        collector.observe_runtime(
            "tool", {"provider_metadata": {"logical_state": logical}}
        )
        collector.end("tool")
        collector.end("session")
        return extract_profile(collector.snapshot())

    def test_stateless_and_empty_state_emit_explicit_zero_in_both_modes(self):
        for mode in ["mcp", "function"]:
            for access in ["stateless", "rw"]:
                with (
                    self.subTest(mode=mode, access=access),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    with patch.dict(
                        os.environ,
                        {
                            "ATSUITE_RUNTIME": mode,
                            "PROVIDER": "",
                            "ATSUITE_SESSION_ROOT": temporary,
                        },
                    ):
                        runtime = StateRuntime()

                    def empty() -> str:
                        return "unchanged business result"

                    tool = UnifiedAgentTool(empty, access=access)
                    with patch(
                        "atsuite_sdk.state.get_state_runtime", return_value=runtime
                    ):
                        self.assertEqual(
                            tool.execute_from_dict({"uid": "same"}),
                            "unchanged business result",
                        )
                    logical = runtime.get_logical_state()
                    self.assertTrue(logical["complete"])
                    self.assertEqual(logical["accesses"], [])
                    metrics = self.profile(logical)["metrics"]
                    for field in [
                        "state_bytes",
                        "state_reads",
                        "state_writes",
                        "state_max_object_bytes",
                    ]:
                        self.assertEqual(metrics[field], 0)
                    self.assertIsNotNone(metrics["state_observation_s"])
                    if mode == "mcp":
                        self.assertFalse((Path(temporary) / ".snapshots").exists())

    def test_nonzero_objects_and_files_are_observed_and_mcp_never_restores(self):
        observations = []
        for mode in ["mcp", "function"]:
            with tempfile.TemporaryDirectory() as temporary:
                with patch.dict(
                    os.environ,
                    {
                        "ATSUITE_RUNTIME": mode,
                        "PROVIDER": "",
                        "ATSUITE_SESSION_ROOT": temporary,
                    },
                ):
                    runtime = StateRuntime()
                state = runtime.register_object(
                    "tensor_workspace",
                    SimpleNamespace(values=[]),
                    __name__,
                    session_scoped=True,
                )

                def write() -> int:
                    state.values.append([1, 2, 3])
                    with runtime.session_open("notebook/entry", "w") as handle:
                        handle.write("value")
                    return len(state.values)

                def read() -> int:
                    return len(state.values)

                with patch("atsuite_sdk.state.get_state_runtime", return_value=runtime):
                    if mode == "mcp":
                        runtime.load_for_tool = unittest.mock.Mock(
                            side_effect=AssertionError("MCP restored")
                        )
                        runtime.save_after_tool = unittest.mock.Mock(
                            side_effect=AssertionError("MCP persisted")
                        )
                    self.assertEqual(
                        UnifiedAgentTool(write, access="rw").execute_from_dict(
                            {"uid": "same"}
                        ),
                        1,
                    )
                    logical = runtime.get_logical_state()
                    self.assertTrue(logical["complete"])
                    self.assertGreater(
                        self.profile(logical)["metrics"]["state_bytes"], 0
                    )
                    self.assertIn(
                        {
                            "object_id": "file:notebook/entry",
                            "operation": "write",
                            "bytes": 5,
                        },
                        logical["accesses"],
                    )
                    observations.append(logical["accesses"])
                    self.assertEqual(
                        UnifiedAgentTool(read, access="r").execute_from_dict(
                            {"uid": "same"}
                        ),
                        1,
                    )
                    self.assertTrue(
                        all(
                            a["operation"] == "read"
                            for a in runtime.get_logical_state()["accesses"]
                        )
                    )
        self.assertEqual(observations[0], observations[1])

    def test_unknown_state_and_failed_calls_do_not_reuse_previous_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(
                os.environ,
                {
                    "ATSUITE_RUNTIME": "mcp",
                    "PROVIDER": "",
                    "ATSUITE_SESSION_ROOT": temporary,
                },
            ):
                runtime = StateRuntime()
            runtime.register_object("opaque", SimpleNamespace(value=object()), __name__)

            def empty() -> str:
                return "ok"

            def fail() -> str:
                raise ValueError("expected")

            with patch("atsuite_sdk.state.get_state_runtime", return_value=runtime):
                UnifiedAgentTool(empty, access="r").execute_from_dict({"uid": "same"})
                self.assertFalse(runtime.get_logical_state()["complete"])
                self.assertIsNone(
                    self.profile(runtime.get_logical_state())["metrics"]["state_bytes"]
                )
                with self.assertRaises(ValueError):
                    UnifiedAgentTool(fail).execute_from_dict({"uid": "same"})
                self.assertTrue(runtime.get_logical_state()["complete"])
                self.assertEqual(runtime.get_logical_state()["accesses"], [])
