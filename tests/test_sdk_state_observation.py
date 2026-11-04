"""Real SDK snapshots distinguish stateless calls, stateful calls and failed measurement."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atsuite.tracing import TraceCollector
from atsuite.workload import extract_profile
from atsuite_sdk.abstract import UnifiedAgentTool
from atsuite_sdk.state import StateRuntime, session_open


class Box:
    def __init__(self):
        self.value = ""


class SDKStateObservationTests(unittest.TestCase):
    def test_resident_and_restored_snapshots_have_the_same_object_and_file_bytes(self):
        measured = []
        for mode in ("mcp", "function"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with patch.dict(os.environ, {"ATSUITE_RUNTIME": mode, "PROVIDER": "",
                                             "ATSUITE_SESSION_ROOT": directory}):
                    runtime = StateRuntime()
                    box = runtime.register_object("box", Box(), __name__, session_scoped=True)

                    def write():
                        box.value = "你好"
                        with session_open("note.txt", "w", encoding="utf-8") as stream:
                            stream.write(box.value)
                        return "saved"

                    def read():
                        with session_open("note.txt", encoding="utf-8") as stream:
                            return stream.read(), box.value

                    with patch("atsuite_sdk.state._RUNTIME", runtime):
                        writer = UnifiedAgentTool(write, access="rw")
                        reader = UnifiedAgentTool(read, access="r")
                        self.assertEqual(writer.execute_from_dict({"uid": "request"}), "saved")
                        written = runtime.get_logical_state()
                        self.assertTrue(written["complete"])
                        self.assertEqual(reader.execute_from_dict({"uid": "request"}), ("你好", "你好"))
                        observed = runtime.get_logical_state()
                        self.assertTrue(observed["complete"])
                        self.assertEqual({a["object_id"] for a in observed["accesses"]},
                                         {f"object:{__name__}:box", "file:note.txt"})
                        file_access = next(a for a in observed["accesses"] if a["object_id"] == "file:note.txt")
                        self.assertEqual(file_access, {"object_id": "file:note.txt", "operation": "read", "bytes": 6})
                        self.assertEqual(observed["accesses"],
                                         [{**a, "operation": "read"} for a in written["accesses"] if a["operation"] == "write"])
                        measured.append(observed["accesses"])
                        path = Path(directory) / ".snapshots" / runtime._snapshot_key("request")
                        self.assertEqual(path.exists(), mode == "function")
        self.assertEqual(measured[0], measured[1])

    def test_stateless_and_failed_business_calls_both_reach_the_collector(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "ATSUITE_RUNTIME": "mcp", "PROVIDER": "", "ATSUITE_SESSION_ROOT": directory,
        }):
            runtime = StateRuntime()
            runtime.register_object("box", Box(), __name__, session_scoped=True)
            with patch("atsuite_sdk.state._RUNTIME", runtime):
                for stateful in (False, True):
                    collector = TraceCollector("r", "s")
                    collector.start("session", "session", "request")
                    collector.start("tool", "tool", "query")

                    def query():
                        if stateful:
                            raise ValueError("Business validation failed")
                        return "read-only immutable input"

                    tool = UnifiedAgentTool(query, access="r" if stateful else "stateless")
                    result = None
                    error = None
                    try:
                        result = tool.execute_from_dict({"uid": "request"})
                    except ValueError as exc:
                        error = exc
                    collector.end("tool", result, error=error)
                    collector.observe_runtime("tool", {"result": result, "provider_metadata": {
                        "logical_state": runtime.get_logical_state()}})
                    collector.end("session", error=error)
                    accesses = collector.spans["tool"]["metadata"]["state_accesses"]
                    self.assertEqual(bool(accesses), stateful)
                    profile = extract_profile(collector.snapshot())
                    self.assertEqual(profile["metrics"]["state_bytes"], sum(a["bytes"] for a in accesses))
                    self.assertEqual(profile["status"], "failed" if stateful else "success")

    def test_unserializable_state_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "ATSUITE_RUNTIME": "mcp", "PROVIDER": "", "ATSUITE_SESSION_ROOT": directory,
        }):
            runtime = StateRuntime()
            box = Box()
            box.value = object()
            runtime.register_object("box", box, __name__)
            with patch("atsuite_sdk.state._RUNTIME", runtime):
                UnifiedAgentTool(lambda: "ok", access="r").execute_from_dict({"uid": "request"})
            self.assertFalse(runtime.get_logical_state()["complete"])


if __name__ == "__main__":
    unittest.main()
