"""Characterization contract tests; no LLM, cloud or Docker required."""

import copy
import tempfile
import unittest
from pathlib import Path

from atsuite.tracing import TraceCollector
from atsuite.workload import WorkloadModel, extract_profile
from atsuite.workload.experiment import evaluate, load_capture


def trace(request="r", arrival=1, end=10, mode="resident"):
    collector = TraceCollector(request, request)
    for key, kind, name, deps, start, stop in [
        ("session", "session", "agent", [], 0, 10),
        ("a", "workflow", "plan", [], 0, 2),
        ("b", "workflow", "work", ["a"], 2, 4),
        ("c", "workflow", "work", ["a"], 2, 4),
        ("d", "workflow", "plan", ["b", "c"], 4, 6),
        ("llm", "llm", "model", [], 0, 2),
        ("tool", "tool", "python", [], 2, 4),
    ]:
        collector.start(
            key,
            kind,
            name,
            parent_id="session" if key != "session" else None,
            dependencies=deps,
        )
        collector.end(key)
        collector.spans[key].update(
            start_ns=start * 10**9, end_ns=stop * 10**9, duration_s=stop - start
        )
    collector.spans["a"]["metadata"]["routes"] = [("work", True), ("work", True)]
    collector.spans["tool"]["metadata"]["state_lifecycle"] = {"mode": mode}
    collector.observe_state_accesses(
        "tool",
        [
            {"object_id": "x", "operation": "read", "bytes": 2000},
            {"object_id": "x", "operation": "read", "bytes": 2000},
            {"object_id": "x", "operation": "write", "bytes": 4000},
        ],
    )
    collector.spans["tool"]["resources"] = {"cpu_seconds": 0.5, "cpu_scope": "tool_invocation",
                                             "cpu_complete": True}
    result = collector.snapshot()
    result.update(arrival_ns=arrival, completed_ns=end)
    return result


class WorkloadTests(unittest.TestCase):
    def test_platform_independent_repeated_accesses_and_structure(self):
        p = extract_profile(trace())
        self.assertEqual(p["metrics"]["state_bytes"], 8000)
        self.assertEqual(p["metrics"]["rho_kb_s"], 4)
        self.assertEqual(p["metrics"]["state_reads"], 2)
        self.assertEqual(p["metrics"]["tool_cpu_s"], 0.5)
        self.assertEqual(p["workflow"]["parallel_peak"], 2)
        self.assertEqual(p["workflow"]["causal_revisits"], {"plan": 1})
        self.assertEqual(
            p["metrics"], extract_profile(trace(mode="restore-persist"))["metrics"]
        )

    def test_unknown_zero_idle_failed_and_incomplete(self):
        t = trace()
        tool = next(s for s in t["spans"] if s["kind"] == "tool")
        del tool["metadata"]["state_accesses"]
        tool.update(state_read_bytes=0, state_write_bytes=0)
        self.assertIsNone(extract_profile(t)["metrics"]["rho_kb_s"])
        tool["metadata"]["state_accesses"] = []
        self.assertEqual(extract_profile(t)["metrics"]["rho_kb_s"], 0)
        next(s for s in t["spans"] if s["kind"] == "llm")["duration_s"] = 0
        self.assertIsNone(extract_profile(t)["metrics"]["rho_kb_s"])
        t["request_status"] = "failed"
        model = WorkloadModel()
        self.assertEqual(model.observe(t, 11)["status"], "failed")
        self.assertEqual(model.predict(12)["sample_count"], 0)
        t["spans"][0]["end_ns"] = None
        self.assertEqual(extract_profile(t)["status"], "incomplete")

    def test_strict_arrival_cutoff_and_delayed_ingestion(self):
        model = WorkloadModel()
        model.observe(trace(), 20)
        self.assertEqual(model.predict(20)["sample_count"], 0)
        self.assertEqual(model.predict(21)["sample_count"], 1)
        prediction = model.begin("next", "s", 15)
        model.observe(trace("future", 16, 30), 30)
        self.assertEqual(prediction["sample_count"], 0)
        self.assertEqual(model.predict(25)["sample_count"], 1)
        self.assertEqual(model.begin("concurrent", "s", 25)["concurrent_sessions"], 1)
        with self.assertRaises(ValueError):
            model.observe(trace("early", 1, 30), 29)

    def test_distributions_persistence_dedup_and_resume(self):
        model = WorkloadModel()
        t = trace()
        partial = copy.deepcopy(t)
        partial["spans"][0]["end_ns"] = None
        model.observe(partial, 10)
        model.observe(t, 11)
        model.observe(t, 12)
        pred = model.predict(13)
        self.assertEqual(pred["metrics"]["rho_kb_s"]["samples"], [4])
        self.assertEqual(pred["workflow"]["branches"]["plan"][1]["probability"], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            self.assertEqual(WorkloadModel.load(path).predict(13), pred)
        t["spans"][-1]["metadata"]["state_accesses"][0]["bytes"] += 1
        with self.assertRaises(ValueError):
            model.observe(t, 14)

    def test_concurrent_evaluation_has_no_future_atg(self):
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate(
                [trace("a", 1, 10), trace("b", 2, 20), trace("c", 11, 30)],
                Path(directory),
            )
            self.assertEqual(
                [r["prediction"]["sample_count"] for r in report["requests"]], [0, 0, 1]
            )
            self.assertEqual(
                report["requests"][1]["prediction"]["concurrent_sessions"], 2
            )
            self.assertEqual(report["requests"][2]["absolute_error"]["rho_kb_s"], 0)
            self.assertTrue((Path(directory) / "distributions.png").is_file())

    def test_archived_datasci_capture_is_conservative(self):
        root = Path("benchmarks/DataSciBench/trace/captured")
        directory = next(root.glob("human_7-*"))
        p = extract_profile(load_capture(directory))
        self.assertEqual(p["status"], "success")
        self.assertGreater(p["metrics"]["llm_idle_s"], 0)
        self.assertIsNone(p["metrics"]["rho_kb_s"])
        self.assertEqual(p["source"], "capture")

    def test_registration_dedup_and_invalid_inputs(self):
        model = WorkloadModel()
        model.begin("r", "r", 1)
        t = trace()
        del t["arrival_ns"]
        model.observe(t, 11)
        model.observe(t, 12)
        self.assertEqual(model.predict(13)["sample_count"], 1)
        with self.assertRaises(ValueError):
            model.begin("bad", "bad", float("nan"))
        with self.assertRaises(ValueError):
            model.predict(-1)
        invalid = extract_profile(trace("bad"))
        invalid["metrics"]["state_bytes"] = -1
        with self.assertRaises(ValueError):
            model.update(invalid, 12)

    def test_bad_access_and_cycle(self):
        c = TraceCollector("r", "s")
        c.start("tool", "tool", "tool")
        with self.assertRaises(ValueError):
            c.observe_state_accesses(
                "tool", [{"object_id": "x", "operation": "read", "bytes": True}]
            )
        t = trace()
        next(s for s in t["spans"] if s["id"] == "a")["dependencies"] = ["d"]
        with self.assertRaises(ValueError):
            extract_profile(t)


if __name__ == "__main__":
    unittest.main()
