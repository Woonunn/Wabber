"""Policy decisions precede price lookup; feedback cannot arrive from the future."""

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from experiments.e3.replay_routing import load_inputs, permute_arrivals, replay


class LocalRoutingReplayTests(unittest.TestCase):
    def test_order_validation_preserves_arrival_slots_and_logical_request_identity(self):
        plan = {"duration_ms": 100, "requests": [
            {"arrival_id": str(i), "task": "task-" + str(i), "relative_arrival_ms": i * 10}
            for i in range(10)
        ]}
        changed = permute_arrivals(plan, 21)
        self.assertEqual(changed, permute_arrivals(plan, 21))
        self.assertNotEqual(changed["requests"], plan["requests"])
        self.assertEqual([r["relative_arrival_ms"] for r in changed["requests"]],
                         [r["relative_arrival_ms"] for r in plan["requests"]])
        self.assertEqual({r["arrival_id"]: r["task"] for r in changed["requests"]},
                         {r["arrival_id"]: r["task"] for r in plan["requests"]})
    def test_selected_costs_cumulative_order_and_only_completed_feedback(self):
        completed, updates, loads, visible = set(), [], [], []

        def decide(rid, task, load, **kwargs):
            visible.append((rid, set(completed)))
            selected = "ali_agentrun" if completed else "ali_fc"
            return {"selected": selected, "reason": "fixture", "prediction_overhead": {
                "cost": 0.01, "total_latency_s": 0, "cost_complete": True},
                "exploration": {"explored": False}, "workload": {"sample_count": len(completed),
                "match_basis": "fixture", "retrieval": {}, "low_confidence": True,
                "missing_context": [], "joint_profiles": []}, "candidates": []}

        def complete(rid, trace, **kwargs):
            self.assertGreater(trace["completed_ns"], trace["arrival_ns"])
            completed.add(rid)
            return {"id": rid, "measurements": {"latency_s": 2}}

        def trace_loader(outcome):
            loads.append(outcome["directory"])
            return {"spans": [{"start_ns": 10, "end_ns": 20}]}

        router = SimpleNamespace(decide=decide, complete=complete,
                                 completed=completed, provider=SimpleNamespace(update=updates.append))
        requests = [{"arrival_id": str(i), "task": str(i), "workload": "fixture",
                     "relative_arrival_ms": ms} for i, ms in enumerate([0, 1000, 10000], 1)]
        outcomes = {(r["arrival_id"], p): {"latency_s": 2, "cost": Decimal(r["arrival_id"]) * multiplier,
                     "directory": p + r["arrival_id"], "success": True}
                    for r in requests for p, multiplier in [("ali_fc", 1), ("ali_agentrun", 100)]}
        with tempfile.TemporaryDirectory() as temporary:
            rows = replay({"requests": requests, "duration_ms": 20000}, dict.fromkeys(["1", "2", "3"], {}),
                          outcomes, router, trace_loader, Path(temporary) / "run")
        self.assertEqual(visible[:2], [("1", set()), ("2", set())])
        self.assertEqual(visible[2], ("3", {"1", "2"}))
        self.assertEqual(loads, ["ali_fc1", "ali_fc2", "ali_agentrun3"])
        self.assertEqual([r["cost_cny"] for r in rows], ["1", "2", "300"])
        self.assertEqual([r["cumulative_cost_cny"] for r in rows], ["1", "3", "303"])
        self.assertEqual(rows[-1]["cumulative_with_routing_cny"], "303.03")
        self.assertEqual(len(updates), 3)
        completed.clear(); loads.clear(); visible.clear(); updates.clear()
        with tempfile.TemporaryDirectory() as temporary:
            sequential = replay({"requests": requests, "duration_ms": 20000}, dict.fromkeys(["1", "2", "3"], {}),
                                outcomes, router, trace_loader, Path(temporary) / "run", feedback_mode="rounds")
        self.assertEqual(visible, [("1", set()), ("2", {"1"}), ("3", {"1", "2"})])
        self.assertEqual([r["cost_cny"] for r in sequential], ["1", "200", "300"])
        self.assertEqual(sequential[-1]["cumulative_cost_cny"], "501")

    def test_missing_price_is_rejected_before_context_or_model_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cloud, corpus, evidence = root / "cloud", root / "corpus", root / "evidence"
            cloud.mkdir(); corpus.mkdir(); (evidence / "priced").mkdir(parents=True)
            (cloud / "FINAL.json").write_text(json.dumps({"final_evidence": str(evidence)}))
            (cloud / "config.json").write_text("{}")
            (corpus / "capture-index.json").write_text(json.dumps({"captures": [{"id": "task"}],
                "requests": [{"arrival_id": "r", "task": "task", "workload": "fixture"}]}))
            for provider in ("ali_fc", "ali_agentrun"):
                directory = cloud / "formal-repaired" / provider
                directory.mkdir(parents=True)
                (directory / "requests.jsonl").write_text(json.dumps({"request_id": "r", "pool": provider,
                    "task": "task", "workload": "fixture", "success": True, "latency_s": 1, "directory": "unused"}))
            (evidence / "priced/request-usage.jsonl").write_text(json.dumps({"request_id": "r", "pool": "ali_fc",
                "priced_cost": {"currency": "CNY", "total": None}}))
            with self.assertRaisesRegex(ValueError, "native price"):
                load_inputs(cloud, corpus)


if __name__ == "__main__":
    unittest.main()
