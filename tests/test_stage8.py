"""Stage 8 provenance and acceptance gates; mocks never count as workload evidence."""

import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp1.data.prepare_mix import audit
from experiments.exp1.runner.mix import input_records
from experiments.exp1.runner.mix import verify_capture


class MixProvenanceTests(unittest.TestCase):
    def fixture(self):
        prompt = "Return the sum."
        catalog = {"source": {"commit_sha": "a" * 40}, "tasks": [{
            "task_id": "bcb1", "task_type": "bcb", "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}]}
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=["Task ID", "Prompt", "Data", "Test", "Ground Truth Function"])
        writer.writeheader()
        writer.writerow({"Task ID": "BigCodeBench/1", "Prompt": prompt, "Data": "args = ([1, 2],)",
                         "Test": "import unittest", "Ground Truth Function": "def task_func(x): return sum(x)"})
        return catalog, stream.getvalue().encode()

    def test_reference_import_is_never_agent_acceptance(self):
        catalog, raw = self.fixture()
        bundle, review = audit(catalog, raw)
        self.assertEqual(bundle["cases"]["bcb1"]["data"], "args = ([1, 2],)")
        self.assertEqual(review["agent_verified_count"], 0)
        self.assertFalse(review["formal_experiment_ready"])
        self.assertEqual(review["tasks"][0]["status"], "pending_sandbox_preflight")

    def test_original_prompt_and_evaluation_must_match(self):
        catalog, raw = self.fixture()
        with self.assertRaises(ValueError):
            audit(catalog, raw.replace(b"Return the sum.", b"Return the product."))
        catalog["tasks"][0]["prompt"] += "changed"
        with self.assertRaises(ValueError):
            audit(catalog, raw)

    def test_prepared_inputs_reject_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.csv").write_bytes(b"x\n1\n")
            records = input_records(root)
            self.assertEqual(records[0]["bytes"], 4)
            (root / "escape").symlink_to("/etc/passwd")
            with self.assertRaises(ValueError):
                input_records(root)

    def test_live_quota_scan_tolerates_deleted_fixture_but_rejects_links(self):
        from unittest.mock import patch
        from benchmarks.DataSciBench.nodes.AgentTools.guards import safe_files

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            present = root / "present"
            present.write_text("x")
            with patch.object(Path, "rglob", return_value=iter([root / "already_deleted", present])):
                self.assertEqual(safe_files(root), [present])
            (root / "link").symlink_to("/etc/passwd")
            with self.assertRaises(ValueError):
                safe_files(root)

    def test_capture_rejects_changed_prompt_and_replay(self):
        task = {"task_id": "bcb1", "prompt": "Original", "prompt_sha256": hashlib.sha256(b"Original").hexdigest(),
                "readiness": {"inputs": {"files": []}}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = {"status": "complete", "task_id": "bcb1", "user_prompt": "Original", "dataset_refs": []}
            (root / "result.json").write_text(json.dumps(state))
            (root / "trace.json").write_text("{}")
            (root / "events.jsonl").write_text(json.dumps({"kind": "run_start", "mode": "replay"}) + "\n")
            with self.assertRaisesRegex(ValueError, "Only live"):
                verify_capture(task, root)
            state["user_prompt"] = "Changed"
            (root / "result.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "original prompt"):
                verify_capture(task, root)


class FunctionAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_function_source_is_preserved_and_tools_own_invocation(self):
        from benchmarks.DataSciBench.langgraph_agent.graph import build_graph
        from benchmarks.DataSciBench.langgraph_agent.trace_recorder import TraceRecorder

        source = "def task_func(x):\n    return ('quoted \\\" value', sum(x))\n"
        calls = []

        class Model:
            async def decide(self, phase, iteration, messages):
                return {"code": source} if phase == "PLAN" else {"verdict": "complete"}

        class Tools:
            async def call(self, name, args, key, wave=None):
                calls.append((name, args))
                if name == "execute_analysis":
                    return {"status": "ok", "step_id": args["step_id"], "branch_ref": key, "artifacts": []}
                return {"status": "ok", "snapshot": 1, "artifacts": []}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorder = TraceRecorder(root / "trace", "r", "s")
            outputs = ["solution.py", "bcb_execution.json", "bcb_result.pkl"]
            graph = build_graph(Model(), Tools(), recorder, outputs, function_task=True)
            result = await graph.ainvoke({"request_id": "r", "session_id": "s", "task_id": "bcb1",
                "user_prompt": "Sum a list", "messages": [], "dataset_refs": [], "iteration": 0,
                "completed": [], "tool_results": {}, "artifacts": [], "snapshot": 0, "workspace_id": ""})
            self.assertEqual(result["status"], "complete")
            analysis = [args for name, args in calls if name == "execute_analysis"]
            self.assertEqual(len(analysis), 1)
            self.assertEqual(analysis[0]["outputs"], outputs)
            exec(analysis[0]["code"], {"output_dir": root})
            self.assertEqual((root / "solution.py").read_text(), source)


class MixArrivalTests(unittest.TestCase):
    def test_reproducible_original_timing_and_prompt_novelty(self):
        from experiments.exp1.data.prepare_mix_arrivals import assign
        from experiments.exp1.runner.arrival_scheduler import Arrival

        tasks = [{"task_id": str(i), "prompt": prompt,
                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "readiness": {"inputs": {"files": []}}}
                 for i, prompt in enumerate(["Calculate mean and variance", "Train a neural classifier"])]
        arrivals = [Arrival(str(i), timestamp, "2026-06-01T00:00:00Z", "part", "copilot", "window")
                    for i, timestamp in enumerate(["0.000001", "9.123456", "20", "25"])]
        windows = [(arrivals, 30), (arrivals, 30)]
        rows = assign(tasks, windows, 8, "stable")
        self.assertEqual(rows, assign(tasks, windows, 8, "stable"))
        self.assertEqual(rows[4]["relative_arrival_ms"], "30.000001")
        self.assertEqual(sum(r["first_prompt"] for r in rows), 2)
        self.assertEqual(len({r["arrival_id"] for r in rows}), 8)
        self.assertFalse(any(r["first_prompt"] for r in rows[4:]))

    def test_unverified_mix_cannot_generate_formal_arrivals(self):
        from experiments.exp1.data.prepare_mix_arrivals import prepare
        from experiments.exp1.data.prepare_mix import sha

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(json.dumps({"tasks": [], "formal_experiment_ready": False}))
            (root / "seal.json").write_text(json.dumps({"manifest_sha256": sha((root / "manifest.json").read_bytes())}))
            with self.assertRaisesRegex(ValueError, "100 distinct"):
                prepare(root, root / "missing-traffic", root / "output", roles=["typical"], seed=8, composition="stable")


class ColdStartTests(unittest.TestCase):
    def router(self, *, default=True, budget=10, epsilon=1):
        from atsuite.provider_model import ProviderModel
        from atsuite.routing import OnlineRouter
        from atsuite.workload import WorkloadModel
        from tests.workload_fixtures import describe
        from tests.stage6_fixtures import bounds
        from tests.test_provider_model import pool

        pools = {name: {**pool(), "environment": name} for name in ["default", "new"]}
        router = OnlineRouter(WorkloadModel(), ProviderModel(), pools, dict.fromkeys(pools, 2),
                              default_pool="default" if default else None, enable_context_routing=True,
                              execution_bounds={name: bounds() for name in pools}, budget=budget,
                              exploration={"epsilon": epsilon, "budget": budget, "seed": 8})
        return router, describe(prompt="Calculate mean and variance", tools=["python"])

    def test_empty_history_explores_without_inventing_predictions(self):
        router, descriptor = self.router()
        decision = router.decide("first", descriptor=descriptor, as_of_ns=1000)
        self.assertEqual(decision["selected"], "new")
        self.assertEqual(decision["exploration"]["reason"], "bounded_cold_start")
        self.assertIsNone(decision["exploration"]["predicted_objective_gain"])
        for candidate in decision["candidates"]:
            self.assertIsNone(candidate["objective_cost"])
            self.assertIsNone(candidate["objective_latency"])
            self.assertFalse(candidate["flags"])
            self.assertIn("cost_prediction_missing", candidate["prediction_flags"])
        self.assertEqual(sum(decision["exploration"]["probabilities"].values()), 1)

    def test_no_default_can_admit_a_bounded_probe(self):
        router, descriptor = self.router(default=False)
        decision = router.decide("first", descriptor=descriptor, as_of_ns=1000)
        self.assertIsNotNone(decision["selected"])
        self.assertTrue(decision["exploration"]["explored"])
        self.assertEqual(sum(decision["exploration"]["probabilities"].values()), 1)

    def test_cold_start_preserves_capacity_budget_environment_and_disable(self):
        for constraint in ("capacity", "budget", "tools", "disabled"):
            router, descriptor = self.router(budget=0.5 if constraint == "budget" else 10,
                                              epsilon=0 if constraint == "disabled" else 1)
            if constraint == "capacity":
                router.limits["new"] = 0
            if constraint == "tools":
                router.execution_bounds["new"]["tools"] = []
            decision = router.decide(constraint, descriptor=descriptor, as_of_ns=1000)
            self.assertFalse(decision["exploration"]["explored"])
            self.assertNotEqual(decision["selected"], "new")

    def test_only_executed_provider_gets_causal_feedback(self):
        from tests.test_workload import trace

        router, descriptor = self.router()
        router.decide("first", descriptor=descriptor, as_of_ns=1000)
        router.complete("first", trace("first", arrival=1000, end=2000), available_ns=2001,
                        actual_cost=0.1, cost_basis="synthetic_usage_estimate")
        self.assertEqual(router.exploration.evidence(router.provider, router.pools["new"], descriptor, 2001)["attempts"], 0)
        self.assertEqual(router.exploration.evidence(router.provider, router.pools["new"], descriptor, 2002)["attempts"], 1)
        self.assertEqual(router.exploration.evidence(router.provider, router.pools["default"], descriptor, 2002)["attempts"], 0)
        self.assertEqual(router.exploration_spent, 0.1)

    def test_joint_controller_can_lease_without_a_priced_plan(self):
        import copy
        from atsuite.routing import JointController, candidate_plans
        from atsuite.workload.descriptor import family_key
        from tests.test_joint_control import DelayedAdapter, FX

        router, descriptor = self.router()
        levels, adapters = {}, {}
        for key, pool in router.pools.items():
            levels[key] = {}
            for target in (0, 1):
                config = copy.deepcopy(pool)
                config["instances"] = None
                config["resources"]["capacity_control"] = {"control": "reserved_concurrency", "target": target}
                levels[key][target] = {"target": target, "pool": config, "session_limit": target * 2,
                                       "scale_delay_s": 0.1, "evidence": ["synthetic readback"]}
            initial = int(key == "default")
            adapters[key] = DelayedAdapter(initial)
            router.set_capacity(key, levels[key][initial]["pool"], initial * 2)
        weights = {"default": 1, "new": 0}
        control = JointController(router, adapters, levels, candidate_plans(levels, [weights]), FX,
                                  period_s=1, cooldown_s=0,
                                  capacity_exploration={"lease_s": 5, "max_cost": {"new": 0.25},
                                                        "evidence": ["synthetic lifecycle bound"]})
        control.descriptors[family_key(descriptor)] = descriptor
        control._refresh(1000)
        proposed = control._probe_plan(None, {family_key(descriptor): 0.1}, 1000, apply=True)
        self.assertEqual(proposed["targets"]["new"], 1)
        self.assertEqual(router.capacity_reserved, {"new": 0.25})
        self.assertEqual(adapters["new"].calls, [])
        self.assertEqual(router.active, {})


class OnlineReportTests(unittest.TestCase):
    def test_caps_are_never_reported_as_actual_cost(self):
        from experiments.exp1.runner.stage8_online import summarize

        decision = {"event": "decision", "request_id": "r", "shadow": False, "selected": "pool",
                    "reason": "provider_exploration", "exploration": {"explored": True, "family": "f"},
                    "candidates": [{"key": "pool", "objective_latency": None}]}
        completion = {"event": "completion", "request_id": "r", "selected": "pool",
                      "accounting": {"available_ns": 10, "shadow": False, "latency_s": 2,
                                     "basis": "execution_cap", "amount": 1, "actual_cost": None, "status": "success"}}
        report = summarize([decision, completion], window=1)
        self.assertIsNone(report["total_infrastructure_actual_cost"])
        self.assertIsNone(report["windows"][0]["mean_actual_execution_cost"])
        self.assertEqual(report["windows"][0]["prediction_coverage"], 0)
        self.assertEqual(report["windows"][0]["exploration_rate"], 1)
        with self.assertRaises(ValueError):
            summarize([decision, completion, completion])

    def test_best_fixed_requires_all_four_complete_matched_baselines(self):
        from experiments.exp1.runner.stage7 import compare

        strategies = ["fixed_aws_lambda", "fixed_aws_agentcore", "fixed_ali_fc", "fixed_ali_agentrun", "cost_aware"]
        reports = [{"strategy": name, "workload_sha256": "w", "comparison_sha256": "c", "deployment_sha256": "d",
                    "cost": {"complete": True, "basis": "actual", "currency": "USD", "total": 10 + i},
                    "exploration_attempts": 0, "selection_proportions": {}}
                   for i, name in enumerate(strategies)]
        self.assertEqual(compare(reports)["best_fixed_provider"], "fixed_aws_lambda")
        self.assertIsNone(compare(reports[1:])["best_fixed_provider"])
        reports[2]["cost"]["complete"] = False
        self.assertIsNone(compare(reports)["best_fixed_provider"])

    def test_replay_failure_comparison_ignores_only_stack_and_time(self):
        from benchmarks.DataSciBench.langgraph_agent.tool_client import bcb_error

        prefix = "Independent BCB tests failed:\ntest_one ... FAIL\nTraceback (most recent call last):\n"
        a = prefix + '  File "worker.py", line 2\n    assert x\nAssertionError: 1 != 2\nRan 1 test in 0.1s\nFAILED (failures=1)'
        b = prefix + '  File "worker.py", line 8\nAssertionError: 1 != 2\nRan 1 test in 0.7s\nFAILED (failures=1)'
        self.assertEqual(bcb_error(a), bcb_error(b))
        self.assertNotEqual(bcb_error(a), bcb_error(b.replace("1 != 2", "2 != 3")))


if __name__ == "__main__":
    unittest.main()
