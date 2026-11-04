"""Independent arrivals, empty initial models, and honest paired cost evidence."""

import asyncio
import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from atsuite.workload import describe
from experiments.e3.independent import (
    ADDITIONS,
    allocate,
    capture_pending,
    cost_balance,
    seal_requests,
)
from experiments.e3.independent_inputs import derive
from experiments.e3.manifest import WORKLOADS, seal
from experiments.e3.run import PROVIDERS, build_router


class IndependentRequestsTests(unittest.TestCase):
    def test_capture_model_forwards_reasoning_effort_and_output_limit(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from benchmarks.DataSciBench.langgraph_agent.model_client import DeepSeekClient

        async def check():
            config = {"seed": 9, "temperature": 0, "top_p": 1, "max_tokens": 32768,
                      "thinking": "enabled", "reasoning_effort": "low"}
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":{}}'), finish_reason="stop")],
                id="unit-test-response", model="deepseek-flash", model_dump=lambda **_: {"usage": {}},
            )
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test-only"}):
                model = DeepSeekClient(MagicMock(), config)
            model.client.chat.completions.create = AsyncMock(return_value=response)
            try:
                await model.decide("PLAN", 0, [])
                request = model.client.chat.completions.create.call_args.kwargs
                self.assertEqual(request["reasoning_effort"], "low")
                self.assertEqual(request["max_tokens"], 32768)
                self.assertEqual(request["extra_body"], {"thinking": {"type": "enabled"}})
            finally:
                await model.client.close()

        asyncio.run(check())

    def test_invalid_agent_instruction_is_rejected_before_any_model_or_tool_initialization(self):
        from experiments.exp1.capture.capture_agent_traces import run

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "tasks.json"
            manifest.write_text('{"tasks":[{"task_id":"one"}]}')
            for field in ("planner_instruction", "validation_contract"):
                with self.assertRaisesRegex(ValueError, "nonempty strings"):
                    asyncio.run(run("one", {field: {}}, root / "unused", manifest=manifest, workload="TravelPlanner"))
            self.assertFalse((root / "unused").exists())

    def test_preflight_initialization_writes_empty_models_without_reading_profiles_or_costs(self):
        from atsuite.provider_model import ProviderModel
        from atsuite.workload import WorkloadModel
        from experiments.e3.profile import initialize

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks = []
            for workload in WORKLOADS:
                path = root / (workload + ".json")
                path.write_text(json.dumps({"tasks": [{"task_id": workload, "tool_definitions": [{"name": "query"}]}]}))
                tasks.append({"id": workload, "task_id": workload, "workload": workload, "manifest": str(path)})
            attempts = [{"task": w, "pool": p, "success": True} for w in WORKLOADS for p in PROVIDERS]
            config = {
                "native_elasticity": {
                    p: {kind: {"cpu": 2, "memory_mb": 4096, "disk_mb": 10240} for kind in ("e3", "datasci")}
                    for p in PROVIDERS
                }
            }
            original_read = Path.read_text

            def read(path, *args, **kwargs):
                if path.name == "fx.json":
                    return '{"date":"fixture", "source":"fixture", "usd_to_cny":7}'
                return original_read(path, *args, **kwargs)

            with (
                patch.object(Path, "read_text", read),
                patch.object(ProviderModel, "update", side_effect=AssertionError("preflight learned")),
                patch.object(WorkloadModel, "update", side_effect=AssertionError("preflight learned")),
            ):
                result = initialize(
                    config,
                    {"tasks": tasks, "requests": [{}] * 213, "duration_ms": 600000},
                    attempts,
                    [{"private_preflight_cost": 9999}],
                    root / "initialized",
                )
            self.assertFalse(WorkloadModel.load(Path(result["workload_model"])).samples)
            self.assertFalse(ProviderModel.load(Path(result["provider_model"])).samples)

    def test_replay_uses_each_requests_actual_capture_seed(self):
        from experiments.e3.fixed_baselines import measure

        calls = []

        async def execute(*args, **kwargs):
            calls.append(args)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            agent = root / "agent.json"
            agent.write_text('{"model":{"seed":37},"planner_instruction":"Use documented room types."}')
            task = {
                "id": "request",
                "task_id": "family",
                "workload": "DataSciBench",
                "manifest": "manifest",
                "capture": "capture",
                "agent_config": str(agent),
            }
            config = {
                "agents": {"DataSciBench": {"model": {"seed": 9}}},
                "targets": {"DataSciBench": {"ali_fc": {"tool": {"provider": "ali_fc"}}}},
            }
            with patch("experiments.exp1.capture.capture_agent_traces.run", side_effect=execute):
                row = asyncio.run(measure(task, config, "ali_fc", root))
            self.assertTrue(row["success"])
            self.assertEqual(calls[0][1]["model"]["seed"], 37)
            self.assertEqual(calls[0][1]["planner_instruction"], "Use documented room types.")

    def test_native_cost_feedback_uses_only_completed_selected_requests_and_accepts_late_revisions(self):
        from types import SimpleNamespace

        from atsuite.provider_model import ProviderModel
        from experiments.e3.feedback import NativeCostFeedback
        from tests.test_provider_model import observation

        provider = ProviderModel()
        provider.update(observation(0))
        router = SimpleNamespace(provider=provider, completed=set(), decisions={"0": {"selected": "aws_lambda"}})
        row = {
            "request_id": "0",
            "pool": "aws_lambda",
            "priced_cost": {"total": 1, "currency": "USD", "lines": [{"category": "memory", "amount": 1}]},
        }
        with tempfile.TemporaryDirectory() as temp:
            feedback = NativeCostFeedback(router, {"native_elasticity": {}}, Path(temp))
            feedback.ingest([row], "native fixture")
            self.assertFalse(feedback.imported)
            router.completed.add("0")
            feedback.ingest([row], "native fixture")
            history = next(iter(provider.samples.values()))
            self.assertEqual(len(history), 2)
            self.assertGreater(history[-1]["available_ns"], history[-1]["completed_ns"])
            self.assertEqual(history[-1]["cost"]["components"], {"memory": 1})
            feedback.ingest([row], "same native fixture")
            self.assertEqual(len(history), 2)
            row["priced_cost"]["total"] = 2
            row["priced_cost"]["lines"][0]["amount"] = 2
            feedback.ingest([row], "late native correction")
            self.assertEqual(len(history), 3)
            row["pool"] = "ali_fc"
            row["priced_cost"]["total"] = 3
            with self.assertRaisesRegex(ValueError, "selected provider"):
                feedback.ingest([row], "wrong provider")

    def test_datasci_program_export_merges_independent_captures_without_relaxing_admission(self):
        from experiments.e3.independent import approve_datasci

        policy = {}

        def events(code):
            return [
                {"kind": "tool", "tool_name": "prepare_workspace", "args": {"task_id": "human_15", "dataset_refs": []}},
                {
                    "kind": "tool",
                    "tool_name": "execute_analysis",
                    "args": {"step_id": "plot", "outputs": ["plot.png"], "snapshot": 0, "seed": 9, "code": code},
                },
            ]

        approve_datasci(policy, {"task_id": "human_15"}, events("plot_a()"))
        approve_datasci(policy, {"task_id": "human_15"}, events("plot_b()"))
        self.assertEqual(len(policy["human_15"]["analyses"]), 2)
        self.assertNotEqual(
            policy["human_15"]["analyses"][0]["code_sha256"], policy["human_15"]["analyses"][1]["code_sha256"]
        )
        changed = events("plot_c()")
        changed[0]["args"]["dataset_refs"] = [{"name": "different.csv"}]
        with self.assertRaisesRegex(ValueError, "dataset versions"):
            approve_datasci(policy, {"task_id": "human_15"}, changed)

    def allocation(self):
        tasks, pairs = [], []
        for workload in WORKLOADS:
            for i in range(6):
                identity = workload + str(i)
                tasks.append({"id": identity, "task_id": identity, "workload": workload})
                pairs.append(
                    {
                        "task": identity,
                        "task_id": identity,
                        "workload": workload,
                        "fc_cost_cny": 1.1 if i == 0 else 0.9 if i == 1 else 1,
                        "mcp_cost_cny": 1,
                    }
                )
        return {"tasks": tasks, "requests": [{}] * 213}, pairs

    def test_24_retained_plus_189_evenly_distributed_before_capture(self):
        source, pairs = self.allocation()
        counts, balance = allocate(source, pairs)
        totals = Counter()
        for task in source["tasks"]:
            self.assertGreaterEqual(counts[task["id"]], 1)
            totals[task["workload"]] += counts[task["id"]] - 1
        self.assertEqual(totals, ADDITIONS)
        self.assertEqual(sum(counts.values()), 213)
        self.assertEqual(balance["aws_balance"], "unmeasured")
        self.assertIn("unmeasured", balance["basis"])
        with self.assertRaisesRegex(ValueError, "identity"):
            pairs[0]["task_id"] = "wrong"
            allocate(source, pairs)

    def test_cold_start_ignores_all_snapshot_paths_and_can_explore_without_samples(self):
        from tests.test_stage7 import router_fixture

        old, descriptor = router_fixture()
        template = copy.deepcopy(next(iter(old.pools.values())))
        public_billing = template.pop("billing")
        config = {
            "pools": {p: {**copy.deepcopy(template), "environment": p, "provider": p} for p in PROVIDERS},
            "limits": dict.fromkeys(PROVIDERS, 64),
            "workload_model": "/must/not/load.json",
            "provider_model": "/must/not/load-either.json",
            "routing": {
                "default_pool": "ali_fc",
                "latency_weight": 0,
                "execution_bounds": {
                    p: {"timeout_s": 900, "currency": "CNY", "tools": ["python"], "evidence": "native configuration"}
                    for p in PROVIDERS
                },
                "exploration": {"epsilon": 1, "seed": 9, "epoch_ns": 123},
            },
            "native_elasticity": {
                "aws_agentcore": {"e3": {"lifecycle_configuration": {"idleRuntimeSessionTimeout": 300}}},
                "ali_agentrun": {"e3": {"session_affinity_config": '{"sessionIdleTimeoutInSeconds":300}'}},
            },
        }
        config["pools"]["aws_lambda"]["billing"] = public_billing
        public_billing["calibration"] = {"evidence": "private preflight", "tool_scale": 999, "reference_id": "secret"}
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("atsuite.workload.WorkloadModel.load", side_effect=AssertionError("old model accessed")),
        ):
            router = build_router(config, "exploration", Path(temp) / "routing.jsonl")
            self.assertFalse(router.workload.samples)
            self.assertFalse(router.workload.initial_profiles)
            self.assertFalse(router.provider.samples)
            self.assertNotIn("calibration", router.pools["aws_lambda"]["billing"])
            self.assertEqual(public_billing["calibration"]["tool_scale"], 999)
            self.assertEqual(router.exploration.config["epoch_ns"], 0)
            decision = router.decide("new", descriptor=descriptor)
            self.assertIn(decision["selected"], PROVIDERS)
            self.assertTrue(decision["exploration"]["explored"])
            self.assertEqual(decision["exploration"]["reason"], "bounded_cold_start")

    def test_independent_assignment_rejects_shared_capture_and_repeated_task(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks, arrivals, assignments = [], [], []
            for i, workload in enumerate(WORKLOADS):
                folder = root / workload
                folder.mkdir()
                (folder / "result.json").write_text('{"status":"complete"}')
                events = [
                    {
                        "kind": "run_start",
                        "task_id": workload,
                        "prompt_sha256": hashlib.sha256(workload.encode()).hexdigest(),
                    },
                    {"kind": "llm", "source": "live_official_api"},
                ]
                (folder / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
                manifest = root / (workload + ".json")
                manifest.write_text(json.dumps({"tasks": [{"task_id": workload, "prompt": workload}]}))
                tasks.append(
                    {
                        "id": workload,
                        "task_id": workload,
                        "workload": workload,
                        "manifest": str(manifest),
                        "capture": str(folder),
                        "inputs": [],
                    }
                )
                arrivals.append({"arrival_id": str(i), "relative_arrival_ms": str(i)})
                assignments.append({"arrival_id": str(i), "task": workload, "workload": workload})
            phases = [{"start_ms": 0, "weights": dict.fromkeys(WORKLOADS, 1)}]
            result = seal(tasks, arrivals, phases, 9, 10, root / "sealed", assignments=assignments)
            self.assertEqual([r["task"] for r in result["requests"]], list(WORKLOADS))
            tasks[1]["capture"] = tasks[0]["capture"]
            with self.assertRaisesRegex(ValueError, "share a Capture"):
                seal(tasks, arrivals, phases, 9, 10, root / "invalid", assignments=assignments)

    def test_each_repeated_family_is_captured_independently_and_resume_is_read_only(self):
        calls = []

        async def live(workload, manifest, ids, config, output, concurrency):
            calls.append((str(manifest), str(output)))
            output.mkdir(parents=True)
            (output / "receipts.json").write_text(
                json.dumps([{"capture": str(output / "attempt-0"), "attempts": [{"success": True}]}])
            )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            agent = root / "agent.json"
            agent.write_text('{"model":{"seed":9}}')
            tasks = [
                {
                    "retained": False,
                    "id": str(i),
                    "task_id": "same-family",
                    "workload": "DataSciBench",
                    "manifest": str(root / f"task-{i}.json"),
                    "agent_config": str(agent),
                }
                for i in range(2)
            ]
            config = {"DataSciBench": {"tool": {"provider": "local"}, "model": {"name": "deepseek-flash"}}}
            with (
                patch("experiments.e3.independent.prepared", return_value={"tasks": tasks}),
                patch("experiments.e3.independent.capture", side_effect=live),
            ):
                result = asyncio.run(capture_pending(root, config))
                asyncio.run(capture_pending(root, config))
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0], calls[1])
            self.assertEqual(result["new_successes"], 2)

    def test_duplicate_live_capture_identity_is_rejected_even_in_different_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = []
            for i in range(2):
                folder = root / str(i)
                folder.mkdir()
                (folder / "events.jsonl").write_text(
                    json.dumps({"kind": "run_start", "request_id": "copied", "config": {"model": {"seed": 9}}})
                )
                entries.append({"id": str(i), "retained": True, "capture": str(folder)})
            with (
                patch("experiments.e3.independent.prepared", return_value={"tasks": entries}),
                self.assertRaisesRegex(ValueError, "identity reused"),
            ):
                seal_requests(root, root / "sealed")

    def test_new_scientific_input_has_its_own_reference_and_arrival_metadata(self):
        import numpy as np

        original = {
            "task_id": "small",
            "readiness": {"inputs": {"files": []}},
            "verification": {"matrix": [[2, 0], [0, 2]], "rhs": [[1], [1]]},
            "tool_definitions": [{"name": "scicom_singular_value_decomposition"}],
        }
        with tempfile.TemporaryDirectory() as temp:
            task, assets = derive(
                {"workload": "ScientificComputation", "task_id": "small"}, {}, original, 1, Path(temp)
            )
            with np.load(task["verification"]["input_npz"]) as inputs:
                solution = np.linalg.solve(inputs["A"], inputs["B"])
                self.assertTrue(np.allclose(inputs["A"] @ solution, inputs["B"]))
            descriptor = describe(prompt=task["prompt"], files=task["readiness"]["inputs"]["files"])
            self.assertEqual(descriptor["hints"], {"rows": 3, "columns": 3, "features": 2})
            self.assertEqual(assets, [task["verification"]["input_npz"]])
            self.assertEqual(original["task_id"], "small")
            self.assertIn("retain the factorization or inverse", task["prompt"])

    def test_cost_balance_requires_complete_native_pairs_and_preserves_currency(self):
        plan = {
            "requests": [{"arrival_id": "a", "task": "independent", "workload": "TravelPlanner"}],
            "balance_tolerance": "0.05",
        }
        rows = [
            {
                "pool": p,
                "request_id": "a",
                "task": "independent",
                "workload": "TravelPlanner",
                "success": True,
                "priced_cost": {"total": cost, "currency": "CNY"},
            }
            for p, cost in (("ali_fc", 1), ("ali_agentrun", 1.04))
        ]
        report = cost_balance(plan, rows)
        self.assertTrue(report["ali"]["within_tolerance"])
        self.assertIsNone(report["aws"]["within_tolerance"])
        rows[1]["priced_cost"]["total"] = None
        self.assertIsNone(cost_balance(plan, rows)["ali"]["within_tolerance"])
        rows[1]["priced_cost"] = {"total": 1, "currency": "USD"}
        with self.assertRaisesRegex(ValueError, "currencies"):
            cost_balance(plan, rows)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            cost_balance(plan, rows[:1] * 2)
