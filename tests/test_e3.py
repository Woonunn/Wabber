import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.e3.agent import build_graph, result_view
from experiments.e3.manifest import WORKLOADS, load, seal
from experiments.e3.report import matrix


class E3Tests(unittest.TestCase):
    def test_state_writes_and_following_reads_keep_native_batch_order(self):
        state = []
        waves = []

        class Model:
            async def decide(self, phase, iteration, messages):
                if iteration == 0:
                    return {
                        "calls": [
                            {"tool": "write", "args": {}},
                            {"tool": "read", "args": {}},
                        ]
                    }
                self_outer.assertIn('"tool_results": [1, 1]', messages[-1]["content"])
                return {"answer": {}}

        class Tools:
            async def call(self, name, args, key, wave=None):
                waves.append(wave)
                if name == "write":
                    await asyncio.sleep(0)
                    state.append("stored")
                return len(state)

        async def validate(*args):
            return {"ok": True}

        self_outer = self
        graph = build_graph(
            Model(),
            Tools(),
            {
                "tool_definitions": [{"name": "write"}, {"name": "read"}],
                "tool_access": {"write": "rw", "read": "r"},
            },
            "stateful batch",
            validate,
        )
        self.assertEqual(
            asyncio.run(graph.ainvoke({"user_prompt": "inspect", "iteration": 0}))[
                "status"
            ],
            "complete",
        )
        self.assertEqual(waves, [None, None])

    def test_corpus_retains_failure_and_selects_first_valid_attempt(self):
        from unittest.mock import patch

        from experiments.e3.corpus import capture

        attempts = []
        environments = []
        tool = {"endpoint": "stale-docker-ip"}

        async def fresh(attempt):
            environments.append(attempt)
            tool["endpoint"] = f"current-docker-ip-{attempt}"

        async def execute(task, config, directory, **kwargs):
            self.assertEqual(config["tool"]["endpoint"], f"current-docker-ip-{len(attempts)}")
            attempts.append(config["model"]["seed"])
            if len(attempts) == 1:
                raise RuntimeError("independent check rejected first answer")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "tasks.json"
            manifest.write_text('{"tasks":[{"task_id":"real-task"}]}')
            with patch("experiments.e3.corpus.run", side_effect=execute):
                rows = asyncio.run(
                    capture(
                        "TravelPlanner",
                        manifest,
                        ["real-task"],
                        {"model": {"seed": 9}, "tool": tool},
                        root / "captures",
                        before_attempt=fresh,
                    )
                )
            self.assertEqual(attempts, [9, 10])
            self.assertEqual(environments, [0, 1])
            self.assertEqual([a["success"] for a in rows[0]["attempts"]], [False, True])
            self.assertTrue(rows[0]["capture"].endswith("attempt-1"))

    def test_original_arrival_slice_preserves_bursts_precision_and_boundary(self):
        from experiments.e3.sequence import arrivals

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "arrivals.csv"
            source.write_text(
                "arrival_id,relative_arrival_ms,original_timestamp_utc\na,99,before\nb,100,first\nc,100,burst\nd,100.000001,precise\ne,699.999999,last\nf,700,outside\n"
            )
            selected = arrivals(source, "100", "600")
            self.assertEqual([r["arrival_id"] for r in selected], ["b", "c", "d", "e"])
            self.assertEqual(
                [r["relative_arrival_ms"] for r in selected],
                ["0", "0", "0.000001", "599.999999"],
            )
            self.assertEqual(selected[2]["original_timestamp_utc"], "precise")

    def test_scientific_artifact_loader_and_compact_results_preserve_actual_state(self):
        from unittest.mock import patch

        from benchmarks.ScientificComputation.langgraph_agent.graph import validate
        from benchmarks.ScientificComputation.nodes.tensor_workspace.implementation import (
            scicom_load_tensor,
            scicom_matrix_inverse,
            scicom_multiply_matrices,
            scicom_scale_matrix,
            scicom_view_tensor,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            problem = {"A": [[2, 0], [0, 4]], "B": [[6], [8]]}
            (inputs / "problem.json").write_text(json.dumps(problem))
            (root / "outside.json").write_text(json.dumps(problem))
            (inputs / "escape.json").symlink_to(root / "outside.json")
            with patch.dict("os.environ", {"ATSUITE_SCIENTIFIC_PROBLEMS": str(inputs)}):
                for invalid in ["../outside", "escape"]:
                    self.assertTrue(
                        json.loads(scicom_load_tensor(invalid, "A", "A"))["isError"]
                    )
                for tensor in ["A", "B"]:
                    self.assertFalse(
                        json.loads(scicom_load_tensor("problem", tensor, tensor))[
                            "isError"
                        ]
                    )
                inverse = json.loads(
                    scicom_matrix_inverse("A", "inverse", summary_only=True)
                )
                self.assertEqual(inverse["result"]["shape"], [2, 2])
                scicom_multiply_matrices("inverse", "B", "solution", summary_only=True)
                self.assertEqual(
                    json.loads(scicom_view_tensor("solution"))["result"], [[3], [2]]
                )
                self.assertTrue(
                    json.loads(scicom_matrix_inverse("A", summary_only=True))["isError"]
                )

            class Tools:
                async def call(self, name, args, key):
                    return scicom_view_tensor(**args)

            task = {
                "report_values": False,
                "verification": {"matrix": problem["A"], "rhs": problem["B"]},
            }
            self.assertTrue(
                asyncio.run(validate(task, {"solution_tensor": "solution"}, Tools()))[
                    "ok"
                ]
            )
            scicom_scale_matrix("solution", 2)
            self.assertFalse(
                asyncio.run(validate(task, {"solution_tensor": "solution"}, Tools()))[
                    "ok"
                ]
            )

    def test_native_config_preserves_environment_and_disables_bootstrap(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from experiments.exp1.runner.deploy_e1a import configure

        api = MagicMock()
        api.get_function_configuration.return_value = {
            "Environment": {"Variables": {"EXISTING": "preserved"}},
            "FunctionArn": "fixture",
            "MemorySize": 3008,
            "Timeout": 240,
        }
        resolved = SimpleNamespace(
            targets={"pool": SimpleNamespace(name="E3AwsLambda")}
        )
        with (
            patch(
                "experiments.exp1.runner.deploy_e1a.resolve_benchmark",
                return_value=resolved,
            ),
            patch(
                "experiments.exp1.runner.deploy_e1a.state_bucket_for_provider",
                return_value="fixture-bucket",
            ),
            patch("atsuite.aws.aws.AWS") as aws,
        ):
            aws.return_value.get_lambda_client.return_value = api
            configure(
                "aws_lambda",
                Path("fixture"),
                environment={"ATSUITE_ALLOW_STATE_BOOTSTRAP": "false"},
            )
        self.assertEqual(
            api.update_function_configuration.call_args.kwargs["Environment"][
                "Variables"
            ],
            {
                "EXISTING": "preserved",
                "PROJECT": "fixture-bucket",
                "ATSUITE_ALLOW_STATE_BOOTSTRAP": "false",
            },
        )

    def test_session_close_does_not_block_concurrent_request_timers(self):
        import threading

        from benchmarks.DataSciBench.langgraph_agent.tool_client import ToolClient

        client = ToolClient.__new__(ToolClient)
        client.rpc_tasks = []
        release = threading.Event()
        closed = []
        client.close = lambda: closed.append(release.wait(1))

        async def close():
            asyncio.get_running_loop().call_later(0.02, release.set)
            await client.aclose()

        asyncio.run(close())
        self.assertEqual(closed, [True])

    def test_model_cannot_inject_state_bootstrap(self):
        class Model:
            async def decide(self, *args):
                return {
                    "tool": "query",
                    "args": {"__atsuite_state_snapshot": {"files": {}}},
                }

        graph = build_graph(
            Model(),
            None,
            {"tool_definitions": [{"name": "query"}]},
            "test",
            None,
            max_iterations=1,
        )
        result = asyncio.run(graph.ainvoke({"user_prompt": "inspect", "iteration": 0}))
        self.assertEqual(result["status"], "failed")

    def test_numeric_replay_accepts_roundoff_and_rejects_changed_results(self):
        from benchmarks.ScientificComputation.langgraph_agent.graph import numeric_equal

        self.assertTrue(
            numeric_equal(
                '{"result":[0.12345678901234]}', '{"result":[0.12345678901235]}'
            )
        )
        for left, right in [
            ([1.0], [1.01]),
            ([1.0], [[1.0]]),
            (True, 1),
            ("1", 1),
            (float("nan"), float("nan")),
        ]:
            self.assertFalse(numeric_equal(left, right))

    def test_git_utc_dates_are_canonical(self):
        source = Path(
            "benchmarks/ClaudeCodeReview/nodes/repo_git/implementation.py"
        ).read_text()
        import ast
        from datetime import datetime

        fn = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "_parse_commits"
        )
        scope = {"datetime": datetime}
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), "<git-parser>", "exec"),
            scope,
        )
        parsed = scope["_parse_commits"](
            "abc\x1fchange\x1f2026-01-01T00:00:00Z\x1fAuthor"
        )
        self.assertEqual(parsed[0]["authored_at"], "2026-01-01T00:00:00+00:00")

    def test_shared_execution_returns_json_records_for_both_transports(self):
        import pandas as pd

        from atsuite_sdk.abstract import UnifiedAgentTool

        def query():
            return pd.DataFrame({"city": ["Myrtle Beach"], "price": [14]})

        tool = UnifiedAgentTool(query)
        self.assertEqual(
            tool.execute_from_dict({"uid": "serialization-check"}),
            [{"city": "Myrtle Beach", "price": 14}],
        )

    def test_real_project_selection_isolated_and_grep_handles_binary_and_file_paths(
        self,
    ):
        import os
        import subprocess
        import sys

        from benchmarks.ClaudeCodeReview.data.materialize_fixture_repo import (
            materialize,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            materialize(root / "first")
            materialize(root / "second")
            p = root / "second/pr_fixture.json"
            v = json.loads(p.read_text())
            v["pull_requests"][0]["id"] = 501
            p.write_text(json.dumps(v))
            (root / "projects.json").write_text(
                json.dumps({"first": "first", "second": "second"})
            )
            (root / "first/fixture_repo/binary.png").write_bytes(b"\x89PNG\x00")
            script = """
from experiments.e3 import review_tools
from atsuite_sdk.abstract import registry
def call(name,uid,**args):return registry.functions[name].execute_from_dict({'uid':uid,**args})
call('repo_git_select_repository','one',project='first')
call('repo_git_select_repository','two',project='second')
assert call('repo_git_list_pull_requests','one')['pull_requests'][0]['id']==101
assert 501 in {p['id'] for p in call('repo_git_list_pull_requests','two')['pull_requests']}
assert call('grep_text','one',pattern='def page',path='pagination.py')['matches']
assert call('grep_text','one',pattern='def page',path='.')['matches']
"""
            env = {
                **os.environ,
                "ATSUITE_DATA_PATH": str(root / "first"),
                "ATSUITE_REVIEW_PROJECTS": str(root),
                "ATSUITE_SESSION_ROOT": str(root / "state"),
                "PROVIDER": "local",
                "ATSUITE_RUNTIME": "function",
            }
            result = subprocess.run(
                [sys.executable, "-c", script], env=env, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_batch_preserves_results_and_parallel_tool_keys(self):
        calls = []

        class Model:
            async def decide(self, phase, iteration, messages):
                if iteration == 0:
                    return {
                        "calls": [
                            {"tool": "left", "args": {}},
                            {"tool": "right", "args": {}},
                        ]
                    }
                return {"answer": {"ok": True}}

        class Tools:
            async def call(self, name, args, key, wave=None):
                calls.append((name, key, wave))
                return {"name": name}

        async def validate(*args):
            return {"ok": True}

        graph = build_graph(
            Model(),
            Tools(),
            {"tool_definitions": [{"name": "left"}, {"name": "right"}], "tool_access": {"left": "r", "right": "r"}},
            "test",
            validate,
        )
        result = asyncio.run(graph.ainvoke({"user_prompt": "inspect", "iteration": 0}))
        self.assertEqual(result["status"], "complete")
        self.assertEqual({c[1] for c in calls}, {"act-0-0", "act-0-1"})
        self.assertEqual({c[2] for c in calls}, {"batch-0"})

    def test_notebook_validates_record_input_at_tool_boundary(self):
        from benchmarks.TravelPlanner.nodes.notebook.implementation import _read_records

        self.assertEqual(
            _read_records('[{"note":"actual tool result"}]').to_dict("records"),
            [{"note": "actual tool result"}],
        )
        for invalid in ("plain notes", "{}", "[1]"):
            with self.assertRaisesRegex(ValueError, "JSON array of objects"):
                _read_records(invalid)

    def test_tool_error_is_recoverable_but_replay_divergence_is_fatal(self):
        from benchmarks.DataSciBench.langgraph_agent.tool_client import (
            DivergenceError,
            ToolExecutionError,
        )

        class Model:
            async def decide(self, phase, iteration, messages):
                if iteration == 0:
                    return {"tool": "query", "args": {}}
                return {"answer": {}}

        class Tools:
            error = ToolExecutionError("invalid input")

            async def call(self, *args):
                raise self.error

        async def validate(*args):
            return {"ok": True}

        tools = Tools()
        graph = build_graph(
            Model(), tools, {"tool_definitions": [{"name": "query"}]}, "test", validate
        )
        self.assertEqual(
            asyncio.run(graph.ainvoke({"user_prompt": "query", "iteration": 0}))[
                "status"
            ],
            "complete",
        )
        tools.error = DivergenceError("changed output")
        with self.assertRaises(DivergenceError):
            asyncio.run(graph.ainvoke({"user_prompt": "query", "iteration": 0}))

    def test_known_workload_context_learns_without_identical_task_id(self):
        from atsuite.workload.descriptor import compatible, distance
        from tests.workload_fixtures import describe

        left = describe(
            prompt="Plan my trip.", tools=["flights"]
        )
        right = describe(
            prompt="Visit another city.", tools=["flights"]
        )
        self.assertEqual(distance(left, right), 1)
        self.assertTrue(compatible(left, right))

    def test_fixed_policy_fallback_needs_bounds_and_respects_admission(self):
        from tests.test_routing import RoutingTests

        router = RoutingTests().router(limit=1, count=0)
        router.default_pool = None
        self.assertEqual(
            router.decide("one", allowed_pools={"aws"}, fallback_pool="aws")[
                "selected"
            ],
            "aws",
        )
        self.assertIsNone(
            router.decide("two", allowed_pools={"aws"}, fallback_pool="aws")["selected"]
        )
        with self.assertRaises(ValueError):
            router.decide("three", allowed_pools=set(), fallback_pool="aws")

    def test_dynamic_agent_hides_validation_and_uses_actual_tool_result(self):
        calls = []

        class Model:
            async def decide(self, phase, iteration, messages):
                self_outer.assertNotIn("secret-oracle", json.dumps(messages))
                if iteration == 0:
                    return {"tool": "lookup", "args": {}}
                self_outer.assertIn("actual-result", json.dumps(messages))
                return {"answer": {"value": "actual-result"}}

        class Tools:
            async def call(self, name, args, key):
                calls.append((name, key))
                return "actual-result"

        async def validate(task, answer, tools):
            return {"ok": answer == {"value": "actual-result"}}

        self_outer = self
        graph = build_graph(
            Model(),
            Tools(),
            {"tool_definitions": [{"name": "lookup"}], "verification": "secret-oracle"},
            "test",
            validate,
        )
        result = asyncio.run(graph.ainvoke({"user_prompt": "inspect", "iteration": 0}))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(calls, [("lookup", "act-0")])

    def test_agent_cannot_override_session_uid(self):
        class Model:
            async def decide(self, *args):
                return {"tool": "lookup", "args": {"uid": "another-user"}}

        graph = build_graph(
            Model(),
            None,
            {"tool_definitions": [{"name": "lookup"}]},
            "test",
            None,
            max_iterations=1,
        )
        result = asyncio.run(graph.ainvoke({"user_prompt": "inspect", "iteration": 0}))
        self.assertEqual(result["status"], "failed")

    def test_seal_reproducible_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks = []
            for workload in WORKLOADS:
                folder = root / workload
                folder.mkdir()
                (folder / "result.json").write_text('{"status":"complete"}')
                (folder / "events.jsonl").write_text(
                    json.dumps(
                        {
                            "kind": "run_start",
                            "task_id": workload,
                            "prompt_sha256": hashlib.sha256(
                                workload.encode()
                            ).hexdigest(),
                        }
                    )
                    + "\n"
                    + '{"kind":"llm","source":"live_official_api"}\n'
                )
                task_manifest = root / (workload + ".json")
                task_manifest.write_text(
                    json.dumps({"tasks": [{"task_id": workload, "prompt": workload}]})
                )
                tasks.append(
                    {
                        "id": workload,
                        "task_id": workload,
                        "workload": workload,
                        "manifest": str(task_manifest),
                        "inputs": [],
                        "capture": str(folder),
                    }
                )
            arrivals = [
                {"arrival_id": str(i), "relative_arrival_ms": i} for i in range(20)
            ]
            phases = [{"start_ms": 0, "weights": dict.fromkeys(WORKLOADS, 1)}]
            design = root / "design.json"
            design.write_text('{"seed":9}')
            first = seal(
                tasks, arrivals, phases, 9, 20, root / "one", artifacts=[design]
            )
            second = seal(
                tasks, arrivals, phases, 9, 20, root / "two", artifacts=[design]
            )
            self.assertEqual(first, second)
            self.assertEqual(load(root / "one"), first)
            design.write_text('{"seed":10}')
            with self.assertRaisesRegex(ValueError, "changed"):
                load(root / "one")
            design.write_text('{"seed":9}')
            Path(tasks[0]["manifest"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                load(root / "one")

    def test_replay_checks_full_business_result(self):
        self.assertNotEqual(result_view('{"result":1}'), result_view('{"result":2}'))
        self.assertNotEqual(result_view("flight A"), result_view("flight B"))

    def test_profiling_does_not_invent_missing_cost(self):
        row = {
            "workload": "TravelPlanner",
            "task": "a",
            "pool": "aws",
            "repeat": 0,
            "latency_s": 2.0,
            "success": True,
        }
        self.assertIsNone(matrix([row])["TravelPlanner"]["aws"]["mean_actual_cost"])
        with self.assertRaises(ValueError):
            matrix([row, row])
        with self.assertRaises(ValueError):
            matrix([{**row, "actual_cost": 0.1}])

    def test_cost_comparison_accepts_only_two_matched_arms(self):
        from experiments.e3.report import compare
        request = {"request_id": "one", "workload": "TravelPlanner", "pool": "ali_fc", "success": True, "latency_s": 1}
        shared = {"manifest_sha256": "same", "native_elasticity": {}, "requests": [request]}
        fixed = {**shared, "strategy": "fixed", "pricing": {"basis": "atsuite_metered_usage_pricing", "complete": True, "total_cny": 2}}
        routed = {**shared, "strategy": "exploration", "pricing": {"basis": "atsuite_metered_usage_pricing", "complete": True, "total_cny": 1}}
        self.assertEqual(compare([fixed, routed])["savings_vs_fixed"], 0.5)
        routed["pricing"]["complete"] = False
        self.assertIsNone(compare([fixed, routed])["savings_vs_fixed"])

    def test_real_numerical_tools_and_independent_validation(self):
        from benchmarks.ScientificComputation.langgraph_agent.graph import validate
        from benchmarks.ScientificComputation.nodes.tensor_workspace.implementation import (
            scicom_create_tensor,
            scicom_matrix_inverse,
            scicom_multiply_matrices,
            scicom_view_tensor,
        )

        scicom_create_tensor(shape=[2, 2], values=[2, 0, 0, 4], name="A")
        scicom_create_tensor(shape=[2, 1], values=[6, 8], name="B")
        scicom_matrix_inverse(name="A", output_name="inv")
        scicom_multiply_matrices(name_a="inv", name_b="B", output_name="X")

        class Tools:
            async def call(self, name, args, key):
                return scicom_view_tensor(**args)

        task = {"verification": {"matrix": [[2, 0], [0, 4]], "rhs": [[6], [8]]}}
        self.assertTrue(
            asyncio.run(
                validate(
                    task, {"solution_tensor": "X", "solution": [[3], [2]]}, Tools()
                )
            )["ok"]
        )
        self.assertFalse(
            asyncio.run(
                validate(
                    task, {"solution_tensor": "X", "solution": [[0], [0]]}, Tools()
                )
            )["ok"]
        )


if __name__ == "__main__":
    unittest.main()
