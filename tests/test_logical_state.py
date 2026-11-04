"""Real local resident/restored workers; no cloud or LLM account required."""

import importlib.util
import ast
import hashlib
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from atsuite.tracing import TraceCollector
from atsuite.workload import extract_profile
from atsuite_sdk.state import StateRuntime
from atsuite_sdk.resources import get_resource_usage
from benchmarks.DataSciBench.nodes.AgentTools import implementation as impl
from benchmarks.DataSciBench.nodes.AgentTools.logical_state import StateObservation


class CanonicalTypeTests(unittest.TestCase):
    def test_selector_types_have_stable_distinct_identities(self):
        import numpy as np

        source = Path("benchmarks/DataSciBench/nodes/AgentTools/sandbox_worker.py")
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "canonical")
        scope = {"hashlib": hashlib}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
        canonical = scope["canonical"]
        self.assertEqual(canonical(np.number), {"python_type": "numpy.number"})
        self.assertNotEqual(canonical(np.float32), canonical(np.float64))
        self.assertNotEqual(canonical(int), canonical(float))


@unittest.skipUnless(
    importlib.util.find_spec("sklearn") and importlib.util.find_spec("joblib"),
    "sandbox numerical dependencies required",
)
class LogicalStateIntegrationTests(unittest.TestCase):
    def execute(self, root, mode):
        data = root / "data" / "human_15"
        data.mkdir(parents=True)
        env = {
            "ATSUITE_RUNTIME": mode,
            "ATSUITE_SESSION_ROOT": str(root / "runtime"),
            "PROVIDER": "",
            "WABBER_EXECUTION_BACKEND": "native",
            "WABBER_DATA_ROOT": str(root / "data"),
            "WABBER_NATIVE_CONFINEMENT": "landlock",
            "PYTHONPATH": str(Path("benchmarks/DataSciBench/nodes/DataSci").resolve()),
        }
        with patch.dict(os.environ, env):
            runtime = StateRuntime()
            workspace = runtime.register_object(
                "agent_workspace", impl.Workspace(), impl.__name__, session_scoped=True
            )
            collector = TraceCollector(mode, mode)
            collector.start("session", "session", "test")
            collector.start("llm", "llm", "fixed")
            collector.end("llm")
            collector.spans["llm"]["duration_s"] = 2
            observations = []

            def call(name, **kwargs):
                span = str(len(observations))
                collector.start(span, "tool", name)
                result = getattr(impl, name).execute_from_dict(
                    dict(uid="test-session", **kwargs)
                )
                collector.end(span, result)
                collector.observe_runtime(span, {"result": result,
                                                "provider_metadata": {"resource_usage": get_resource_usage()}})
                if name in {"execute_analysis", "merge_artifacts", "validate_result"}:
                    usage = collector.spans[span]["resources"]
                    self.assertGreaterEqual(usage["cpu_seconds"], 0)
                    self.assertEqual(usage["cpu_scope"], "isolated_worker_process")
                    self.assertEqual(usage["cpu_interval"], "call_delta")
                    self.assertFalse(usage["cpu_complete"])
                    self.assertFalse(usage["includes_children"])
                else:
                    self.assertEqual(collector.spans[span]["resources"]["cpu_scope"], "runtime_calling_thread")
                observations.append(result["logical_state"])
                self.assertTrue(
                    result["logical_state"]["complete"], result["logical_state"]
                )
                return result

            with (
                patch("atsuite_sdk.state._RUNTIME", runtime),
                patch.object(impl, "get_state_runtime", return_value=runtime),
                patch.object(impl, "workspace", workspace),
                patch.object(impl, "_mcp_kernel", None),
            ):
                try:
                    call("prepare_workspace", task_id="human_15", dataset_refs=[])
                    if mode == "mcp":
                        self.assertFalse(
                            (
                                runtime.session_dir("test-session") / "v0" / "state.pkl"
                            ).exists()
                        )
                    branch = call(
                        "execute_analysis",
                        step_id="a",
                        snapshot=0,
                        seed=1,
                        call_id="a",
                        code="""state_out['value'] = 17
import numpy as np
import pandas as pd
state_out['matrix'] = np.arange(12).reshape(3,4)
state_out['frame'] = pd.DataFrame({'x':[1,2,3], 'y':[4.0,5.0,6.0]})
(output_dir/'a.json').write_text('[1,2,3]')
import matplotlib.pyplot as plt
fig,ax=plt.subplots();ax.bar([2022,2023],[4800,7800],color='red',edgecolor='black');fig.savefig(output_dir/'bar_chart.png');plt.close(fig)
fig,ax=plt.subplots()
for values in ([1000,1100],[2000,2800],[2700,3000]):ax.plot([1,2],values,linestyle='-.')
fig.savefig(output_dir/'line_chart.png');plt.close(fig)
fig,ax=plt.subplots()
for values,color in (([1500,1600,1700],'red'),([1800,1900],'green'),([2000,2100],'blue')):ax.scatter(values,values,s=values,alpha=.5,color=color)
fig.savefig(output_dir/'scatter_chart.png');plt.close(fig)
""",
                        outputs=[
                            "a.json",
                            "bar_chart.png",
                            "line_chart.png",
                            "scatter_chart.png",
                        ],
                    )
                    call(
                        "merge_artifacts",
                        branches=[
                            {
                                "step_id": "a",
                                "branch_ref": branch["branch_ref"],
                                "artifacts": [
                                    {
                                        k: a[k]
                                        for k in ("name", "signature", "comparison")
                                    }
                                    for a in branch["artifacts"]
                                ],
                            }
                        ],
                        snapshot=0,
                    )
                    call("read_workspace", snapshot=1)
                    call(
                        "validate_result",
                        snapshot=1,
                        outputs=[
                            "bar_chart.png",
                            "line_chart.png",
                            "scatter_chart.png",
                        ],
                    )
                    for step in ("b", "c"):
                        call(
                            "execute_analysis",
                            step_id=step,
                            snapshot=1,
                            seed=1,
                            call_id=step,
                            code="state_out['value'] = shared['a']['value'] + 1\n(output_dir/'b.json').write_text('[4]')",
                            outputs=["b.json"],
                        )
                    if mode == "mcp":
                        self.assertFalse(
                            (
                                runtime.session_dir("test-session") / "v1" / "state.pkl"
                            ).exists()
                        )
                    else:
                        self.assertTrue(
                            (
                                runtime.session_dir("test-session") / "v1" / "state.pkl"
                            ).exists()
                        )
                    with self.assertRaises(impl.AnalysisFailure) as validation_failure:
                        impl.validate_result.execute_from_dict(
                            {
                                "uid": "test-session",
                                "snapshot": 1,
                                "outputs": ["missing.png"],
                            }
                        )
                    failed_validation = validation_failure.exception.result
                    self.assertTrue(failed_validation["logical_state"]["complete"])
                    failed_span = "failed-validation"
                    collector.start(failed_span, "tool", "validate_result")
                    collector.end(
                        failed_span,
                        failed_validation,
                        error="expected validation failure",
                    )
                    collector.observe_runtime(
                        failed_span, {"result": failed_validation}
                    )
                    self.assertEqual(
                        len(collector.spans[failed_span]["metadata"]["state_accesses"]),
                        6,
                    )
                    collector.end("session")
                    profile = extract_profile(collector.snapshot())
                    self.assertIsNotNone(profile["metrics"]["rho_kb_s"])
                    self.assertGreater(profile["metrics"]["state_bytes"], 0)
                    with self.assertRaises(impl.AnalysisFailure) as failed:
                        impl.execute_analysis.execute_from_dict(
                            {
                                "uid": "test-session",
                                "step_id": "failed",
                                "snapshot": 1,
                                "seed": 1,
                                "call_id": "failed",
                                "code": 'raise ValueError("expected")',
                                "outputs": [],
                            }
                        )
                    self.assertFalse(
                        failed.exception.result["logical_state"]["complete"]
                    )
                    return observations, profile
                finally:
                    if impl._mcp_kernel:
                        impl._mcp_kernel.close()
                    for kernel in impl._faas_kernels.values():
                        kernel.close()
                    impl._faas_kernels.clear()

    def test_resident_and_restored_have_identical_logical_accesses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resident, rp = self.execute(root / "mcp", "mcp")
            restored, fp = self.execute(root / "function", "function")
            self.assertEqual(
                [len(o["accesses"]) for o in resident], [3, 8, 14, 1, 6, 9, 9]
            )
            self.assertEqual(rp["metrics"]["state_reads"], 36)
            self.assertEqual(rp["metrics"]["state_writes"], 20)
            self.assertIsNotNone(rp["metrics"]["state_observation_s"])
            self.assertEqual(
                [o["accesses"] for o in resident], [o["accesses"] for o in restored]
            )
            self.assertEqual(rp["metrics"]["state_bytes"], fp["metrics"]["state_bytes"])
            self.assertEqual(rp["metrics"]["rho_kb_s"], fp["metrics"]["rho_kb_s"])
            reads = [
                [
                    a
                    for a in o["accesses"]
                    if a["object_id"] == "canonical/state" and a["operation"] == "read"
                ]
                for o in resident[-2:]
            ]
            self.assertEqual(reads[0], reads[1])
            self.assertTrue(reads[0])
            artifact_reads = [
                a
                for a in resident[-1]["accesses"]
                if a["object_id"] == "canonical/artifacts/a.json"
            ]
            self.assertEqual(
                artifact_reads,
                [
                    {
                        "object_id": "canonical/artifacts/a.json",
                        "operation": "read",
                        "bytes": 7,
                    }
                ],
            )
            evidence = os.getenv("ATSUITE_TEST_ARTIFACT_DIR")
            if evidence:
                Path(evidence).mkdir(parents=True, exist_ok=True)
                (Path(evidence) / "logical-state-parity.json").write_text(
                    json.dumps(
                        {
                            "resident": resident,
                            "restored": restored,
                            "resident_profile": rp,
                            "restored_profile": fp,
                        },
                        indent=2,
                    )
                )


class LogicalStateTests(unittest.TestCase):
    def test_measurement_failure_is_missing_not_zero(self):
        observation = StateObservation()
        observation.record("x", "read", lambda: 12)

        def fail():
            raise ValueError("not serializable")

        observation.record("x", "write", fail)
        collector = TraceCollector("r", "s")
        collector.start("tool", "tool", "tool")
        collector.observe_runtime("tool", {"result": observation.attach({})})
        self.assertNotIn("state_accesses", collector.spans["tool"]["metadata"])
        self.assertFalse(
            collector.spans["tool"]["metadata"]["logical_state"]["complete"]
        )

    def test_concurrent_calls_keep_separate_access_logs(self):
        barrier = threading.Barrier(2)

        @impl.observe_workspace("r")
        def measured(name):
            impl._observations.get().record(name, "read", lambda: 13)
            barrier.wait(timeout=5)
            return {"status": "ok"}

        with (
            patch.object(impl, "workspace", impl.Workspace()),
            ThreadPoolExecutor(2) as pool,
        ):
            results = list(pool.map(measured, ("left", "right")))
        self.assertEqual(
            [[a["object_id"] for a in r["logical_state"]["accesses"]] for r in results],
            [["workspace/metadata", "left"], ["workspace/metadata", "right"]],
        )

    def test_malformed_runtime_measurement_stays_unknown(self):
        collector = TraceCollector("r", "s")
        collector.start("tool", "tool", "tool")
        collector.observe_state_accesses("tool", [])
        collector.observe_runtime(
            "tool",
            {
                "result": {
                    "logical_state": {
                        "contract": "datasci-tool-snapshot-v1",
                        "complete": True,
                        "accesses": [{"object_id": "x", "operation": [], "bytes": 1}],
                    }
                }
            },
        )
        self.assertNotIn("state_accesses", collector.spans["tool"]["metadata"])
        self.assertFalse(
            collector.spans["tool"]["metadata"]["logical_state"]["complete"]
        )

    def test_old_worker_does_not_claim_complete(self):
        observation = StateObservation()
        observation.include({"status": "ok"})
        self.assertFalse(observation.complete)


if __name__ == "__main__":
    unittest.main()
