"""Cloud preparation cannot change a sealed sequence or accept a metadata-only task count."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from atsuite.pipeline import BuildConfig
from experiments.exp1.data.prepare_e1b import sha
from experiments.exp1.data.prepare_mix_cloud import prepare
from experiments.exp1.runner.run_e1b_cloud import rendering_hash


class E2BCloudTests(unittest.TestCase):
    def test_bulk_usage_collection_parses_logs_linearly_across_sessions(self):
        from unittest.mock import Mock
        from experiments.exp1.runner.provider_usage import BatchCloudWatch

        collector=BatchCloudWatch.__new__(BatchCloudWatch)
        collector._usage_groups={}
        collector.discover_agentcore_usage_log_groups=Mock(return_value=['usage'])
        events=[{'message':json.dumps({'attributes':{'session.id':str(i)},
            'event_timestamp':i,'metrics':{'agent.runtime.vcpu.hours.used':i/1000,
                                          'agent.runtime.memory.gb_hours.used':i/100}})} for i in range(100)]
        collector._filter_log_group_events=Mock(return_value=events)
        with patch('experiments.exp1.runner.provider_usage.json.loads',wraps=json.loads) as parse:
            for i in range(100):
                result=collector.get_agentcore_usage_from_logs('runtime',str(i),0,1)
                self.assertEqual(result['vcpu_hours'],i/1000)
                self.assertEqual(result['log_entries'],1)
            self.assertLessEqual(parse.call_count,200)
        collector._filter_log_group_events.assert_called_once()

    def test_explicit_base_image_is_inherited(self):
        original = BuildConfig.from_dict({"python_version": "3.12.12", "base_image": "python@sha256:pinned"})
        self.assertEqual(BuildConfig.from_dict({}, original), original)
        self.assertEqual(BuildConfig.from_dict({"base_image": "other"}, original).base_image, "other")
        self.assertEqual(BuildConfig().base_image, "")

    def test_count_header_cannot_replace_100_distinct_validated_prompts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "tasks.json"
            source.write_text(json.dumps({"reference_verified_count": 100, "formal_experiment_ready": True, "tasks": []}))
            (root / "arrivals.json").write_text("[]")
            (root / "manifest.json").write_text(json.dumps({"mix_manifest": str(source),
                "mix_manifest_sha256": sha(source), "arrivals_sha256": sha(root / "arrivals.json")}))
            (root / "seal.json").write_text(json.dumps({"manifest_sha256": sha(root / "manifest.json")}))
            with self.assertRaisesRegex(ValueError, "100 validated"):
                prepare(root, root / "out")
            self.assertFalse((root / "out").exists())
            (root / "arrivals.json").write_text("[{}]")
            with self.assertRaisesRegex(ValueError, "assignments changed"):
                prepare(root, root / "out")

    def test_mix_uses_strict_environment_references_without_old_rendering_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(json.dumps({"task_manifest": "sealed-mix"}))
            self.assertIsNone(rendering_hash(root))

    def test_faas_evicts_idle_task_interpreters_without_canonical_state_reuse(self):
        from benchmarks.DataSciBench.nodes.AgentTools import implementation as impl

        kernels=[]
        class Kernel:
            def __init__(self, root, inputs, resident):
                self.closed=False
                self.resident=resident
                kernels.append(self)
            def run(self, *args):return {"status":"ok"},None
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            runtime=SimpleNamespace(runtime='function',root=root)
            with (patch.object(impl,'get_state_runtime',return_value=runtime),
                  patch.object(impl,'NativeKernel',Kernel),
                  patch.dict(impl._faas_kernels,{},clear=True),
                  patch.dict(impl._faas_kernel_users,{},clear=True),
                  patch.dict('os.environ',{'WABBER_EXECUTION_BACKEND':'native'})):
                for name in ('a','a','b'):impl.sandbox({},root,inputs=root/name)
                self.assertEqual(len(kernels),2)
                self.assertTrue(kernels[0].closed)
                self.assertFalse(kernels[1].closed)
                self.assertFalse(any(k.resident for k in kernels))
                self.assertEqual(len(impl._faas_kernels),1)

    def test_nltk_lookup_failure_retains_resource_identity_not_worker_home(self):
        from benchmarks.DataSciBench.langgraph_agent.tool_client import semantic, ToolClient

        error="Resource punkt_tab not found\nhttps://www.nltk.org/data.html\nAttempted to load tokenizers/punkt_tab/english/\n  Searched in:\n - /tmp/worker-one/nltk_data"
        a={'status':'error','error_type':'LookupError','error':error}
        b={**a,'error':error.replace('worker-one','worker-two')}
        self.assertEqual(semantic(a),semantic(b))
        self.assertNotEqual(semantic(a),semantic({**b,'error':b['error'].replace('punkt_tab','other_resource')}))
        client=ToolClient.__new__(ToolClient)
        client.renderings={}
        self.assertEqual(client.comparable({'errors':{'tool_errors':[a]}}),
                         client.comparable({'errors':{'tool_errors':[b]}}))
        self.assertNotEqual(client.comparable(a),client.comparable({**b,'error_type':'ValueError'}))

    def test_faas_cache_never_evicts_an_inflight_interpreter(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from benchmarks.DataSciBench.nodes.AgentTools import implementation as impl

        entered,release=threading.Event(),threading.Event()
        kernels={}
        class Kernel:
            def __init__(self,root,inputs,resident):
                self.name=inputs.name;self.closed=False;kernels[self.name]=self
            def run(self,*args):
                if self.name=='a':
                    entered.set()
                    if not release.wait(5):raise TimeoutError('Test did not release active worker')
                    if self.closed:raise AssertionError('In-flight worker was closed')
                return {'status':'ok'},None
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with (patch.object(impl,'get_state_runtime',return_value=SimpleNamespace(runtime='function',root=root)),
                  patch.object(impl,'NativeKernel',Kernel),patch.dict(impl._faas_kernels,{},clear=True),
                  patch.dict(impl._faas_kernel_users,{},clear=True),
                  patch.dict('os.environ',{'WABBER_EXECUTION_BACKEND':'native'}),ThreadPoolExecutor(1) as executor):
                future=executor.submit(impl.sandbox,{},root,root/'a')
                try:
                    self.assertTrue(entered.wait(5))
                    impl.sandbox({},root,inputs=root/'b')
                    self.assertFalse(kernels['a'].closed)
                finally:release.set()
                self.assertEqual(future.result()[0]['status'],'ok')
