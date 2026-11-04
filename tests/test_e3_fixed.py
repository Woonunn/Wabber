"""Checks for fixed baseline measurement, independent of cloud pricing or routing predictions."""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.e3.fixed_baselines import measure


class FixedMeasurementTests(unittest.TestCase):
    def test_failed_requests_keep_queue_time_and_unknown_cost(self):
        async def fail(*args, **kwargs):
            raise RuntimeError("provider failed")

        task = {
            "id": "task",
            "task_id": "task",
            "workload": "TravelPlanner",
            "manifest": "manifest",
            "capture": "capture",
        }
        config = {"agents": {"TravelPlanner": {}}, "targets": {"TravelPlanner": {"provider": {"tool": {}}}}}
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("experiments.exp1.capture.capture_agent_traces.run", side_effect=fail),
        ):
            row = asyncio.run(
                measure(task, config, "provider", Path(temporary), arrival_ns=time.time_ns() - 2_000_000_000)
            )
        self.assertFalse(row["success"])
        self.assertGreaterEqual(row["latency_s"], 2)
        self.assertIsNone(row["actual_cost"])
        self.assertEqual(row["error"]["message"], "provider failed")

    def test_success_uses_frozen_capture_and_fixed_target(self):
        calls = []

        async def execute(*args, **kwargs):
            calls.append((args, kwargs))

        task = {
            "id": "task",
            "task_id": "task",
            "workload": "ScientificComputation",
            "manifest": "manifest",
            "capture": "capture",
        }
        config = {
            "agents": {"ScientificComputation": {"model": {"seed": 9}}},
            "targets": {"ScientificComputation": {"provider": {"tool": {"provider": "provider"}}}},
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("experiments.exp1.capture.capture_agent_traces.run", side_effect=execute),
        ):
            row = asyncio.run(measure(task, config, "provider", Path(temporary)))
        self.assertTrue(row["success"])
        self.assertEqual(calls[0][0][3], Path("capture"))
        self.assertEqual(calls[0][0][1]["tool"]["provider"], "provider")
        self.assertNotIn("router", calls[0][1])


class SelectedPoolTests(unittest.TestCase):
    def test_measure_reads_compressed_routing_decision(self):
        from atsuite.tracing.storage import write_json
        async def execute(task_id,agent,output,*args,**kwargs):
            write_json(output/'routing_decision.json',{'selected':'actual'})
        task={'id':'task','task_id':'task','workload':'TravelPlanner','manifest':'manifest','capture':'capture'}
        config={'agents':{'TravelPlanner':{}},'targets':{'TravelPlanner':{'fallback':{'tool':{}}}}}
        with tempfile.TemporaryDirectory() as temporary,patch('experiments.exp1.capture.capture_agent_traces.run',side_effect=execute):
            row=asyncio.run(measure(task,config,'fallback',Path(temporary),router=object(),routing_context={}))
        self.assertTrue(row['success'])
        self.assertEqual(row['pool'],'actual')
