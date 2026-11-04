"""Exercise real framework loops with deterministic HTTP responses and business-only tools."""

import asyncio
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import httpx
from agents import Agent, OpenAIChatCompletionsModel, RunHooks, function_tool
from agents.tracing import set_trace_processors
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from atsuite.runtime import FunctionRuntimeAdapter
from atsuite.tracing import TraceCollector, TracingAdapter, export_atg
from atsuite.tracing.langchain import LangChainTracingAdapter
from atsuite.tracing.langgraph import LangGraphTracingAdapter
from atsuite.tracing.openai_agents import OpenAIAgentsTracingAdapter
from atsuite.tracing.tools import RuntimeToolset
from atsuite.tracing.topology import tool_topology
from atsuite_sdk.response import ToolResponse
from atsuite_sdk.workflow import Trace


def model_transport(*, failure=False, batches=(2,), tool_name="echo"):
    count = 0

    def respond(request):
        nonlocal count
        count += 1
        if failure:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "fixture model failure",
                        "type": "invalid_request_error",
                    }
                },
            )
        calls = (
            [
                {
                    "id": f"call_{count}_{i}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps({"value": str(i)}),
                    },
                }
                for i in range(batches[count - 1])
            ]
            if count <= len(batches)
            else []
        )
        message = (
            {"role": "assistant", "content": None, "tool_calls": calls}
            if calls
            else {"role": "assistant", "content": "done"}
        )
        return httpx.Response(
            200,
            json={
                "id": f"chat_{count}",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if calls else "stop",
                        "message": message,
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    return httpx.MockTransport(respond)


class Callback(BaseCallbackHandler):
    def __init__(self):
        self.tools = 0

    def on_tool_start(self, *args, **kwargs):
        self.tools += 1


class Hooks(RunHooks):
    def __init__(self):
        self.tools = 0

    async def on_tool_start(self, *args):
        self.tools += 1


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Unit tests keep SDK traces local; the library preserves the user's exporters.
        set_trace_processors([])

    async def run_agent(self, harness, function, *, transport=None, callback=None, sdk_config=None):
        collector = TraceCollector("request", "session")
        async with httpx.AsyncClient(transport=transport or model_transport()) as http:
            if harness == "openai":
                client = AsyncOpenAI(
                    api_key="fixture",
                    base_url="https://fixture.invalid",
                    http_client=http,
                    max_retries=0,
                )
                adapter = OpenAIAgentsTracingAdapter(collector)
                agent = Agent(
                    name="fixture",
                    model=OpenAIChatCompletionsModel("fixture", client),
                    tools=[function],
                )
                result = await adapter.ainvoke(agent, "call both tools", hooks=callback, run_config=sdk_config)
                self.assertEqual(result.final_output, "done")
            else:
                model = ChatOpenAI(
                    model="fixture",
                    api_key="fixture",
                    base_url="https://fixture.invalid",
                    http_async_client=http,
                    max_retries=0,
                )
                adapter = LangChainTracingAdapter(collector)
                agent = create_agent(model, [function])
                result = await adapter.ainvoke(
                    agent,
                    {"messages": [{"role": "user", "content": "call both tools"}]},
                    {"callbacks": [callback] if callback else []},
                )
                self.assertEqual(result["messages"][-1].content, "done")
        self.assertIsInstance(adapter, TracingAdapter)
        return collector

    async def test_business_only_tools_parallel_callbacks_and_export(self):
        for harness in ("langchain", "openai"):
            with self.subTest(harness=harness):
                active = peak = 0

                async def echo(value: str) -> str:
                    """Return the supplied value."""
                    nonlocal active, peak
                    active += 1
                    peak = max(peak, active)
                    await asyncio.sleep(0.01)
                    active -= 1
                    return value

                native = function_tool(echo) if harness == "openai" else tool(echo)
                callback = Hooks() if harness == "openai" else Callback()
                collector = await self.run_agent(harness, native, callback=callback)
                trace = collector.snapshot()
                self.assertEqual(peak, 2)
                self.assertEqual(callback.tools, 2)
                self.assertEqual(trace["diagnostics"], [])
                self.assertTrue(trace["summary"]["complete"])
                self.assertTrue(trace["summary"]["successful"])
                self.assertEqual(trace["summary"]["llm_calls"], 2)
                tools = [s for s in trace["spans"] if s["kind"] == "tool"]
                self.assertEqual(len(tools), 2)
                self.assertEqual(tool_topology(trace)["edges"], [])
                for span in tools:
                    self.assertGreater(span["resources"]["peak_rss_bytes"], 0)
                    self.assertGreaterEqual(span["resources"]["cpu_seconds"], 0)
                    self.assertFalse(span["resources"]["cpu_complete"])
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "atg.json"
                    export_atg(trace, path)
                    parsed = Trace.from_file(path)
                    self.assertEqual(sum(n.node_type == "tool_use" for n in parsed.nodes), 2)
                if harness == "openai":
                    self.assertEqual(tools[0]["dependencies"], tools[1]["dependencies"])
                    self.assertEqual(collector.spans[tools[0]["dependencies"][0]]["kind"], "llm")

    async def test_runtime_measurements_are_attached_without_business_instrumentation(
        self,
    ):
        def echo(value: str) -> str:
            """Return the supplied value."""
            raise AssertionError("The business function must execute in the Runtime")

        for harness in ("langchain", "openai"):
            with self.subTest(harness=harness):
                runtime = FunctionRuntimeAdapter("local")
                runtime.connect({"targets": {"t": "https://fixture.invalid"}})
                client = MagicMock()
                client.invoke_result.return_value = ToolResponse(
                    result="echoed",
                    provider_metadata={
                        "resource_usage": {
                            "cpu_seconds": 0.25,
                            "cpu_scope": "runtime_calling_thread",
                        },
                        "state_sync": {
                            "load_bytes": 1,
                            "save_bytes": 2,
                            "branch_read_bytes": 0,
                            "branch_write_bytes": 0,
                        },
                    },
                )
                runtime._client = MagicMock(return_value=client)
                async with RuntimeToolset(runtime, "t") as toolset:
                    native = (toolset.openai_tools if harness == "openai" else toolset.langchain_tools)([echo])[0]
                    collector = await self.run_agent(harness, native)
                tools = [s for s in collector.spans.values() if s["kind"] == "tool"]
                self.assertEqual(len(tools), 2)
                for span in tools:
                    self.assertEqual(span["resources"]["cpu_seconds"], 0.25)
                    self.assertEqual(span["state_read_bytes"], 1)
                    self.assertEqual(span["state_write_bytes"], 2)
                    self.assertEqual(span["metadata"]["runtime_uid"], toolset.uid)
                self.assertEqual(client.invoke_result.call_count, 2)
                for call in client.invoke_result.call_args_list:
                    self.assertEqual(call.args[1]["uid"], toolset.uid)

    async def test_sdk_model_failure_finishes_trace_and_resets_context(self):
        from atsuite.tracing.context import active_adapter

        collector = TraceCollector("failed", "session")
        adapter = OpenAIAgentsTracingAdapter(collector)
        async with httpx.AsyncClient(transport=model_transport(failure=True)) as http:
            client = AsyncOpenAI(api_key="fixture", http_client=http, max_retries=0)
            agent = Agent(name="failed", model=OpenAIChatCompletionsModel("fixture", client))
            with self.assertRaises(Exception):
                await adapter.ainvoke(agent, "fail")
        trace = collector.snapshot()
        self.assertFalse(trace["summary"]["successful"])
        self.assertTrue(all(s["end_ns"] is not None for s in trace["spans"]))
        self.assertIsNone(active_adapter.get())
        self.assertTrue(any(s["kind"] == "llm" and s["status"] == "error" for s in trace["spans"]))

    async def test_sdk_sensitive_data_setting_also_redacts_runtime_payloads(self):
        def echo(value: str, uid: str) -> str:
            """Return the supplied value."""
            return value

        runtime = FunctionRuntimeAdapter("local")
        runtime.connect({"targets": {"t": "https://fixture.invalid"}})
        client = MagicMock()
        client.invoke_result.return_value = ToolResponse(
            result="PRIVATE_BUSINESS_RESULT",
            provider_metadata={
                "mcp_result": {"secret": "PRIVATE_BUSINESS_RESULT"},
                "resource_usage": {"cpu_seconds": 0.25},
            },
        )
        runtime._client = MagicMock(return_value=client)
        async with RuntimeToolset(runtime, "t") as tools:
            collector = await self.run_agent(
                "openai",
                tools.openai_tools([echo])[0],
                sdk_config={"trace_include_sensitive_data": False},
            )
        trace = collector.snapshot()
        self.assertNotIn("PRIVATE_BUSINESS_RESULT", json.dumps(trace))
        for span in trace["spans"]:
            self.assertIsNone(span["input"])
            self.assertIsNone(span["output"])
        self.assertEqual(
            [s["resources"]["cpu_seconds"] for s in trace["spans"] if s["kind"] == "tool"],
            [0.25, 0.25],
        )

    async def test_cancelled_proxy_drains_rpc_before_releasing_session(self):
        def echo(value: str, uid: str) -> str:
            """Return the supplied value."""
            return value

        started, finished = threading.Event(), threading.Event()

        def invoke(*args, **kwargs):
            started.set()
            time.sleep(0.05)
            finished.set()
            return ToolResponse(result="done")

        runtime = FunctionRuntimeAdapter("local")
        runtime.connect({"targets": {"t": "https://fixture.invalid"}})
        client = MagicMock()
        client.invoke_result.side_effect = invoke
        runtime._client = MagicMock(return_value=client)
        runtime.close_session = MagicMock(side_effect=lambda session: self.assertTrue(finished.is_set()))
        async with RuntimeToolset(runtime, "t") as tools:
            task = asyncio.create_task(tools.langchain_tools([echo])[0].ainvoke({"value": "x"}))
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(finished.is_set())
            runtime.close_session.assert_not_called()
        runtime.close_session.assert_called_once()

    async def test_parallel_batch_joins_before_next_tool_in_both_frameworks(self):
        async def echo(value: str) -> str:
            """Return the supplied value."""
            await asyncio.sleep(0.01)
            return value

        for harness in ("langchain", "openai"):
            with self.subTest(harness=harness):
                native = function_tool(echo) if harness == "openai" else tool(echo)
                collector = await self.run_agent(harness, native, transport=model_transport(batches=(2, 1)))
                trace = collector.snapshot()
                self.assertEqual(trace["diagnostics"], [])
                self.assertTrue(trace["summary"]["complete"])
                tools = [s for s in trace["spans"] if s["kind"] == "tool"]
                graph = tool_topology(trace)
                edges = {(edge["source"], edge["target"]) for edge in graph["edges"]}
                self.assertEqual(
                    edges,
                    {
                        (tools[0]["id"], tools[2]["id"]),
                        (tools[1]["id"], tools[2]["id"]),
                    },
                )

    async def test_sdk_handled_tool_error_is_retained(self):
        @function_tool
        async def echo(value: str) -> str:
            """Return the supplied value, unless it is zero."""
            if value == "0":
                raise ValueError("fixture business failure")
            return value

        collector = await self.run_agent("openai", echo)
        tools = [s for s in collector.spans.values() if s["kind"] == "tool"]
        self.assertEqual([s["status"] for s in tools], ["error", "ok"])
        self.assertIn("fixture business failure", tools[0]["error"])
        self.assertTrue(collector.snapshot()["summary"]["successful"])

    async def test_concurrent_requests_keep_separate_collectors(self):
        @function_tool
        async def echo(value: str) -> str:
            """Return the supplied value."""
            await asyncio.sleep(0.01)
            return value

        left, right = await asyncio.gather(self.run_agent("openai", echo), self.run_agent("openai", echo))
        self.assertFalse(set(left.spans) & set(right.spans))
        for collector in (left, right):
            self.assertTrue(collector.snapshot()["summary"]["successful"])
            self.assertEqual(collector.snapshot()["summary"]["tool_calls"], 2)

    async def test_nested_runner_without_inherited_hooks_still_records_its_model(self):
        from agents import Runner

        @function_tool
        async def delegate(value: str) -> str:
            """Ask a specialist to process the value."""
            async with httpx.AsyncClient(transport=model_transport(batches=())) as http:
                client = AsyncOpenAI(api_key="fixture", http_client=http)
                specialist = Agent(
                    name="specialist",
                    model=OpenAIChatCompletionsModel("fixture", client),
                )
                return (await Runner.run(specialist, value)).final_output

        collector = await self.run_agent(
            "openai",
            delegate,
            transport=model_transport(batches=(1,), tool_name="delegate"),
        )
        trace = collector.snapshot()
        self.assertEqual(trace["diagnostics"], [])
        self.assertEqual(trace["summary"]["llm_calls"], 3)
        self.assertEqual(trace["summary"]["tool_calls"], 1)
        self.assertTrue(trace["summary"]["complete"])

    async def test_create_agent_checkpoint_resume_keeps_tool_origins(self):
        from langgraph.checkpoint.memory import InMemorySaver

        @tool
        async def echo(value: str) -> str:
            """Return the supplied value."""
            return value

        collector = TraceCollector("resume", "session")
        adapter = LangChainTracingAdapter(collector)
        config = {"configurable": {"thread_id": "resume"}}
        async with httpx.AsyncClient(transport=model_transport()) as http:
            model = ChatOpenAI(model="fixture", api_key="fixture", http_async_client=http)
            graph = create_agent(model, [echo], checkpointer=InMemorySaver(), interrupt_before=["tools"])
            await adapter.ainvoke(graph, {"messages": [{"role": "user", "content": "go"}]}, config)
            self.assertFalse(collector.snapshot()["summary"]["complete"])
            await adapter.ainvoke(graph, None, config)
        trace = collector.snapshot()
        self.assertEqual(trace["diagnostics"], [])
        self.assertTrue(trace["summary"]["complete"])
        self.assertTrue(trace["summary"]["successful"])
        self.assertEqual(trace["summary"]["tool_calls"], 2)

    def test_langchain_reuses_langgraph_adapter(self):
        self.assertTrue(issubclass(LangChainTracingAdapter, LangGraphTracingAdapter))

    def test_runtime_proxy_resolves_business_module_annotations(self):
        from typing import Literal

        namespace = {"Category": Literal["a", "b"]}
        exec(
            'from __future__ import annotations\ndef echo(value: Category) -> str:\n    """Echo a category."""\n    return value',
            namespace,
        )
        runtime = FunctionRuntimeAdapter("local")
        tools = RuntimeToolset(runtime, "t")
        langchain_tool = tools.langchain_tools([namespace["echo"]])[0]
        openai_tool = tools.openai_tools([namespace["echo"]])[0]
        self.assertEqual(langchain_tool.args_schema.model_json_schema()["properties"]["value"]["enum"], ["a", "b"])
        self.assertEqual(openai_tool.params_json_schema["properties"]["value"]["enum"], ["a", "b"])

    def test_tool_projection_preserves_fork_join_and_removes_transitive_edges(self):
        collector = TraceCollector("r", "s")
        for rid, kind, parent, deps in [
            ("session", "session", None, []),
            ("prepare", "tool", "session", []),
            ("model", "llm", "session", ["prepare"]),
            ("left", "tool", "session", ["model"]),
            ("right", "tool", "session", ["model"]),
            ("join", "workflow", "session", ["left", "right"]),
            ("merge", "tool", "join", ["prepare"]),
        ]:
            collector.start(rid, kind, rid, parent_id=parent, dependencies=deps)
            collector.end(rid)
        edges = {(edge["source"], edge["target"]) for edge in tool_topology(collector.snapshot())["edges"]}
        self.assertEqual(
            edges,
            {
                ("prepare", "left"),
                ("prepare", "right"),
                ("left", "merge"),
                ("right", "merge"),
            },
        )
