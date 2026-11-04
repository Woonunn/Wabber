"""Task cleanup releases native sessions only after outstanding tools finish."""

import asyncio
import threading
import unittest
from unittest.mock import MagicMock, patch

import httpx
from botocore.credentials import Credentials

from atsuite.mcp.mcp import AgentCoreMCPClient, MCPClient
from atsuite.runtime import MCPRuntimeAdapter, RuntimeSession
from atsuite_sdk.response import ToolResponse
from benchmarks.DataSciBench.langgraph_agent.tool_client import ToolClient


class NativeSessionCleanupTests(unittest.TestCase):
    def test_mcp_delete_uses_session_and_negotiated_protocol_and_is_idempotent(self):
        with patch("atsuite.mcp.mcp.httpx.Client") as http:
            client = MCPClient("https://example.test", "task")
        client.session_id = "native-session"
        client.protocol_version = "2025-06-18"
        http.return_value.delete.return_value = httpx.Response(
            204, request=httpx.Request("DELETE", "https://example.test/mcp")
        )
        client.close()
        client.close()
        http.return_value.delete.assert_called_once_with(
            "https://example.test/mcp",
            headers={"mcp-session-id": "native-session", "mcp-protocol-version": "2025-06-18"},
            timeout=30,
        )
        self.assertIsNone(client.session_id)
        self.assertFalse(client._initialized)

    def test_mcp_delete_failure_is_visible_and_transport_is_still_closed(self):
        for status in (403, 404, 405, 500):
            with self.subTest(status=status), patch("atsuite.mcp.mcp.httpx.Client") as http:
                client = MCPClient("https://example.test", "task")
                client.session_id = "native-session"
                http.return_value.delete.return_value = httpx.Response(
                    status, request=httpx.Request("DELETE", "https://example.test/mcp")
                )
                if status == 404:
                    client.close()
                    self.assertIsNone(client.session_id)
                else:
                    with self.assertRaises(httpx.HTTPStatusError):
                        client.close()
                    self.assertEqual(client.session_id, "native-session")
                http.return_value.close.assert_called_once()

    def test_agentcore_stops_correct_runtime_session_with_signed_request(self):
        client = AgentCoreMCPClient.__new__(AgentCoreMCPClient)
        client.url = (
            "https://bedrock-agentcore.us-west-2.amazonaws.com/"
            "runtimes/arn%3Aaws%3Abedrock-agentcore%3Aus-west-2%3A123456789012%3Aruntime%2Ftest/"
            "invocations?qualifier=TEST"
        )
        client.region = "us-west-2"
        client.session_id = "a-native-session-id-longer-than-33-characters"
        client._initialized = True
        client._credentials = Credentials("test-key", "test-secret", "test-token").get_frozen_credentials()
        stop_url = client.url.replace("/invocations?", "/stopruntimesession?")
        for status in (403, 404, 200):
            with self.subTest(status=status), patch("atsuite.mcp.mcp.requests.post") as post:
                client.session_id = "a-native-session-id-longer-than-33-characters"
                post.return_value = httpx.Response(status, request=httpx.Request("POST", stop_url))
                if status == 403:
                    with self.assertRaises(httpx.HTTPStatusError):
                        client.close()
                    self.assertIsNotNone(client.session_id)
                else:
                    client.close()
                    client.close()
                    self.assertIsNone(client.session_id)
                post.assert_called_once()
                self.assertEqual(post.call_args.args, (stop_url,))
                headers = {k.lower(): v for k, v in post.call_args.kwargs["headers"].items()}
                self.assertEqual(
                    headers["x-amzn-bedrock-agentcore-runtime-session-id"],
                    "a-native-session-id-longer-than-33-characters",
                )
                self.assertIn("/us-west-2/bedrock-agentcore/aws4_request", headers["authorization"])
                self.assertEqual(headers["x-amz-security-token"], "test-token")
                self.assertEqual(post.call_args.kwargs["data"], "{}")
                self.assertEqual(post.call_args.kwargs["timeout"], 30)

    def test_cleanup_attempts_all_task_sessions_and_preserves_other_tasks(self):
        adapter = MCPRuntimeAdapter("ali_agentrun")
        adapter.connect({"targets": {name: "https://example.test" for name in ("one", "two", "three")}})
        clients = {name: MagicMock() for name in ("one", "two", "three")}
        other = MagicMock()
        adapter._clients = {f"{name}:task": client for name, client in clients.items()}
        adapter._clients["one:other-task"] = other
        clients["two"].close.side_effect = RuntimeError("release denied")
        with self.assertRaisesRegex(RuntimeError, "release denied"):
            adapter.cleanup_run("task")
        for client in clients.values():
            client.close.assert_called_once()
        other.close.assert_not_called()
        self.assertEqual(adapter._clients, {"one:other-task": other})

    def test_failed_initialize_releases_partially_opened_session(self):
        adapter = MCPRuntimeAdapter("ali_agentrun")
        adapter.connect({"targets": {"server": "https://example.test"}})
        client = MagicMock(_initialized=False)
        client.initialize.side_effect = RuntimeError("initialize failed")
        adapter._clients["server:task"] = client
        with self.assertRaisesRegex(RuntimeError, "initialize failed"):
            adapter.open_session(adapter.targets["server"], "task")
        client.close.assert_called_once()
        self.assertFalse(adapter._clients)


class TaskCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_cancellation_still_waits_for_tools_and_releases_sessions(self):
        finished = asyncio.Event()
        closed = []
        client = ToolClient.__new__(ToolClient)
        pending = asyncio.create_task(finished.wait())
        client.rpc_tasks = [pending]
        client.close = lambda: closed.append(True)
        closing = asyncio.create_task(client.aclose())
        try:
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(0)
            self.assertFalse(closed)
            self.assertFalse(pending.cancelled())
            finished.set()
            with self.assertRaises(asyncio.CancelledError):
                await closing
        finally:
            finished.set()
            await asyncio.gather(pending, closing, return_exceptions=True)
        self.assertEqual(closed, [True])

    async def test_cancelled_tool_finishes_and_records_result_before_session_release(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []
        recorder = MagicMock(request_id="request")
        adapter = MagicMock()
        adapter.targets = {"server": object()}
        adapter.open_session.return_value = RuntimeSession("task", "server", "native-session")

        def invoke(request):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release the tool")
            calls.append("result_received")
            return ToolResponse(result={"status": "ok", "answer": 42})

        adapter.invoke.side_effect = invoke
        adapter.cleanup_run.side_effect = lambda uid: calls.append("session_released")
        with patch(
            "benchmarks.DataSciBench.langgraph_agent.tool_client.create_runtime_adapter", return_value=adapter
        ):
            client = ToolClient(
                {"provider": "ali_agentrun", "family": "session", "target_id": "server", "endpoint": "fixture"},
                recorder,
            )
        running = asyncio.create_task(client.call("query", {}, "call"))
        closing = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            adapter.cleanup_run.assert_not_called()
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
            closing = asyncio.create_task(client.aclose())
            await asyncio.sleep(0)
            adapter.cleanup_run.assert_not_called()
            release.set()
            await closing
        finally:
            release.set()
            await asyncio.gather(running, *([closing] if closing else []), return_exceptions=True)
        self.assertEqual(calls, ["result_received", "session_released"])
        event_names = [call.args[0] for call in recorder.emit.call_args_list]
        self.assertLess(event_names.index("tool"), event_names.index("tool_sessions_closed"))
        self.assertFalse(client.sessions)

    async def test_cleanup_waits_for_cancelled_session_open_and_releases_it(self):
        entered = threading.Event()
        release = threading.Event()
        adapter = MagicMock()
        adapter.targets = {name: object() for name in ("one", "two")}

        def open_session(target, uid):
            if target is adapter.targets["two"]:
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test did not release initialization")
            return RuntimeSession(uid, "server", "native-session")

        adapter.open_session.side_effect = open_session
        with patch(
            "benchmarks.DataSciBench.langgraph_agent.tool_client.create_runtime_adapter", return_value=adapter
        ):
            client = ToolClient(
                {"provider": "ali_agentrun", "family": "session", "target_id": "one",
                 "targets": {name: {"endpoint": "fixture"} for name in ("one", "two")},
                 "tool_targets": {"query": "two"}},
                MagicMock(request_id="request"),
            )
        running = asyncio.create_task(client.call("query", {}, "call"))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
            closing = asyncio.create_task(client.aclose())
            await asyncio.sleep(0)
            adapter.cleanup_run.assert_not_called()
            release.set()
            await closing
        finally:
            release.set()
            await asyncio.gather(running, return_exceptions=True)
        adapter.invoke.assert_not_called()
        adapter.cleanup_run.assert_called_once_with(client.uid)


if __name__ == "__main__":
    unittest.main()
