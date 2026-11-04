from __future__ import annotations

import os
import time
import uuid
import threading
from contextlib import ExitStack
from dataclasses import replace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol


EndpointMap = Mapping[str, Any]


@dataclass(frozen=True)
class RuntimeCapabilities:
    family: str
    requires_open_session: bool
    platform_handles_state_concurrency: bool
    supports_external_state_ops: bool
    observability: str = ""


@dataclass(frozen=True)
class RuntimeTarget:
    target_id: str
    family: str
    endpoint: str
    resources: Dict[str, Any] = field(default_factory=dict)
    tool_manifest: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeSession:
    uid: str
    target_id: str
    provider_session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InvocationRequest:
    target_id: str
    tool_name: str
    args: Dict[str, Any]
    uid: str
    call_id: str = ""
    session: Optional[RuntimeSession] = None
    timeout: Optional[float] = None
    run_id: str = ""


from atsuite_sdk.response import ToolResponse as InvocationResult


@dataclass(frozen=True)
class StateKey:
    service: str
    uid: str
    name: str

    def as_path(self) -> str:
        return f"{self.service}/{self.uid}/{self.name}".strip("/")


@dataclass(frozen=True)
class StatePrefix:
    service: str
    uid: str = ""

    def as_path(self) -> str:
        return f"{self.service}/{self.uid}".strip("/")


class RuntimeAdapter(Protocol):
    def capabilities(self) -> RuntimeCapabilities:
        ...

    def connect(self, endpoint_map: EndpointMap) -> None:
        ...

    def open_session(self, target: RuntimeTarget, uid: str) -> RuntimeSession:
        ...

    def invoke(self, request: InvocationRequest) -> InvocationResult:
        ...

    def close_session(self, session: RuntimeSession) -> dict | None:
        ...

    def cleanup_run(self, uid: str) -> list[dict] | None:
        ...


class StateBackend(Protocol):
    def upload(self, key: StateKey, local_path: Path) -> None:
        ...

    def download(self, key: StateKey, local_path: Path) -> None:
        ...

    def append(self, key: StateKey, data: bytes) -> int:
        ...

    def read(self, key: StateKey) -> bytes:
        ...

    def delete(self, key: StateKey) -> None:
        ...

    def clear(self, prefix: StatePrefix) -> None:
        ...

    def sizeof(self, prefix: StatePrefix) -> int:
        ...


class _BaseRuntimeAdapter:
    def __init__(self, provider: str):
        self.provider = provider
        self.endpoint_map: EndpointMap = {}
        self.targets: Dict[str, RuntimeTarget] = {}
        self._clients_lock = threading.RLock()

    def connect(self, endpoint_map: EndpointMap) -> None:
        self.endpoint_map = endpoint_map
        raw_targets = endpoint_map.get("targets", {})
        self.targets = {}
        for target_id, raw in raw_targets.items():
            if isinstance(raw, str):
                endpoint = raw
                resources: Dict[str, Any] = {}
                manifest: Dict[str, Any] = {}
                family = str(endpoint_map.get("family") or self.capabilities().family)
            elif isinstance(raw, Mapping):
                endpoint = str(raw.get("endpoint") or raw.get("url") or "")
                resources = dict(raw.get("resources") or raw.get("runtime") or {})
                manifest = dict(raw.get("tool_manifest") or {})
                family = str(raw.get("family") or endpoint_map.get("family") or self.capabilities().family)
            else:
                continue
            self.targets[str(target_id)] = RuntimeTarget(
                target_id=str(target_id),
                family=family,
                endpoint=endpoint.rstrip("/"),
                resources=resources,
                tool_manifest=manifest,
            )

    def _target(self, target_id: str) -> RuntimeTarget:
        try:
            return self.targets[target_id]
        except KeyError as exc:
            raise ValueError(f"Missing endpoint for target: {target_id}") from exc

    def close_session(self, session: RuntimeSession) -> dict | None:
        return None

    def cleanup_run(self, uid: str) -> list[dict] | None:
        return None


class FunctionRuntimeAdapter(_BaseRuntimeAdapter):
    def __init__(self, provider: str):
        super().__init__(provider)
        self._clients: Dict[str, Any] = {}

    def capabilities(self) -> RuntimeCapabilities:
        observability = {
            "ali_fc": "ali_sls",
            "aws_lambda": "aws_lambda_cloudwatch",
            "gcp_faas": "gcp_cloud_logging",
        }.get(self.provider, self.provider)
        return RuntimeCapabilities(
            family="faas",
            requires_open_session=False,
            platform_handles_state_concurrency=False,
            supports_external_state_ops=True,
            observability=observability,
        )

    def open_session(self, target: RuntimeTarget, uid: str) -> RuntimeSession:
        return RuntimeSession(uid=uid, target_id=target.target_id)

    def _client(self, target: RuntimeTarget):
        with self._clients_lock:
            if target.target_id not in self._clients:
                from atsuite.faas.function import FunctionClient

                self._clients[target.target_id] = FunctionClient(target.endpoint)
            return self._clients[target.target_id]

    def invoke(self, request: InvocationRequest) -> InvocationResult:
        target = self._target(request.target_id)
        request_id = request.call_id or uuid.uuid4().hex
        run_id = request.run_id or uuid.uuid4().hex
        args = dict(request.args)
        if args.get("uid") not in (None, request.uid):
            raise ValueError("Invocation uid does not match Tool arguments")
        args["uid"] = request.uid
        start = time.perf_counter()
        try:
            client = self._client(target)
            if hasattr(client, "invoke_result"):
                result = client.invoke_result(
                    request.tool_name, args, request_id=request_id,
                    timeout=request.timeout, run_id=run_id,
                )
                return replace(result, provider_metadata={**result.provider_metadata, "runtime_provider": self.provider})
            provider_request_id, is_stateful = client.invoke(request.tool_name, args, request_id=request_id)
        except Exception as exc:
            return InvocationResult(
                request_id=request_id, run_id=run_id,
                status="timeout" if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower() else "error",
                error=f"{type(exc).__name__}: {exc}",
                client_elapsed_ms=(time.perf_counter() - start) * 1000,
                provider_metadata={"exception_type": type(exc).__name__},
            )
        return InvocationResult(
            provider_request_id=provider_request_id or "", request_id=request_id, run_id=run_id,
            client_elapsed_ms=(time.perf_counter() - start) * 1000,
            provider_metadata={"is_stateful": bool(is_stateful)},
        )

    def close(self):
        with self._clients_lock:
            clients=list(self._clients.values());self._clients.clear()
        for client in clients:
            client.session.close()


class MCPRuntimeAdapter(_BaseRuntimeAdapter):
    def __init__(self, provider: str):
        super().__init__(provider)
        self._clients: Dict[str, Any] = {}

    def capabilities(self) -> RuntimeCapabilities:
        observability = {
            "ali_agentrun": "ali_sls",
            "aws_agentcore": "aws_agentcore_cloudwatch",
            "gcp_mcp": "gcp_cloud_logging",
        }.get(self.provider, self.provider)
        return RuntimeCapabilities(
            family="session",
            requires_open_session=True,
            platform_handles_state_concurrency=True,
            supports_external_state_ops=False,
            observability=observability,
        )

    def _client(self, target: RuntimeTarget, uid: str):
        with self._clients_lock:
            key = f"{target.target_id}:{uid}"
            if key in self._clients:
                return self._clients[key]
            if "bedrock-agentcore" in target.endpoint or self.provider == "aws_agentcore":
                from atsuite.mcp.mcp import AgentCoreMCPClient

                client = AgentCoreMCPClient(target.endpoint)
            else:
                from atsuite.mcp.mcp import MCPClient

                client = MCPClient(target.endpoint, uid)
            self._clients[key] = client
            return client

    def open_session(self, target: RuntimeTarget, uid: str) -> RuntimeSession:
        client = self._client(target, uid)
        initialize_request_id = None
        if not getattr(client, "_initialized", False):
            try:
                initialize_request_id = client.initialize()
            except BaseException:
                self.close_session(RuntimeSession(uid=uid, target_id=target.target_id))
                raise
        return RuntimeSession(
            uid=uid,
            target_id=target.target_id,
            provider_session_id=str(getattr(client, "session_id", "") or ""),
            metadata={"initialize_request_id": initialize_request_id or ""},
        )

    def close_session(self, session: RuntimeSession) -> dict | None:
        with self._clients_lock:
            client = self._clients.pop(f"{session.target_id}:{session.uid}", None)
        if client is not None and hasattr(client, "close"):
            client.close()
            metadata = getattr(client, "last_close_metadata", None)
            if isinstance(metadata, dict):
                return {"target_id": session.target_id, **metadata}
        return None

    def cleanup_run(self, uid: str) -> list[dict]:
        receipts = []

        def release(target_id):
            receipt = self.close_session(RuntimeSession(uid=uid, target_id=target_id))
            if receipt is not None:
                receipts.append(receipt)

        with ExitStack() as stack:
            for target_id in self.targets:
                stack.callback(release, target_id)
        return receipts

    def invoke(self, request: InvocationRequest) -> InvocationResult:
        target = self._target(request.target_id)
        request_id = request.call_id or uuid.uuid4().hex
        run_id = request.run_id or uuid.uuid4().hex
        args = dict(request.args)
        if args.get("uid") not in (None, request.uid):
            raise ValueError("Invocation uid does not match Tool arguments")
        args["uid"] = request.uid
        start = time.perf_counter()
        try:
            client = self._client(target, request.uid)
            if hasattr(client, "invoke_result"):
                result = client.invoke_result(
                    request.tool_name, args, request_id=request_id,
                    timeout=request.timeout, run_id=run_id,
                )
                return replace(result, provider_metadata={**result.provider_metadata, "runtime_provider": self.provider, "session_id": str(getattr(client, "session_id", "") or "")})
            provider_request_id = client.invoke(request.tool_name, args, request_id=request_id)
        except Exception as exc:
            return InvocationResult(
                request_id=request_id, run_id=run_id,
                status="timeout" if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower() else "error",
                error=f"{type(exc).__name__}: {exc}",
                client_elapsed_ms=(time.perf_counter() - start) * 1000,
                provider_metadata={"exception_type": type(exc).__name__},
            )
        return InvocationResult(
            provider_request_id=provider_request_id or "", request_id=request_id, run_id=run_id,
            client_elapsed_ms=(time.perf_counter() - start) * 1000,
            provider_metadata={"session_id": str(getattr(client, "session_id", "") or "")},
        )


class MCPGatewayRuntimeAdapter(MCPRuntimeAdapter):
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            family="session",
            requires_open_session=True,
            platform_handles_state_concurrency=True,
            supports_external_state_ops=False,
            observability="mcp_gateway",
        )


class GatewayClient:
    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None):
        self.base_url = (base_url or os.environ.get("MCP_GATEWAY_URL", "")).rstrip("/")
        self.token = token if token is not None else os.environ.get("MCP_GATEWAY_TOKEN", "")
        if not self.base_url:
            raise RuntimeError("MCP_GATEWAY_URL is required for mcp_gateway deploy")

    def register_target(
        self,
        *,
        name: str,
        image: str,
        resources: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = requests.post(
            f"{self.base_url}/targets",
            json={
                "name": name,
                "image": image,
                "resources": dict(resources),
                "manifest": dict(manifest),
            },
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        endpoint = str(payload.get("endpoint") or payload.get("url") or "").strip()
        if not endpoint:
            raise RuntimeError("MCP-Gateway registration response is missing endpoint")
        return endpoint.rstrip("/")


def create_runtime_adapter(provider: str, family: str) -> RuntimeAdapter:
    provider_key = str(provider).strip().lower()
    family_key = str(family).strip().lower()
    if provider_key == "mcp_gateway":
        return MCPGatewayRuntimeAdapter(provider_key)
    if family_key == "faas":
        return FunctionRuntimeAdapter(provider_key)
    if family_key in {"session", "mcp_serverless"}:
        return MCPRuntimeAdapter(provider_key)
    raise ValueError(f"Unsupported runtime family: {family}")
