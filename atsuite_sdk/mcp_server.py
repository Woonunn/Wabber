import importlib
import uuid
import traceback
from datetime import datetime, timezone
from fastmcp.tools.tool import ToolResult
from atsuite_sdk.events import emit_event,runtime_metadata
import json
import os
import sys
import inspect
import time

from contextvars import ContextVar
from contextlib import asynccontextmanager
from functools import wraps
from fastmcp import FastMCP
from pathlib import Path
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.concurrency import run_in_threadpool
import uvicorn

from atsuite_sdk.abstract import registry
from atsuite_sdk.state import get_state_runtime

_REQUEST_BREAKDOWN: ContextVar[dict | None] = ContextVar(
    "atsuite_mcp_request_breakdown", default=None
)
_REQUEST_BREAKDOWN_STATE_KEY = "atsuite_mcp_request_breakdown"

_concurrent_request_threshold_ms = 15
_completed_requests: list[dict] = []


def _now_ns() -> int:
    return time.perf_counter_ns()


def _wall_ns() -> int:
    return time.time_ns()


def _ns_to_ms(delta_ns: int) -> float:
    return round(delta_ns / 1_000_000, 3)


def _wall_from_monotonic_delta(ctx: dict, event_ns: int | None) -> int | None:
    request_wall_ns = ctx.get("request_wall_ns")
    request_start_ns = ctx.get("request_start_ns")
    if request_wall_ns is None or request_start_ns is None or event_ns is None:
        return None
    return int(request_wall_ns) + max(0, int(event_ns) - int(request_start_ns))


def _extract_jsonrpc_info(body: bytes) -> tuple[str, str]:
    jsonrpc_id = ""
    method = ""
    if not body:
        return jsonrpc_id, method
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return jsonrpc_id, method
    if not isinstance(payload, dict):
        return jsonrpc_id, method
    value = payload.get("id")
    if value is not None:
        jsonrpc_id = str(value)
    method = str(payload.get("method") or "")
    return jsonrpc_id, method


def _build_mcp_breakdown_payload(
    ctx: dict | None,
    *,
    request_end_ns: int,
    status: int,
) -> dict | None:
    if not ctx:
        return None
    request_start_ns = ctx.get("request_start_ns")
    tool_start_ns = ctx.get("tool_start_ns")
    tool_end_ns = ctx.get("tool_end_ns")
    if request_start_ns is None:
        return None

    app_e2e_ms = _ns_to_ms(request_end_ns - int(request_start_ns))

    global _completed_requests
    threshold_ns = _concurrent_request_threshold_ms * 1_000_000
    tool_exec_ms = 0.0
    if tool_start_ns is not None and tool_end_ns is not None:
        tool_exec_ms = _ns_to_ms(int(tool_end_ns) - int(tool_start_ns))
        last_tool_end_ns = 0
        for req in _completed_requests:
            prev_start = req.get("request_start_ns", 0)
            prev_end = req.get("tool_end_ns", 0)
            if abs(int(request_start_ns) - int(prev_start)) < threshold_ns:
                last_tool_end_ns = max(last_tool_end_ns, int(prev_end))
        if last_tool_end_ns > 0:
            app_e2e_ms = max(0.1, _ns_to_ms(int(tool_end_ns) - last_tool_end_ns))
    method = ctx.get("jsonrpc_method", "")
    request_wall_ns = ctx.get("request_wall_ns")

    if method == "initialize":
        return {
            "event": "atsuite_mcp_initialize",
            "request_id": str(ctx.get("request_id") or ""),
            "jsonrpc_id": str(ctx.get("jsonrpc_id") or ""),
            "method": method,
            "status": int(status),
            "app_e2e_ms": app_e2e_ms,
            "tool_exec_ms": 0.0,
            "state_sync_overhead_ms": 0.0,
            "framework_overhead_ms": app_e2e_ms,
            "pre_tool_ms": app_e2e_ms,
            "post_tool_ms": 0.0,
            "request_wall_ns": request_wall_ns,
            "request_start_wall_ns": _wall_from_monotonic_delta(ctx, int(request_start_ns)),
            "request_end_wall_ns": _wall_from_monotonic_delta(ctx, request_end_ns),
            "service_name": os.environ.get(
                "K_SERVICE", os.environ.get("ATSUITE_MCP_NAME", "atsuite-mcp")
            ),
            "timestamp_ms": int(time.time() * 1000),
        }

    if tool_start_ns is None or tool_end_ns is None:
        return None

    tool_exec_ms = _ns_to_ms(int(tool_end_ns) - int(tool_start_ns))
    pre_tool_ms = _ns_to_ms(int(tool_start_ns) - int(request_start_ns))
    post_tool_ms = _ns_to_ms(request_end_ns - int(tool_end_ns))
    framework_overhead_ms = round(max(0.0, app_e2e_ms - tool_exec_ms), 3)
    state_sync_overhead_ms = round(
        float(ctx.get("state_sync_overhead_ms") or 0.0),
        3,
    )
    
    return {
        "event": "atsuite_mcp_breakdown",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_ns": request_end_ns,
        "run_id": str(ctx.get("run_id") or ""),
        "request_id": str(ctx.get("request_id") or ""),
        "jsonrpc_id": str(ctx.get("jsonrpc_id") or ""),
        "tool_name": str(ctx.get("tool_name") or ""),
        "status": int(status),
        "app_e2e_ms": app_e2e_ms,
        "tool_exec_ms": tool_exec_ms,
        "state_sync_overhead_ms": state_sync_overhead_ms,
        "framework_overhead_ms": framework_overhead_ms,
        "request_wall_ns": request_wall_ns,
        "request_start_wall_ns": _wall_from_monotonic_delta(ctx, int(request_start_ns)),
        "request_end_wall_ns": _wall_from_monotonic_delta(ctx, request_end_ns),
        "tool_start_wall_ns": _wall_from_monotonic_delta(ctx, int(tool_start_ns)),
        "tool_end_wall_ns": _wall_from_monotonic_delta(ctx, int(tool_end_ns)),
        "pre_tool_ms": pre_tool_ms,
        "post_tool_ms": post_tool_ms,
        "service_name": os.environ.get(
            "K_SERVICE", os.environ.get("ATSUITE_MCP_NAME", "atsuite-mcp")
        ),
        "timestamp_ms": int(time.time() * 1000),
    }


def _set_request_breakdown_state(request: Request, ctx: dict) -> None:
    try:
        setattr(request.state, _REQUEST_BREAKDOWN_STATE_KEY, ctx)
    except Exception:
        return


def _get_request_breakdown_state(request: object | None) -> dict | None:
    if request is None:
        return None
    try:
        value = getattr(request.state, _REQUEST_BREAKDOWN_STATE_KEY, None)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _get_active_request_breakdown() -> dict | None:
    try:
        from fastmcp.server.dependencies import get_http_request

        request = get_http_request()
    except Exception:
        request = None
    request_ctx = _get_request_breakdown_state(request)
    if isinstance(request_ctx, dict):
        return request_ctx
    ctx = _REQUEST_BREAKDOWN.get()
    if isinstance(ctx, dict):
        return ctx
    return None


def _emit_mcp_breakdown(ctx: dict | None, *, request_end_ns: int, status: int) -> None:
    global _completed_requests
    if not ctx or ctx.get("breakdown_emitted"):
        return
    breakdown = _build_mcp_breakdown_payload(
        ctx,
        request_end_ns=request_end_ns,
        status=status,
    )
    if breakdown is None:
        return
    ctx["breakdown_emitted"] = True
    print(json.dumps(breakdown, ensure_ascii=False), flush=True)

    request_start_ns = ctx.get("request_start_ns")
    tool_start_ns = ctx.get("tool_start_ns")
    tool_end_ns = ctx.get("tool_end_ns")
    if request_start_ns is not None and tool_start_ns is not None and tool_end_ns is not None:
        _completed_requests.append({
            "request_start_ns": request_start_ns,
            "tool_end_ns": tool_end_ns,
            "tool_exec_ms": breakdown.get("tool_exec_ms", 0.0),
        })
        if len(_completed_requests) > 100:
            _completed_requests = _completed_requests[-50:]


MCP_DIR = Path(__file__).resolve().parent.parent / "mcp"
CURRENT_DIR = Path.cwd()
if MCP_DIR.exists():
    for tool_dir in sorted(MCP_DIR.iterdir()):
        if tool_dir.is_dir() and (tool_dir / "implementation.py").exists():
            prev_cwd = os.getcwd()
            os.chdir(tool_dir)

            sys.path.insert(0, str(tool_dir))
            mod_name = f"{tool_dir.name}_implementation"
            spec = importlib.util.spec_from_file_location(
                mod_name, tool_dir / "implementation.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            sys.path.pop(0)

            os.chdir(prev_cwd)

@asynccontextmanager
async def runtime_lifespan(server):
    emit_event("runtime_started")
    try:
        yield {}
    finally:
        emit_event("runtime_stopped")


server = FastMCP(name=os.environ.get("ATSUITE_MCP_NAME", "atsuite-mcp"),lifespan=runtime_lifespan)


def _load_allowed_tools() -> set[str] | None:
    manifest_path = os.environ.get("ATSUITE_MANIFEST_PATH", "/app/atsuite-manifest.json")
    if not manifest_path:
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    allowed = payload.get("allowed_tools")
    if not isinstance(allowed, list):
        return None
    return {str(item).strip() for item in allowed if str(item).strip()}


ALLOWED_TOOLS = _load_allowed_tools()


class MCPRequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("x-request-id", "") or uuid.uuid4().hex
        run_id = request.headers.get("x-run-id", "") or uuid.uuid4().hex
        body = await request.body()
        jsonrpc_id, jsonrpc_method = _extract_jsonrpc_info(body)

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)
        request_start_ns = _now_ns()
        token = _REQUEST_BREAKDOWN.set(
            {
                "request_id": request_id,
                "run_id": run_id,
                "jsonrpc_id": jsonrpc_id,
                "jsonrpc_method": jsonrpc_method,
                "request_start_ns": request_start_ns,
                "request_wall_ns": _wall_ns(),
            }
        )
        ctx = _REQUEST_BREAKDOWN.get()
        if isinstance(ctx, dict):
            _set_request_breakdown_state(request, ctx)
        response = None
        try:
            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response
        finally:
            request_end_ns = _now_ns()
            duration_ms = _ns_to_ms(request_end_ns - request_start_ns)
            payload = {
                "event": "atsuite_mcp_request",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_ns": request_end_ns,
                "request_id": request_id,
                "run_id": run_id,
                "jsonrpc_id": jsonrpc_id,
                "jsonrpc_method": jsonrpc_method,
                "method": request.method,
                "path": request.url.path,
                "status": int(response.status_code) if response is not None else 500,
                "duration_ms": round(duration_ms, 3),
                "request_start_wall_ns": _wall_from_monotonic_delta(
                    _get_active_request_breakdown() or {},
                    request_start_ns,
                ),
                "request_end_wall_ns": _wall_from_monotonic_delta(
                    _get_active_request_breakdown() or {},
                    request_end_ns,
                ),
                "service_name": os.environ.get(
                    "K_SERVICE", os.environ.get("ATSUITE_MCP_NAME", "atsuite-mcp")
                ),
                "timestamp_ms": int(time.time() * 1000),
            }
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            _emit_mcp_breakdown(
                _get_active_request_breakdown(),
                request_end_ns=request_end_ns,
                status=int(response.status_code) if response is not None else 500,
            )
            _REQUEST_BREAKDOWN.reset(token)


class FailedToolResult(ToolResult):
    def to_mcp_result(self):
        result = super().to_mcp_result()
        result.isError = True
        return result


def wrap_tool(tool_obj):
    func = tool_obj.func
    tool_name = getattr(tool_obj, "name", func.__name__)

    @wraps(func)
    async def handler(**kwargs):
        call_args = dict(kwargs)
        ctx = _get_active_request_breakdown()
        if ctx is not None:
            ctx["tool_name"] = tool_name
            ctx["tool_start_ns"] = _now_ns()
        started = _now_ns()
        request_id = str((ctx or {}).get("request_id") or uuid.uuid4().hex)
        run_id = str((ctx or {}).get("run_id") or uuid.uuid4().hex)
        metrics = {}
        resources = {}
        logical_state = {}
        def execute():
            try:
                return tool_obj.execute_from_dict(call_args)
            finally:
                from atsuite_sdk.resources import get_resource_usage
                resources.update(get_resource_usage())
                metrics.update(get_state_runtime().get_sync_metrics())
                logical_state.update(get_state_runtime().get_logical_state())
        try:
            result = await run_in_threadpool(execute)
            elapsed_ms = _ns_to_ms(_now_ns() - started)
            emit_event("tool_execution", run_id=run_id, request_id=request_id, uid=call_args.get("uid"),
                       tool_name=tool_name, status="ok", duration_ms=elapsed_ms)
            return ToolResult(content=result, structured_content={"result": result}, meta={"atsuite": {
                "result_envelope": True, "request_id": request_id, "run_id": run_id,
                "execution_elapsed_ms": elapsed_ms, "state_sync": metrics, "runtime": runtime_metadata(),
                "resource_usage": resources, "logical_state": logical_state,
            }})
        except Exception as exc:
            emit_event("tool_execution", run_id=run_id, request_id=request_id, uid=call_args.get("uid"),
                       tool_name=tool_name, status="timeout" if isinstance(exc, TimeoutError) else "error",
                       duration_ms=_ns_to_ms(_now_ns() - started), error=str(exc), traceback=traceback.format_exc())
            return FailedToolResult(content=str(exc), structured_content={"result": getattr(exc, "result", None)},
                                    meta={"atsuite": {"result_envelope": True, "request_id": request_id, "run_id": run_id,
                                                      "status": "timeout" if isinstance(exc, TimeoutError) else "error",
                                                      "execution_elapsed_ms": _ns_to_ms(_now_ns() - started), "state_sync": metrics, "runtime": runtime_metadata(), "resource_usage": resources, "logical_state": logical_state}})
        finally:
            if ctx is not None:
                ctx["tool_end_ns"] = _now_ns()
                ctx["state_sync_overhead_ms"] = metrics.get("total_ms")

    sig_params = list(inspect.signature(func).parameters.values())
    if "uid" not in [p.name for p in sig_params]:
        sig_params.append(
            inspect.Parameter("uid", inspect.Parameter.KEYWORD_ONLY, default=None)
        )
    if "__atsuite_state_snapshot" not in [p.name for p in sig_params]:
        sig_params.append(
            inspect.Parameter(
                "__atsuite_state_snapshot",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
            )
        )
    handler.__signature__ = inspect.Signature(sig_params)
    return handler


for tool in registry.functions.values():
    if ALLOWED_TOOLS is not None and tool.name not in ALLOWED_TOOLS:
        continue
    server.tool(wrap_tool(tool))


if __name__ == "__main__":
    # ATSUITE_MCP_STATELESS: 无状态 HTTP
    #   - AgentCore 部署必须设为 true（agentcore 自身维护状态）
    stateless = os.environ.get("ATSUITE_MCP_STATELESS", "false").lower() in ("true", "1")

    provider = os.environ.get("PROVIDER", "").strip().lower()
    transport = os.environ.get("ATSUITE_MCP_TRANSPORT", "http")
    host = "0.0.0.0"
    port = int(os.environ.get("ATSUITE_MCP_PORT", "8000"))
    path = os.environ.get("ATSUITE_MCP_PATH", "/mcp")
    if transport in ("http", "streamable-http", "sse"):
        json_response = None
        if provider == "aws_agentcore" and transport in ("http", "streamable-http"):
            json_response = True
        app = server.http_app(
            path=path,
            json_response=json_response,
            stateless_http=stateless,
            transport=transport,
            middleware=[Middleware(MCPRequestLogMiddleware)],
        )
        uvicorn.run(app, host=host, port=port)
    else:
        server.run(
            transport=transport,
            host=host,
            port=port,
            path=path,
            stateless_http=stateless,
        )
