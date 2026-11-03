import importlib
import json
import os
import time
import uuid
import traceback
from datetime import datetime, timezone

import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Type, Optional
from urllib.parse import urlparse

from atsuite_sdk.abstract import registry
from atsuite_sdk.events import emit_event,runtime_metadata
from atsuite_sdk.state import get_state_runtime

module_name = os.environ.get("ATSUITE_NODE_MODULE", "implementation")
importlib.import_module(module_name)

TOOLS = registry.functions


def _now_ns() -> int:
    return time.perf_counter_ns()

def _wall_ns() -> int:
    return time.time_ns()


def _ns_to_ms(delta_ns: int) -> float:
    return round(delta_ns / 1_000_000, 3)


def _wall_from_monotonic_delta(ctx: Dict[str, object], event_ns: int | None) -> int | None:
    request_wall_ns = ctx.get("request_wall_ns")
    request_start_ns = ctx.get("request_start_ns")
    if request_wall_ns is None or request_start_ns is None or event_ns is None:
        return None
    return int(request_wall_ns) + max(0, int(event_ns) - int(request_start_ns))


def _extract_xray_root_trace_id() -> str:
    header = os.environ.get("_X_AMZN_TRACE_ID", "").strip()
    if not header:
        return ""
    for part in header.split(";"):
        key, sep, value = part.partition("=")
        if sep and key.strip() == "Root":
            return value.strip()
    return ""


def _build_function_breakdown_payload(
    ctx: Dict[str, object] | None,
    *,
    request_end_ns: int,
    status: int,
) -> Dict[str, object] | None:
    if not ctx:
        return None
    request_start_ns = ctx.get("request_start_ns")
    tool_start_ns = ctx.get("tool_start_ns")
    tool_end_ns = ctx.get("tool_end_ns")
    if request_start_ns is None or tool_start_ns is None or tool_end_ns is None:
        return None

    app_e2e_ms = _ns_to_ms(request_end_ns - int(request_start_ns))
    tool_exec_ms = _ns_to_ms(int(tool_end_ns) - int(tool_start_ns))
    pre_tool_ms = _ns_to_ms(int(tool_start_ns) - int(request_start_ns))
    post_tool_ms = _ns_to_ms(request_end_ns - int(tool_end_ns))
    framework_overhead_ms = round(max(0.0, app_e2e_ms - tool_exec_ms), 3)
    state_sync_overhead_ms = round(
        float(ctx.get("state_sync_overhead_ms") or 0.0),
        3,
    )

    request_wall_ns = ctx.get("request_wall_ns")
    
    return {
        "event": "atsuite_function_breakdown",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_ns": request_end_ns,
        "run_id": str(ctx.get("run_id") or ""),
        "request_id": str(ctx.get("request_id") or ""),
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
        "trace_id": _extract_xray_root_trace_id(),
        "service_name": os.environ.get(
            "AWS_LAMBDA_FUNCTION_NAME",
            os.environ.get("K_SERVICE", "atsuite-function"),
        ),
        "timestamp_ms": int(time.time() * 1000),
    }


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


_ALLOWED_TOOLS = _load_allowed_tools()
if _ALLOWED_TOOLS is not None:
    TOOLS = {name: tool for name, tool in TOOLS.items() if name in _ALLOWED_TOOLS}


def create_handler() -> Type[BaseHTTPRequestHandler]:
    class ToolHandler(BaseHTTPRequestHandler):
        server_version = "ATSUITEFUNCTION/1.0"

        def log_message(self, format, *args):
            return

        def _clean_path(self) -> str:
            """规范化请求路径，去除访问时的查询参数和尾部斜杠，解决aws lambda web adapter 的请求路径问题"""
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")
            return path if path else "/"

        def _send_json(self, status: int, payload: Dict, custom_headers: Optional[Dict] = None):
            request_id = getattr(self, "request_id", "")
            run_id = getattr(self, "run_id", "")
            payload.setdefault("request_id", request_id)
            payload.setdefault("run_id", run_id)
            if status >= 400:
                emit_event("function_error", request_id=request_id, run_id=run_id, http_status=status, error=payload.get("error"))
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if custom_headers:
                for key, value in custom_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self._clean_path()
            if path in ("/", "/health"):
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"ok")
                return

            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            self.request_id = self.headers.get("X-Request-Id", "").strip() or uuid.uuid4().hex
            self.run_id = self.headers.get("X-Run-Id", "").strip() or uuid.uuid4().hex
            path = self._clean_path()
            if path not in ("/run", "/invoke"):
                self._send_json(404, {"error": f"not found (path={self.path!r})"})
                return

            request_id, run_id = self.request_id, self.run_id
            trace_header = self.headers.get("X-Cloud-Trace-Context", "").strip()
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            if request_id and project_id:
                trace_full = ""
                if trace_header:
                    part = trace_header.split(";")[0].strip().split("/")[0].strip()
                    if part:
                        trace_full = f"projects/{project_id}/traces/{part}"
                print(json.dumps({"request_id": request_id, "trace": trace_full, "message": "atsuite_request"}, ensure_ascii=False), flush=True)

            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0:
                    raise ValueError("negative Content-Length")
                raw = self.rfile.read(length).decode("utf-8") if length else ""
                payload = json.loads(raw) if raw else {}
            except (ValueError, UnicodeDecodeError):
                self._send_json(400, {"error": "invalid JSON body or Content-Length"})
                return

            if not isinstance(payload, dict) or not isinstance(payload.get("args", {}), dict):
                self._send_json(400, {"error": "payload and args must be objects", "request_id": request_id, "run_id": run_id})
                return
            tool_name = payload.get("tool")
            args = payload.get("args", {})

            if not isinstance(tool_name, str) or not tool_name:
                self._send_json(400, {"error": "tool name must be a nonempty string"})
                return
            
            tool_obj = TOOLS.get(tool_name)
            if not tool_obj:
                self._send_json(404, {"error": f"Tool '{tool_name}' not found"})
                return
            is_stateful = str(tool_obj.stateful).lower()

            request_ctx: Dict[str, object] = {
                "request_id": request_id,
                "run_id": run_id,
                "tool_name": str(tool_name),
                "request_start_ns": _now_ns(),
                "request_wall_ns": _wall_ns(),
            }
            status = 200
            native_request_id = None
            try:
                native_context = json.loads(self.headers.get("x-amzn-lambda-context", "{}"))
                if isinstance(native_context, dict):
                    native_request_id = native_context.get("request_id")
            except (ValueError, TypeError):
                pass
            response_metadata = {"native_request_id": native_request_id, "runtime": runtime_metadata()}
            try:
                request_ctx["tool_start_ns"] = _now_ns()
                try:
                    result = TOOLS[tool_name].execute_from_dict(args)
                finally:
                    from atsuite_sdk.resources import get_resource_usage
                    response_metadata["resource_usage"] = get_resource_usage()
                    request_ctx["tool_end_ns"] = _now_ns()
                    request_ctx["state_sync_overhead_ms"] = (
                        get_state_runtime().get_sync_metrics().get("total_ms", 0.0)
                    )
                if hasattr(result, 'to_dict'):
                    result = result.to_dict(orient="records")
            except Exception as e:
                status = 500
                breakdown = _build_function_breakdown_payload(
                    request_ctx,
                    request_end_ns=_now_ns(),
                    status=status,
                )
                if breakdown is not None:
                    print(json.dumps(breakdown, ensure_ascii=False), flush=True)
                elapsed_ms = _ns_to_ms(int(request_ctx["tool_end_ns"]) - int(request_ctx["tool_start_ns"]))
                emit_event("tool_execution", run_id=run_id, request_id=request_id, uid=args.get("uid"),
                           tool_name=tool_name, status="timeout" if isinstance(e, TimeoutError) else "error",
                           duration_ms=elapsed_ms, error=str(e), traceback=traceback.format_exc())
                self._send_json(500, {"status": "timeout" if isinstance(e, TimeoutError) else "error", "error": str(e),
                                     "result": getattr(e, "result", None), "request_id": request_id, "run_id": run_id,
                                     "execution_elapsed_ms": elapsed_ms, "metadata": {**response_metadata, "state_sync": get_state_runtime().get_sync_metrics()}}, custom_headers={"X-Request-Id": request_id})
                return

            breakdown = _build_function_breakdown_payload(
                request_ctx,
                request_end_ns=_now_ns(),
                status=status,
            )
            if breakdown is not None:
                print(json.dumps(breakdown, ensure_ascii=False), flush=True)
            elapsed_ms = _ns_to_ms(int(request_ctx["tool_end_ns"]) - int(request_ctx["tool_start_ns"]))
            emit_event("tool_execution", run_id=run_id, request_id=request_id, uid=args.get("uid"),
                       tool_name=tool_name, status="ok", duration_ms=elapsed_ms)
            self._send_json(200, {"status": "ok", "result": result, "error": "", "request_id": request_id,
                                 "run_id": run_id, "execution_elapsed_ms": elapsed_ms, "metadata": {**response_metadata, "state_sync": get_state_runtime().get_sync_metrics()}},
                            custom_headers={"X-Tool-Stateful": is_stateful, "X-Request-Id": request_id})

    return ToolHandler


def main():
    host = os.environ.get("FC_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("FC_SERVER_PORT", "9000"))
    handler = create_handler()
    server = ThreadingHTTPServer((host, port), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
