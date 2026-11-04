"""Propagate request-local Runtime observations without instrumenting business tools."""

from contextvars import ContextVar
from dataclasses import replace
from functools import wraps

active_adapter = ContextVar("wabber_tracing_adapter", default=None)


def observe_invocation(invoke):
    """Associate a Runtime response with the native tool span in the calling context."""

    @wraps(invoke)
    def observed(runtime, request):
        adapter = active_adapter.get()
        if adapter is None:
            return invoke(runtime, request)
        if not request.run_id:
            request = replace(request, run_id=adapter.collector.request_id)
        with adapter.runtime_call(request) as span_id:
            result = invoke(runtime, request)
            transport = result.to_dict()
            if not adapter.include_data:
                raw = transport.get("result")
                raw = raw if isinstance(raw, dict) else {}
                worker = raw.get("worker_usage") or raw.get("kernel_execution", {}).get("worker_usage")
                transport["result"] = {"worker_usage": worker} if worker else None
                transport["provider_metadata"] = {
                    key: value
                    for key, value in transport["provider_metadata"].items()
                    if key
                    in {
                        "resource_usage",
                        "state_sync",
                        "runtime_provider",
                        "session_id",
                    }
                }
            adapter.collector.observe_runtime(span_id, transport)
            return result

    return observed
