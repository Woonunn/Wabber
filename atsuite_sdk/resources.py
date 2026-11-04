"""Call-scoped CPU observations; never infer process CPU from elapsed time."""

from contextlib import contextmanager
from contextvars import ContextVar
import copy
import os
import resource
import sys
import threading
import time

_USAGE = ContextVar("atsuite_resource_usage", default=None)
_ACTIVE = ContextVar("atsuite_cpu_call", default=None)


def get_resource_usage() -> dict:
    """Read the last call in this execution context, including failed calls."""
    return copy.deepcopy(_USAGE.get() or {})


@contextmanager
def observe_call():
    """Measure the calling thread only, safely across concurrent MCP calls.

    Child processes, executor threads and native library threads are excluded.
    This is a partial observation, never a Tool total or provider billing meter.
    """
    _USAGE.set({})
    parent = _ACTIVE.get()
    active = {"nested_cpu": 0.0, "thread_id": threading.get_ident()}
    token = _ACTIVE.set(active)
    clock = getattr(time, "thread_time", None)
    try:
        started = clock() if clock else None
    except (OSError, ValueError):
        started = None
    try:
        yield
    finally:
        try:
            seconds = clock() - started if started is not None else None
        except (OSError, ValueError):
            seconds = None
        _ACTIVE.reset(token)
        if parent is not None and parent["thread_id"] == active["thread_id"]:
            if seconds is None:
                parent["nested_cpu"] = None
            elif parent["nested_cpu"] is not None:
                parent["nested_cpu"] += seconds
        if seconds is not None:
            seconds = max(0.0, seconds - active["nested_cpu"]) if active["nested_cpu"] is not None else None
        _USAGE.set({"cpu_seconds": seconds, "cpu_scope": "runtime_calling_thread",
                    "cpu_complete": False, "cpu_interval": "call_delta",
                    "includes_children": False, "includes_other_threads": False,
                    "includes_nested_tool_calls": False,
                    "pid": os.getpid(), "thread_id": threading.get_ident(),
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                        1 if sys.platform == "darwin" else 1024),
                    "rss_scope": "runtime process lifetime high-water mark; shared by concurrent calls"})
