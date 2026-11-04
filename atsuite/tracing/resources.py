"""Fallback observations for ordinary in-process tools; Runtime workers take precedence."""

import os
import resource
import sys
import time
from pathlib import Path


def resource_start() -> float:
    """Sample the harness process CPU clock, including its other active calls."""
    return time.process_time()


def resource_end(started: float) -> dict:
    """Report process-wide observations without claiming per-tool CPU attribution."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "cpu_seconds": max(0.0, time.process_time() - started),
        "cpu_scope": "harness_process_during_tool",
        "cpu_complete": False,
        "cpu_interval": "call_delta",
        "includes_children": False,
        "includes_other_threads": True,
        "includes_concurrent_calls": True,
        "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "rss_scope": "harness process lifetime high-water mark",
        "pid": os.getpid(),
    }
    try:
        result["rss_bytes"] = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        result["rss_bytes"] = None
    return result
