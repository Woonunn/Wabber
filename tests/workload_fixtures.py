"""Explicit synthetic identities, never evidence about deployed implementations."""

from atsuite.workload.descriptor import describe as build_context


def describe(*, tools=(), **kwargs):
    """Give test tools a pinned implementation and test Harness version."""
    kwargs.setdefault("harness", {"id": "test-harness", "revision": "synthetic-v1"})
    kwargs.setdefault("model", {"name": "synthetic-model", "revision": "v1"})
    kwargs.setdefault("control", {"max_iterations": 10})
    return build_context(
        tools=[
            {
                "name": t,
                "definition": {"name": t, "parameters": {}},
                "implementation": {"id": "test:" + t, "revision": "synthetic-v1"},
            }
            if isinstance(t, str)
            else t
            for t in tools
        ],
        **kwargs,
    )


def stable_prediction(prediction):
    """Elapsed prediction time is measured anew, not part of prediction equivalence."""
    import copy

    result = copy.deepcopy(prediction)
    result.get("retrieval", {}).pop("latency_s", None)
    return result


def calibrate(model, descriptor, *, available_ns=1):
    """Synthetic independent residuals for tests exercising calibrated routing."""
    model.errors = [
        {
            "request_id": "calibration:" + str(i),
            "available_ns": available_ns,
            "descriptor": descriptor,
            "revision": model.retriever.revision,
            "metrics": {
                k: 0.1 for k in ("tool_elapsed_s", "state_bytes", "llm_idle_s")
            },
        }
        for i in range(10)
    ]
