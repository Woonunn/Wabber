"""Numerical agent using the existing session tensor workspace."""

import json
import math

import numpy as np

from experiments.e3.agent import build_graph as build_tool_graph
from experiments.e3.agent import result_view


def numeric_equal(left, right):
    """Accept roundoff while keeping shape, metadata, strings and flags exact."""

    def decode(value):
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except ValueError:
                pass
        return value

    left, right = decode(left), decode(right)
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, int) and isinstance(right, int):
            return left == right
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(numeric_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(numeric_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


async def validate(task, answer, tools):
    """Check stored results and the reported solution with an independent linear solve."""
    verification = task["verification"]
    if "input_npz" in verification:
        with np.load(verification["input_npz"], allow_pickle=False) as inputs:
            matrix, rhs = inputs["A"], inputs["B"]
    else:
        matrix = np.asarray(verification["matrix"], dtype=float)
        rhs = np.asarray(verification["rhs"], dtype=float)
    reference = np.linalg.solve(matrix, rhs)
    stored = result_view(
        await tools.call("scicom_view_tensor", {"name": answer.get("solution_tensor", "")}, "verify-solution")
    )
    try:
        actual = np.asarray(stored["result"], dtype=float)
        reported = np.asarray(answer["solution"], dtype=float) if task.get("report_values", True) else actual
        ok = (
            not stored["isError"]
            and actual.shape == reference.shape == reported.shape
            and np.allclose(actual, reference, rtol=1e-7, atol=1e-9)
            and np.allclose(reported, reference, rtol=1e-7, atol=1e-9)
            and np.allclose(matrix @ actual, rhs, rtol=1e-7, atol=1e-9)
        )
    except (KeyError, TypeError, ValueError):
        ok = False
    return {"ok": bool(ok), "basis": "independent numpy.linalg.solve and residual; stored tensor readback"}


def build_graph(model, tools, task, max_iterations=40):
    return build_tool_graph(
        model,
        tools,
        task,
        "Solve the scientific problem with the supplied tools. Retain and reuse tensors. "
        + (
            'Report {"solution_tensor":"stored name", "solution":[[...],...]} as answer.'
            if task.get("report_values", True)
            else "Load the supplied problem inputs with scicom_load_tensor. Use summary_only=true and output_name for "
            'stored matrix operations to avoid printing large tensors. Report {"solution_tensor":"stored name"}; '
            "the actual stored tensor is independently checked, so do not copy numeric arrays into the answer."
        ),
        validate,
        max_iterations,
    )
