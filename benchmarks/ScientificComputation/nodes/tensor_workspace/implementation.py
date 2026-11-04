import json
import os
import re
import time
from pathlib import Path
from typing import Annotated, Any, Optional

import numpy as np
from pydantic import Field

from atsuite_sdk.abstract import registry
from atsuite_sdk.events import emit_event
from atsuite_sdk.state import register_state_object


class TensorWorkspace:
    def __init__(self) -> None:
        self.tensors: dict[str, list[Any]] = {}

    def _arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(value) for name, value in self.tensors.items()}

    def has(self, name: str) -> bool:
        return name in self.tensors

    def names(self) -> list[str]:
        return list(self.tensors.keys())

    def load(self, name: str) -> np.ndarray:
        if name not in self.tensors:
            raise ValueError("The tensor name is not found in the store.")
        return np.asarray(self.tensors[name])

    def save(self, name: str, value: Any) -> None:
        self.tensors[name] = np.asarray(value).tolist()

    def delete(self, name: str) -> None:
        if name not in self.tensors:
            raise ValueError(f"Tensor '{name}' not found in the store.")
        self.tensors.pop(name)


tensor_workspace = register_state_object("tensor_workspace", TensorWorkspace(), session_scoped=True)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _success(result: Any, **extra: Any) -> str:
    payload = {"isError": False, "result": _to_jsonable(result)}
    payload.update({key: _to_jsonable(value) for key, value in extra.items()})
    return json.dumps(payload, ensure_ascii=False)


def _error(tool: str, exc: Exception, **extra: Any) -> str:
    payload = {"isError": True, "error": str(exc), "tool": tool}
    payload.update({key: _to_jsonable(value) for key, value in extra.items()})
    return json.dumps(payload, ensure_ascii=False)


def _store_tensor_if_requested(output_name: Optional[str], value: Any) -> None:
    if output_name:
        tensor_workspace.save(output_name, value)


def _store_named_tensors(outputs: dict[str, Any]) -> None:
    for output_name, value in outputs.items():
        _store_tensor_if_requested(output_name, value)


@registry.tool(stateful=True)
def scicom_create_tensor(
    shape: Annotated[
        list[int], Field(min_length=1, description="Tensor shape as list of integers")
    ],
    values: Annotated[
        list[float],
        Field(min_length=1, description="Flat list of floats to fill the tensor"),
    ],
    name: str,
    uid: Optional[str] = None,
) -> str:
    try:
        normalized_shape = [int(x) for x in shape]
        normalized_values = [float(x) for x in values]
        if len(normalized_values) != int(np.prod(normalized_shape)):
            raise ValueError("Shape does not match number of values.")
        tensor = np.array(normalized_values, dtype=float).reshape(normalized_shape)
        tensor_workspace.save(name, tensor)
        return _success(tensor, tensor_name=name)
    except Exception as exc:
        return _error("scicom_create_tensor", exc, tensor_name=name)


@registry.tool(access="r")
def scicom_view_tensor(name: str, uid: Optional[str] = None) -> str:
    try:
        return _success(tensor_workspace.load(name), tensor_name=name)
    except Exception as exc:
        return _error("scicom_view_tensor", exc, tensor_name=name)


@registry.tool(stateful=True)
def scicom_load_tensor(problem: str, tensor: str, name: str, uid: Optional[str] = None) -> str:
    """Load A or B from an immutable problem input into the tensor workspace, returning its shape."""
    try:
        if not re.fullmatch(r"[a-z0-9-]+", problem) or tensor not in {"A", "B"}:
            raise ValueError("Unknown problem input")
        root = Path(os.environ["ATSUITE_SCIENTIFIC_PROBLEMS"]).resolve()
        path = (root / (problem + ".json")).resolve()
        path.relative_to(root)
        if path.exists():
            value = np.asarray(json.loads(path.read_text())[tensor], dtype=float)
        else:
            path = (root / (problem + ".npz")).resolve()
            path.relative_to(root)
            with np.load(path, allow_pickle=False) as inputs:
                value = np.asarray(inputs[tensor], dtype=float)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError("Problem input must be a finite matrix")
        tensor_workspace.save(name, value)
        return _success({"tensor_name": name, "shape": list(value.shape)})
    except Exception as exc:
        return _error("scicom_load_tensor", exc)


@registry.tool(access="r")
def scicom_list_tensor_names(uid: Optional[str] = None) -> str:
    try:
        tensor_names = tensor_workspace.names()
        return _success(tensor_names, count=len(tensor_names))
    except Exception as exc:
        return _error("scicom_list_tensor_names", exc)


@registry.tool(stateful=True)
def scicom_delete_tensor(name: str, uid: Optional[str] = None) -> str:
    try:
        tensor_workspace.delete(name)
        return _success(f"Tensor '{name}' deleted successfully.", tensor_name=name)
    except Exception as exc:
        return _error("scicom_delete_tensor", exc, tensor_name=name)


@registry.tool(stateful=True)
def scicom_add_matrices(
    name_a: str,
    name_b: str,
    output_name: Optional[str] = None,
    uid: Optional[str] = None,
) -> str:
    try:
        result = np.add(tensor_workspace.load(name_a), tensor_workspace.load(name_b))
        _store_tensor_if_requested(output_name, result)
        return _success(result, tool="scicom_add_matrices")
    except Exception as exc:
        return _error("scicom_add_matrices", exc)


@registry.tool(stateful=True)
def scicom_subtract_matrices(
    name_a: str,
    name_b: str,
    output_name: Optional[str] = None,
    uid: Optional[str] = None,
) -> str:
    try:
        result = np.subtract(
            tensor_workspace.load(name_a), tensor_workspace.load(name_b)
        )
        _store_tensor_if_requested(output_name, result)
        return _success(result, tool="scicom_subtract_matrices")
    except Exception as exc:
        return _error("scicom_subtract_matrices", exc)


@registry.tool(stateful=True)
def scicom_multiply_matrices(
    name_a: str,
    name_b: str,
    output_name: Optional[str] = None,
    uid: Optional[str] = None,
    summary_only: bool = False,
) -> str:
    try:
        if summary_only and not output_name:
            raise ValueError("Summary output requires a stored tensor name")
        result = np.matmul(tensor_workspace.load(name_a), tensor_workspace.load(name_b))
        _store_tensor_if_requested(output_name, result)
        return _success({"tensor_name": output_name, "shape": list(result.shape)} if summary_only else result, tool="scicom_multiply_matrices")
    except Exception as exc:
        return _error("scicom_multiply_matrices", exc)


@registry.tool(stateful=True)
def scicom_scale_matrix(
    name: str,
    scale_factor: float,
    in_place: bool = True,
    output_name: Optional[str] = None,
    uid: Optional[str] = None,
) -> str:
    try:
        result = tensor_workspace.load(name) * float(scale_factor)
        if in_place:
            tensor_workspace.save(name, result)
        _store_tensor_if_requested(output_name, result)
        return _success(result, tool="scicom_scale_matrix")
    except Exception as exc:
        return _error("scicom_scale_matrix", exc)


@registry.tool(stateful=True)
def scicom_matrix_inverse(
    name: str,
    output_name: Optional[str] = None,
    uid: Optional[str] = None,
    summary_only: bool = False,
) -> str:
    try:
        if summary_only and not output_name:
            raise ValueError("Summary output requires a stored tensor name")
        matrix = tensor_workspace.load(name)
        result = np.linalg.inv(matrix)
        _store_tensor_if_requested(output_name, result)
        return _success({"tensor_name": output_name, "shape": list(result.shape)} if summary_only else result, tool="scicom_matrix_inverse")
    except Exception as exc:
        return _error("scicom_matrix_inverse", exc)


@registry.tool(stateful=True)
def scicom_transpose(
    name: str,
    output_name: Optional[str] = None,
    uid: Optional[str] = None,
) -> str:
    try:
        result = tensor_workspace.load(name).T
        _store_tensor_if_requested(output_name, result)
        return _success(result, tool="scicom_transpose")
    except Exception as exc:
        return _error("scicom_transpose", exc)


@registry.tool(access="r")
def scicom_determinant(name: str, uid: Optional[str] = None) -> str:
    try:
        return _success(
            float(np.linalg.det(tensor_workspace.load(name))), tool="scicom_determinant"
        )
    except Exception as exc:
        return _error("scicom_determinant", exc)


@registry.tool(access="r")
def scicom_rank(name: str, uid: Optional[str] = None) -> str:
    try:
        matrix = tensor_workspace.load(name)
        _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
        if singular_values.size == 0:
            result = 0
        else:
            threshold = (
                np.max(singular_values) * max(matrix.shape) * np.finfo(float).eps
            )
            result = int(np.sum(singular_values > threshold))
        return _success(result, tool="scicom_rank")
    except Exception as exc:
        return _error("scicom_rank", exc)


@registry.tool(stateful=True)
def scicom_compute_eigen(
    name: str,
    eigenvalues_output_name: Optional[str] = None,
    eigenvectors_output_name: Optional[str] = None,
    uid: Optional[str] = None,
) -> str:
    try:
        matrix = tensor_workspace.load(name)
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        _store_named_tensors(
            {
                eigenvalues_output_name or "": eigenvalues,
                eigenvectors_output_name or "": eigenvectors,
            }
        )
        return _success(
            {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors},
            tool="scicom_compute_eigen",
        )
    except Exception as exc:
        return _error("scicom_compute_eigen", exc)


@registry.tool(stateful=True)
def scicom_qr_decompose(
    name: str,
    q_output_name: Optional[str] = None,
    r_output_name: Optional[str] = None,
    uid: Optional[str] = None,
) -> str:
    try:
        matrix = tensor_workspace.load(name)
        q, r = np.linalg.qr(matrix)
        _store_named_tensors(
            {
                q_output_name or "": q,
                r_output_name or "": r,
            }
        )
        return _success({"q": q, "r": r}, tool="scicom_qr_decompose")
    except Exception as exc:
        return _error("scicom_qr_decompose", exc)


@registry.tool(stateful=True)
def scicom_find_orthonormal_basis(
    name: str,
    output_name: Optional[str] = None,
    uid: Optional[str] = None,
) -> str:
    try:
        q, _ = np.linalg.qr(tensor_workspace.load(name))
        _store_tensor_if_requested(output_name, q)
        return _success(q, tool="scicom_find_orthonormal_basis")
    except Exception as exc:
        return _error("scicom_find_orthonormal_basis", exc)


@registry.tool(stateful=True)
def scicom_svd_decompose(
    name: str,
    u_output_name: Optional[str] = None,
    s_output_name: Optional[str] = None,
    v_t_output_name: Optional[str] = None,
    uid: Optional[str] = None,
    summary_only: bool = False,
) -> str:
    """Compute and retain an SVD; summary_only avoids returning large factors to the LLM."""
    try:
        if summary_only and not all((u_output_name, s_output_name, v_t_output_name)):
            raise ValueError("summary_only requires names for all three stored factors")
        matrix = tensor_workspace.load(name)
        started = time.perf_counter()
        u, s, v_t = np.linalg.svd(matrix)
        emit_event("scientific_factorization", uid=uid, tensor=name, shape=list(matrix.shape),
                   algorithm="numpy.linalg.svd", compute_s=time.perf_counter() - started)
        _store_named_tensors(
            {
                u_output_name or "": u,
                s_output_name or "": s,
                v_t_output_name or "": v_t,
            }
        )
        result = (
            {"u_shape": list(u.shape), "s_shape": list(s.shape), "v_t_shape": list(v_t.shape)}
            if summary_only else {"u": u, "s": s, "v_t": v_t}
        )
        return _success(result, tool="scicom_svd_decompose")
    except Exception as exc:
        return _error("scicom_svd_decompose", exc)


@registry.tool(stateful=True)
def scicom_solve_from_svd(
    u_name: str,
    s_name: str,
    v_t_name: str,
    rhs_name: str,
    output_name: str,
    uid: Optional[str] = None,
) -> str:
    """Reuse stored SVD factors to solve a full-rank system for multiple response vectors."""
    try:
        u, s, vt, rhs = (tensor_workspace.load(n) for n in (u_name, s_name, v_t_name, rhs_name))
        if s.ndim != 1 or rhs.ndim != 2 or rhs.shape[0] != u.shape[0]:
            raise ValueError("Incompatible factor or response shapes")
        if not np.isfinite(s).all() or np.any(s <= np.finfo(float).eps * max(u.shape) * s.max()):
            raise ValueError("The system must have full numerical rank")
        rank = len(s)
        solution = vt[:rank].T @ ((u[:, :rank].T @ rhs) / s[:, None])
        tensor_workspace.save(output_name, solution)
        return _success({"tensor_name": output_name, "shape": list(solution.shape)})
    except Exception as exc:
        return _error("scicom_solve_from_svd", exc)


@registry.tool(stateful=True)
def scicom_change_basis(
    name: str,
    new_basis: list[list[float]],
    output_name: Optional[str] = None,
    uid: Optional[str] = None,
) -> str:
    try:
        matrix = tensor_workspace.load(name)
        basis = np.asarray(new_basis, dtype=float)
        result = np.linalg.inv(basis) @ matrix @ basis
        _store_tensor_if_requested(output_name, result)
        return _success(result, tool="scicom_change_basis")
    except Exception as exc:
        return _error("scicom_change_basis", exc)


@registry.tool(stateful=True)
def scicom_vector_project(
    name: str,
    new_vector: list[float],
    output_name: Optional[str] = None,
    uid: Optional[str] = None,
) -> str:
    try:
        vector = np.asarray(new_vector, dtype=float)
        result = (
            np.dot(tensor_workspace.load(name), vector)
            / np.linalg.norm(vector)
            * vector
        )
        _store_tensor_if_requested(output_name, result)
        return _success(result, tool="scicom_vector_project")
    except Exception as exc:
        return _error("scicom_vector_project", exc)


@registry.tool(access="r")
def scicom_vector_dot_product(
    name_a: str, name_b: str, uid: Optional[str] = None
) -> str:
    try:
        result = np.dot(tensor_workspace.load(name_a), tensor_workspace.load(name_b))
        return _success(result, tool="scicom_vector_dot_product")
    except Exception as exc:
        return _error("scicom_vector_dot_product", exc)


@registry.tool(stateful=True)
def scicom_vector_cross_product(
    name_a: str,
    name_b: str,
    output_name: Optional[str] = None,
    uid: Optional[str] = None,
) -> str:
    try:
        result = np.cross(tensor_workspace.load(name_a), tensor_workspace.load(name_b))
        _store_tensor_if_requested(output_name, result)
        return _success(result, tool="scicom_vector_cross_product")
    except Exception as exc:
        return _error("scicom_vector_cross_product", exc)
