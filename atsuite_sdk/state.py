from __future__ import annotations

import base64
import contextvars
import copy
import fcntl
import hashlib
import tempfile
import threading
import uuid
from contextlib import ExitStack
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

_SKIP = object()
_CURRENT_UID = contextvars.ContextVar("atsuite_current_uid", default="default")
_EXECUTION = contextvars.ContextVar("atsuite_state_execution", default=None)
_SYNC_TIMING = contextvars.ContextVar("atsuite_state_sync_timing", default=None)
_LOGICAL_STATE = contextvars.ContextVar("atsuite_logical_state", default=None)


def _safe_uid(uid: Optional[str]) -> str:
    raw = str(uid or "default")
    cleaned = "".join(ch if ch.isascii() and (ch.isalnum() or ch in ("-", "_")) else "_" for ch in raw)
    if cleaned != raw or len(cleaned) > 120:
        return cleaned[:40] + "-" + hashlib.sha256(raw.encode()).hexdigest()
    return cleaned or "default"


class SessionObject:
    """Resolve a FaaS registered object against the invocation's private copy."""
    def __init__(self, prototype):
        object.__setattr__(self, "_prototype", prototype)

    def _current(self):
        ctx = _EXECUTION.get()
        return ctx["objects"][id(self)] if ctx is not None else self._prototype

    def __getattr__(self, name):
        return getattr(self._current(), name)

    def __setattr__(self, name, value):
        setattr(self._current(), name, value)


class _ReadWriteLock:
    """One in-memory lock per state domain inside an isolated MCP Session."""
    def __init__(self):
        self.condition = threading.Condition()
        self.readers = 0
        self.writer = False
        self.waiting_writers = 0

    @contextmanager
    def hold(self, access, timeout):
        read = access == "r"
        with self.condition:
            if not read:
                self.waiting_writers += 1
            try:
                ready = self.condition.wait_for(
                    lambda: not self.writer and (self.waiting_writers == 0 if read else self.readers == 0),
                    timeout=timeout,
                )
                if not ready:
                    raise TimeoutError("MCP Session state access timed out")
                if read:
                    self.readers += 1
                else:
                    self.writer = True
            finally:
                if not read:
                    self.waiting_writers -= 1
                self.condition.notify_all()
        try:
            yield
        finally:
            with self.condition:
                if read:
                    self.readers -= 1
                else:
                    self.writer = False
                self.condition.notify_all()


def load_manifest():
    path = Path(os.getenv("ATSUITE_MANIFEST_PATH", "/app/atsuite-manifest.json"))
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Tool manifest must be an object")
    return payload


class StateRuntime:
    """State runtime for MCP/FaaS stateful tools.

    MCP keeps live objects in its isolated Session environment.
    FaaS restores and commits snapshots under local and distributed writer locks.
    """

    def __init__(self) -> None:
        self.runtime = os.getenv("ATSUITE_RUNTIME", "").strip().lower() or "function"
        self.provider = os.getenv("PROVIDER", "").strip().lower()
        self.bucket = os.getenv("PROJECT", "atsuite")
        self.service_name = (
            os.getenv("ATSUITE_STATE_SERVICE", "").strip()
            or os.getenv("ATSUITE_MCP_NAME", "").strip()
            or load_manifest().get("state_service")
            or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
            or os.getenv("K_SERVICE")
            or "atsuite-service"
        )
        self.root = Path(os.getenv("ATSUITE_SESSION_ROOT", "/tmp/atsuite_state"))
        self.root.mkdir(parents=True, exist_ok=True)

        self._registry: Dict[str, Dict[str, object]] = {}
        self._defaults: Dict[str, Dict[str, Any]] = {}
        self._storage = None
        self._legacy_lock = threading.RLock()
        self._storage_error = None
        self._mcp_guard = threading.Lock()
        self._mcp_locks = {}
        self._mcp_uid = None
        self._mcp_started = False

        if self.runtime != "mcp" and (
            self.provider.startswith("ali")
           
            or self.provider.startswith("aws")
            or self.provider.startswith("gcp")
        ):
            try:
                from atsuite_sdk.storage import create_storage

                kwargs = {"provider": self.provider, "bucket": self.bucket}
                self._storage = create_storage(**kwargs)
            except Exception as e:
                self._storage_error = e

    def register_object(self, name: str, obj: object, module_name: Optional[str] = None, *, session_scoped: bool = False):
        if session_scoped and self.runtime != "mcp":
            obj = SessionObject(obj)
        if module_name is None:
            module_name = self.service_name
        module = module_name
        mod_entry = self._registry.setdefault(module, {})
        mod_entry[name] = obj

        defaults = self._defaults.setdefault(module, {})
        if self.runtime != "mcp":
            defaults[name] = self._extract_object_state(obj._prototype if isinstance(obj, SessionObject) else obj)
        return obj

    def list_registered(self, module_name: str) -> Dict[str, object]:
        return self._registry.get(module_name, {})

    def _resolve_uid(self, uid: Optional[str]) -> str:
        if uid is None:
            return _safe_uid(self._mcp_uid if self.runtime == "mcp" and self._mcp_uid is not None else _CURRENT_UID.get())
        return _safe_uid(uid)

    @contextmanager
    def uid_context(self, uid: Optional[str]):
        token = _CURRENT_UID.set(self._resolve_uid(uid))
        try:
            yield
        finally:
            _CURRENT_UID.reset(token)

    def session_dir(self, uid: Optional[str] = None) -> Path:
        ctx = _EXECUTION.get()
        if ctx is not None and self._resolve_uid(uid) == ctx["uid"]:
            return ctx["workspace"]
        return self.root / self.service_name / self._resolve_uid(uid) / "workspace"

    def session_path(self, relpath: str, uid: Optional[str] = None) -> Path:
        rel = Path(relpath)
        if rel.is_absolute():
            raise ValueError("session path must be relative")

        base = self.session_dir(uid)
        path = (base / rel).resolve()
        base_resolved = base.resolve()
        if not path.is_relative_to(base_resolved):
            raise ValueError("session path escapes session directory")
        return path

    def session_open(self, relpath: str, mode: str = "r", *args, uid: Optional[str] = None, **kwargs):
        path = self.session_path(relpath, uid=uid)
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            path.parent.mkdir(parents=True, exist_ok=True)
        return path.open(mode, *args, **kwargs)

    def reset_logical_state(self) -> None:
        """A call that fails before observation must not inherit the previous call."""
        _LOGICAL_STATE.set(None)

    def get_logical_state(self) -> dict:
        """Return logical snapshot demand, independently of physical state transfers."""
        return copy.deepcopy(_LOGICAL_STATE.get() or {})

    @contextmanager
    def observe_logical_state(self, access: str, uid: Optional[str]):
        """Measure SDK-managed object/file snapshots; stateless calls explicitly emit []."""
        observation = {"contract": "sdk-tool-snapshot-v1", "complete": True,
                       "scope": "SDK-managed persistent objects and session files",
                       "accesses": [], "observation_s": 0.0, "errors": []}
        _LOGICAL_STATE.set(observation)

        def record(operation):
            started = time.perf_counter()
            try:
                for module, named in self._registry.items():
                    for name, obj in named.items():
                        target = obj._current() if isinstance(obj, SessionObject) else obj
                        state = {}
                        for key, value in vars(target).items():
                            if key.startswith("_") or callable(value):
                                continue
                            encoded = self._to_json_compatible(value)
                            if encoded is _SKIP:
                                raise ValueError("Unmeasured registered state")
                            state[key] = encoded
                        if state:
                            size = len(json.dumps(state, ensure_ascii=False, allow_nan=False).encode("utf-8"))
                            observation["accesses"].append({"object_id": f"object:{module}:{name}",
                                                           "operation": operation, "bytes": size})
                base = self.session_dir(uid).resolve()
                for path in sorted(base.rglob("*")):
                    if path.is_file():
                        if not path.resolve().is_relative_to(base):
                            raise ValueError("State file escapes session directory")
                        observation["accesses"].append({"object_id": "file:" + path.relative_to(base).as_posix(),
                                                       "operation": operation, "bytes": path.stat().st_size})
            except Exception as exc:
                observation["complete"] = False
                observation["errors"].append(type(exc).__name__)
            finally:
                observation["observation_s"] += time.perf_counter() - started

        if access in {"r", "rw"}:
            record("read")
        try:
            yield
        finally:
            if access in {"w", "rw"}:
                record("write")
            _LOGICAL_STATE.set(observation)

    def reset_sync_metrics(self) -> None:
        _SYNC_TIMING.set({"load_ms": 0.0, "save_ms": 0.0,
                          "load_bytes": 0, "save_bytes": 0, "branch_read_bytes": 0, "branch_write_bytes": 0})

    def get_sync_metrics(self) -> Dict[str, Any]:
        raw = _SYNC_TIMING.get()
        if not isinstance(raw, dict):
            raw = {"load_ms": 0.0, "save_ms": 0.0}
        load_ms = self._coerce_metric_ms(raw.get("load_ms"))
        save_ms = self._coerce_metric_ms(raw.get("save_ms"))
        return {
            "load_ms": round(load_ms, 3),
            "save_ms": round(save_ms, 3),
            "total_ms": round(load_ms + save_ms, 3),
            **{key: raw.get(key) for key in ("load_bytes", "save_bytes", "branch_read_bytes", "branch_write_bytes")},
        }

    def record_transfer_bytes(self, key: str, size: int | None) -> None:
        """Count completed snapshot/branch I/O at its actual storage boundary."""
        if key not in {"load_bytes", "save_bytes", "branch_read_bytes", "branch_write_bytes"} or (size is not None and size < 0):
            raise ValueError("Invalid state transfer counter")
        raw = _SYNC_TIMING.get()
        if isinstance(raw, dict):
            previous = raw.get(key, 0)
            raw[key] = previous + size if previous is not None and size is not None else None

    @staticmethod
    def _coerce_metric_ms(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _record_sync_timing(self, key: str, start_ns: int) -> None:
        raw = _SYNC_TIMING.get()
        if not isinstance(raw, dict):
            raw = {"load_ms": 0.0, "save_ms": 0.0}
        raw[key] = self._coerce_metric_ms(raw.get(key)) + (
            (time.perf_counter_ns() - start_ns) / 1_000_000
        )
        _SYNC_TIMING.set(raw)

    @contextmanager
    def _mcp_execution(self, uid, access, domain, snapshot):
        session_uid = self._resolve_uid(uid)
        with self._mcp_guard:
            if self._mcp_uid is None:
                self._mcp_uid = session_uid
            elif self._mcp_uid != session_uid:
                raise ValueError("MCP execution environment belongs to another Session; route this uid to its own environment")
            lock = self._mcp_locks.setdefault(domain or self.service_name, _ReadWriteLock()) if access != "stateless" else None
        self.reset_sync_metrics()
        with ExitStack() as stack:
            stack.enter_context(self.uid_context(uid))
            if access != "stateless":
                stack.enter_context(lock.hold(access, float(os.getenv("ATSUITE_STATE_LOCK_TIMEOUT", "30"))))
            if snapshot is not None:
                if access not in {"w", "rw"}:
                    raise ValueError("Only write calls may seed state")
                self.prime_snapshot(uid, snapshot)
            if access != "stateless":
                with self._mcp_guard:
                    self._mcp_started = True
            self.session_dir(uid).mkdir(parents=True, exist_ok=True)
            yield

    @contextmanager
    def execution(self, tool_module, uid, access="rw", domain="", snapshot=None):
        if snapshot is not None and os.getenv("ATSUITE_ALLOW_STATE_BOOTSTRAP", "true").lower() == "false":
            raise ValueError("State bootstrap is disabled for closed Trace replay")
        if access not in {"r", "w", "rw", "stateless"}:
            raise ValueError(f"Invalid Tool access: {access}")
        if self.runtime == "mcp":
            with self._mcp_execution(uid, access, domain, snapshot):
                yield
            return
        if self._storage_error is not None and access != "stateless":
            raise RuntimeError("State storage initialization failed") from self._storage_error
        session_uid = self._resolve_uid(uid)
        domain = _safe_uid(domain) if domain else ""
        lock_key = f"{self.service_name}/{session_uid}"
        lock_path = self.root / ".locks" / (hashlib.sha256(lock_key.encode()).hexdigest() + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            if access != "stateless":
                handle = stack.enter_context(lock_path.open("a+"))
                deadline = time.monotonic() + float(os.getenv("ATSUITE_STATE_LOCK_TIMEOUT", "30"))
                mode = fcntl.LOCK_SH if access == "r" else fcntl.LOCK_EX
                while True:
                    try:
                        fcntl.flock(handle, mode | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"State lock timeout: {lock_key}")
                        time.sleep(0.01)
                stack.callback(fcntl.flock, handle, fcntl.LOCK_UN)
                if self._storage is not None and access in {"w", "rw"}:
                    # shortcut: crashed writers retain this lock; remove only after confirming the writer has stopped.
                    remote_key = lock_key + ".lock"
                    owner = json.dumps({"owner": uuid.uuid4().hex, "created_utc_ns": time.time_ns()}).encode()
                    while not self._storage.try_create(remote_key, owner):
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Cloud state lock timeout: {remote_key}")
                        time.sleep(0.05)
                    stack.callback(self._storage.deleteobj, remote_key)
            legacy = any(not isinstance(obj, SessionObject) for named in self._registry.values() for obj in named.values())
            if legacy:
                stack.enter_context(self._legacy_lock)
            invocation_root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="call-", dir=self.root)))
            workspace = (self.root / self.service_name / session_uid / "workspace"
                         if access in {"w", "rw"} and os.getenv("ATSUITE_EPHEMERAL_WORKSPACE", "false").lower() != "true" else invocation_root / "workspace")
            workspace.parent.mkdir(parents=True, exist_ok=True)
            ctx = {"uid": session_uid, "domain": domain, "workspace": workspace, "objects": {}}
            for named in self._registry.values():
                for obj in named.values():
                    if isinstance(obj, SessionObject):
                        ctx["objects"][id(obj)] = copy.deepcopy(obj._prototype)
            token = _EXECUTION.set(ctx)
            stack.callback(_EXECUTION.reset, token)
            stack.enter_context(self.uid_context(uid))
            self.reset_sync_metrics()
            if snapshot is not None:
                if access not in {"w", "rw"}:
                    raise ValueError("Only write calls may prime state")
                self.prime_snapshot(uid, snapshot)
            if access != "stateless":
                self.load_for_tool(tool_module, uid)
            else:
                ctx["workspace"].mkdir()
            yield
            if access in {"w", "rw"}:
                self.save_after_tool(tool_module, uid)

    def load_for_tool(self, tool_module: str, uid: Optional[str]) -> None:
        if self.runtime == "mcp":
            return
        sync_start_ns = time.perf_counter_ns()
        try:
            snap = self._get_snapshot(uid)
            self._restore_files(uid, snap.get("files") or {})
            all_vars = snap.get("vars") or {}
            for module, named in self._registry.items():
                module_vars = all_vars.get(module) or {}
                for name, obj in named.items():
                    target = module_vars.get(name)
                    if target is None:
                        target = next((values[name] for values in all_vars.values() if name in values), None)
                    if target is None:
                        target = self._defaults.get(module, {}).get(name, {})
                    self._apply_object_state(obj._current() if isinstance(obj, SessionObject) else obj, copy.deepcopy(target))
        finally:
            self._record_sync_timing("load_ms", sync_start_ns)

    def save_after_tool(self, tool_module: str, uid: Optional[str]) -> None:
        if self.runtime == "mcp":
            return
        sync_start_ns = time.perf_counter_ns()
        try:
            variables = {
                module: {name: self._extract_object_state(obj._current() if isinstance(obj, SessionObject) else obj)
                         for name, obj in named.items()}
                for module, named in self._registry.items()
            }
            self._set_snapshot(uid, {"version": 1, "vars": variables, "files": self._capture_files(uid)})
        finally:
            self._record_sync_timing("save_ms", sync_start_ns)

    def _snapshot_key(self, uid: Optional[str]) -> str:
        return f"{self.service_name}/{self._resolve_uid(uid)}.json"

    def _get_snapshot(self, uid: Optional[str]) -> Dict[str, Any]:
        key = self._snapshot_key(uid)
        try:
            if self._storage is not None:
                content = self._storage.read(key)
            else:
                path = self.root / ".snapshots" / key
                content = path.read_bytes() if path.exists() else b""
        except BaseException:
            self.record_transfer_bytes("load_bytes", None)
            raise
        self.record_transfer_bytes("load_bytes", len(content.encode("utf-8") if isinstance(content, str) else content))
        if not content:
            return {}
        payload = json.loads(content)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError(f"Invalid state snapshot: {key}")
        return payload

    def _set_snapshot(self, uid: Optional[str], snapshot: Dict[str, Any]) -> None:
        key = self._snapshot_key(uid)
        blob = json.dumps(snapshot, ensure_ascii=False).encode("utf-8")
        try:
            if self._storage is not None:
                self._storage.put(key, blob)
            else:
                path = self.root / ".snapshots" / key
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + "." + uuid.uuid4().hex)
                try:
                    temporary.write_bytes(blob)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)

        except BaseException:
            self.record_transfer_bytes("save_bytes", None)
            raise
        self.record_transfer_bytes("save_bytes", len(blob))

    def prime_snapshot(self, uid: Optional[str], snapshot: Dict[str, Any]) -> None:
        if not isinstance(snapshot, dict):
            raise ValueError("State snapshot must be an object")
        if self.runtime != "mcp":
            self._set_snapshot(uid, snapshot)
            return
        # Explicit bootstrap only; never use snapshots to restore ordinary MCP calls.
        with self._mcp_guard:
            if self._mcp_started:
                raise ValueError("Cannot reseed a running MCP Session")
            all_vars = snapshot.get("vars") or {}
            for module, named in self._registry.items():
                for name, obj in named.items():
                    values = (all_vars.get(module) or {}).get(name)
                    if values is None:
                        values = next((entry[name] for entry in all_vars.values() if name in entry), None)
                    if values is not None:
                        self._apply_object_state(obj, values)
            self._restore_files(uid, snapshot.get("files") or {})
            self._mcp_started = True

    def _capture_files(self, uid: Optional[str]) -> Dict[str, str]:
        base = self.session_dir(uid)
        if not base.exists():
            return {}
        files: Dict[str, str] = {}
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(base.resolve()):
                raise ValueError("State file escapes session directory")
            rel = path.relative_to(base).as_posix()
            files[rel] = base64.b64encode(path.read_bytes()).decode("ascii")
        return files

    def _restore_files(self, uid: Optional[str], files: Dict[str, str]) -> None:
        base = self.session_dir(uid)
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)

        for rel, encoded in files.items():
            path = self.session_path(rel, uid=uid)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(encoded.encode("ascii")))

    def _extract_object_state(self, obj: object) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for key, value in vars(obj).items():
            if key.startswith("_"):
                continue
            if callable(value):
                continue
            encoded = self._to_json_compatible(value)
            if encoded is _SKIP:
                continue
            data[key] = encoded
        return data

    def _apply_object_state(self, obj: object, data: Dict[str, Any]) -> None:
        for key in list(vars(obj)):
            if not key.startswith("_"):
                delattr(obj, key)
        for key, value in data.items():
            setattr(obj, key, value)

    def _to_json_compatible(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            out = []
            for item in value:
                converted = self._to_json_compatible(item)
                if converted is _SKIP:
                    return _SKIP
                out.append(converted)
            return out
        if isinstance(value, tuple):
            out = []
            for item in value:
                converted = self._to_json_compatible(item)
                if converted is _SKIP:
                    return _SKIP
                out.append(converted)
            return out
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for k, v in value.items():
                if not isinstance(k, str):
                    return _SKIP
                converted = self._to_json_compatible(v)
                if converted is _SKIP:
                    return _SKIP
                out[k] = converted
            return out
        return _SKIP


_RUNTIME = StateRuntime()


def get_state_runtime() -> StateRuntime:
    return _RUNTIME


def register_state_object(name: str, obj: object, module_name: Optional[str] = None, *, session_scoped: bool = False):
    if module_name is None:
        import inspect
        caller_frame = inspect.currentframe()
        if caller_frame and caller_frame.f_back:
            caller_module = caller_frame.f_back.f_globals.get("__name__")
            if caller_module:
                module_name = caller_module
    return _RUNTIME.register_object(name, obj, module_name=module_name, session_scoped=session_scoped)


def session_path(relpath: str, uid: Optional[str] = None) -> str:
    return str(_RUNTIME.session_path(relpath, uid=uid))


def session_open(relpath: str, mode: str = "r", *args, uid: Optional[str] = None, **kwargs):
    return _RUNTIME.session_open(relpath, mode, *args, uid=uid, **kwargs)
