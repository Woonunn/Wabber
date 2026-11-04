"""Versioned arrival context and strict workload implementation identities."""

import copy
import hashlib
import json
import math
from pathlib import PurePath

from atsuite.workload.profile import number

NUMERIC = {
    "prompt_chars",
    "prompt_lines",
    "code_blocks",
    "file_count",
    "input_bytes",
    "unknown_file_sizes",
}
LEGACY_FIELDS = NUMERIC | {"file_types", "tools", "category", "hints"}
CONTEXT_FIELDS = (LEGACY_FIELDS - {"category"}) | {
    "version",
    "prompt",
    "files",
    "tool_definitions",
    "harness",
    "model",
    "control",
    "session",
}


def fingerprint(value: object) -> str:
    """Hash canonical JSON, including implementation revisions and definitions."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _identity(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"id", "revision"}
        and all(isinstance(v, str) and v.strip() for v in value.values())
    )


def describe(
    *,
    prompt="",
    files=(),
    tools=(),
    harness=None,
    model=None,
    control=None,
    session=None,
    hints=None,
) -> dict:
    """Freeze only information available before execution.

    Tools need name, definition and implementation {id, revision}. Name-only tools
    and absent Harness identities remain readable but cannot retrieve history.
    File usage is deliberately unspecified unless supplied in arrival metadata.
    """
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be text")
    tools = list(tools)
    files = copy.deepcopy(list(files))
    definitions = sorted(
        (copy.deepcopy(t) for t in tools if isinstance(t, dict)),
        key=lambda t: t["name"],
    )
    names = [t["name"] if isinstance(t, dict) else t for t in tools]
    file_hints = {
        key: max(f[key] for f in files if number(f.get(key)))
        for key in ("rows", "columns", "features")
        if any(number(f.get(key)) for f in files)
    }
    result = {
        "version": 2,
        "prompt": prompt,
        "files": files,
        "tool_definitions": definitions,
        "harness": copy.deepcopy(harness),
        "model": copy.deepcopy(model),
        "control": copy.deepcopy(control),
        "session": copy.deepcopy(session),
        "prompt_chars": len(prompt),
        "prompt_lines": len(prompt.splitlines()),
        "code_blocks": prompt.count("```"),
        "file_count": len(files),
        "input_bytes": sum(f["bytes"] for f in files if number(f.get("bytes"))),
        "unknown_file_sizes": sum(not number(f.get("bytes")) for f in files),
        "file_types": sorted({PurePath(f["name"]).suffix.lower() for f in files}),
        "tools": sorted(names),
        "hints": {**file_hints, **dict(hints or {})},
    }
    validate_descriptor(result)
    return result


def validate_descriptor(value: dict) -> None:
    """Accept old snapshots for audit only; never treat labels as semantic samples."""
    if not isinstance(value, dict) or set(value) not in (
        LEGACY_FIELDS,
        LEGACY_FIELDS | {"semantic"},
        CONTEXT_FIELDS,
    ):
        raise ValueError("Descriptor must contain only arrival-time features")
    if any(not number(value[k]) for k in NUMERIC):
        raise ValueError("Invalid descriptor feature")
    if value.get("category") is not None and not isinstance(value["category"], str):
        raise ValueError("Invalid category")
    for field in ("tools", "file_types"):
        if not isinstance(value[field], list) or any(
            not isinstance(v, str) or not v for v in value[field]
        ):
            raise ValueError("Invalid descriptor names")
    if len(set(value["tools"])) != len(value["tools"]):
        raise ValueError("Duplicate tools")
    if not isinstance(value["hints"], dict) or any(
        k not in {"cpu_s", "memory_mb", "state_bytes", "rows", "columns", "features"}
        or not number(v)
        for k, v in value["hints"].items()
    ):
        raise ValueError("Invalid resource hints")
    if set(value) != CONTEXT_FIELDS:
        return
    if (
        type(value["version"]) is not int
        or value["version"] != 2
        or not isinstance(value["prompt"], str)
    ):
        raise ValueError("Invalid request context version/prompt")
    files = value["files"]
    allowed_file = {
        "name",
        "bytes",
        "sha256",
        "rows",
        "columns",
        "features",
        "schema",
        "description",
        "usage",
        "path",
    }
    if not isinstance(files, list) or any(
        not isinstance(f, dict)
        or not set(f) <= allowed_file
        or not isinstance(f.get("name"), str)
        or any(
            k in f and f[k] is not None and not number(f[k])
            for k in ("bytes", "rows", "columns", "features")
        )
        for f in files
    ):
        raise ValueError("Invalid arrival file metadata")
    if value["harness"] is not None and not _identity(value["harness"]):
        raise ValueError("Harness needs a stable id and immutable revision")
    definitions = value["tool_definitions"]
    if (
        not isinstance(definitions, list)
        or any(
            not isinstance(t, dict)
            or set(t) != {"name", "definition", "implementation"}
            or not isinstance(t["name"], str)
            or not isinstance(t["definition"], dict)
            or not _identity(t["implementation"])
            for t in definitions
        )
        or len({t["name"] for t in definitions}) != len(definitions)
        or not {t["name"] for t in definitions} <= set(value["tools"])
    ):
        raise ValueError("Invalid versioned tool definitions")
    for field in ("model", "control", "session"):
        if value[field] is not None and not isinstance(value[field], dict):
            raise ValueError("Invalid arrival " + field)
    expected = {
        "prompt_chars": len(value["prompt"]),
        "prompt_lines": len(value["prompt"].splitlines()),
        "code_blocks": value["prompt"].count("```"),
        "file_count": len(files),
        "input_bytes": sum(f["bytes"] for f in files if number(f.get("bytes"))),
        "unknown_file_sizes": sum(not number(f.get("bytes")) for f in files),
        "file_types": sorted({PurePath(f["name"]).suffix.lower() for f in files}),
    }
    if any(value[k] != v for k, v in expected.items()):
        raise ValueError("Context summary differs from raw arrival inputs")
    fingerprint(value)


def compatibility_key(value: dict | None) -> str | None:
    """A missing implementation identity never creates a shared unknown group."""
    if not value or value.get("version") != 2 or not _identity(value.get("harness")):
        return None
    definitions = value.get("tool_definitions", [])
    if len(definitions) != len(value["tools"]) or any(
        not _identity(t.get("implementation")) for t in definitions
    ):
        return None
    return fingerprint(
        {
            "tools": sorted(definitions, key=lambda t: t["name"]),
            "harness": value["harness"],
        }
    )


def compatible(left: dict | None, right: dict | None) -> bool:
    """Exact capability and Harness equality is a gate, never a score."""
    key = compatibility_key(left)
    return key is not None and key == compatibility_key(right)


def context_key(value: dict | None) -> str:
    """Cache/group key includes full context, model, policy and session state."""
    return fingerprint(value)


def resource_distance(left: dict, right: dict) -> float:
    """Compare declared scale without filling unknown sizes with zero."""
    parts = [
        abs(math.log2(1 + left["file_count"]) - math.log2(1 + right["file_count"])) / 4
    ]
    if not left["unknown_file_sizes"] and not right["unknown_file_sizes"]:
        parts.append(
            abs(
                math.log2(1 + left["input_bytes"]) - math.log2(1 + right["input_bytes"])
            )
            / 4
        )
    elif left["unknown_file_sizes"] != right["unknown_file_sizes"]:
        parts.append(1.0)
    for key in set(left["hints"]) | set(right["hints"]):
        x, y = left["hints"].get(key), right["hints"].get(key)
        parts.append(
            abs(math.log2(1 + x) - math.log2(1 + y)) / 4
            if x is not None and y is not None
            else 1.0
        )
    return max(parts)


def distance(left: dict, right: dict) -> float:
    """Conservative exact-context match for callers without a semantic model."""
    return (
        0.0
        if compatible(left, right) and context_key(left) == context_key(right)
        else 1.0
    )


def family_key(value: dict | None) -> str:
    """Group offered load without task categories or legacy intent labels."""
    if value is not None:
        validate_descriptor(value)
    return context_key(value)
