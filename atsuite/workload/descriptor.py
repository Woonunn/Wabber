"""Arrival-time computational intent and resource descriptors; no observed DAGs."""

import hashlib
import json
import math
import re
from pathlib import PurePath

from atsuite.workload.profile import number

# shortcut: fixed lexical cues are incomplete; keep unknown intents uncertain until independently validated semantics are available.
INTENTS = {
    "training": r"\b(train(?:ing)?|fit|predictive|classifier|classification|regression|lstm|neural|random forest|svm|xgboost|clustering|k-means)\b|训练|拟合|分类|回归|聚类",
    "iterative": r"\b(iterat\w*|bootstrap\w*|monte carlo|simulat\w*|optimi[sz]\w*|convergen\w*|epochs?|cross.validat\w*|grid search)\b|迭代|自助采样|蒙特卡洛|优化|交叉验证",
    "statistics": r"\b(statisti\w*|analysis|analy[sz]e|mean|median|variance|correlation|hypothesis|t.test|anova|gini|inequality|percentile|summari[sz]e|summary)\b|统计|均值|方差|相关|检验",
    "visualization": r"\b(plot\w*|chart\w*|visuali[sz]\w*|histogram|scatter|heatmap|lorenz|figure)\b|绘图|可视化|图表",
    "processing": r"\b(clean\w*|transform\w*|filter\w*|join|merge|aggregat\w*|preprocess\w*|sort\w*|deduplicat\w*|imput\w*)\b|清洗|转换|过滤|合并|排序|去重",
    "stateful": r"\b(shared (?:state|array|data)|retain\w*|checkpoint\w*|intermediate|persist\w*|cache\w*|reuse\w*|in.memory)\b|共享状态|保留|中间结果|持久化|缓存|复用",
}
NUMERIC = {
    "prompt_chars",
    "prompt_lines",
    "code_blocks",
    "file_count",
    "input_bytes",
    "unknown_file_sizes",
}
LEGACY_FIELDS = NUMERIC | {"file_types", "tools", "category", "hints"}


def describe(*, prompt="", files=(), tools=(), category=None, hints=None) -> dict:
    """Extract a versioned, deterministic multi-intent signature from supplied inputs."""
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be text")
    intents = [
        name for name, pattern in INTENTS.items() if re.search(pattern, prompt, re.I)
    ]
    # Generic analysis words should not mask a more specific computational operation.
    primary = next(
        (
            name
            for name in (
                "training",
                "iterative",
                "visualization",
                "statistics",
                "processing",
            )
            if name in intents
        ),
        "unknown",
    )
    result = {
        "prompt_chars": len(prompt),
        "prompt_lines": len(prompt.splitlines()),
        "code_blocks": prompt.count("```"),
        "file_count": len(files),
        "input_bytes": sum(f["bytes"] for f in files if number(f.get("bytes"))),
        "unknown_file_sizes": sum(not number(f.get("bytes")) for f in files),
        "file_types": sorted({PurePath(f["name"]).suffix.lower() for f in files}),
        "tools": sorted(set(tools)),
        "category": category,
        "hints": dict(hints or {}),
        "semantic": {"version": 1, "primary": primary, "intents": sorted(intents)},
    }
    validate_descriptor(result)
    return result


def validate_descriptor(value: dict) -> None:
    """Accept legacy snapshots; reject outcome features and malformed arrival hints."""
    if not isinstance(value, dict) or set(value) not in (
        LEGACY_FIELDS,
        LEGACY_FIELDS | {"semantic"},
    ):
        raise ValueError("Descriptor must contain only arrival-time features")
    if any(not number(value[k]) for k in NUMERIC):
        raise ValueError("Invalid descriptor feature")
    if value["category"] is not None and not isinstance(value["category"], str):
        raise ValueError("Invalid category")
    for field in ("tools", "file_types"):
        if not isinstance(value[field], list) or any(
            not isinstance(v, str) for v in value[field]
        ):
            raise ValueError("Invalid descriptor names")
    if not isinstance(value["hints"], dict) or any(
        k not in {"cpu_s", "memory_mb", "state_bytes", "rows", "columns", "features"}
        or not number(v)
        for k, v in value["hints"].items()
    ):
        raise ValueError("Invalid resource hints")
    if "semantic" in value:
        s = value["semantic"]
        if (
            not isinstance(s, dict)
            or set(s) != {"version", "primary", "intents"}
            or type(s["version"]) is not int
            or s["version"] != 1
            or not isinstance(s["intents"], list)
            or any(not isinstance(i, str) or i not in INTENTS for i in s["intents"])
            or len(set(s["intents"])) != len(s["intents"])
            or s["primary"] not in {*INTENTS, "unknown"}
            or (s["primary"] != "unknown" and s["primary"] not in s["intents"])
        ):
            raise ValueError("Invalid computational intent")


def legacy_distance(left: dict, right: dict) -> float:
    """Stage 6 descriptor distance, retained for snapshot compatibility and ablation."""
    parts = [
        abs(math.log2(1 + left[k]) - math.log2(1 + right[k])) / 4
        for k in ("prompt_chars", "file_count", "input_bytes")
    ]
    for key in ("file_types", "tools"):
        a, b = set(left[key]), set(right[key])
        parts.append(1 - len(a & b) / len(a | b) if a | b else 0)
    for key in set(left["hints"]) | set(right["hints"]):
        a, b = left["hints"].get(key), right["hints"].get(key)
        parts.append(
            abs(math.log2(1 + a) - math.log2(1 + b)) / 4
            if a is not None and b is not None
            else 1
        )
    if left["category"] and right["category"]:
        parts.append(float(left["category"] != right["category"]))
    return sum(parts) / len(parts)


def resource_distance(left: dict, right: dict) -> float:
    """Compare declared resource scales without prompt text, intent or file encoding.

    Unknown file sizes are missing evidence, not zero-byte inputs. File extensions
    do not establish a compute or memory requirement.
    """
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
    """Recognize an observed intent combination, then compare its resource scale."""
    if left["category"] != right["category"]:
        return 1.0
    a, b = left.get("semantic"), right.get("semantic")
    if (left["category"] and a and b and a["primary"] == b["primary"] == "unknown"
            and set(left["tools"]) == set(right["tools"])
            and set(left["file_types"]) == set(right["file_types"])):
        return resource_distance(left, right)
    if a is None and b is None:
        return legacy_distance(left, right)
    if (
        not a
        or not b
        or not a["intents"]
        or not b["intents"]
        or a["primary"] == "unknown"
        or b["primary"] == "unknown"
        or set(a["intents"]) != set(b["intents"])
    ):
        return 1.0
    # Different capabilities/encodings can inform an uncertain prior, not a known family.
    if set(left["tools"]) != set(right["tools"]) or set(left["file_types"]) != set(
        right["file_types"]
    ):
        return 1.0
    return resource_distance(left, right)


def family_key(value: dict | None) -> str:
    """Stable intent/resource label independent of task IDs and superficial prompt size."""
    if value is None:
        return "global"
    validate_descriptor(value)
    coarse = {k: value[k] for k in ("file_types", "tools", "category")}
    sizes = ("input_bytes", "file_count")
    if "semantic" in value:
        coarse["semantic"] = {
            "version": value["semantic"]["version"],
            "intents": sorted(value["semantic"]["intents"]),
        }
        coarse["hints"] = {
            k: int(math.log2(1 + v) // 4) for k, v in value["hints"].items()
        }
    else:
        sizes += ("prompt_chars",)
    coarse.update({k: int(math.log2(1 + value[k]) // 4) for k in sizes})
    return hashlib.sha256(json.dumps(coarse, sort_keys=True).encode()).hexdigest()[:16]
