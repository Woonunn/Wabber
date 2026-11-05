"""Small in-process empirical workload model with causal prediction cutoffs."""

import copy
import json
import math
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from atsuite.tracing.storage import read_text, write_json
from atsuite.workload.descriptor import compatible, context_key, model_settings, validate_descriptor
from atsuite.workload.profile import METRICS, extract_profile, number
from atsuite.workload.retrieval import ContextBackend, ContextRetriever


def distribution(values: list, total: int) -> dict:
    """Return exact empirical samples and interpolated quantiles, without imputation."""
    values = sorted(values)

    def quantile(q):
        if not values:
            return None
        position = (len(values) - 1) * q
        lower = int(position)
        return values[lower] + (
            values[min(lower + 1, len(values) - 1)] - values[lower]
        ) * (position - lower)

    return {
        "n": len(values),
        "missing": total - len(values),
        "coverage": len(values) / total if total else 0,
        "samples": values,
        "mean": sum(values) / len(values) if values else None,
        "p10": quantile(0.1),
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "low_confidence": len(values) < 10 or len(values) < total,
    }


class WorkloadModel:
    """Conditional joint execution scenarios with causal feedback calibration."""

    def __init__(
        self,
        initial_profiles: list[dict] | None = None,
        *,
        retrieval_config=None,
        retriever=None,
    ) -> None:
        # shortcut: retain exact samples in memory; use bounded summaries when volume warrants it.
        self.initial_profiles = copy.deepcopy(initial_profiles or [])
        self.retrieval_config = copy.deepcopy(retrieval_config)
        self.retriever = retriever or ContextRetriever(
            ContextBackend(retrieval_config) if retrieval_config else None
        )
        self.pending = {}
        self.errors = []
        self.samples = {}
        self.active = {}
        self.lock = threading.RLock()

    def begin(
        self,
        request_id: str,
        session_id: str,
        arrival_ns: int | None = None,
        *,
        descriptor=None,
    ) -> dict:
        """Freeze a prediction before executing a request; count distinct active sessions."""
        arrival_ns = time.time_ns() if arrival_ns is None else arrival_ns
        if not number(arrival_ns):
            raise ValueError("Invalid arrival timestamp")
        with self.lock:
            if request_id in self.active or request_id in self.samples:
                raise ValueError("Request ID already registered")
            prediction = self.predict(arrival_ns, descriptor=descriptor)
            self.active[request_id] = {
                "session_id": session_id,
                "arrival_ns": arrival_ns,
                "descriptor": copy.deepcopy(descriptor),
            }
            prediction["concurrent_sessions"] = len(
                {v["session_id"] for v in self.active.values()}
            )
            self.record_prediction(request_id, prediction)
            prediction["request_id"] = request_id
            return prediction

    def observe(
        self, trace: dict, available_ns: int | None = None, *, descriptor=None
    ) -> dict:
        """Extract and ingest a completed trace; retain failed/incomplete audits separately."""
        return self.update(extract_profile(trace), available_ns, descriptor=descriptor)

    def update(
        self, profile: dict, available_ns: int | None = None, *, descriptor=None
    ) -> dict:
        """Ingest once, allowing an incomplete checkpoint to later become complete."""
        available_ns = time.time_ns() if available_ns is None else available_ns
        profile = copy.deepcopy(profile)
        if descriptor is not None:
            profile["descriptor"] = copy.deepcopy(descriptor)
        if profile.get("descriptor") is not None:
            validate_descriptor(profile["descriptor"])
        if not number(available_ns):
            raise ValueError("Invalid availability timestamp")
        if profile.get("status") not in {"success", "failed", "incomplete"}:
            raise ValueError("Invalid profile status")
        for value in profile["metrics"].values():
            if value is not None and not number(value):
                raise ValueError("Invalid profile measurement")
        for key in ("arrival_ns", "completed_ns"):
            if profile.get(key) is not None and not number(profile[key]):
                raise ValueError("Invalid request timestamp")
        if (
            profile.get("arrival_ns") is not None
            and profile.get("completed_ns") is not None
            and profile["completed_ns"] < profile["arrival_ns"]
        ):
            raise ValueError("Completion precedes arrival")
        with self.lock:
            key = profile["request_id"]
            profile = copy.deepcopy(profile)
            registered = self.active.get(key, {})
            frozen = self.pending.get(key)
            if frozen and frozen.get("descriptor") is not None:
                if profile.get("descriptor") is None:
                    profile["descriptor"] = copy.deepcopy(frozen["descriptor"])
                elif context_key(profile["descriptor"]) != context_key(
                    frozen["descriptor"]
                ):
                    raise ValueError(
                        "Completion context differs from frozen arrival context"
                    )
            if profile.get("arrival_ns") is None:
                profile["arrival_ns"] = registered.get("arrival_ns")
            previous = self.samples.get(key)
            if profile.get("arrival_ns") is None and previous:
                profile["arrival_ns"] = previous["profile"].get("arrival_ns")
            if previous and previous["profile"]["status"] != "incomplete":
                if previous["profile"] == profile:
                    return copy.deepcopy(profile)
                raise ValueError("Conflicting completed request")
            if (
                profile.get("completed_ns") is not None
                and profile["completed_ns"] > available_ns
            ):
                raise ValueError("Observation precedes request completion")
            if (
                profile["status"] != "incomplete"
                or profile.get("completed_ns") is not None
            ):
                self.active.pop(key, None)
            if frozen and profile["status"] != "incomplete":
                errors = {
                    k: abs(profile["metrics"][k] - frozen["metrics"][k]["p50"])
                    for k in METRICS
                    if number(profile["metrics"].get(k))
                    and number(frozen["metrics"][k]["p50"])
                    and profile["status"] == "success"
                }
                self.errors.append(
                    {
                        "request_id": key,
                        "available_ns": available_ns,
                        "descriptor": copy.deepcopy(profile.get("descriptor")),
                        "revision": frozen["retrieval"]["revision"],
                        "metrics": errors,
                    }
                )
                self.pending.pop(key, None)
            self.samples[key] = {"available_ns": available_ns, "profile": profile}
            return copy.deepcopy(profile)

    def predict(self, as_of_ns: int | None = None, *, descriptor=None) -> dict:
        """Use only observations available strictly before arrival, never the new ATG."""
        as_of_ns = time.time_ns() if as_of_ns is None else as_of_ns
        if descriptor is not None:
            validate_descriptor(descriptor)
        if not number(as_of_ns):
            raise ValueError("Invalid prediction timestamp")
        with self.lock:
            eligible = [
                s["profile"]
                for s in self.samples.values()
                if s["available_ns"] < as_of_ns
                and (
                    number(s["profile"].get("completed_ns"))
                    and s["profile"]["completed_ns"] < as_of_ns
                )
            ]
            eligible = [
                p for p in eligible if compatible(descriptor, p.get("descriptor"))
            ]
            history = sorted(
                eligible,
                key=lambda p: (
                    -self.samples[p["request_id"]]["available_ns"],
                    p["request_id"],
                ),
            )
            # Initial profiling is subject to the same identity and availability gates.
            priors = [
                p
                for p in self.initial_profiles
                if p.get("status") == "success"
                and number(p.get("available_ns"))
                and p["available_ns"] < as_of_ns
                and number(p.get("completed_ns"))
                and p["completed_ns"] < as_of_ns
                and compatible(descriptor, p.get("descriptor"))
            ]
            known_ids = {p["request_id"] for p in history}
            priors = {
                p["request_id"]: p for p in priors if p["request_id"] not in known_ids
            }
            samples, retrieval = self.retriever.select(
                descriptor, history + list(priors.values())
            )
            # Do not trust a backend to enforce isolation on our behalf.
            samples = [
                p for p in samples
                if p.get("status") == "success" and compatible(descriptor, p.get("descriptor"))
            ][:20]
            basis = retrieval["match_basis"] if samples else "cold_start"
            n = len(samples)
            metrics = {
                k: distribution(
                    [p["metrics"][k] for p in samples if number(p["metrics"].get(k))], n
                )
                for k in METRICS
            }
            names = sorted({name for p in samples for name in p["workflow"]["calls"]})
            transitions = Counter()
            routes = defaultdict(Counter)
            for p in samples:
                nodes = {s["id"]: s["name"] for s in p["workflow"]["nodes"]}
                transitions.update(
                    (nodes[a], nodes[b]) for a, b in p["workflow"]["edges"]
                )
                for route in p["workflow"]["routes"]:
                    routes[route["node"]][tuple(route["targets"])] += 1
            calibration = self._calibration(descriptor, as_of_ns, metrics)
            missing = [k for k, v in metrics.items() if v["missing"] or not v["n"]]
            missing_context = [
                k
                for k in ("model", "control")
                if not descriptor or descriptor.get(k) is None
            ]
            attempts = sum(p["status"] != "incomplete" for p in eligible)
            failures = sum(p["status"] == "failed" for p in eligible)
            return {
                "as_of_ns": as_of_ns,
                "sample_count": n,
                "match_basis": basis,
                "eligible_success_count": sum(p["status"] == "success" for p in history),
                "unknown_request": not samples,
                "descriptor": copy.deepcopy(descriptor),
                "joint_profiles": copy.deepcopy(samples),
                "scenario_weights": [1 / n] * n if n else [],
                "retrieval": retrieval,
                "uncertainty": calibration,
                "missing_metrics": missing,
                "missing_context": missing_context,
                "observed_failure_rate": failures / attempts if attempts else None,
                "status_counts": dict(Counter(p["status"] for p in eligible)),
                "low_confidence": n < 3
                or not calibration["calibrated"]
                or bool(missing_context)
                or bool(descriptor and descriptor["unknown_file_sizes"])
                or any(p["source"] == "capture" for p in samples),
                "metrics": metrics,
                "workflow": {
                    "tool_calls": {
                        name: distribution(
                            [p["workflow"]["tool_calls"].get(name, 0) for p in samples],
                            n,
                        )
                        for name in sorted(
                            {
                                name
                                for p in samples
                                for name in p["workflow"]["tool_calls"]
                            }
                        )
                    },
                    "calls": {
                        name: distribution(
                            [p["workflow"]["calls"].get(name, 0) for p in samples], n
                        )
                        for name in names
                    },
                    "presence_probability": {
                        name: sum(name in p["workflow"]["calls"] for p in samples) / n
                        for name in names
                    },
                    "causal_revisits": {
                        name: distribution(
                            [
                                p["workflow"]["causal_revisits"].get(name, 0)
                                for p in samples
                            ],
                            n,
                        )
                        for name in names
                    },
                    "dependency_edges": [
                        {"source": a, "target": b, "count": count}
                        for (a, b), count in sorted(transitions.items())
                    ],
                    "branches": {
                        name: [
                            {
                                "targets": list(targets),
                                "count": count,
                                "probability": count / sum(counts.values()),
                            }
                            for targets, count in sorted(counts.items())
                        ]
                        for name, counts in routes.items()
                    },
                    "parallel_peak": distribution(
                        [
                            p["workflow"]["parallel_peak"]
                            for p in samples
                            if p["workflow"]["parallel_peak"] is not None
                        ],
                        n,
                    ),
                },
                "tool_durations": {
                    name: distribution(
                        [
                            v
                            for p in samples
                            for v in p["tool_durations"].get(name, [])
                            if number(v)
                        ],
                        sum(len(p["tool_durations"].get(name, [])) for p in samples),
                    )
                    for name in sorted(
                        {name for p in samples for name in p["tool_durations"]}
                    )
                },
            }

    def record_prediction(self, request_id: str, prediction: dict) -> None:
        """Retain pre-execution forecasts for independent, subsequent residuals."""
        with self.lock:
            if request_id in self.pending and context_key(
                self.pending[request_id].get("descriptor")
            ) != context_key(prediction.get("descriptor")):
                raise ValueError("Request prediction context already frozen")
            if request_id not in self.samples:
                self.pending.setdefault(request_id, copy.deepcopy(prediction))

    def _calibration(self, descriptor, as_of_ns, metrics):
        errors = [
            e
            for e in self.errors
            if e["available_ns"] < as_of_ns
            and e["revision"] == self.retriever.revision
            and compatible(descriptor, e["descriptor"])
            and model_settings(e["descriptor"].get("model")) == model_settings(descriptor.get("model"))
            and e["descriptor"].get("control") == descriptor.get("control")
        ]
        intervals = {}
        for key in METRICS:
            residuals = sorted(
                e["metrics"][key] for e in errors if number(e["metrics"].get(key))
            )
            # Finite sample absolute residual interval, never embedding confidence.
            rank = math.ceil((len(residuals) + 1) * 0.9)
            radius = (
                residuals[rank - 1]
                if len(residuals) >= 10 and rank <= len(residuals)
                else None
            )
            center = metrics[key]["p50"]
            intervals[key] = {
                "n": len(residuals),
                "lower": max(0, center - radius)
                if center is not None and radius is not None
                else None,
                "upper": center + radius
                if center is not None and radius is not None
                else None,
            }
        return {
            "method": "prequential_absolute_residual",
            "nominal_coverage": 0.9,
            "calibrated": all(
                intervals[k]["lower"] is not None
                for k in ("tool_elapsed_s", "state_bytes", "llm_idle_s")
            ),
            "intervals": intervals,
            "missing_information": "Intervals assume exchangeable future residuals; no coverage guarantee under drift",
        }

    def save(self, path: Path) -> None:
        """Atomically persist samples; active requests are intentionally process-local."""
        with self.lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            write_json(
                Path(path),
                {
                    "schema_version": 2,
                    "retrieval_config": self.retrieval_config,
                    "errors": self.errors,
                    "samples": list(self.samples.values()),
                    "initial_profiles": self.initial_profiles,
                },
            )

    @classmethod
    def load(
        cls, path: Path, *, retriever=None, retrieval_config=None
    ) -> "WorkloadModel":
        """Load a model snapshot without changing observation availability times."""
        payload = json.loads(read_text(Path(path)))
        if payload.get("schema_version") not in (1, 2):
            raise ValueError("Unsupported workload model schema")
        model = cls(
            payload.get("initial_profiles"),
            retrieval_config=retrieval_config
            if retrieval_config is not None
            else payload.get("retrieval_config"),
            retriever=retriever,
        )
        model.errors = payload.get("errors", [])
        for sample in payload["samples"]:
            model.update(sample["profile"], sample["available_ns"])
        return model
