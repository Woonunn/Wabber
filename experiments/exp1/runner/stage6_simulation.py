"""Synthetic mechanism study using the existing controller and Tool-level simulator."""

import argparse
import copy
from collections import Counter
import json
from pathlib import Path

from atsuite.provider_model import ProviderModel
from atsuite.routing import JointController, OnlineRouter, candidate_plans
from atsuite.routing.objective import pareto
from atsuite.routing.simulation import ServiceSurface, SimulatedAdapter, simulate
from atsuite.workload import WorkloadModel, describe, extract_profile


def build(latency_weight, policy):
    cards = json.loads(Path("experiments/exp1/configs/stage6-tariffs.json").read_text())
    levels, training, validation, workflows, descriptors = {}, [], [], {}, {}
    provider, workload = ProviderModel(), WorkloadModel()
    for task, tool_s, idle_s, state in [
        ("compute", 4, 0.1, 0),
        ("state", 1, 0.1, 10**6),
        ("idle", 0.2, 5, 1000),
    ]:
        access = [{"object_id": "state", "operation": "write", "bytes": state}]
        workflows[task] = {
            "nodes": [
                {"id": "llm", "kind": "llm", "name": "model", "wait_s": idle_s},
                {
                    "id": "tool",
                    "kind": "tool",
                    "name": "python",
                    "metadata": {"state_accesses": access},
                    "resources": {"cpu_seconds": tool_s / 2},
                },
            ],
            "edges": [["llm", "tool"]],
        }
        descriptor = describe(
            prompt={
                "compute": "Train a classifier",
                "state": "Sort and retain shared data",
                "idle": "Calculate the mean",
            }[task],
            files=[{"name": "input.csv", "bytes": max(1, state)}],
            tools=["python"],
        )
        descriptors[task] = descriptor
        trace = {
            "schema_version": 1,
            "request_id": task,
            "session_id": task,
            "arrival_ns": 1,
            "completed_ns": 10,
            "diagnostics": [],
            "source": "synthetic",
            "spans": [
                {
                    "id": "session",
                    "name": "agent",
                    "kind": "session",
                    "start_ns": 0,
                    "end_ns": int((tool_s + idle_s) * 1e9),
                    "duration_s": tool_s + idle_s,
                    "status": "ok",
                    "metadata": {},
                },
                {
                    "id": "llm",
                    "name": "model",
                    "kind": "llm",
                    "start_ns": 0,
                    "end_ns": int(idle_s * 1e9),
                    "duration_s": idle_s,
                    "status": "ok",
                    "metadata": {},
                    "parent_id": "session",
                },
                {
                    "id": "tool",
                    "name": "python",
                    "kind": "tool",
                    "start_ns": int(idle_s * 1e9),
                    "end_ns": int((tool_s + idle_s) * 1e9),
                    "duration_s": tool_s,
                    "status": "ok",
                    "metadata": {"state_accesses": access},
                    "resources": {"cpu_seconds": tool_s / 2},
                    "parent_id": "session",
                },
            ],
        }
        p = extract_profile(trace)
        p["reference_id"] = "stage6-synthetic-v1"
        p["descriptor"] = descriptor
        workload.update(p, 100, descriptor=descriptor)
        for name, memory, speed in [("cheap", 1024, 1), ("fast", 4096, 0.2)]:
            levels.setdefault(name, {})
            for target in (1, 2, 4):
                card = copy.deepcopy(cards["aws_lambda"])
                card["rates"]["pool_per_s"] = 0.00001 * target * (memory / 1024)
                card["rates"]["scale_event"] = 0.00001
                card["source"] = [
                    "synthetic reservation surcharge plus documented Lambda on-demand rate"
                ]
                card["calibration"] = {
                    "tool_scale": speed,
                    "reference_id": "stage6-synthetic-v1",
                    "evidence": ["synthetic independent profiling"],
                }
                pool = {
                    "provider": "aws_lambda",
                    "environment": "synthetic-" + name,
                    "resources": {
                        "cpu": 2,
                        "memory_mb": memory,
                        "disk_mb": 512,
                        "capacity_control": {
                            "control": "reserved_concurrency",
                            "target": target,
                        },
                    },
                    "instances": None,
                    "sessions_per_instance": 1,
                    "invocations_per_instance": 1,
                    "billing": card,
                }
                levels[name][target] = {
                    "pool": pool,
                    "session_limit": target * 3,
                    "tool_slots": target,
                    "evidence": ["synthetic capacity"],
                    "scale_delay_s": 0.5,
                    "target": target,
                }
                for i in range(3):
                    provider.update(
                        {
                            "kind": "request",
                            "id": f"{name}-{target}-{task}-{i}",
                            "task_id": "training-" + task,
                            "profile": p,
                            "pool": pool,
                            "load_rps": i,
                            "status": "success",
                            "arrival_ns": 1,
                            "completed_ns": 10,
                            "available_ns": 100,
                            "measurements": {
                                "latency_s": idle_s + speed * tool_s,
                                "tool_s": speed * tool_s,
                                "state_sync_s": 0,
                                "cold_start_s": 0,
                                "queue_s": 0,
                            },
                            "cost": None,
                        }
                    )
                for run, destination in [
                    ("train", training),
                    ("independent", validation),
                ]:
                    for load in (
                        0.0,
                        0.01,
                        0.2,
                        0.4,
                        0.6,
                        0.8,
                        1.0,
                        1.2,
                        1.4,
                        1.6,
                        1.8,
                        2.0,
                    ):
                        for concurrent in range(1, target * 3 + 1):
                            for i in range(10):
                                identity = f"{run}-{name}-{target}-{task}-{load}-{concurrent}-{i}"
                                destination.append(
                                    {
                                        "id": identity,
                                        "request_id": identity,
                                        "source_run": run,
                                        "pool": pool,
                                        "task_id": task,
                                        "tool": "python",
                                        "load_rps": load,
                                        "concurrent_sessions": concurrent,
                                        "service_s": tool_s * speed * (1 + i / 100),
                                        "cold_start_s": 0,
                                        "status": "success",
                                        "queue_excluded": True,
                                        "evidence": ["synthetic independent samples"],
                                        "basis": "synthetic",
                                    }
                                )
    bounds = {
        k: {
            "timeout_s": 60,
            "max_cost": 1,
            "currency": "USD",
            "tools": ["python"],
            "evidence": ["synthetic bounded executor"],
        }
        for k in levels
    }
    router = OnlineRouter(
        workload,
        provider,
        {k: v[1]["pool"] for k, v in levels.items()},
        {},
        default_pool="cheap",
        execution_bounds=bounds,
        latency_weight=latency_weight,
    )
    adapters = {
        k: SimulatedAdapter("reserved_concurrency", 1, {1: 0.5, 2: 0.5, 4: 0.5})
        for k in levels
    }
    fx = {
        "currency": "USD",
        "rates": {"USD": 1},
        "source": "identity",
        "date": "2026-10-30",
    }
    controller = JointController(
        router,
        adapters,
        levels,
        candidate_plans(
            levels,
            [
                {"cheap": 1, "fast": 0},
                {"cheap": 0, "fast": 1},
                {"cheap": 0.5, "fast": 0.5},
            ],
        ),
        fx,
        policy=policy,
        period_s=2,
        horizon_s=20,
        observation_s=5,
        cooldown_s=3,
    )
    surface = ServiceSurface(training, validation)
    arrivals = []
    for i, at in enumerate(
        [float(i * 5 + 1) for i in range(4)]
        + [22 + i * 1.1 for i in range(24)]
        + [55 + i * 5 for i in range(4)]
    ):
        task = list(workflows)[i % len(workflows)]
        arrivals.append(
            {
                "id": "unseen-" + str(i),
                "task_id": task,
                "at_s": at,
                "descriptor": descriptors[task],
            }
        )
    return controller, arrivals, workflows, surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for policy in ("static", "routing", "independent", "joint"):
        for weight in (0, 0.00001, 0.001) if policy == "joint" else (0.00001,):
            controller, arrivals, workflows, surface = build(weight, policy)
            report = simulate(
                controller,
                arrivals,
                workflows,
                surface,
                seed=6,
                end_s=90,
                start_ns=10**9,
            )
            report["latency_weight"] = weight
            # Full validation strata are reproducible from the generator and need not duplicate every pool identity.
            report["validation_summary"] = {
                "count": len(report["validation"]),
                "passed": all(v["passed"] for v in report["validation"]),
            }
            report.pop("validation")
            (args.output / f"{policy}-{weight}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            reports.append(report)
            print(
                policy, weight, report["admitted"], report["cost"]["total"], flush=True
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    points = []
    for r in reports:
        name = f"{r['policy']}:{r['latency_weight']}"
        if r["cost"]["total"] is not None and r["p95_s"] is not None:
            axes[0, 0].scatter(
                r["cost"]["total"],
                r["p95_s"],
                label=f"{name} ({r['admitted']}/{r['offered']})",
            )
            points.append(
                {
                    "policy": name,
                    "objective_cost": r["cost"]["total"],
                    "objective_latency": r["p95_s"],
                    "admission_rate": r["admission_rate"],
                }
            )
        latencies = r["latency"]["samples"]
        axes[0, 1].plot(
            latencies,
            [(i + 1) / len(latencies) for i in range(len(latencies))],
            label=name,
        )
        axes[1, 0].plot(
            [p["at_s"] for p in r["cost_timeline"]],
            [p["total"] for p in r["cost_timeline"]],
            label=name,
        )
    names = [f"{r['policy']}:{r['latency_weight']}" for r in reports]
    routed = [
        Counter(row["pool"] or "rejected" for row in r["requests"]) for r in reports
    ]
    bottom = [0] * len(reports)
    for pool_name in ("cheap", "fast", "rejected"):
        values = [c[pool_name] for c in routed]
        axes[0, 2].bar(names, values, bottom=bottom, label=pool_name)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0, 2].set(ylabel="Offered requests by route / rejection")
    axes[0, 2].tick_params(axis="x", rotation=30)
    axes[1, 2].bar(
        names, [r["success_rate"] for r in reports], label="success / offered"
    )
    axes[1, 2].set(ylim=(0, 1), ylabel="Completion fraction; compare before costs")
    axes[1, 2].tick_params(axis="x", rotation=30)
    for p in ("cheap", "fast"):
        timeline = [r for r in reports[-1]["capacity_timeline"] if r["pool"] == p]
        axes[1, 1].step(
            [r["at_s"] for r in timeline],
            [r["target"] for r in timeline],
            where="post",
            label=p,
        )
    axes[0, 0].set(
        xlabel="Synthetic infrastructure cost (USD)",
        ylabel="Successful-request p95 (s)",
    )
    axes[0, 1].set(xlabel="Latency (s)", ylabel="CDF")
    axes[1, 0].set(xlabel="Time (s)", ylabel="Cumulative infrastructure estimate (USD)")
    axes[1, 1].set(
        xlabel="Time (s)", ylabel="Capacity control target (not physical instances)"
    )
    for ax in axes.flat:
        ax.legend(fontsize=6)
    fig.suptitle(
        "Synthetic mechanism study; inspect admission/failure rates, not cloud savings"
    )
    fig.tight_layout()
    fig.savefig(args.output / "pareto-capacity.png", dpi=150)
    fig.savefig(args.output / "pareto-capacity.pdf")
    plt.close(fig)
    (args.output / "pareto.json").write_text(
        json.dumps(
            {
                "scope": "synthetic only; compare equal completed workloads",
                "points": points,
                "frontiers_by_admitted_count": {
                    str(n): pareto(
                        [p for p, r in zip(points, reports) if r["admitted"] == n]
                    )
                    for n in {r["admitted"] for r in reports}
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
