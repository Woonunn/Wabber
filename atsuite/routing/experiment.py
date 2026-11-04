"""Reproducible joint-control configuration, comparison plots and evidence preflight."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from atsuite.provider_model import ProviderModel
from atsuite.provider_model.e1a import load_e1a
from atsuite.provider_model.e1b import load_e1b
from atsuite.routing import JointController, OnlineRouter, candidate_plans
from atsuite.routing.simulation import (
    ServiceSurface,
    SimulatedAdapter,
    load_sealed_workload,
    simulate,
)
from atsuite.tracing.storage import read_text
from atsuite.workload import WorkloadModel
from atsuite.workload.model import distribution

STRATEGIES = (
    "fixed_aws_lambda",
    "fixed_aws_agentcore",
    "fixed_ali_fc",
    "fixed_ali_agentrun",
    "cost_aware",
    "random_exploration",
    "exploration_aware",
    "full_wabber",
)


def strategy_config(config: dict, strategy: str) -> tuple[dict, str]:
    """Keep sealed workload/deployments unchanged and construct one comparison policy."""
    if strategy not in STRATEGIES:
        raise ValueError("Unknown Stage 7 strategy")
    config = copy.deepcopy(config)
    config["strategy"] = strategy
    routing = config.setdefault("routing", {})
    control = config.setdefault("control", {})
    if strategy != "full_wabber":
        control.pop("capacity_exploration", None)
    if strategy.startswith("fixed_"):
        provider = strategy.removeprefix("fixed_")
        keys = [
            k
            for k, p in config["pools"].items()
            if p["levels"][0]["pool"]["provider"] == provider
        ]
        if len(keys) != 1:
            raise ValueError(
                "Fixed strategy requires exactly one validated provider pool"
            )
        key = keys[0]
        routing["default_pool"] = key
        routing.pop("exploration", None)
        config["weights"] = [{k: float(k == key) for k in config["pools"]}]
        policy = "static"
    elif strategy == "cost_aware":
        routing.pop("exploration", None)
        policy = "routing"
    else:
        if not routing.get("exploration"):
            raise ValueError(
                "Exploration strategy requires explicit epsilon, budget and bounded executors"
            )
        routing["exploration"]["strategy"] = (
            "random" if strategy == "random_exploration" else "contextual"
        )
        policy = "joint" if strategy == "full_wabber" else "routing"
    return config, policy


def implementation_hashes() -> dict:
    """Seal controller/model sources as well as experimental data."""
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "atsuite/routing").glob("*.py"))
    paths += list((root / "atsuite/provider_model").glob("*.py"))
    paths += list((root / "atsuite/workload").glob("*.py"))
    paths += [
        root / p
        for p in (
            "atsuite/provider_model/model.py",
            "atsuite/provider_model/observation.py",
            "atsuite/workload/model.py",
            "atsuite/workload/profile.py",
            "experiments/exp1/runner/run_joint.py",
            "experiments/exp1/runner/stage7.py",
            "experiments/exp1/runner/arrival_scheduler.py",
            "experiments/exp1/capture/capture_agent_traces.py",
            "benchmarks/DataSciBench/langgraph_agent/graph.py",
            "benchmarks/DataSciBench/langgraph_agent/validators.py",
            "benchmarks/DataSciBench/langgraph_agent/trace_replayer.py",
            "benchmarks/DataSciBench/nodes/AgentTools/implementation.py",
            "benchmarks/DataSciBench/nodes/DataSci/implementation.py",
            "benchmarks/DataSciBench/langgraph_agent/tool_client.py",
        )
    ]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def control_identity(config: dict) -> str:
    """Bind simulation and cloud admission to the same model, rates and controller."""
    keys = (
        "pools",
        "weights",
        "fx",
        "routing",
        "control",
        "initial_task_rates",
        "strategy",
    )
    value = {k: config.get(k) for k in keys}
    value["implementation_sha256"] = implementation_hashes()
    value["models"] = {
        k: hashlib.sha256(read_text(Path(config[k])).encode()).hexdigest()
        for k in ("workload_model", "provider_model")
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def build_controller(config: dict, adapters: dict, *, policy="joint", log_path=None):
    """Build independent model/controller instances for each comparison strategy."""
    if config.get("strategy"):
        config, policy = strategy_config(config, config["strategy"])
    levels = {
        key: {level["target"]: level for level in value["levels"]}
        for key, value in config["pools"].items()
    }
    if any(len(levels[k]) != len(v["levels"]) for k, v in config["pools"].items()):
        raise ValueError("Duplicate capacity target")
    pools = {k: levels[k][v["initial"]]["pool"] for k, v in config["pools"].items()}
    router = OnlineRouter(
        WorkloadModel.load(Path(config["workload_model"]), retrieval_config=config.get("retrieval")),
        ProviderModel.load(Path(config["provider_model"])),
        pools,
        {},
        **{**config.get("routing", {}), "fx": config["fx"]},
        log_path=log_path,
    )
    return JointController(
        router,
        adapters,
        levels,
        candidate_plans(levels, config["weights"]),
        config["fx"],
        policy=policy,
        **config.get("control", {}),
    )


def plot_comparison(reports: list[dict], output: Path) -> None:
    """Write a standalone figure with admission, latency and cost provenance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    names = [r["policy"] for r in reports]
    values = [
        (
            "Infrastructure cost (estimated " + reports[0]["cost"]["currency"] + ")",
            [r["cost"]["total"] for r in reports],
        ),
        ("Successful-request p95 (s)", [r["p95_s"] for r in reports]),
        ("Unsatisfied / offered", [r["unsatisfied_rate_offered"] for r in reports]),
        ("Admission / offered", [r["admission_rate"] for r in reports]),
        ("Success / offered", [r["success_rate"] for r in reports]),
        ("Scaling requests", [len(r["scaling_events"]) for r in reports]),
    ]
    for ax, (title, numbers) in zip(axes.flat, values):
        ax.bar(names, [v if v is not None else 0 for v in numbers])
        for i, value in enumerate(numbers):
            if value is None:
                ax.text(i, 0, "unknown", rotation=90, va="bottom", ha="center")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(
        "Synthetic smoke test — no cloud benefit evidence"
        if any(r.get("synthetic") for r in reports)
        else "Simulation only — no observed mixed-cloud savings"
    )
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


def audit_evidence(rows: list[dict], provenance: dict) -> dict:
    """Report missing Stage 5 evidence without fabricating queues/capacity/prices."""
    groups = {}
    for row in rows:
        if row["kind"] != "request":
            continue
        key = (
            row["pool"]["provider"],
            row.get("task_id", "unknown"),
            json.dumps(row["pool"], sort_keys=True),
            row["load_rps"],
        )
        groups.setdefault(key, []).append(row)
    strata = []
    for (provider, task, _, rate), values in sorted(groups.items()):
        pool = values[0]["pool"]
        latencies = [
            r["measurements"]["latency_s"]
            for r in values
            if r["measurements"].get("latency_s") is not None
        ]
        flags = []
        if "capacity_control" not in pool["resources"]:
            flags.append("no_calibrated_capacity_control_identity")
        for field in ("queue_s", "cold_start_s", "service_slot_s"):
            if any(r["measurements"].get(field) is None for r in values):
                flags.append("missing_" + field)
        if any(not r.get("cost") or not r["cost"]["complete"] for r in values):
            flags.append("incomplete_infrastructure_cost")
        strata.append(
            {
                "provider": provider,
                "task": task,
                "load_rps": rate,
                "latency": distribution(latencies, len(values)),
                "flags": flags,
            }
        )
    return {
        "status": "blocked"
        if not strata or any(s["flags"] for s in strata)
        else "needs_independent_validation",
        "strata": strata,
        "provenance": provenance,
        "real_cloud_joint_experiment_completed": False,
        "actual_cost_savings": None,
        "independent_simulator_validation": "requires distinct source runs and Tool service/cold/queue measurements",
    }


def main() -> None:
    """Audit E1 evidence or run a validated, seeded four-policy simulation."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--e1b", type=Path)
    source.add_argument("--e1a", type=Path)
    source.add_argument("--simulation", type=Path)
    source.add_argument("--readback-deployment", type=Path)
    source.add_argument("--sealed-workload", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.sealed_workload:
        arrivals, workflows, provenance = load_sealed_workload(args.sealed_workload)
        for name, value in (
            ("arrivals", arrivals),
            ("workflows", workflows),
            ("provenance", provenance),
        ):
            (args.output / (name + ".json")).write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n"
            )
        print(args.output)
        return
    if args.readback_deployment:
        from dotenv import load_dotenv

        from experiments.exp1.data.prepare_e1b import ROOT, sha
        from experiments.exp1.runner.run_joint import cloud_adapters

        load_dotenv(ROOT / ".env", override=False)
        entries = json.loads(args.readback_deployment.read_text())["results"]
        adapters = cloud_adapters(
            entries, {"pools": {e["provider"]: {} for e in entries}}
        )
        records = []
        for key, adapter in adapters.items():
            try:
                records.append(
                    {
                        "provider": key,
                        "resource": adapter.resource,
                        "readback": adapter.read().to_dict(),
                        "configuration": adapter.configuration,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - retain each independent provider read failure.
                records.append({"provider": key, "error_type": type(exc).__name__})
        result = {
            "deployment_sha256": sha(args.readback_deployment),
            "mutations": False,
            "pools": records,
        }
        (args.output / "readback.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        print(args.output)
        return
    if args.e1a or args.e1b:
        rows, provenance = load_e1a(args.e1a) if args.e1a else load_e1b(args.e1b)
        report = audit_evidence(rows, provenance)
        (args.output / "preflight.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(
            [s["provider"] + "\n" + s["task"] for s in report["strata"]],
            [s["latency"]["p95"] or 0 for s in report["strata"]],
        )
        ax.tick_params(axis="x", labelsize=6, rotation=70)
        ax.set_ylabel("Observed p95 latency (s)")
        ax.set_title("Existing independent-pool evidence; not joint-run outcomes")
        fig.tight_layout()
        fig.savefig(args.output / "latency_coverage.png", dpi=160)
        plt.close(fig)
    else:
        config = json.loads(args.simulation.read_text())
        import hashlib

        paths = [args.simulation] + [
            Path(config[k])
            for k in ("training", "validation", "provider_model", "workload_model")
        ]
        if not config.get("sealed_workload"):
            paths.extend(Path(config[k]) for k in ("arrivals", "workflows"))
        provenance = {
            str(p): hashlib.sha256(read_text(p).encode()).hexdigest() for p in paths
        }
        provenance["implementation_sha256"] = implementation_hashes()
        training, validation = [
            json.loads(read_text(Path(config[k]))) for k in ("training", "validation")
        ]
        surface = ServiceSurface(
            training, validation, **config.get("validation_thresholds", {})
        )
        if config.get("sealed_workload"):
            arrivals, workflows, sealed = load_sealed_workload(
                Path(config["sealed_workload"])
            )
            if config["end_s"] < sealed["window_duration_s"]:
                raise ValueError(
                    "The simulator must preserve the sealed trailing idle window"
                )
            provenance["sealed_workload"] = sealed
        else:
            workflows = json.loads(read_text(Path(config["workflows"])))
            arrivals = json.loads(read_text(Path(config["arrivals"])))
        reports = []
        policies = (
            [strategy_config(config, config["strategy"])[1]]
            if config.get("strategy")
            else ("static", "routing", "independent", "joint")
        )
        for policy in policies:
            adapters = {
                k: SimulatedAdapter(
                    v["control"],
                    v["initial"],
                    {level["target"]: level["scale_delay_s"] for level in v["levels"]},
                )
                for k, v in config["pools"].items()
            }
            controller = build_controller(
                config,
                adapters,
                policy=policy,
                log_path=args.output / (policy + ".jsonl"),
            )
            report = simulate(
                controller,
                arrivals,
                workflows,
                surface,
                seed=config["seed"],
                end_s=config["end_s"],
                start_ns=config["start_ns"],
            )
            report["provenance"] = provenance
            report["control_identity"] = control_identity(config)
            (args.output / (policy + ".json")).write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            reports.append(report)
        plot_comparison(reports, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
