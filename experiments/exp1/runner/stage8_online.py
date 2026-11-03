"""Rolling causal routing diagnostics and an explicitly synthetic cold-start mechanism check."""

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import fields
import json
from pathlib import Path

from experiments.exp1.runner.arrival_scheduler import stats
from experiments.exp1.runner.mix import write
from experiments.exp1.data.prepare_mix import ROOT, sha


def summarize(events: list[dict], window: int = 50) -> dict:
    """Separate actual execution charges, simulated usage, and unpriced infrastructure."""
    if type(window) is not int or window < 1:
        raise ValueError("Window must be a positive request count")
    decisions = {e["request_id"]: e for e in events if e["event"] == "decision" and not e["shadow"]}
    completions = sorted((e for e in events if e["event"] == "completion"
                          and e["request_id"] in decisions and not e["accounting"]["shadow"]),
                         key=lambda e: e["accounting"]["available_ns"])
    if len({e["request_id"] for e in completions}) != len(completions):
        raise ValueError("Duplicate completion evidence")
    cumulative, windows, family_counts = defaultdict(float), [], defaultdict(Counter)
    for offset in range(0, len(completions), window):
        batch = completions[offset:offset + window]
        routes, errors, latencies = Counter(), [], []
        costs, known_costs, explored, fallback, successes = [], 0, 0, 0, 0
        for event in batch:
            decision = decisions[event["request_id"]]
            accounting = event["accounting"]
            selected = event["selected"]
            if selected != decision["selected"]:
                raise ValueError("Feedback is not from the selected provider")
            candidate = next(c for c in decision["candidates"] if c["key"] == selected)
            latency = accounting["latency_s"]
            if latency is not None:
                latencies.append(latency)
                if candidate["objective_latency"] is not None:
                    errors.append(abs(candidate["objective_latency"] - latency))
            routes[selected] += 1
            family = decision["exploration"].get("family", "unclassified")
            family_counts[family][selected] += 1
            explored += bool(decision["exploration"]["explored"])
            fallback += decision["reason"] == "validated_default"
            successes += accounting["status"] == "success"
            cumulative[accounting["basis"]] += accounting["amount"]
            if accounting["basis"] == "attributable_execution_cost" and accounting["actual_cost"] is not None:
                costs.append(accounting["actual_cost"])
                known_costs += 1
        n = len(batch)
        windows.append({"completed": offset + n, "requests": n, "latency_s": stats(latencies),
                        "latency_prediction_absolute_error_s": stats(errors),
                        "prediction_coverage": len(errors) / n,
                        "routing_proportions": {key: value / n for key, value in routes.items()},
                        "exploration_rate": explored / n, "fallback_rate": fallback / n,
                        "success_rate": successes / n,
                        "mean_actual_execution_cost": sum(costs) / n if known_costs == n else None,
                        "cumulative_accounted_by_basis": dict(cumulative)})
    return {"decisions": len(decisions), "completed": len(completions),
            "rejected": sum(e["selected"] is None for e in decisions.values()),
            "windows": windows, "family_provider_evidence": {key: dict(v) for key, v in family_counts.items()},
            "total_infrastructure_actual_cost": None, "savings_vs_best_fixed_provider": None,
            "scope": "Completion-order diagnostics; full costs and savings require independently executed matched baselines and complete billing"}


def smoke(output: Path, count: int = 300) -> dict:
    """Use existing independently split synthetic service samples; never populate Mix timings."""
    from atsuite.routing.exploration import Exploration
    from atsuite.routing.simulation import simulate
    from experiments.exp1.runner.stage6_simulation import build

    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for composition in ("stable", "shift"):
        for policy in ("routing", "joint"):
            controller, template, workflows, surface = build(0.001, policy)
            controller.period_s = 15
            router = controller.router
            router.workload.samples.clear()
            router.provider.samples.clear()
            router.budget = 1000
            router.exploration = Exploration({"epsilon": 0.5, "budget": 300, "seed": 8})
            router.log_path = output / f"{composition}-{policy}.jsonl"
            descriptors = {r["task_id"]: r["descriptor"] for r in template}
            tasks = list(workflows)
            arrivals = []
            for i in range(count):
                task = tasks[i % len(tasks)] if composition == "stable" else tasks[0 if i < count // 2 else 2]
                arrivals.append({"id": f"cold-{i}", "task_id": task,
                                 "descriptor": descriptors[task], "at_s": 1 + i * 15})
            result = simulate(controller, arrivals, workflows, surface, seed=8,
                              end_s=count * 15 + 30, start_ns=10**9)
            events = [json.loads(line) for line in router.log_path.read_text().splitlines()]
            report = {"composition": composition, "policy": policy, "synthetic": True,
                      "surface_validation_strata": len(result["validation"]),
                      "simulation": {k: v for k, v in result.items() if k != "validation"},
                      "online": summarize(events)}
            write(output / f"{composition}-{policy}.json", report)
            reports.append({"composition": composition, "policy": policy,
                            "completed": report["online"]["completed"],
                            "exploration": result["exploration"]})
    summary = {"reports": reports,
               "scope": "Synthetic mechanism validation with empty history; not the formal 100-Prompt E2 experiment or a cloud saving"}
    write(output / "summary.json", summary)
    plot_smoke(output)
    return summary


def plot_smoke(output: Path) -> None:
    """Plot measured simulator decisions, with an explicit synthetic scope label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    for axis, (composition, policy) in zip(axes.flat, (
        ("stable", "routing"), ("stable", "joint"), ("shift", "routing"), ("shift", "joint")
    )):
        rows = json.loads((output / f"{composition}-{policy}.json").read_text())["online"]["windows"]
        x = [row["completed"] for row in rows]
        axis.plot(x, [row["exploration_rate"] for row in rows], marker="o", label="Exploration")
        axis.plot(x, [row["fallback_rate"] for row in rows], marker="s", label="Fallback")
        axis.plot(x, [row["routing_proportions"].get("fast", 0) for row in rows], linestyle="--", label="Fast pool share")
        if composition == "shift":
            axis.axvline(150, color="gray", alpha=0.5, linewidth=1)
        axis.set(title=f"{composition} / {policy}", ylim=(-0.03, 1.03), xlabel="Completed requests")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("Share in 50-request window")
    axes[1, 0].set_ylabel("Share in 50-request window")
    figure.suptitle("Synthetic cold-start checks — not real cloud cost or convergence evidence")
    figure.tight_layout()
    figure.savefig(output / "online-trends.png", dpi=180)
    figure.savefig(output / "online-trends.pdf")
    plt.close(figure)


async def execute_local(sequence: Path, config_path: Path, output: Path, *, duration_ms: int | None = None,
                        concurrency: int = 4) -> dict:
    """Replay sealed real arrivals on local FaaS and learn only from completed real Tools."""
    from atsuite.provider_model import ProviderModel
    from atsuite.workload import WorkloadModel
    from experiments.exp1.capture.capture_agent_traces import run
    from experiments.exp1.runner.arrival_scheduler import Arrival, ms_ns, run_open_loop
    from experiments.exp1.runner.mix import input_records, verify_capture

    plan = json.loads((sequence / "manifest.json").read_text())
    seal = json.loads((sequence / "seal.json").read_text())
    if sha((sequence / "manifest.json").read_bytes()) != seal["manifest_sha256"]:
        raise ValueError("Sequence seal changed")
    if sha((sequence / "arrivals.json").read_bytes()) != plan["arrivals_sha256"]:
        raise ValueError("Arrival assignments changed")
    manifest_path = Path(plan["mix_manifest"])
    if sha(manifest_path.read_bytes()) != plan["mix_manifest_sha256"]:
        raise ValueError("Task manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("reference_verified_count", 0) < 100:
        raise ValueError("Strict local replay requires all 100 environment references")
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    for task in tasks.values():
        if input_records(ROOT / "benchmarks/DataSciBench/data" / task["task_id"]) != task["readiness"]["inputs"]["files"]:
            raise ValueError("Prepared inputs changed")
        if verify_capture(task, Path(task["replay_reference"]["directory"]), require_live=False) != task["replay_reference"]:
            raise ValueError("Environment reference changed")
    config = json.loads(config_path.read_text())
    if config["tool"]["provider"] != "local" or config["tool"]["family"] != "faas" or config["local_pool"]["provider"] != "local":
        raise ValueError("This local validation entrypoint cannot dispatch cloud or shared MCP Sessions")
    duration = plan["duration_ms"] if duration_ms is None else duration_ms
    if not 0 < duration <= plan["duration_ms"]:
        raise ValueError("Choose an explicit prefix within the sealed time axis")
    rows = [r for r in json.loads((sequence / "arrivals.json").read_text()) if ms_ns(r["relative_arrival_ms"]) < ms_ns(duration)]
    by_id = {r["arrival_id"]: r for r in rows}
    workload, provider, offered = WorkloadModel(), ProviderModel(), {}
    output.mkdir(parents=True, exist_ok=False)

    def observe(event):
        if event["kind"] == "request_submitted":
            offered[event["arrival_id"]] = event["wall_ns"]

    async def handler(arrival, started):
        started()
        task = tasks[by_id[arrival.arrival_id]["task_id"]]
        now = offered[arrival.arrival_id]
        rate = sum(now - 30e9 < stamp <= now for stamp in offered.values()) / 30
        return await run(task["task_id"], config, output / "requests" / arrival.arrival_id,
                         capture=Path(task["replay_reference"]["directory"]), workload_model=workload,
                         provider_model=provider, provider_context={"pool": config["local_pool"], "load_rps": rate},
                         manifest=manifest_path, arrival_ns=now)

    try:
        result = await run_open_loop([Arrival(**{f.name: r[f.name] for f in fields(Arrival)}) for r in rows],
                                    handler, output / "schedule", duration_ms=duration, concurrency=concurrency,
                                    observer=observe, queue_capacity=1000,
                                    metadata={"sequence_sha256": seal["manifest_sha256"], "prefix_ms": duration,
                                              "scope": "Real local Tool replay and online updates; no cross-provider cost claim"})
        write(output / "report.json", result)
        return result
    finally:
        workload.save(output / "workload_model.json")
        provider.save(output / "provider_model.json")


def main() -> None:
    """Summarize real controller logs or run the separate synthetic mechanism check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["summarize", "smoke", "local"])
    parser.add_argument("--events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--duration-ms", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke(args.output)
    elif args.mode == "local":
        if not args.sequence or not args.config:
            parser.error("local requires --sequence and --config")
        asyncio.run(execute_local(args.sequence, args.config, args.output,
                                  duration_ms=args.duration_ms, concurrency=args.concurrency))
    else:
        if not args.events:
            parser.error("summarize requires --events")
        write(args.output, summarize([json.loads(line) for line in args.events.read_text().splitlines()]))


if __name__ == "__main__":
    main()
