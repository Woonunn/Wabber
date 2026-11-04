"""Stage 7 sealed comparison plans, family holdouts and attributable run reports."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from atsuite.routing.cost import total_cost
from atsuite.routing.experiment import (
    STRATEGIES,
    implementation_hashes,
    strategy_config,
)
from atsuite.routing.generalization import TOOLS, evaluate
from atsuite.workload import describe

ROOT = Path(__file__).resolve().parents[3]
PANEL = (
    "human_15",
    "csv_excel_34",
    "human_142",
    "human_12",
    "human_7",
    "csv_excel_29",
    "bcb409",
    "bcb735",
    "bcb700",
    "bcb294",
    "bcb871",
    "bcb969",
)
COMPONENTS = {"execution", "storage", "retention", "scaling", "shared"}


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prompt_panel():
    """Select original DataSciBench prompts from the pinned local catalog, without rewrites."""
    catalog = load(ROOT / "benchmarks/DataSciBench/tasks/catalog.json")
    prepared = {
        t["task_id"]: t
        for t in load(ROOT / "benchmarks/DataSciBench/tasks/manifest.json")["tasks"]
    }
    by_id = {t["task_id"]: t for t in catalog["tasks"]}
    cases = []
    for task_id in PANEL:
        task = by_id[task_id]
        if hashlib.sha256(task["prompt"].encode()).hexdigest() != task["prompt_sha256"]:
            raise ValueError("Catalog prompt hash changed")
        inputs = prepared.get(task_id, {}).get("readiness", {}).get("inputs", {})
        cases.append(
            {
                "task_id": task_id,
                "prompt": task["prompt"],
                "prompt_sha256": task["prompt_sha256"],
                "source_path": task["source_path"],
                "inputs_prepared": inputs.get("status") == "prepared",
                "descriptor": describe(
                    prompt=task["prompt"], files=inputs.get("files", []), tools=TOOLS
                ),
            }
        )
    return {
        "source": catalog["source"],
        "cases": cases,
        "scope": "Original prompts; unprepared cases need the existing dataset/output validation pipeline before cloud execution",
    }


def make_plan(config, workload, output):
    """Emit all strategies against identical frozen arrivals, harness, Tool and model inputs."""
    workload = Path(workload)
    files = [workload / "manifest.json", workload / "planned_requests.csv"]
    # Existing manifests may call the sealed arrival table requests.csv.
    if not files[1].exists():
        files[1] = workload / "requests.csv"
    files += [Path(config[k]) for k in ("workload_model", "provider_model")]
    files += [
        ROOT / "benchmarks/DataSciBench/tasks" / name
        for name in ("catalog.json", "manifest.json", "datasets.lock.json")
    ]
    files += [ROOT / item["path"] for item in config.get("evidence", [])]
    identity = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    strategies = []
    for strategy in STRATEGIES:
        variant, policy = strategy_config(config, strategy)
        path = Path(output) / (strategy + ".json")
        write(path, variant)
        strategies.append(
            {
                "strategy": strategy,
                "controller_policy": policy,
                "config": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "inputs_sha256": identity,
        "implementation_sha256": implementation_hashes(),
        "strategies": strategies,
        "scope": "Execution plan only; run each strategy through run_joint preflight and its matching Pilot",
        "arrival_clock": "identical relative timestamps, independent wall-clock run starts",
        "cloud_runs_completed": False,
    }


def collect(run_dir, config, charges):
    """Join exclusive cost evidence to real decision logs, retaining incomplete totals as unknown."""
    run_dir = Path(run_dir)
    run = load(run_dir / "run.json")
    events = [
        json.loads(line)
        for line in (run_dir / "control.jsonl").read_text().splitlines()
    ]
    outcomes_path = run_dir / "request_metrics.jsonl"
    outcomes = (
        [json.loads(line) for line in outcomes_path.read_text().splitlines()]
        if outcomes_path.exists()
        else []
    )
    decisions = [e for e in events if e["event"] == "decision" and not e["shadow"]]
    completions = [e for e in events if e["event"] == "completion"]
    costs = total_cost(charges, config["fx"], components=COMPONENTS)
    required = {(r["provider"], r["resource"]) for r in run.get("cost_resources", [])}
    covered = set(map(tuple, costs["resources"]))
    full_window = True
    for resource in required:
        for component in COMPONENTS:
            spans = sorted(
                {
                    (c["start_ns"], c["end_ns"])
                    for c in charges
                    if (c["provider"], c["resource"]) == resource
                    and c["component"] == component
                }
            )
            cursor = run["started_ns"]
            for start, end in spans:
                if start != cursor:
                    full_window = False
                cursor = end
            full_window &= cursor == run.get("finished_ns")
    costs["complete"] &= bool(required) and covered == required and full_window
    if not costs["complete"]:
        costs["total"] = None
    counts, timeline = Counter(), []
    for d in decisions:
        counts[d["selected"]] += 1
        timeline.append(
            {
                "request_id": d["request_id"],
                "decision_ns": d["decision_ns"],
                "selected": d["selected"],
                "exploration": d.get("exploration"),
                "selection_counts": dict(counts),
            }
        )
    explored = {
        d["request_id"] for d in decisions if d.get("exploration", {}).get("explored")
    }
    accounted = [
        c["accounting"]
        for c in completions
        if c["request_id"] in explored and c.get("accounting")
    ]
    admitted = sum(counts[k] for k in config["pools"])
    return {
        "strategy": run.get("strategy", run["policy"]),
        "workload_sha256": run["workload_sha256"],
        "config_sha256": run["config_sha256"],
        "comparison_sha256": run.get("comparison_sha256"),
        "deployment_sha256": run.get("deployment_sha256"),
        "cost": costs,
        "metrics": run.get("metrics"),
        "selection_counts": {k: counts[k] for k in config["pools"]},
        "selection_proportions": {
            k: counts[k] / admitted if admitted else 0 for k in config["pools"]
        },
        "decision_timeline": timeline,
        "request_outcomes": outcomes,
        "capacity_events": [
            e
            for e in events
            if e["event"].startswith(("scaling_", "capacity_", "exploration_capacity_"))
        ],
        "exploration_attempts": len(explored),
        "exploration_execution_cost_by_basis": {
            b: sum(c["amount"] for c in accounted if c["basis"] == b)
            for b in {c["basis"] for c in accounted}
        },
        "scope": "Run-specific measured outcomes; cost includes every configured resource and shared control charges only when evidence is complete",
        "realized_counterfactual_gain": None,
    }


def compare(reports: list[dict]) -> dict:
    """Report matched-run cost/latency differences with explicit accounting provenance."""
    if not reports or len({r["workload_sha256"] for r in reports}) != 1:
        raise ValueError(
            "Comparison requires the same sealed workload and arrival schedule"
        )
    for field in ("comparison_sha256", "deployment_sha256"):
        if (
            not all(r.get(field) for r in reports)
            or len({r[field] for r in reports}) != 1
        ):
            raise ValueError(
                "Comparison requires the same sealed deployment, harness, model priors and objective"
            )
    baseline = next((r for r in reports if r["strategy"] == "cost_aware"), None)
    if baseline is None:
        raise ValueError("A no-exploration cost_aware baseline is required")
    if len({r["strategy"] for r in reports}) != len(reports):
        raise ValueError("Use one independently executed run per strategy and seed")
    fixed = [r for r in reports if r["strategy"].startswith("fixed_")]
    best_fixed = min(fixed, key=lambda r: r["cost"]["total"]) if (
        {r["strategy"] for r in fixed} == {s for s in STRATEGIES if s.startswith("fixed_")}
        and all(r["cost"]["complete"] and r["cost"]["basis"] in {"actual", "estimated"} for r in fixed)
        and len({(r["cost"]["currency"], r["cost"]["basis"]) for r in fixed}) == 1
    ) else None
    comparisons = []
    for report in reports:
        a, b = baseline["cost"], report["cost"]
        comparable = (
            a["complete"]
            and b["complete"]
            and a["currency"] == b["currency"]
            and a["basis"] == b["basis"]
            and a["basis"] in {"actual", "estimated"}
        )
        am, bm = baseline.get("metrics") or {}, report.get("metrics") or {}
        comparisons.append(
            {
                "strategy": report["strategy"],
                "cost_basis": b["basis"],
                "cost_saving_vs_no_exploration": a["total"] - b["total"]
                if comparable
                else None,
                "cost_saving_vs_best_fixed": best_fixed["cost"]["total"] - b["total"]
                if best_fixed and b["complete"] and (b["currency"], b["basis"]) == (
                    best_fixed["cost"]["currency"], best_fixed["cost"]["basis"]
                ) else None,
                "p95_reduction_s": am["p95_s"] - bm["p95_s"]
                if am.get("p95_s") is not None and bm.get("p95_s") is not None
                else None,
                "success_rate": bm.get("success_rate"),
                "exploration_attempts": report["exploration_attempts"],
                "selection_proportions": report["selection_proportions"],
            }
        )
    return {
        "baseline": "cost_aware",
        "best_fixed_provider": best_fixed["strategy"] if best_fixed else None,
        "workload_sha256": baseline["workload_sha256"],
        "comparisons": comparisons,
        "scope": "Differences between complete matched runs; report failures and repeat across seeds, not per-request counterfactuals",
    }


def smoke(output, seed=7):
    """Run production routing with hidden synthetic outcomes; learn only from chosen pools."""
    from atsuite.routing.exploration import Exploration
    from atsuite.routing.simulation import simulate
    from experiments.exp1.runner.stage6_simulation import build

    hashes = implementation_hashes()
    reports = []
    for strategy in ("none", "random", "contextual"):
        controller, arrivals, workflows, surface = build(0.001, "routing")
        router = controller.router
        controller.period_s = 10
        controller.plans = [
            p for p in controller.plans if all(t == 1 for t in p["targets"].values())
        ]
        # Hide target execution evidence; tariffs/calibration remain the arrival-time prior.
        router.provider.samples = {
            k: v
            for k, v in router.provider.samples.items()
            if v[0]["pool"]["environment"] == "synthetic-cheap"
        }
        for i, sample in enumerate(list(router.workload.samples.values())):
            for j in range(2):
                p = copy.deepcopy(sample["profile"])
                p["request_id"] = f"prior-{i}-{j}"
                router.workload.update(p, 100)
        router.budget = 100
        if strategy != "none":
            router.exploration = Exploration(
                {"strategy": strategy, "epsilon": 0.8, "budget": 20, "seed": seed}
            )
        router.log_path = Path(output) / (strategy + ".jsonl")
        # Sequential arrivals isolate learning from load rejection; every strategy gets the same stream.
        arrivals = [
            {
                "id": f"request-{i}",
                "task_id": arrivals[0]["task_id"],
                "descriptor": arrivals[0]["descriptor"],
                "at_s": 1 + i * 10,
            }
            for i in range(60)
        ]
        report = simulate(
            controller,
            arrivals,
            workflows,
            surface,
            seed=seed,
            end_s=610,
            start_ns=10**9,
        )
        report["strategy"] = strategy
        report["implementation_sha256"] = hashes
        report["arrival_sha256"] = hashlib.sha256(
            json.dumps(arrivals, sort_keys=True).encode()
        ).hexdigest()
        report["validation_summary"] = {
            "strata": len(report.pop("validation")),
            "basis": "synthetic",
        }
        write(Path(output) / (strategy + ".json"), report)
        reports.append(
            {
                k: report[k]
                for k in (
                    "strategy",
                    "synthetic",
                    "admitted",
                    "p95_s",
                    "cost",
                    "exploration",
                )
            }
        )
    return {
        "seed": seed,
        "implementation_sha256": hashes,
        "reports": reports,
        "scope": "Synthetic mechanism check, not a cloud benefit claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("panel", "plan", "holdout", "collect", "compare", "smoke")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--charges", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--rows", type=Path)
    source.add_argument("--e1b", type=Path)
    parser.add_argument("--reports", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    required = {
        "plan": ("config", "workload"),
        "collect": ("run", "config", "charges"),
        "compare": ("reports",),
        "holdout": ("config",),
    }
    if any(getattr(args, key) is None for key in required.get(args.action, ())):
        parser.error("Missing arguments for " + args.action)
    if args.action == "holdout" and not (args.rows or args.e1b):
        parser.error("holdout requires --rows or --e1b")
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = implementation_hashes()
    if args.action == "panel":
        result = prompt_panel()
    elif args.action == "plan":
        result = make_plan(load(args.config), args.workload, args.output)
    elif args.action == "collect":
        result = collect(args.run, load(args.config), load(args.charges))
    elif args.action == "compare":
        result = compare([load(p) for p in args.reports])
    elif args.action == "smoke":
        result = smoke(args.output, args.seed)
    else:
        config = load(args.config)
        panel = None if config.get("descriptors") else prompt_panel()
        descriptors = config.get("descriptors") or {
            t["task_id"]: t["descriptor"] for t in panel["cases"]
        }
        if args.e1b:
            from atsuite.provider_model.e1b import load_e1b

            rows, provenance = load_e1b(args.e1b)
        else:
            rows = load(args.rows)
            provenance = {
                "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest()
            }
        reports = []
        present = {r.get("task_id") for r in rows}
        labels = config.get("family_labels")
        if labels is None:
            raise ValueError("Holdout requires independent family_labels; task classes are not prediction features")
        families = {v for t, v in labels.items() if t in present}
        for family in sorted(families):
            for mode in ("context", "embedding"):
                report = evaluate(
                    rows,
                    descriptors,
                    config["tariffs"],
                    held_family=family,
                    descriptor_mode=mode,
                    retrieval_config=config.get("retrieval"),
                    history_limit=config.get("history_limit", 1000),
                    fx=config.get("fx"),
                    family_labels=labels,
                    history_sampling=config.get("history_sampling", "reservoir"),
                    sampling_seed=config.get("sampling_seed", 7),
                )
                fold = (
                    hashlib.sha256(family.encode()).hexdigest()[:16]
                    if labels is not None
                    else family
                )
                report["prediction_file"] = f"{fold}-{mode}.json"
                write(args.output / report["prediction_file"], report)
                reports.append({k: v for k, v in report.items() if k != "predictions"})
        result = {
            "reports": reports,
            "scope": "Family-held-out historical prediction diagnostics; no mixed-routing performance",
            "provenance": provenance,
            "implementation_sha256": implementation_hashes(),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "panel": panel,
        }
    result["implementation_sha256"] = source_hashes
    write(args.output / "report.json", result)
    print(args.output)


if __name__ == "__main__":
    main()
