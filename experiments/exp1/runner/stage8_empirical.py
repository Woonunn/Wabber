"""Causal, fixed-outcome replay of the four paired E2-B provider runs.

This evaluates the working-tree router, not a new cloud execution. Unselected
outcomes and delayed billing evidence are never exposed to the online models.
"""

import argparse
import copy
from collections import Counter, deque
from decimal import Decimal
import hashlib
import heapq
import json
from pathlib import Path
import gzip

from atsuite.provider_model.model import ProviderModel
from atsuite.routing.router import OnlineRouter
from atsuite.tracing.storage import read_text
from atsuite.workload.descriptor import describe, family_key
from atsuite.workload.model import WorkloadModel, distribution
from experiments.exp1.runner.mix import write

PROVIDERS = ("aws_lambda", "aws_agentcore", "ali_fc", "ali_agentrun")
TOOLS = ["prepare_workspace", "execute_analysis", "merge_artifacts", "validate_result"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(router: OnlineRouter, arrivals: list[dict], outcome) -> list[dict]:
    """Reveal exactly one selected trace at its completion; ties remain invisible."""
    pending, rows, recent = [], [], deque()

    def finish(before):
        while pending and pending[0][0] < before:
            completed, index, trace = heapq.heappop(pending)
            row = rows[index]
            router.complete(row["request_id"], trace, available_ns=completed,
                            actual_pool=row["selected"])
            row["feedback_ns"] = completed
            row["cpu_observation"] = router.workload.samples[row["request_id"]]["profile"].get("cpu_observation")

    for index, arrival in enumerate(arrivals):
        now = arrival["at_ns"]
        finish(now)
        while recent and recent[0] <= now - 60 * 10**9:
            recent.popleft()
        recent.append(now)
        # A neutral sequence ID keeps task IDs and hashes out of random policy draws.
        rid = f"replay-{index:04d}"
        decision = router.decide(rid, as_of_ns=now, arrival_ns=now,
                                 descriptor=arrival["descriptor"], load_rps=len(recent) / 60)
        selected = decision["selected"]
        candidate = next((c for c in decision["candidates"] if c["key"] == selected), None)
        row = {"request_id": rid, "arrival_id": arrival["arrival_id"],
               "at_ns": now, "segment": arrival["segment"],
               "first_prompt": arrival["first_prompt"], "first_family": arrival["first_family"],
               "family": family_key(arrival["descriptor"]),
               "selected": selected, "reason": decision["reason"],
               "explored": decision["exploration"]["explored"],
               "exploration_reason": decision["exploration"].get("reason"),
               "predicted_latency_s": candidate["objective_latency"] if candidate else None,
               "predicted_cost_cny": candidate["objective_cost"] if candidate else None,
               "confidence": candidate["prediction"].get("confidence") if candidate else None,
               "visible_feedback": len(router.completed),
               "candidates": {c["key"]: {
                   "flags": c["flags"], "prediction_flags": c["prediction_flags"],
                   "sample_count": c["prediction"]["sample_count"],
                   "missing_cost_features": (c["prediction"].get("analytical_cost") or {}).get("missing"),
                   "confidence": c["prediction"].get("confidence"),
                   "evidence": c["evidence"],
               } for c in decision["candidates"]}}
        rows.append(row)
        if selected is None:
            continue
        trace, elapsed = outcome(arrival["arrival_id"], selected)
        if elapsed <= 0:
            raise ValueError("Nonpositive recorded duration")
        trace = copy.deepcopy(trace)
        completed = now + round(elapsed * 10**9)
        trace.update(request_id=rid, arrival_ns=now, completed_ns=completed)
        row["latency_s"] = elapsed
        heapq.heappush(pending, (completed, index, trace))
    finish(float("inf"))
    if len(router.completed) != sum(r["selected"] is not None for r in rows):
        raise ValueError("Feedback count differs from admitted requests")
    return rows


def summarize(rows: list[dict]) -> dict:
    """Report prediction coverage alongside errors, without filling unknown costs."""
    completed = [r for r in rows if "feedback_ns" in r]
    errors = [abs(r["predicted_latency_s"] - r["latency_s"]) for r in completed
              if r["predicted_latency_s"] is not None]
    return {
        "requests": len(rows), "completed": len(completed),
        "routes": dict(Counter(r["selected"] for r in completed)),
        "reasons": dict(Counter(r["reason"] for r in rows)),
        "exploration_rate": sum(r["explored"] for r in rows) / len(rows) if rows else None,
        "fallback_rate": sum(r["reason"] == "validated_default" for r in rows) / len(rows) if rows else None,
        "exploitation_rate": sum(r["reason"] == "cost_latency_objective" for r in rows) / len(rows) if rows else None,
        "latency_s": distribution([r["latency_s"] for r in completed], len(rows)),
        "latency_prediction_coverage": len(errors) / len(rows) if rows else None,
        "latency_prediction_mae_s": sum(errors) / len(errors) if errors else None,
        "cost_prediction_coverage": sum(r["predicted_cost_cny"] is not None for r in rows) / len(rows) if rows else None,
        "total_infrastructure_cost": None, "net_savings": None,
    }


def run(root: Path, sequence: Path, output: Path) -> dict:
    """Verify the sealed paired dataset, freeze assumptions, then run ablations."""
    output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def read(path, expected=None):
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError(f"Changed sealed input: {path}")
        sources[str(path)] = value
        return json.loads(path.read_text())

    seal = read(sequence / "seal.json")
    plan = read(sequence / "manifest.json", seal["manifest_sha256"])
    assignments = read(sequence / "arrivals.json", plan["arrivals_sha256"])
    manifest = read(Path(plan["mix_manifest"]), plan["mix_manifest_sha256"])
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    arrivals = []
    for a in assignments:
        task = tasks[a["task_id"]]
        if hashlib.sha256(task["prompt"].encode()).hexdigest() != a["prompt_sha256"]:
            raise ValueError("Prompt differs from sealed assignment")
        arrivals.append({**a, "at_ns": 10**9 + int(Decimal(a["relative_arrival_ms"]) * 10**6),
                         "descriptor": describe(prompt=task["prompt"], tools=TOOLS,
                                                files=task["readiness"]["inputs"]["files"])})
    if len({a["arrival_id"] for a in arrivals}) != len(arrivals) or any(
        a["at_ns"] > b["at_ns"] for a, b in zip(arrivals, arrivals[1:])
    ):
        raise ValueError("Arrival sequence is duplicated or unsorted")
    audit = read(root / "execution-audit.json")
    if audit["status"] != "passed" or audit["total_formal_requests"] != 4 * len(arrivals):
        raise ValueError("Formal execution audit is incomplete")
    trace_hashes = {str(Path(e["path"]).resolve()): e["sha256"] for e in audit["evidence"]}
    pools, metrics = {}, {}
    for provider in PROVIDERS:
        directory = root / f"formal-{provider}" / provider
        run_info = read(directory / "run.json")
        if run_info["experiment"] != "E2-B" or run_info["pilot"] or run_info["validation_only"]:
            raise ValueError("Not a formal E2-B run")
        actual, image = run_info["deployment"]["actual"], run_info["deployment"]["image"]
        resources = {**actual, "image": image["id"], "architecture": image["architecture"]}
        if provider == "aws_lambda":
            resources["disk_mb"] = actual["ephemeral_storage"]["Size"]
        affinity = actual.get("session_affinity_config") or {}
        if isinstance(affinity, str):
            affinity = json.loads(affinity)
        pools[provider] = {"provider": provider, "environment": image["id"],
                           "resources": resources, "instances": None,
                           "sessions_per_instance": affinity.get("sessionConcurrencyPerInstance"),
                           "invocations_per_instance": actual.get("instance_concurrency")
                           if isinstance(actual.get("instance_concurrency"), int) else None}
        metric_path = directory / "request_metrics.jsonl"
        sources[str(metric_path)] = digest(metric_path)
        records = [json.loads(line) for line in metric_path.read_text().splitlines()]
        metrics[provider] = {r["arrival_id"]: r for r in records}
        if len(records) != len(arrivals) or len(metrics[provider]) != len(records):
            raise ValueError("Incomplete or duplicated outcome table")
        for a in arrivals:
            r = metrics[provider][a["arrival_id"]]
            if any(str(r[k]) != str(a[k]) for k in ("task_id", "prompt_sha256", "relative_arrival_ms")) or r["status"] != "succeeded":
                raise ValueError("Unpaired or unsuccessful formal request")

    tariffs = read(Path("experiments/exp1/configs/stage6-tariffs.json"))
    for provider in ("ali_fc", "ali_agentrun"):
        tariffs[provider]["rates"].update(cpu_s=0.00011, memory_gb_s=0.00011 * 0.15,
                                            invocation=0.00011 * 0.0075, disk_gb_s=0.00011 * 0.05)
        tariffs[provider]["free_disk_gb"] = 10
        tariffs[provider]["source"].append("experiments/exp1/configs/stage8-standard-rates.json")
    read(Path("experiments/exp1/configs/stage8-standard-rates.json"))
    cost_root = root / "standard-cost-20261103-v1"
    fx = read(cost_root / "fx.json")
    conversion = {"currency": "CNY", "rates": {"CNY": 1, "USD": float(fx["usd_to_cny"])},
                  "date": fx["date"], "source": fx["source"]}
    bounds = {p: {"evidence": str(root / f"pilot-v2-{p}" / p), "currency": "CNY",
                  "timeout_s": 120, "max_cost": 1, "tools": TOOLS} for p in PROVIDERS}
    policies = [("no-exploration", "aws_lambda", 0, 8),
                ("exploration-seed8", "aws_lambda", 0.1, 8),
                ("exploration-seed18", "aws_lambda", 0.1, 18),
                ("exploration-fc-default", "ali_fc", 0.1, 8)]
    config = {"policies": policies, "pools": pools, "tariffs": tariffs, "fx": conversion,
              "bounds": bounds, "limits": {p: 32 for p in PROVIDERS},
              "budget_cny": 1000, "exploration_budget_cny": 300,
              "cost_weight": 1, "latency_weight": 0.001,
              "initial_history": "empty; readiness only, no target performance history",
              "limits_basis": "Replay admission scenario matching the original harness cap, not physical capacity",
              "budget_basis": "Conservative 1 CNY reservation per request; no delayed bill feedback",
              "cost_scope": "Analytical forecasts only; configured storage/tail/zero-control assumptions are not measured totals",
              "method": "Fixed recorded outcome per arrival/provider; changed load and capacity effects are not modeled"}
    for directory in ("atsuite/routing", "atsuite/provider_model", "atsuite/workload"):
        for path in Path(directory).glob("*.py"):
            sources[str(path)] = digest(path)
    sources[__file__] = digest(Path(__file__))
    write(output / "config.json", config)
    write(output / "pre-replay-sources.json", sources)

    def outcome(arrival_id, provider):
        path = root / f"formal-{provider}" / provider / "requests" / arrival_id / "trace.json.gz"
        sources[str(path)] = digest(path)
        if sources[str(path)] != trace_hashes[str(path.resolve())]:
            raise ValueError("Original execution trace changed")
        return json.loads(read_text(path)), metrics[provider][arrival_id]["end_to_end_ms"] / 1000

    reports = {}
    for name, default, epsilon, seed in policies:
        router = OnlineRouter(WorkloadModel(), ProviderModel(tariffs=tariffs), pools,
                              config["limits"], default_pool=default, execution_bounds=bounds,
                              budget=1000, fx=conversion,
                              exploration={"epsilon": epsilon, "budget": 300, "seed": seed})
        rows = replay(router, arrivals, outcome)
        reports[name] = {"overall": summarize(rows),
                         "windows": [summarize([r for r in rows if r["segment"] == s]) for s in range(4)],
                         "first_prompts": summarize([r for r in rows if r["first_prompt"]]),
                         "repeat_prompts": summarize([r for r in rows if not r["first_prompt"]]),
                         "first_families": summarize([r for r in rows if r["first_family"]]),
                         "feedback_by_pool": dict(Counter(k[0] for k in router.provider.samples))}
        with gzip.open(output / f"{name}-requests.json.gz", "wt") as stream:
            json.dump(rows, stream, allow_nan=False)
        write(output / f"{name}.json", reports[name])
        print(name, reports[name]["overall"]["routes"], reports[name]["overall"]["reasons"], flush=True)
    billing = read(cost_root / "final-v3/summary.json")
    ledger = {(r["provider"], r["arrival_id"]): r for r in read(cost_root / "final-v3/requests.json")}
    for name, *_ in policies:
        with gzip.open(output / f"{name}-requests.json.gz", "rt") as stream:
            rows = json.load(stream)
        known, unknown = Decimal(0), Counter()
        for row in rows:
            provider = row["selected"]
            if provider is None:
                continue
            amount = ledger[(provider, row["arrival_id"])].get("runtime_standard_cost")
            if amount is None:
                unknown[provider] += 1
            else:
                known += Decimal(amount) * (Decimal(fx["usd_to_cny"]) if provider.startswith("aws") else 1)
        reports[name]["standard_runtime_cost"] = {
            "known_subtotal_cny": str(known), "unpriced_requests": dict(unknown),
            "complete_total_cny": str(known) if not unknown else None,
            "scope": "Post-replay selected measured runtime subtotal; not infrastructure total or online feedback",
        }
    baselines = {p: {"latency_s": distribution([r["end_to_end_ms"] / 1000 for r in metrics[p].values()], len(arrivals)),
                     "standard_cost": billing["providers"][p]} for p in PROVIDERS}
    result = {"method": config["method"], "policies": reports, "fixed_baselines": baselines,
              "full_cost_winner": None, "dynamic_capacity_validated": False,
              "causal_feedback": "Selected provider only, strictly before subsequent decision; no billing fed to models"}
    write(output / "summary.json", result)
    write(output / "sources.json", sources)
    write(output / "seal.json", {str(p): digest(p) for p in output.iterdir() if p.is_file()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/stage8/cloud-stable-4typical-v1"))
    parser.add_argument("--sequence", type=Path, default=Path("results/stage8/e2-stable-4typical-v1"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.sequence, args.output)


if __name__ == "__main__":
    main()
