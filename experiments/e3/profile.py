"""Run a predeclared paired pilot on real tool backends without routing or scaling."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from atsuite.tracing.storage import read_text
from experiments.e3.manifest import WORKLOADS, load
from experiments.e3.report import matrix
from experiments.e3.run import PROVIDERS
from experiments.exp1.capture.capture_agent_traces import run


def initialize(config: dict, plan: dict, attempts: list[dict], priced: list[dict], output: Path) -> dict:
    """Validate preflight readiness, then discard its learned profiles and costs."""
    import copy
    import hashlib

    from atsuite.provider_model import ProviderModel
    from atsuite.workload import WorkloadModel

    output.mkdir(parents=True, exist_ok=False)
    tasks = {t["id"]: t for t in plan["tasks"]}
    expected = {(w, p) for w in WORKLOADS for p in PROVIDERS}
    if any(r["task"] not in tasks for r in attempts):
        raise ValueError("Unknown preflight task")
    if {(tasks[r["task"]]["workload"], r["pool"]) for r in attempts} != expected:
        raise ValueError("Preflight must validate every workload/provider pair")
    if not all(r["success"] for r in attempts):
        raise ValueError("Preflight contains failures")
    config = copy.deepcopy(config)
    config["pools"] = {}
    for p in PROVIDERS:
        a = config["native_elasticity"][p]["e3"]
        resources = {"cpu": a.get("cpu"), "memory_mb": a.get("memory_mb"), "disk_mb": a.get("disk_mb")}
        resources["workload_overrides"] = {
            "DataSciBench": {
                k: config["native_elasticity"][p]["datasci"].get(k) for k in ("cpu", "memory_mb", "disk_mb")
            }
        }
        config["pools"][p] = {
            "provider": p,
            "environment": hashlib.sha256(
                json.dumps(config["native_elasticity"][p], sort_keys=True).encode()
            ).hexdigest(),
            "resources": resources,
            "instances": None,
            "sessions_per_instance": 1 if p in {"aws_agentcore", "ali_agentrun"} else None,
            "invocations_per_instance": 1 if p in {"aws_lambda", "ali_fc"} else None,
        }
    workload, provider = WorkloadModel(), ProviderModel(minimum_samples=3)
    supported = set()
    for task in tasks.values():
        declared = next(
            t for t in json.loads(Path(task["manifest"]).read_text())["tasks"] if t["task_id"] == task["task_id"]
        )
        tools = (
            ["prepare_workspace", "execute_analysis", "merge_artifacts", "validate_result"]
            if task["workload"] == "DataSciBench"
            else [d["name"] for d in declared["tool_definitions"]]
        )
        supported.update(tools)
    workload.save(output / "workload_model.json")
    provider.save(output / "provider_model.json")
    config.update(
        fixed_provider="ali_fc",
        workload_model=str(output / "workload_model.json"),
        provider_model=str(output / "provider_model.json"),
        limits=dict.fromkeys(PROVIDERS, 64),
        load_rps=len(plan["requests"]) / (plan["duration_ms"] / 1000),
    )
    fx = json.loads(Path("results/stage8/cloud-stable-4typical-v1/standard-cost-20261103-v1/fx.json").read_text())
    config["routing"] = {
        "default_pool": "ali_fc",
        "cost_weight": 1.0,
        "latency_weight": 0.0,
        "fx": {
            "date": fx["date"],
            "source": fx["source"],
            "currency": "CNY",
            "rates": {"CNY": 1, "USD": float(fx["usd_to_cny"])},
        },
        "execution_bounds": {
            p: {
                "timeout_s": 900,
                "currency": "CNY" if p.startswith("ali") else "USD",
                "tools": sorted(supported),
                "evidence": "provider capabilities validated in preflight; task costs not imported",
            }
            for p in PROVIDERS
        },
        "exploration": {
            "strategy": "contextual",
            "epsilon": 0.35,
            "seed": 9,
            "min_samples": 3,
            "window_s": 3600,
            "optimism": 1.0,
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "initialization.json").write_text(
        json.dumps(
            {
                "source": "empty formal models; preflight retained for readiness audit only",
                "attempts": len(attempts),
                "preflight_priced_rows_not_imported": len(priced),
                "workload_samples": 0,
                "provider_samples": 0,
                "exploration_attempts": 0,
                "formal_outcomes_used": False,
                "task_id_matching": False,
                "fixed_provider": "ali_fc",
                "strategies": ["fixed", "exploration"],
            },
            indent=2,
        )
        + "\n"
    )
    return config


async def profile(sequence: Path, config: dict, output: Path) -> dict:
    """Use the sealed pilot task IDs, preserving failures and unknown actual costs."""
    plan = load(sequence)
    tasks = {t["id"]: t for t in plan["tasks"]}
    pilot = config["pilot_tasks"]
    if len(set(pilot)) != len(pilot) or not set(pilot) <= tasks.keys():
        raise ValueError("Pilot must be predeclared unique task IDs")
    if {tasks[t]["workload"] for t in pilot} != {t["workload"] for t in tasks.values()}:
        raise ValueError("Pilot must cover all workloads")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        for task_id in pilot:
            task = tasks[task_id]

            async def attempt(provider, repeat):
                keys = [k for k, p in config["pools"].items() if p["provider"] == provider]
                if len(keys) != 1:
                    raise ValueError("Exactly one pool per provider required")
                key = keys[0]
                target = config["targets"][task["workload"]][key]
                agent = {**config["agents"][task["workload"]], "tool": target["tool"]}
                directory = output / task_id / key / f"repeat-{repeat}"
                started_epoch = time.time()
                start = time.perf_counter()
                row = {
                    "workload": task["workload"],
                    "task": task_id,
                    "pool": key,
                    "repeat": repeat,
                    "directory": str(directory),
                    "started_epoch": started_epoch,
                    "success": False,
                    "actual_cost": None,
                    "billing_evidence": None,
                }
                try:
                    await run(
                        task["task_id"],
                        agent,
                        directory,
                        Path(task["capture"]),
                        manifest=Path(task["manifest"]),
                        workload=task["workload"],
                    )
                    row["success"] = True
                except Exception as exc:
                    row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                row["latency_s"] = time.perf_counter() - start
                row["completed_epoch"] = time.time()
                profile_path = directory / "workload_profile.json"
                if profile_path.exists() or profile_path.with_suffix(".json.gz").exists():
                    row["profile"] = json.loads(read_text(profile_path))
                return row

            for repeat in range(3):
                rows.extend(await asyncio.gather(*(attempt(provider, repeat) for provider in PROVIDERS)))
    finally:
        (output / "attempts.json").write_text(json.dumps(rows, indent=2) + "\n")
    report = matrix(rows)
    (output / "matrix.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(override=False)
    print(json.dumps(asyncio.run(profile(args.sequence, json.loads(args.config.read_text()), args.output))))


if __name__ == "__main__":
    main()
