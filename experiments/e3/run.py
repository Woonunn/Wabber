"""Two E3 arms: one fixed Provider and online routing with exploration."""

import argparse
import asyncio
import copy
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from atsuite.provider_model import ProviderModel
from atsuite.routing.router import OnlineRouter
from atsuite.tracing.storage import read_text
from atsuite.workload import WorkloadModel
from experiments.e3.feedback import NativeCostFeedback
from experiments.e3.fixed_baselines import OfficialEvaluator, measure
from experiments.e3.manifest import load
from experiments.exp1.runner.arrival_scheduler import Arrival, run_open_loop, stats

PROVIDERS = ("aws_lambda", "aws_agentcore", "ali_fc", "ali_agentrun")
STRATEGIES = ("fixed", "exploration")


def build_router(config: dict, strategy: str, log_path: Path) -> OnlineRouter:
    """Start each formal arm with empty learned state and fixed native resources."""
    if strategy not in STRATEGIES:
        raise ValueError("E3 supports only fixed and exploration")
    if set(config["pools"]) != set(PROVIDERS) or any(k in config for k in ("control", "levels", "scaling")):
        raise ValueError("Four native execution pools required; active scaling is forbidden")
    routing = copy.deepcopy(config["routing"])
    if "budget" in routing or "budget" in routing.get("exploration", {}):
        raise ValueError("E3 does not use financial budgets")
    if not routing.get("exploration"):
        raise ValueError("Online routing requires exploration")
    routing["exploration"].pop("epoch_ns", None)
    pools = copy.deepcopy(config["pools"])
    for pool in pools.values():
        if pool.get("billing"):
            pool["billing"].pop("calibration", None)
    router = OnlineRouter(
        WorkloadModel(retrieval_config=config.get("retrieval")),
        ProviderModel(minimum_samples=3),
        pools,
        config["limits"],
        **routing,
        log_path=log_path,
    )
    assert not router.workload.samples and not router.workload.initial_profiles and not router.provider.samples
    router.session_retention_s = {
        "aws_agentcore": max(
            a["lifecycle_configuration"]["idleRuntimeSessionTimeout"]
            for a in config["native_elasticity"]["aws_agentcore"].values()
        ),
        "ali_agentrun": max(
            json.loads(a["session_affinity_config"])["sessionIdleTimeoutInSeconds"]
            for a in config["native_elasticity"]["ali_agentrun"].values()
        ),
    }
    return router


class NativeOccupancy:
    """Read Ali's retained native Sessions; never change instances or elasticity."""

    def __init__(self, router, config, path):
        from atsuite.ali.ali import Ali

        self.client = Ali().get_fc_client()
        self.router, self.config, self.path = router, config, path
        self.owned = set()

    async def refresh(self):
        from alibabacloud_fc20230330.models import ListSessionsRequest

        def query():
            sessions = set()
            for a in self.config["native_elasticity"]["ali_agentrun"].values():
                token = None
                while True:
                    page = self.client.list_sessions(
                        a["runtime_name"], ListSessionsRequest(limit=100, next_token=token, session_status="Active")
                    ).body.to_map()
                    sessions.update(s["sessionId"] for s in page.get("sessions", []) if s["sessionStatus"] == "Active")
                    nxt = page.get("nextToken")
                    if not nxt:
                        break
                    if nxt == token:
                        raise RuntimeError("Session pagination did not advance")
                    token = nxt
            return sessions

        event = {"observed_ns": time.time_ns()}
        try:
            sessions = await asyncio.to_thread(query)
            external = len(sessions - self.owned)
            event.update(native_active=len(sessions), owned=len(sessions & self.owned), external=external)
        except Exception as exc:
            external = self.config["limits"]["ali_agentrun"]
            event.update(error=str(exc), external=external)
        self.router.observe_occupancy("ali_agentrun", external)
        with self.path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    async def watch(self):
        while True:
            await asyncio.sleep(10)
            await self.refresh()

    def completed(self, row):
        if row["pool"] != "ali_agentrun":
            return
        for line in read_text(Path(row["directory"]) / "events.jsonl").splitlines():
            event = json.loads(line)
            if event["kind"] == "tool":
                sid = event["transport"].get("provider_metadata", {}).get("session_id")
                if sid:
                    self.owned.add(sid)


async def execute(sequence: Path, config: dict, strategy: str, output: Path) -> dict:
    """Run one unchanged 10-minute request stream and retain every failure."""
    if strategy not in STRATEGIES or config["fixed_provider"] not in PROVIDERS:
        raise ValueError("Invalid two-arm strategy or fixed Provider")
    plan = load(sequence)
    output.mkdir(parents=True, exist_ok=False)
    tasks = {t["id"]: t for t in plan["tasks"]}
    first = next(t for t in plan["tasks"] if t["workload"] == "TravelPlanner")
    task = next(t for t in json.loads(Path(first["manifest"]).read_text())["tasks"] if t["task_id"] == first["task_id"])
    from benchmarks.TravelPlanner.langgraph_agent import graph as travel

    oracle = OfficialEvaluator(task, output / "oracle.log")
    original = travel.validate
    router = observer = watcher = cost_feedback = cost_watcher = None
    rows, submitted = [], {}
    try:
        startup = await oracle.start()
        travel.validate = oracle.validate
        if strategy == "exploration":
            router = build_router(config, strategy, output / "routing.jsonl")
            observer = NativeOccupancy(router, config, output / "native-occupancy.jsonl")
            await observer.refresh()
            watcher = asyncio.create_task(observer.watch())
            cost_feedback = NativeCostFeedback(router, config, output)
            cost_watcher = asyncio.create_task(cost_feedback.watch())
        (output / "execution.json").write_text(
            json.dumps(
                {
                    "strategy": strategy,
                    "manifest_sha256": json.loads((sequence / "seal.json").read_text())["manifest_sha256"],
                    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "native_elasticity": config["native_elasticity"],
                    "config": config,
                    "oracle": startup,
                    "accounting": "ATSuite native usage pricing; no financial budget; no instance control",
                    "initial_learning_state": {"workload_samples": 0, "provider_samples": 0, "exploration_attempts": 0},
                },
                indent=2,
            )
            + "\n"
        )
        assignments = {r["arrival_id"]: r for r in plan["requests"]}
        arrivals = [Arrival(**{k: r[k] for k in Arrival.__dataclass_fields__}) for r in plan["requests"]]

        def observe(event):
            if event["kind"] == "request_submitted":
                submitted[event["arrival_id"]] = event["wall_ns"]

        async def handler(arrival, started):
            assignment = assignments[arrival.arrival_id]
            started()
            context = {
                "request_id": arrival.arrival_id,
                "arrival_ns": submitted[arrival.arrival_id],
                "targets": config["targets"][assignment["workload"]],
                "load_rps": config["load_rps"],
            }
            row = await measure(
                tasks[assignment["task"]],
                config,
                config["fixed_provider"],
                output / "requests" / arrival.arrival_id,
                request_id=arrival.arrival_id,
                arrival_ns=submitted[arrival.arrival_id],
                router=router,
                routing_context=context,
            )
            rows.append(row)
            with (output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if observer:
                observer.completed(row)
            if not row["success"]:
                raise RuntimeError(json.dumps(row.get("error")))
            return {"success": True}

        summary = await run_open_loop(
            arrivals,
            handler,
            output / "schedule",
            duration_ms=plan["duration_ms"],
            concurrency=config["concurrency"],
            observer=observe,
            metadata={
                "strategy": strategy,
                "fixed_provider": config["fixed_provider"],
                "native_elasticity": config["native_elasticity"],
            },
        )
        result = {
            "strategy": strategy,
            "attempts": len(rows),
            "successes": sum(r["success"] for r in rows),
            "latency_s": stats([r["latency_s"] for r in rows]),
            "scheduler": summary,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if router:
            if cost_watcher:
                cost_watcher.cancel()
                await asyncio.gather(cost_watcher, return_exceptions=True)
                cost_watcher = None
            result["native_cost_feedback"] = {
                "imported_requests": len(cost_feedback.imported),
                "last_error": cost_feedback.failed,
                "scope": "only formal completed requests; unavailable native costs remain unknown",
            }
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            router.workload.save(output / "workload_model.json")
            router.provider.save(output / "provider_model.json")
        return result
    finally:
        if cost_watcher:
            cost_watcher.cancel()
            await asyncio.gather(cost_watcher, return_exceptions=True)
        if watcher:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        travel.validate = original
        await oracle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("sequence", "config", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(".env")
    with asyncio.Runner() as runner:
        runner.get_loop().set_default_executor(ThreadPoolExecutor(max_workers=128))
        print(
            json.dumps(
                runner.run(execute(args.sequence, json.loads(args.config.read_text()), args.strategy, args.output))
            )
        )


if __name__ == "__main__":
    main()
