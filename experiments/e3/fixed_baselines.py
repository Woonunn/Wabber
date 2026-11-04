"""Measured fixed-provider baselines over the sealed E3 arrival stream."""

import argparse
import asyncio
import hashlib
import io
import json
import os
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path


class OfficialEvaluator:
    """Reuse the unmodified official checker in one isolated process, with bounded memory."""

    def __init__(self, task: dict, output: Path):
        command = task["verification"]["command"]
        self.python = command[0]
        self.upstream = command[command.index("--upstream") + 1]
        self.output = output
        self.lock = asyncio.Lock()
        self.pending = set()

    async def start(self) -> dict:
        self.log = self.output.open("w")
        begin = time.perf_counter()
        self.process = await asyncio.create_subprocess_exec(
            self.python,
            "-m",
            "experiments.e3.fixed_baselines",
            "--mode",
            "oracle",
            "--upstream",
            self.upstream,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.log,
            env={
                **os.environ,
                "GRADIO_ANALYTICS_ENABLED": "False",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        message = await asyncio.wait_for(self.process.stdout.readline(), 180)
        if json.loads(message).get("ready") is not True:
            raise RuntimeError("Independent evaluator did not start")
        return {
            "startup_s": time.perf_counter() - begin,
            "basis": "official modules cached; same checker and inputs; no provider prewarm",
        }

    async def validate(self, task, answer, tools):
        async def transaction():
            async with self.lock:
                token = uuid.uuid4().hex
                self.process.stdin.write(
                    (
                        json.dumps({"id": token, "command": task["verification"]["command"], "answer": answer}) + "\n"
                    ).encode()
                )
                await self.process.stdin.drain()
                response = json.loads(await asyncio.wait_for(self.process.stdout.readline(), 120))
                if response.get("id") != token or "error" in response:
                    raise RuntimeError(
                        "Independent evaluator failed: " + str(response.get("error", "protocol mismatch"))
                    )
                return response["result"]

        job = asyncio.create_task(transaction())
        self.pending.add(job)
        job.add_done_callback(self.pending.discard)
        return await asyncio.shield(job)

    async def close(self):
        if self.pending:
            await asyncio.gather(*self.pending, return_exceptions=True)
        if self.process.returncode is None:
            self.process.stdin.close()
            await asyncio.wait_for(self.process.wait(), 30)
        self.log.close()


def oracle_worker(upstream: Path) -> None:
    """Keep the official evaluator bridge unchanged and repeat its exact entrypoint."""
    import importlib

    from experiments.e3.travel_eval import main as evaluate

    incoming, outgoing = sys.stdin, sys.stdout
    os.chdir(upstream / "evaluation")
    sys.path[:0] = [str(upstream), str(upstream / "evaluation")]
    with redirect_stdout(sys.stderr):
        importlib.import_module("commonsense_constraint")
        importlib.import_module("hard_constraint")
    print(json.dumps({"ready": True}), file=outgoing, flush=True)
    for line in incoming:
        request = json.loads(line)
        original_argv = sys.argv
        try:
            command = request["command"]
            if command[1:3] != ["-m", "experiments.e3.travel_eval"] or command[command.index("--upstream") + 1] != str(
                upstream
            ):
                raise ValueError("Unexpected evaluator command")
            sys.argv = [command[2], *command[3:]]
            sys.stdin = io.StringIO(json.dumps(request["answer"]))
            output = io.StringIO()
            with redirect_stdout(output):
                evaluate()
            response = {"id": request["id"], "result": json.loads(output.getvalue())}
        except Exception as exc:
            response = {"id": request["id"], "error": {"type": type(exc).__name__, "message": str(exc)}}
        finally:
            sys.argv, sys.stdin = original_argv, incoming
        print(json.dumps(response), file=outgoing, flush=True)


async def measure(
    task: dict,
    config: dict,
    provider: str,
    output: Path,
    *,
    request_id=None,
    arrival_ns=None,
    router=None,
    routing_context=None,
) -> dict:
    from atsuite.tracing.storage import read_text
    from experiments.exp1.capture.capture_agent_traces import run

    start = time.perf_counter()
    row = {
        "workload": task["workload"],
        "task_id": task["task_id"],
        "task": task["id"],
        "pool": provider,
        "request_id": request_id,
        "started_epoch": time.time(),
        "directory": str(output),
        "success": False,
        "actual_cost": None,
        "billing_evidence": None,
    }
    agent_config = (
        json.loads(Path(task["agent_config"]).read_text())
        if task.get("agent_config")
        else config["agents"][task["workload"]]
    )
    agent = {**agent_config, "tool": config["targets"][task["workload"]][provider]["tool"]}
    try:
        await run(
            task["task_id"],
            agent,
            output,
            Path(task["capture"]),
            manifest=Path(task["manifest"]),
            workload=task["workload"],
            arrival_ns=arrival_ns,
            **({"router": router, "routing_context": routing_context} if router is not None else {}),
        )
        row["success"] = True
    except Exception as exc:
        row["error"] = {"type": type(exc).__name__, "message": str(exc)}
    row.update(latency_s=time.perf_counter() - start, completed_epoch=time.time())
    decision = output / "routing_decision.json"
    if decision.exists() or decision.with_suffix(".json.gz").exists():
        row["pool"] = json.loads(read_text(decision)).get("selected")
    if arrival_ns is not None:
        row["latency_s"] = (time.time_ns() - arrival_ns) / 1e9
    profile = output / "workload_profile.json"
    if profile.exists() or profile.with_suffix(".json.gz").exists():
        row["profile"] = json.loads(read_text(profile))
    return row


async def execute(sequence: Path, config: dict, output: Path, mode: str, providers: list[str]) -> dict:
    from benchmarks.TravelPlanner.langgraph_agent import graph as travel
    from experiments.e3.manifest import load
    from experiments.exp1.runner.arrival_scheduler import stats

    plan = load(sequence)
    output.mkdir(parents=True, exist_ok=False)
    first = next(t for t in plan["tasks"] if t["workload"] == "TravelPlanner")
    task = next(t for t in json.loads(Path(first["manifest"]).read_text())["tasks"] if t["task_id"] == first["task_id"])
    oracle = OfficialEvaluator(task, output / "oracle.log")
    original = travel.validate
    rows = []
    try:
        startup = await oracle.start()
        travel.validate = oracle.validate
        (output / "oracle-startup.json").write_text(json.dumps(startup, indent=2) + "\n")
        (output / "execution.json").write_text(
            json.dumps(
                {
                    "source_manifest_sha256": json.loads((sequence / "seal.json").read_text())["manifest_sha256"],
                    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "mode": mode,
                    "providers": providers,
                    "native_elasticity": config["native_elasticity"],
                    "config": config,
                    "scope": "real Tool replay; exact recorded LLM idle; official verifier cached in isolated process; no LLM API calls or scaling controller",
                },
                indent=2,
            )
            + "\n"
        )
        if mode != "preflight":
            raise ValueError("Use experiments.e3.run for the two formal arms")
        slots = {p: asyncio.Semaphore(4) for p in providers}

        async def one(task, provider):
            async with slots[provider]:
                row = await measure(task, config, provider, output / task["id"] / provider)
                rows.append(row)
                (output / "attempts.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(
                    json.dumps({k: row.get(k) for k in ["task_id", "pool", "success", "latency_s", "error"]}),
                    flush=True,
                )

        await asyncio.gather(*(one(t, p) for t in plan["tasks"] for p in providers))
        summary = {}
        for provider in providers:
            group = [r for r in rows if r["pool"] == provider]
            summary[provider] = {
                "attempts": len(group),
                "successes": sum(r["success"] for r in group),
                "success_rate": sum(r["success"] for r in group) / len(group),
                "latency_s": stats([r["latency_s"] for r in group]),
                "workload_requests": dict(Counter(r["workload"] for r in group)),
                "actual_total_cost": None,
            }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        travel.validate = original
        await oracle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["oracle", "preflight"], required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--sequence", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--providers", nargs="+", default=["aws_lambda", "aws_agentcore", "ali_fc", "ali_agentrun"])
    args = parser.parse_args()
    if args.mode == "oracle":
        oracle_worker(args.upstream.resolve())
        return
    from dotenv import load_dotenv

    load_dotenv(".env")
    with asyncio.Runner() as runner:
        runner.get_loop().set_default_executor(ThreadPoolExecutor(max_workers=128))
        print(
            json.dumps(
                runner.run(
                    execute(args.sequence, json.loads(args.config.read_text()), args.output, args.mode, args.providers)
                )
            )
        )


if __name__ == "__main__":
    main()
