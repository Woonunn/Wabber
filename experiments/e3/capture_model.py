"""Run every prepared request once with a new model, retaining unsuccessful tasks."""

import argparse
import asyncio
import copy
import fcntl
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

from dotenv import load_dotenv

from atsuite.tracing.storage import read_text
from experiments.e3.independent import prepared
from experiments.e3.manifest import digest
from experiments.exp1.capture.capture_agent_traces import run

ROOT = Path(__file__).resolve().parents[2]
TOOLS_IMAGE = "sha256:2298dd8ea96a7e03cbc362b0ee7c55d41eedd4f216a1eee9b559ff3820a45f1a"


def write(path: Path, value: dict) -> None:
    """Atomically publish progress without risking the previous index."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def outcome(directory: Path) -> dict:
    """Audit recording completeness independently of business or API success."""
    events = [json.loads(line) for line in read_text(directory / "events.jsonl").splitlines()]
    state = json.loads(read_text(directory / "result.json"))
    trace = json.loads(read_text(directory / "trace.json"))
    complete = (
        bool(events) and events[0]["kind"] == "run_start"
        and [e["seq"] for e in events] == list(range(len(events)))
        and all(e["request_id"] == trace["request_id"] for e in events)
        and bool(trace.get("completed_ns")) and trace.get("sessions_closed") is True
        and bool(trace["spans"]) and all(s.get("end_ns") is not None for s in trace["spans"])
        and any(e["kind"] == "tool_sessions_closed" for e in events)
    )
    errors = [e for e in events if e["kind"] == "run_error"]
    failure = None
    if state["status"] != "complete":
        last = errors[-1] if errors else {}
        llm_errors = [e for e in events if e["kind"] == "llm_error"]
        if last.get("error", "").startswith("Agent did not complete:"):
            failure = "task"
        elif last.get("error_type") == "TimeoutError":
            failure = "timeout"
        elif llm_errors and llm_errors[-1].get("error_type") == last.get("error_type"):
            failure = "model_output" if llm_errors[-1].get("response") is not None else "model_api"
        else:
            failure = "runtime"
    return {
        "capture_status": "complete" if complete else "incomplete",
        "task_status": state["status"], "failure_origin": failure,
        "request_id": trace["request_id"], "trace": str(directory.resolve()),
        "validation": state.get("validation"), "error": state.get("error"),
        "model_response_ids": [e["response_id"] for e in events if e["kind"] == "llm"],
        "tool_calls": sum(e["kind"] == "tool" for e in events),
        "usage": {key: sum((e.get("usage") or {}).get(key, 0) or 0 for e in events
                                         if e["kind"] in {"llm", "llm_error"})
                  for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
    }


async def command(*args: str, check: bool = True) -> str:
    """Run a local Docker command without shell interpolation."""
    process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await process.communicate()
    if check and process.returncode:
        raise RuntimeError(err.decode()[-2000:])
    return (out + err).decode()


async def ready(endpoint: str, process, workload: str | None) -> None:
    """Verify that the request has reached its own healthy Tool runtime."""
    def probe():
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=1) as response:
                if workload and json.load(response)["workload"] != workload:
                    raise ValueError("Wrong workload endpoint")
        except urllib.error.HTTPError as exc:
            if workload or exc.code != 404:
                raise

    for _ in range(120):
        if process and process.poll() is not None:
            raise RuntimeError("Local Tool service exited")
        try:
            await asyncio.to_thread(probe)
            return
        except (OSError, urllib.error.URLError):
            await asyncio.sleep(.5)
    raise TimeoutError("Local Tool service did not become ready")


def select_requests(plan: dict, request_ids: list[str]) -> dict:
    """Restrict execution and arrival metadata to an explicitly frozen request list."""
    tasks = {task["id"]: task for task in plan["tasks"]}
    if not request_ids or len(set(request_ids)) != len(request_ids) or not set(request_ids) <= tasks.keys():
        raise ValueError("Selection requires unique known request IDs")
    return {**plan, "tasks": [tasks[identity] for identity in request_ids],
            "requests": [r for r in plan["requests"] if r["task"] in request_ids]}


async def capture_all(source: Path, output: Path, model: dict, concurrency: int = 2,
                      *, request_ids: list[str] | None = None) -> dict:
    """Reuse prepared inputs and actual agent settings; replace only the model and local runtime."""
    if concurrency < 1:
        raise ValueError("Positive concurrency required")
    plan = prepared(source)
    reference = json.loads((source / "capture-index.json").read_text())
    old = {t["id"]: t for t in reference["captures"]}
    if len(plan["tasks"]) != 213 or set(old) != {t["id"] for t in plan["tasks"]}:
        raise ValueError("Expected the same 213 independent requests")
    if request_ids is not None:
        plan = select_requests(plan, request_ids)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "active.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return await _capture(plan, old, source, output, model, concurrency)


async def _capture(plan, old, source, output, model, concurrency):
    support = source / "local-capture"
    sandbox = (await command("docker", "image", "inspect", "wabber-independent-datasci:local", "--format", "{{.Id}}")).strip()
    await command("docker", "image", "inspect", TOOLS_IMAGE)
    policy = {"source_plan_sha256": digest(source / "plan.json"), "model": model,
              "tools_image": TOOLS_IMAGE, "datasci_image": sandbox, "concurrency": concurrency,
              "tool_service_sha256": digest(support / "tool_service.py"),
              "attempts_per_task": 1, "requests": plan["requests"],
              "scheduling": "bounded local capture; original arrivals retained as metadata, not replayed in real time"}
    policy_path = output / "policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
        raise ValueError("Capture policy changed; use a new output directory")
    write(policy_path, policy)
    network = "wabber-model-" + digest(policy_path)[:12]
    await command("docker", "network", "create", "--internal", network)
    slots, heavy = asyncio.Semaphore(concurrency), asyncio.Lock()
    from benchmarks.TravelPlanner.langgraph_agent import graph as travel
    original_validate = travel.validate

    async def validate(*args):
        async with heavy:
            return await original_validate(*args)

    travel.validate = validate
    rows = []

    def index():
        summary = {"expected": len(plan["tasks"]), "recorded": len(rows),
                   "capture_status": dict(Counter(r["capture_status"] for r in rows)),
                   "task_status": dict(Counter(r["task_status"] for r in rows)),
                   "failure_origin": dict(Counter(r["failure_origin"] for r in rows if r.get("failure_origin")))}
        value = {"summary": summary, "requests": plan["requests"], "captures": sorted(rows, key=lambda r: r["id"])}
        write(output / "capture-index.json", value)
        return value

    async def one(number, entry):
        folder = output / "requests" / entry["id"]
        receipt = folder / "receipt.json"
        if receipt.exists():
            row = json.loads(receipt.read_text())
            if row["capture_status"] == "complete":
                row.update(await asyncio.to_thread(outcome, folder / "trace"))
            rows.append(row)
            index()
            return
        async with slots:
            if (output / "STOP_AFTER_ACTIVE").exists():
                return
            # Scientific solves, DataSci workers and the official Travel evaluator share limited host RAM.
            async with heavy if entry["workload"] in {"DataSciBench", "ScientificComputation"} else nullcontext():
                if (output / "STOP_AFTER_ACTIVE").exists():
                    return
                if folder.exists():
                    raise ValueError("Unfinished prior attempt must be inspected: " + str(folder))
                if shutil.disk_usage(output).free < 2 * 1024**3:
                    raise RuntimeError("Less than 2 GiB free; preserving all records and stopping")
                folder.mkdir(parents=True)
                scratch, state = folder / "scratch", folder / "state"
                scratch.mkdir()
                state.mkdir()
                process = container = None
                workload = entry["workload"]
                row = {"id": entry["id"], "workload": workload, "task_id": entry["task_id"],
                       "manifest": entry["manifest"], "source_trace": old[entry["id"]]["trace"],
                       "capture_status": "incomplete", "task_status": "not_started", "started_epoch": time.time()}
                log = (folder / "service.log").open("w")
                try:
                    if workload == "DataSciBench":
                        endpoint = f"http://127.0.0.1:{20400 + number}"
                        env = {**os.environ, "PYTHONPATH": str(ROOT), "PROVIDER": "",
                               "ATSUITE_STATE_SERVICE": "independent-datasci", "ATSUITE_SESSION_ROOT": str(state),
                               "WABBER_DATA_ROOT": str(ROOT / "benchmarks/DataSciBench/data"),
                               "WABBER_SANDBOX_IMAGE": sandbox, "TMPDIR": str(scratch),
                               "ATSUITE_MANIFEST_PATH": str(support / "tool-manifests/DataSciBench.json"),
                               "ATSUITE_ALLOW_STATE_BOOTSTRAP": "false", "WABBER_CODE_TIMEOUT": "240",
                               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                        for name in ("WABBER_NATIVE_CONFINEMENT", "WABBER_EXECUTION_BACKEND"):
                            env.pop(name, None)
                        process = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-m",
                            "experiments.exp1.capture.local_tool_service", "--port", str(20400 + number), "--family", "mcp"],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    else:
                        container = "wab-model-" + network[-12:] + "-" + str(number)
                        args = ["docker", "run", "-d", "--name", container, "--network", network,
                                "--cpus", "2", "--memory", "4g", "--memory-swap", "4g", "--read-only",
                                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                                "-v", f"{scratch}:/tmp:rw", "-e", f"CAPTURE_WORKLOAD={workload}",
                                "-e", "PYTHONPATH=/root/Wabber", "-e", "ATSUITE_MANIFEST_PATH=/capture-manifest.json",
                                "-v", f"{support / 'tool-manifests' / (workload + '.json')}:/capture-manifest.json:ro",
                                "-v", f"{support / 'tool_service.py'}:/capture-tool-service.py:ro",
                                "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1"]
                        paths = ["atsuite", "atsuite_sdk", "experiments", "benchmarks/" + workload,
                                 {"TravelPlanner": "results/e3-independent/local-capture/travel-inputs",
                                  "ScientificComputation": "results/e3-independent/runtime-inputs/scientific",
                                  "ClaudeCodeReview": "results/e3-stateful/review"}[workload]]
                        for relative in paths:
                            args.extend(["-v", f"{ROOT / relative}:{ROOT / relative}:ro"])
                        await command(*args, "--entrypoint", "python", TOOLS_IMAGE, "-u", "/capture-tool-service.py")
                        actual = json.loads(await command("docker", "inspect", container))[0]
                        write(folder / "docker-readback.json", actual)
                        endpoint = "http://" + actual["NetworkSettings"]["Networks"][network]["IPAddress"] + ":8080"
                    await ready(endpoint, process, workload if container else None)
                    config = copy.deepcopy(old[entry["id"]]["actual_agent_config"])
                    config["model"] = {**model, "seed": config["model"]["seed"]}
                    config["tool"] = {"provider": "local", "family": "session", "target_id": "independent-" + workload,
                                      "endpoint": endpoint, "timeout": 240,
                                      "execution_scope": "Local Docker MCP Session; independent live model capture"}
                    write(folder / "config.json", config)
                    try:
                        await run(entry["task_id"], config, folder / "trace", manifest=Path(entry["manifest"]), workload=workload)
                    except Exception as exc:
                        row["execution_error"] = {"type": type(exc).__name__, "message": str(exc)}
                    row.update(await asyncio.to_thread(outcome, folder / "trace"))
                    if workload == "DataSciBench":
                        artifacts = list(state.glob("independent-datasci/*/workspace/v*/artifacts"))
                        if artifacts:
                            def archive():
                                with tarfile.open(folder / "generated-artifacts.tar.gz", "w:gz") as tar:
                                    for path in artifacts:
                                        tar.add(path, arcname=path.parent.name + "/artifacts")
                            await asyncio.to_thread(archive)
                except Exception as exc:
                    row["capture_error"] = {"type": type(exc).__name__, "message": str(exc)}
                finally:
                    if process:
                        process.terminate()
                        try:
                            await asyncio.to_thread(process.wait, 15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            await asyncio.to_thread(process.wait)
                    if container:
                        log.write(await command("docker", "logs", container, check=False))
                        await command("docker", "rm", "-f", container, check=False)
                    log.close()
                    if row["capture_status"] == "complete" and "capture_error" not in row:
                        shutil.rmtree(scratch)
                        shutil.rmtree(state)
                    row["completed_epoch"] = time.time()
                    write(receipt, row)
                    rows.append(row)
                    print(json.dumps(index()["summary"]), flush=True)
    try:
        await asyncio.gather(*(one(i, entry) for i, entry in enumerate(plan["tasks"])))
    finally:
        travel.validate = original_validate
        await command("docker", "network", "rm", network, check=False)
    return index()


def main() -> None:
    """Capture all 213 requests without retries selected by business success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/e3-independent")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "benchmarks/DataSciBench/config/langgraph_openrouter_gpt_oss_20b.json")
    parser.add_argument("--selection", type=Path, help="Frozen JSON selection containing request ids")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    selected = json.loads(args.selection.read_text())["ids"] if args.selection else None
    result = asyncio.run(capture_all(args.source.resolve(), args.output.resolve(), json.loads(args.config.read_text())["model"],
                                    request_ids=selected))
    print(json.dumps(result["summary"], ensure_ascii=False))
    if result["summary"]["capture_status"] != {"complete": result["summary"]["expected"]}:
        raise SystemExit("Some recordings need inspection; every attempt is retained")


if __name__ == "__main__":
    main()
