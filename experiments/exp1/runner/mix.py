"""Real isolated DataSciBench-Mix preflight, LangGraph capture and acceptance gate."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import gzip
from pathlib import Path
import shutil
import subprocess
import tarfile
import uuid

from experiments.exp1.data.prepare_mix import ROOT, sha

CONTRACT = (
    "Implement the original task_func as self-contained source in output_dir / 'solution.py'. "
    "Use one analysis step with outputs ['solution.py','bcb_execution.json','bcb_result.pkl']. "
    "Write solution.py using a Python string; do not write bcb_execution.json or bcb_result.pkl yourself. "
    "The Tool automatically imports your solution and actually invokes task_func on the "
    "prepared original upstream arguments, then writes bcb_execution.json. "
    "The independent validator runs the complete original upstream test suite in a fresh worker. "
    "Keep the original signature and tuple return contract exactly. No reference solutions, "
    "test inspection, monkeypatching the validator, or replacement inputs. "
    "Do not load fixture_args.pkl yourself. All paths passed to task_func are used as given. "
    "Do not call task_func separately or manufacture extra work."
)


def write(path: Path, value: dict) -> None:
    """Atomically persist metadata; individual run directories are never reused."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def input_records(directory: Path) -> list[dict]:
    """Hash only regular bounded files; reject links from sandbox outputs."""
    records = []
    total = 0
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Unsafe prepared input: " + str(path))
        if path.is_file():
            total += path.stat().st_size
            if total > 1024**3 or len(records) >= 256:
                raise ValueError("Prepared input quota exceeded")
            records.append({"name": path.relative_to(directory).as_posix(),
                            "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size})
    return records


def verify_capture(task: dict, directory: Path, *, require_live: bool = True) -> dict:
    """Accept only matching live LangGraph traces with actual calls and independent tests."""
    from atsuite.tracing.storage import read_text, trace_path

    state = json.loads(read_text(directory / "result.json"))
    events = [json.loads(line) for line in read_text(directory / "events.jsonl").splitlines()]
    trace = json.loads(read_text(directory / "trace.json"))
    if state.get("status") != "complete":
        raise ValueError("Agent execution failed: " + str(state.get("feedback") or state.get("error")))
    if (state.get("task_id") != task["task_id"]
            or state.get("user_prompt") != task["prompt"]
            or sha(state["user_prompt"].encode()) != task["prompt_sha256"]
            or state.get("dataset_refs") != task["readiness"]["inputs"]["files"]):
        raise ValueError("Capture does not match the original prompt and sealed inputs")
    starts = [e for e in events if e["kind"] == "run_start"]
    mode = "capture" if require_live else "environment_reference"
    if len(starts) != 1 or starts[0]["mode"] != mode:
        raise ValueError("Only live Agent capture counts as new Prompt validation")
    source = "live_official_api" if require_live else "environment_reference_frozen_decision"
    if not any(e["kind"] == "llm" and e.get("source") == source for e in events):
        raise ValueError("Missing real model execution")
    if trace["request_id"] != state["request_id"] or not trace["spans"]:
        raise ValueError("Missing matching tracing evidence")
    validation = state.get("validation", {})
    checks = validation.get("checks", {})
    if (validation.get("status") != "ok" or checks.get("tests_run", 0) < 1
            or checks.get("function_calls", 0) < 1 or checks.get("original_input_invocations") != 1):
        raise ValueError("No complete independent invocation and test evidence")
    expected_test = task.get("validation", {}).get("hashes", {}).get("testcase")
    if not expected_test or checks.get("test_sha256") != expected_test:
        raise ValueError("Independent test source changed")
    calls = [e for e in events if e["kind"] == "tool" and e.get("tool_name") == "execute_analysis"
             and e.get("result", {}).get("status") == "ok"]
    if not any(e["result"].get("bcb_invocation_s", -1) >= 0 for e in calls):
        raise ValueError("Missing measured original function call")
    return {"directory": str(directory), "checks": checks,
            "hashes": {name: sha(trace_path(directory / name).read_bytes())
                       for name in ("result.json", "events.jsonl", "trace.json", "workload_profile.json")}}


def seal(manifest_path: Path, captures: list[Path], output: Path, data: Path, *, references: bool = False) -> dict:
    """Freeze distinct validated Prompts and all evidence before arrival assignment."""
    manifest = json.loads(manifest_path.read_text())
    selected, rejected, seen = [], [], set()
    for task in manifest["tasks"]:
        if references and verify_capture(task, Path(task["agent_validation"]["directory"])) != task["agent_validation"]:
            raise ValueError("Original Agent evidence changed")
        if input_records(data / task["task_id"]) != task["readiness"]["inputs"]["files"]:
            raise ValueError("Prepared input changed: " + task["task_id"])
        for root in captures:
            directory = root / task["task_id"]
            if not directory.exists():
                continue
            try:
                evidence = verify_capture(task, directory, require_live=not references)
            except (ValueError, KeyError, OSError) as exc:
                rejected.append({"task_id": task["task_id"], "directory": str(directory), "reason": str(exc)})
                continue
            if task["prompt_sha256"] not in seen:
                selected.append({**task, "replay_reference" if references else "agent_validation": evidence})
                seen.add(task["prompt_sha256"])
            break
    if output.exists():
        raise FileExistsError("Never overwrite a sealed workload")
    output.mkdir(parents=True)
    result = {**manifest, "tasks": selected, "task_count": len(selected),
              "agent_verified_count": len(seen), "formal_experiment_ready": len(seen) >= 100,
              "reference_verified_count": len(seen) if references else 0,
              "preflight_manifest_sha256": sha(manifest_path.read_bytes()), "rejected": rejected}
    write(output / "manifest.json", result)
    write(output / "seal.json", {"manifest_sha256": sha((output / "manifest.json").read_bytes())})
    return result


def export(manifest_path: Path, output: Path, data: Path) -> dict:
    """Archive exact inputs, traces and actual returned artifacts without host unpickling."""
    from atsuite.tracing.storage import read_text, trace_path

    manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or (output / "evidence.tar.gz").exists():
        raise FileExistsError("Never overwrite exported evidence")
    files = set()
    for task in manifest["tasks"]:
        if input_records(data / task["task_id"]) != task["readiness"]["inputs"]["files"]:
            raise ValueError("Prepared input changed")
        files.update(data / task["task_id"] / f["name"] for f in task["readiness"]["inputs"]["files"])
        for key in ("agent_validation", "replay_reference"):
            evidence = task[key]
            directory = Path(evidence["directory"])
            if verify_capture(task, directory, require_live=key == "agent_validation") != evidence:
                raise ValueError("Evidence changed")
            files.update(trace_path(directory / name) for name in evidence["hashes"])
        original = Path(task["agent_validation"]["directory"])
        events = [json.loads(line) for line in read_text(original / "events.jsonl").splitlines()]
        uid = next(e["uid"] for e in events if e["kind"] == "tool_session_open")
        state = json.loads(read_text(original / "result.json"))
        matches = list((ROOT / "results/stage8").glob(f"runtime*/datasci-agent/{uid}/workspace/v{state['snapshot']}/artifacts"))
        if len(matches) != 1:
            raise ValueError("Missing unique actual artifact directory")
        artifacts = []
        for record in state["artifacts"]:
            path = matches[0] / record["name"]
            digest = sha(path.read_bytes())
            if digest != record["sha256"]:
                raise ValueError("Executed artifact checksum mismatch")
            files.add(path)
            artifacts.append({"path": str(path.relative_to(ROOT)), "sha256": digest})
        task["executed_artifacts"] = artifacts
    bundle = output / "evidence.tar.gz"
    with bundle.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in sorted(files):
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Evidence must contain regular files only")
                name = path.resolve().relative_to(ROOT).as_posix()
                info = archive.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)
    manifest["bundle"] = {"path": str(bundle.resolve().relative_to(ROOT)), "sha256": sha(bundle.read_bytes()),
                          "files": len(files), "bytes": bundle.stat().st_size}
    write(output / "manifest.json", manifest)
    write(output / "seal.json", {"manifest_sha256": sha((output / "manifest.json").read_bytes())})
    return {"tasks": len(manifest["tasks"]), "bundle": manifest["bundle"]}


def preflight(bundle: Path, image: str, output: Path, data: Path, concurrency: int,
              task_ids: list[str] | None = None) -> dict:
    """Audit all 172 candidates and actually test each BCB reference in Docker."""
    if not image.startswith("sha256:"):
        raise ValueError("Pin the sandbox image ID")
    output.mkdir(parents=True, exist_ok=False)
    catalog = json.loads((ROOT / "benchmarks/DataSciBench/tasks/catalog.json").read_text())
    review = json.loads(bundle.with_name("audit.json").read_text())
    by_id = {t["task_id"]: t for t in catalog["tasks"]}

    def one(record):
        task_id = record["task_id"]
        if record["status"] != "pending_sandbox_preflight" or (task_ids is not None and task_id not in task_ids):
            return record, None
        directory = output / task_id
        directory.mkdir(mode=0o777)
        directory.chmod(0o777)
        name = "wabber-mix-preflight-" + uuid.uuid4().hex
        command = ["docker", "run", "--name", name, "--network", "none", "--read-only",
                   "--cpus", "2", "--memory", "4g", "--memory-swap", "4g", "--pids-limit", "128",
                   "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65532:65532",
                   "--ulimit", "fsize=536870912:536870912", "--tmpfs", "/tmp:rw,nosuid,size=256m",
                   "-v", f"{bundle.resolve()}:/review/bundle.json.gz:ro",
                   "-v", f"{directory.resolve()}:/work:rw", "--entrypoint", "python", image,
                   "/opt/wabber/bcb_checks.py", task_id, "/review/bundle.json.gz", "/work"]
        try:
            with (directory / "container.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
            result = json.loads((directory / "preflight.json").read_text())
            record = {**record, "status": result["status"], "preflight": result}
            if result["status"] != "reference_validated":
                record["reasons"] = [result.get("error", "Reference check failed")]
                return record, None
            files = input_records(directory / "inputs")
            destination = data / task_id
            if destination.exists():
                if input_records(destination) != files:
                    raise ValueError("Existing prepared inputs differ; refusing overwrite")
            else:
                shutil.copytree(directory / "inputs", destination)
            task = {**by_id[task_id], "output_files": ["solution.py", "bcb_execution.json", "bcb_result.pkl"],
                    "validation_contract": CONTRACT,
                    "readiness": {"status": "reference_validated_not_agent_validated",
                                  "inputs": {"status": "prepared", "files": files}},
                    "validation": {"source": review["source"], "hashes": record["hashes"]}}
            return record, task
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            return {**record, "status": "preflight_failed", "reasons": [str(exc)]}, None
        finally:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    tasks, records = [], []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for record, task in pool.map(one, review["tasks"]):
            records.append(record)
            if task:
                tasks.append(task)
            print(record["task_id"], record["status"], flush=True)
    manifest = {"schema_version": 1, "name": "DataSciBench-Mix", "source": review["source"],
                "sandbox_image": image, "seed": 0, "tasks": tasks, "task_count": len(tasks),
                "agent_verified_count": 0, "formal_experiment_ready": False}
    write(output / "manifest.json", manifest)
    write(output / "audit.json", {**review, "tasks": records})
    return manifest


async def capture(manifest_path: Path, config_path: Path, output: Path, concurrency: int,
                  task_ids: list[str] | None = None, *, references: bool = False) -> dict:
    """Run new model decisions through the existing LangGraph and real Tool RPC."""
    from experiments.exp1.capture.capture_agent_traces import run
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    selected = [t for t in manifest["tasks"] if task_ids is None or t["task_id"] in task_ids]
    if task_ids is not None and set(task_ids) != {t["task_id"] for t in selected}:
        raise ValueError("Requested tasks must pass preflight")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(task):
        async with semaphore:
            directory = output / task["task_id"]
            try:
                original = Path(task["agent_validation"]["directory"]) if references else None
                if references:
                    if verify_capture(task, original) != task["agent_validation"]:
                        raise ValueError("Source capture changed since sealing")
                result = await run(task["task_id"], config, directory, capture=original,
                                   calibration=references, manifest=manifest_path)
                result["evidence"] = verify_capture(task, directory, require_live=not references)
                result["prompt_sha256"] = task["prompt_sha256"]
            except Exception as exc:
                result = {"task_id": task["task_id"], "status": "failed", "error": str(exc)}
            write(output / (task["task_id"] + ".json"), result)
            print(task["task_id"], result["status"], flush=True)
            return result

    results = await asyncio.gather(*(one(task) for task in selected))
    verified = {r["prompt_sha256"] for r in results if r["status"] == "complete"}
    report = {"manifest_sha256": sha(manifest_path.read_bytes()), "config_sha256": sha(config_path.read_bytes()),
              "results": results, "agent_verified_count": 0 if references else len(verified),
              "reference_count": len(verified) if references else 0,
              "formal_experiment_ready": not references and len(verified) >= 100}
    write(output / "report.json", report)
    return report


def main() -> None:
    """Run preflight or real capture, keeping reference and Agent evidence distinct."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "capture", "references", "seal", "reference-seal", "export"])
    parser.add_argument("--bundle", type=Path, default=ROOT / "benchmarks/DataSciBench/tasks/mix/bcb_cases.json.gz")
    parser.add_argument("--image")
    parser.add_argument("--data", type=Path, default=ROOT / "benchmarks/DataSciBench/data")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "benchmarks/DataSciBench/config/langgraph_experiment.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--task", action="append")
    parser.add_argument("--captures", type=Path, action="append")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 16:
        parser.error("concurrency must be between 1 and 16")
    load_dotenv(ROOT / ".env", override=False)
    if args.mode == "preflight":
        if not args.image:
            parser.error("preflight requires --image")
        result = preflight(args.bundle, args.image, args.output, args.data, args.concurrency, args.task)
    elif args.mode in {"capture", "references"}:
        if not args.manifest:
            parser.error("capture requires --manifest")
        result = asyncio.run(capture(args.manifest, args.config, args.output, args.concurrency, args.task,
                                     references=args.mode == "references"))
    elif args.mode == "export":
        if not args.manifest:
            parser.error("export requires --manifest")
        result = export(args.manifest, args.output, args.data)
    else:
        if not args.manifest or not args.captures:
            parser.error("seal requires --manifest and --captures")
        result = seal(args.manifest, args.captures, args.output, args.data, references=args.mode == "reference-seal")
    print(json.dumps({k: v for k, v in result.items() if k not in {"tasks", "results"}}))


if __name__ == "__main__":
    main()
