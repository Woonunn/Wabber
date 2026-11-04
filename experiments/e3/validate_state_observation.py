"""Execute the 213 captured Tool sequences locally and audit state observation."""

import argparse
import copy
import importlib
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from atsuite.tracing.storage import read_text, write_json

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "results/e3-independent/capture-index.json"


def tool_events(capture: dict) -> list[dict]:
    """Read the original executed calls, including business failures and retries."""
    return [e for e in map(json.loads, read_text(Path(capture["trace"]) / "events.jsonl").splitlines())
            if e["kind"] == "tool"]


def expected_state(capture: dict, events: list[dict]) -> dict:
    """Independent census from the four workloads' persistent object contracts."""
    workload = capture["workload"]
    if workload == "TravelPlanner":
        state_tools = [e["tool_name"] for e in events if e["tool_name"].startswith("notebook_")]
        objects = ["notebook"] if state_tools else []
    else:
        state_tools = [e["tool_name"] for e in events]
        objects = {"DataSciBench": ["workspace", "canonical/state", "branch state/artifacts"],
                   "ScientificComputation": ["tensor_workspace"],
                   "ClaudeCodeReview": ["claude_code_review_state", "test results/checkouts"]}[workload]
    return {"task": capture["id"], "workload": workload, "tool_calls": len(events),
            "expected_stateful": bool(state_tools), "state_tools": sorted(set(state_tools)),
            "expected_objects": objects}


def load_tools(workload: str) -> object:
    """Load real implementations in a fresh per-request process."""
    if workload == "TravelPlanner":
        previous = Path.cwd()
        os.chdir(ROOT / "results/e3-travel-v2")
        try:
            for node in ("notebook", "flights", "restaurants", "attractions", "accommodations",
                         "cities", "google_distance_matrix"):
                importlib.import_module(f"benchmarks.TravelPlanner.nodes.{node}.implementation")
        finally:
            os.chdir(previous)
    elif workload == "ClaudeCodeReview":
        importlib.import_module("experiments.e3.review_tools")
    elif workload == "ScientificComputation":
        importlib.import_module("benchmarks.ScientificComputation.nodes.tensor_workspace.implementation")
    else:
        return importlib.import_module("benchmarks.DataSciBench.nodes.AgentTools.implementation")
    return None


def validate_request(capture: dict, output: Path, mode: str) -> dict:
    """Observe actual executions; never fill an absent measurement with zero."""
    from atsuite.tracing import TraceCollector
    from atsuite.workload import extract_profile
    from atsuite_sdk.abstract import registry
    from atsuite_sdk.resources import get_resource_usage
    from atsuite_sdk.state import get_state_runtime

    events = tool_events(capture)
    census = expected_state(capture, events)
    implementation = load_tools(capture["workload"])
    runtime = get_state_runtime()
    collector = TraceCollector(capture["id"], capture["id"])
    collector.start("session", "session", "local-state-verification")
    mismatches, failed, observations, unexpected_errors = [], [], [], []
    produced, rebound = {}, []
    try:
        for index, event in enumerate(events):
            name = event["tool_name"]
            print(json.dumps({"tool_start": event["key"], "name": name}), flush=True)
            tool = registry.functions[name]
            span = str(index)
            call_args = copy.deepcopy(event["args"])
            if name == "merge_artifacts":
                for branch in call_args["branches"]:
                    current = produced.get(branch["branch_ref"])
                    if current is not None:
                        artifacts = [{k: a[k] for k in ("name", "signature", "comparison")} for a in current["artifacts"]]
                        assert sorted(a["name"] for a in artifacts) == sorted(a["name"] for a in branch["artifacts"])
                        if artifacts != branch["artifacts"]:
                            rebound.append({"key": event["key"], "branch_ref": branch["branch_ref"]})
                            branch["artifacts"] = artifacts
            collector.start(span, "tool", name, inputs=call_args)
            error, result = None, None
            try:
                result = tool.execute_from_dict({**call_args, "uid": capture["id"]})
                if name == "execute_analysis" and result.get("branch_ref"):
                    produced[result["branch_ref"]] = result
            except Exception as exc:
                error = exc
                result = getattr(exc, "result", None)
                failed.append({"key": event["key"], "type": type(exc).__name__, "error": str(exc)[:500]})
                original = event["result"]
                expected_error = (event.get("transport", {}).get("status") in {"error", "timeout"}
                                  or isinstance(original, dict) and original.get("status") in {"error", "timeout"})
                if not expected_error:
                    unexpected_errors.append(event["key"])
            collector.end(span, result, error=error)
            collector.observe_runtime(span, {"result": result, "provider_metadata": {
                "logical_state": runtime.get_logical_state(), "resource_usage": get_resource_usage()}})
            metadata = collector.spans[span]["metadata"]
            if capture["workload"] == "DataSciBench":
                assert metadata.get("logical_state", {}).get("contract") == "datasci-tool-snapshot-v1", \
                    "SDK workspace metadata cannot replace the DataSci worker's state observation"
            observed = metadata.get("state_accesses")
            if not isinstance(observed, list):
                raise AssertionError(f"Unobserved state at {capture['id']}:{event['key']}: {metadata.get('logical_state')}")
            expected_call_state = (capture["workload"] != "TravelPlanner" or name.startswith("notebook_"))
            if bool(observed) != expected_call_state:
                raise AssertionError(f"State classification differs at {capture['id']}:{event['key']}")
            assert all(isinstance(a["bytes"], int) and a["bytes"] >= 0 for a in observed)
            observations.extend(observed)
            print(json.dumps({"tool_end": event["key"], "state_bytes": sum(a["bytes"] for a in observed)}), flush=True)
            # Business equivalence is disclosed separately from instrumentation correctness.
            from benchmarks.DataSciBench.langgraph_agent.tool_client import semantic
            from experiments.e3.agent import result_view
            view = semantic if capture["workload"] == "DataSciBench" else result_view
            if result is not None and view(result) != view(event["result"]):
                mismatches.append(event["key"])
        collector.end("session", error=None if capture["business_success"] else "Original business failure")
        trace = collector.snapshot()
        profile = extract_profile(trace)
        assert profile["metrics"]["state_bytes"] == sum(a["bytes"] for a in observations)
        assert (profile["metrics"]["state_bytes"] > 0) == census["expected_stateful"]
        assert all(profile["metrics"][key] is not None for key in
                   ("state_bytes", "state_reads", "state_writes", "state_max_object_bytes"))
        write_json(output / "trace.json", trace)
        write_json(output / "profile.json", profile)
        assert not unexpected_errors, f"Original business paths were not reproduced: {unexpected_errors}; {failed[:2]}"
        result = {**census, "runtime": mode, "stateful": bool(observations),
                  "state_bytes": profile["metrics"]["state_bytes"], "state_accesses": observations,
                  "state_reads": profile["metrics"]["state_reads"], "state_writes": profile["metrics"]["state_writes"],
                  "business_output_differences": mismatches, "business_tool_errors": failed,
                  "rebound_artifact_references": rebound,
                  "source_business_success": capture["business_success"], "classification_correct": True}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        if implementation is not None:
            if implementation._mcp_kernel:
                implementation._mcp_kernel.close()
            for kernel in implementation._faas_kernels.values():
                kernel.close()


def worker(capture: dict, output: Path, mode: str) -> None:
    """Keep all mutable workspaces, imports and resident state local to this request."""
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="state-audit-") as temporary:
        os.environ.update(ATSUITE_RUNTIME=mode, PROVIDER="", ATSUITE_SESSION_ROOT=temporary,
                          WABBER_EXECUTION_BACKEND="native", WABBER_NATIVE_CONFINEMENT="landlock",
                          WABBER_CODE_TIMEOUT="600",
                          WABBER_DATA_ROOT=str(ROOT / "benchmarks/DataSciBench/data"),
                          ATSUITE_REVIEW_PROJECTS=str(ROOT / "results/e3-stateful/review"),
                          ATSUITE_DATA_PATH=str(ROOT / "results/e3-stateful/review/requests"),
                          ATSUITE_SCIENTIFIC_PROBLEMS=str(Path(capture["task_manifest"]).parent))
        validate_request(capture, output, mode)


def main() -> None:
    """Write the expected census first, then compare every actual call against it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", choices=("mcp", "function"), default="mcp")
    parser.add_argument("--request", action="append")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    corpus = json.loads(CORPUS.read_text())
    captures = {c["id"]: c for c in corpus["captures"]}
    if args.worker:
        worker(captures[args.request[0]], args.output, args.runtime)
        return
    selected = [captures[r["task"]] for r in corpus["requests"] if not args.request or r["task"] in args.request]
    assert len(selected) == (len(set(args.request)) if args.request else 213), "Every requested case must exist"
    args.output.mkdir(parents=True, exist_ok=True)
    census = [expected_state(c, tool_events(c)) for c in selected]
    expected_path = args.output / "expected.json"
    if expected_path.exists():
        assert json.loads(expected_path.read_text()) == census, "Resume must use the same census"
    else:
        expected_path.write_text(json.dumps(census, indent=2) + "\n")
    ledger = args.output / "requests.jsonl"
    if ledger.exists():
        ledger.replace(args.output / "requests.previous.jsonl")

    def execute(capture):
        output = args.output / capture["id"]
        if (output / "result.json").exists():
            previous = json.loads((output / "result.json").read_text())
            assert previous["classification_correct"] and previous["task"] == capture["id"]
            return previous
        if output.exists():
            output.rename(output.with_name(output.name + f".failed-{time.time_ns()}"))
        image = ("wabber-independent-datasci-sdk:verified" if capture["workload"] == "DataSciBench"
                 else "wabber-e3-stateful-tests:local")
        command = ["docker", "run", "--rm", "--network", "none", "--user", "0:0", "--cap-drop", "ALL",
                   "--security-opt", "seccomp=unconfined", "--entrypoint", "python",
                   "--mount", f"type=bind,src={ROOT},dst={ROOT},readonly",
                   "--mount", f"type=bind,src={args.output.resolve()},dst={args.output.resolve()}",
                   "--workdir", str(ROOT), "--env", f"PYTHONPATH={ROOT}:{ROOT / 'benchmarks/DataSciBench/nodes/DataSci'}",
                   "--env", "OPENBLAS_NUM_THREADS=1", "--env", "OMP_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1",
                   image, "-c", "import runpy,sys; sys.path.append('/root/Wabber/.venv/lib/python3.12/site-packages'); "
                   "runpy.run_module('experiments.e3.validate_state_observation', run_name='__main__')",
                   "--worker", "--request", capture["id"], "--runtime", args.runtime, "--output", str(output.resolve())]
        with (args.output / (capture["id"] + ".log")).open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        if completed.returncode:
            return {"task": capture["id"], "workload": capture["workload"], "classification_correct": False,
                    "error_log": str(args.output / (capture["id"] + ".log"))}
        return json.loads((output / "result.json").read_text())

    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(execute, c) for c in selected]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            with (args.output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps({"complete": len(results), "correct": sum(x["classification_correct"] for x in results),
                              "last": row["task"], "ok": row["classification_correct"]}), flush=True)
    summary = {"requests": len(results), "correct": sum(r["classification_correct"] for r in results),
               "expected_stateful": sum(c["expected_stateful"] for c in census),
               "observed_stateful": sum(r.get("stateful", False) for r in results),
               "unknown": sum(r.get("state_bytes") is None for r in results),
               "tool_calls": sum(c["tool_calls"] for c in census),
               "workloads": dict(Counter(c["workload"] for c in census)),
               "business_output_differences": sum(len(r.get("business_output_differences", [])) for r in results),
               "runtime": args.runtime, "cloud_calls": 0, "model_calls": 0}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    assert summary["requests"] == summary["correct"] and summary["unknown"] == 0, summary


if __name__ == "__main__":
    main()
