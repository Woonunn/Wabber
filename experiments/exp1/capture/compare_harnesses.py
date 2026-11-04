"""Run one original DataSciBench task with three native harnesses and local Docker tools."""

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values

from atsuite.runtime import create_runtime_adapter
from atsuite.tracing import TraceCollector, export_atg
from atsuite.tracing.langchain import LangChainTracingAdapter
from atsuite.tracing.openai_agents import OpenAIAgentsTracingAdapter
from atsuite.tracing.storage import read_text
from atsuite.tracing.tools import RuntimeToolset
from atsuite.tracing.topology import tool_topology

ROOT = Path(__file__).resolve().parents[3]
SYSTEM = """Complete the user's original data-analysis task using the available workspace tools.
First prepare_workspace with the task_id and dataset_refs provided below. Its snapshot starts at 0.
execute_analysis runs Python in an isolated Docker worker against an immutable snapshot. Supply a unique
step_id and call_id, seed=0, code, and the exact filenames that code produces in outputs. Python receives
output_dir (Path), inputs_dir (Path), shared (committed Python values), state_out (dict for new values),
artifact_path(name) (committed artifact filename), and seed. Store outputs under output_dir. Use matplotlib
Agg. Write real computations and charts; never claim success without execution and validation.
Choose your own grouping and dependencies. Independent analyses may execute concurrently against the same
snapshot. Sibling branches cannot see one another's uncommitted state or artifacts.
After analyses, merge_artifacts commits successful branches and increments snapshot. Each branch reference
must contain step_id, branch_ref, and artifacts copied from its execute_analysis result; each artifact
reference contains ONLY name, signature, comparison. Merge all successful independent branches before
advancing their shared snapshot. Use shared[step_id] for committed state_out from a prior step.
Finally call validate_result with ALL originally requested output filenames and the latest snapshot.
Finish only if validation returns status=ok. Do not generate tracing or resource-monitoring code.
"""


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def tool_peak(trace):
    points = []
    for span in trace["spans"]:
        if span["kind"] == "tool" and span["end_ns"] > span["start_ns"]:
            points.extend([(span["start_ns"], 1), (span["end_ns"], -1)])
    active = peak = 0
    for _, delta in sorted(points):
        active += delta
        peak = max(peak, active)
    return peak


async def native_run(harness, task, config, output, api_key):
    from agents import Agent, ModelSettings, OpenAIChatCompletionsModel
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from openai import AsyncOpenAI

    from benchmarks.DataSciBench.nodes.AgentTools import implementation

    output.mkdir()
    collector = TraceCollector(uuid4().hex, uuid4().hex, output)
    adapter = (LangChainTracingAdapter if harness == "langchain" else OpenAIAgentsTracingAdapter)(collector)
    functions = [
        getattr(implementation, name)
        for name in (
            "prepare_workspace",
            "execute_analysis",
            "merge_artifacts",
            "read_workspace",
            "validate_result",
        )
    ]
    runtime = create_runtime_adapter("local", "faas")
    runtime.connect({"targets": {"datasci": config["tool"]["endpoint"]}})

    async def record_response(response):
        await response.aread()
        with (output / "model-responses.jsonl").open("a") as stream:
            stream.write(json.dumps({"status": response.status_code, "response": response.json()}) + "\n")

    references = [
        {key: ref[key] for key in ("name", "sha256", "bytes")} for ref in task["readiness"]["inputs"]["files"]
    ]
    instructions = SYSTEM + "\nWorkspace task_id: " + task["task_id"] + "\ndataset_refs: " + json.dumps(references)
    async with (
        RuntimeToolset(runtime, "datasci") as tools,
        httpx.AsyncClient(
            event_hooks={"response": [record_response]},
            timeout=180,
        ) as http,
    ):
        if harness == "langchain":
            model = ChatOpenAI(
                model="deepseek-flash",
                api_key=api_key,
                base_url="https://api.deepseek.com",
                temperature=0,
                top_p=1,
                max_tokens=16000,
                max_retries=2,
                extra_body={"thinking": {"type": "disabled"}},
                http_async_client=http,
            )
            agent = create_agent(model, tools.langchain_tools(functions), system_prompt=instructions)
            result = await adapter.ainvoke(
                agent,
                {"messages": [{"role": "user", "content": task["prompt"]}]},
                {"recursion_limit": 80, "max_concurrency": 3},
            )
            answer = result["messages"][-1].content
        else:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                http_client=http,
                max_retries=2,
                timeout=180,
            )
            agent = Agent(
                name="DataSciBench",
                instructions=instructions,
                tools=tools.openai_tools(functions),
                model=OpenAIChatCompletionsModel("deepseek-flash", client),
                model_settings=ModelSettings(
                    temperature=0,
                    top_p=1,
                    max_tokens=16000,
                    extra_body={"thinking": {"type": "disabled"}},
                ),
            )
            result = await adapter.ainvoke(agent, task["prompt"], max_turns=20)
            answer = result.final_output
    write_json(output / "result.json", {"answer": answer})


def validate_run(harness, output, task, state_root):
    trace = json.loads(read_text(output / "trace.json"))
    if not trace["summary"]["complete"] or not trace["summary"]["successful"]:
        raise AssertionError(f"{harness}: incomplete or unsuccessful trace: {trace['diagnostics']}")
    tools = [span for span in trace["spans"] if span["kind"] == "tool"]
    validations = [
        span["metadata"].get("runtime", {}).get("result") for span in tools if span["name"] == "validate_result"
    ]
    validation = validations[-1] if validations else None
    if (
        not validation
        or validation.get("status") != "ok"
        or not validation.get("checks", {}).get("chart_geometry_verified")
    ):
        raise AssertionError(f"{harness}: original task did not pass independent chart checks")
    if any(not span["resources"] or "peak_rss_bytes" not in span["resources"] for span in tools):
        raise AssertionError(f"{harness}: missing automatic CPU/RSS observations")
    uid = tools[0]["metadata"]["runtime_uid"]
    sources = [
        path
        for path in state_root.rglob("artifacts")
        if uid in path.parts and all((path / name).is_file() for name in task["output_files"])
    ]
    if len(sources) != 1:
        raise AssertionError(f"{harness}: cannot uniquely locate validated artifacts")
    shutil.copytree(sources[0], output / "artifacts")
    write_json(output / "validation.json", validation)
    topology = tool_topology(trace)
    write_json(output / "tool-topology.json", topology)
    export_atg(trace, output / "trace.atg.json")
    return {
        "harness": harness,
        "task_id": task["task_id"],
        "prompt_sha256": task["prompt_sha256"],
        "validated": True,
        "tool_parallel_peak": tool_peak(trace),
        **trace["summary"],
        "topology": topology,
    }


def render(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    layouts = []
    for row in rows:
        graph = row["topology"]
        ranks = {}
        pending = {node["id"] for node in graph["nodes"]}
        while pending:
            ready = [
                node["id"]
                for node in graph["nodes"]
                if node["id"] in pending
                and all(edge["source"] in ranks for edge in graph["edges"] if edge["target"] == node["id"])
            ]
            if not ready:
                raise AssertionError("Tool topology contains a cycle")
            for rid in ready:
                ranks[rid] = max(
                    (ranks[e["source"]] + 1 for e in graph["edges"] if e["target"] == rid),
                    default=0,
                )
                pending.remove(rid)
        positions = {}
        for rank in set(ranks.values()):
            group = [node["id"] for node in graph["nodes"] if ranks[node["id"]] == rank]
            for index, rid in enumerate(group):
                positions[rid] = ((index - (len(group) - 1) / 2) * 1.25, -rank * 1.2)
        layouts.append((positions, max(ranks.values(), default=0)))
    depth = max(rank for _, rank in layouts)
    figure, axes = plt.subplots(1, 3, figsize=(17, max(7, depth * 1.25 + 2)), facecolor="#f7f9fc")
    for axis, row, (positions, _) in zip(axes, rows, layouts):
        graph = row["topology"]
        axis.set_title(
            f"{row['harness']}\n{row['llm_calls']} model calls / {row['tool_calls']} tools / tool peak {row['tool_parallel_peak']}",
            fontsize=12,
            pad=18,
        )
        for edge in graph["edges"]:
            start, end = positions[edge["source"]], positions[edge["target"]]
            axis.add_patch(
                FancyArrowPatch(
                    (start[0], start[1] - 0.31),
                    (end[0], end[1] + 0.31),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color="#7a8799",
                    linewidth=1.2,
                )
            )
        for index, node in enumerate(graph["nodes"], 1):
            x, y = positions[node["id"]]
            failed = node["status"] != "ok"
            axis.add_patch(
                FancyBboxPatch(
                    (x - 0.57, y - 0.3),
                    1.14,
                    0.6,
                    boxstyle="round,pad=0.03",
                    facecolor="#ffe4e4" if failed else "#e7efff",
                    edgecolor="#c63d3d" if failed else "#3864aa",
                )
            )
            name = node["name"].replace("_", "\n", 1)
            inputs = node.get("input")
            step = inputs.get("step_id", "") if isinstance(inputs, dict) else ""
            detail = f"\n{step}" if step else ""
            axis.text(
                x,
                y,
                f"{index}. {name}{detail}\n{'FAILED / ' if failed else ''}{node['duration_s']:.2f}s",
                ha="center",
                va="center",
                fontsize=8,
            )
        axis.set_xlim(-2, 2)
        axis.set_ylim(-depth * 1.2 - 0.7, 0.8)
        axis.axis("off")
    figure.suptitle(
        "DataSciBench human_15 | DeepSeek V4.1 Flash | local Docker\nObserved tool calls; model/control nodes projected out; arrows show causal prerequisites",
        fontsize=14,
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output / "comparison.svg", bbox_inches="tight")
    figure.savefig(output / "comparison.png", dpi=140, bbox_inches="tight")
    plt.close(figure)


async def compare(output, image):
    from agents.tracing import set_trace_processors
    from openai import AsyncOpenAI

    from experiments.exp1.capture.capture_agent_traces import run

    output.mkdir(parents=True, exist_ok=False)
    key = os.getenv("DEEPSEEK_API_KEY") or dotenv_values(ROOT / ".env").get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("Configure DEEPSEEK_API_KEY outside source control")
    async with AsyncOpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=30) as client:
        models = (await client.models.list()).model_dump(mode="json")
    write_json(output / "models.json", models)
    if not any(model.get("id") == "deepseek-flash" and "V4.1" in model.get("name", "") for model in models["data"]):
        raise ValueError("The provider did not identify deepseek-flash as DeepSeek V4.1")
    task = next(
        task
        for task in json.loads((ROOT / "benchmarks/DataSciBench/tasks/manifest.json").read_text())["tasks"]
        if task["task_id"] == "human_15"
    )
    write_json(output / "task.json", task)
    state_root = output / "runtime-state"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = {
        "max_iterations": 3,
        "max_parallel": 3,
        "session_timeout_s": 900,
        "model": {
            "name": "deepseek-flash",
            "seed": 0,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 16000,
            "thinking": "disabled",
            "retries": 2,
        },
        "tool": {
            "provider": "local",
            "family": "faas",
            "target_id": "datasci",
            "endpoint": f"http://127.0.0.1:{port}",
            "timeout": 240,
            "execution_scope": "local Docker only; no cloud deployment",
        },
    }
    write_json(output / "config.json", {**config, "sandbox_image": image})
    env = {
        **os.environ,
        "WABBER_SANDBOX_IMAGE": image,
        "WABBER_DATA_ROOT": str(ROOT / "benchmarks/DataSciBench/data"),
        "ATSUITE_SESSION_ROOT": str(state_root),
        "WABBER_CODE_TIMEOUT": "120",
        "PROVIDER": "",
        "ATSUITE_BRANCH_PROVIDER": "",
        "WABBER_EXECUTION_BACKEND": "docker",
    }
    env.pop("DEEPSEEK_API_KEY", None)
    # This validation uses only the Wabber local destination, with no OpenAI export credentials.
    set_trace_processors([])
    with (output / "runtime.log").open("w") as log:
        service = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.exp1.capture.local_tool_service",
                "--family",
                "faas",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                if service.poll() is not None:
                    raise RuntimeError("Local Tool service exited; inspect runtime.log")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Local Tool service did not start")
                    await asyncio.sleep(0.2)
            rows = []
            for harness in ("langgraph", "langchain", "openai_agents"):
                print(f"Running {harness}: human_15", flush=True)
                directory = output / harness
                async with asyncio.timeout(900):
                    if harness == "langgraph":
                        prior_key = os.environ.get("DEEPSEEK_API_KEY")
                        os.environ["DEEPSEEK_API_KEY"] = key
                        try:
                            await run("human_15", config, directory)
                        finally:
                            if prior_key is None:
                                os.environ.pop("DEEPSEEK_API_KEY", None)
                            else:
                                os.environ["DEEPSEEK_API_KEY"] = prior_key
                    else:
                        await native_run(harness, task, config, directory, key)
                rows.append(validate_run(harness, directory, task, state_root))
                write_json(output / "comparison.json", rows)
                print(
                    f"Validated {harness}: {rows[-1]['tool_calls']} tools, peak {rows[-1]['tool_parallel_peak']}",
                    flush=True,
                )
            render(rows, output)
        finally:
            service.terminate()
            try:
                service.wait(timeout=15)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image",
        required=True,
        help="Pinned sha256 Docker image containing the DataSci worker",
    )
    args = parser.parse_args()
    if not args.image.startswith("sha256:"):
        parser.error("--image must be a pinned sha256 image ID")
    asyncio.run(compare(args.output.resolve(), args.image))


if __name__ == "__main__":
    main()
