"""Prepare E3-only native cloud targets using the existing ATSuite builder/deployer."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def prepare(travel: Path, review: Path, scientific: Path, capture: Path, benchmark: Path | None = None) -> dict:
    template = ROOT / "benchmarks/E3Tools"
    benchmark = (benchmark or template).resolve()
    if benchmark != template.resolve():
        shutil.copytree(template / "nodes", benchmark / "nodes", ignore=shutil.ignore_patterns("__pycache__"))
    payload = benchmark / "data/payload"
    payload.mkdir(parents=True, exist_ok=False)
    for workload, nodes in {
        "TravelPlanner": (
            "flights",
            "accommodations",
            "restaurants",
            "attractions",
            "cities",
            "google_distance_matrix",
            "notebook",
        ),
        "ScientificComputation": ("tensor_workspace",),
        "ClaudeCodeReview": ("repo_git", "repo_files"),
    }.items():
        for node in nodes:
            source = ROOT / "benchmarks" / workload / "nodes" / node
            shutil.copytree(
                source, payload / "benchmarks" / workload / "nodes" / node, ignore=shutil.ignore_patterns("__pycache__")
            )
    (payload / "travel").mkdir()
    travel_manifest = json.loads((travel / "TravelPlanner.json").read_text())
    database = json.loads(Path(__file__).with_name("travel_source.lock.json").read_text())["database"]
    for name, expected in database.items():
        source = travel / name
        if not source.exists():
            source = next(Path(p) for p in travel_manifest["assets"] if Path(p).name == name)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("Travel database changed: " + name)
        shutil.copyfile(source, payload / "travel" / name)
    (payload / "scientific").mkdir()
    scientific_manifest = json.loads(scientific.read_text())
    for task in scientific_manifest["tasks"]:
        if task.get("report_values") is False:
            name = task["task_id"] + (".npz" if "input_npz" in task["verification"] else ".json")
            source = next(Path(p) for p in scientific_manifest["assets"] if Path(p).name == name)
            shutil.copyfile(source, payload / "scientific" / name)
    (payload / "review").mkdir()
    shutil.copyfile(review / "projects.json", payload / "review/projects.json")
    for project in json.loads((review / "projects.json").read_text()):
        shutil.copytree(review / project, payload / "review" / project, ignore=shutil.ignore_patterns("__pycache__"))
    tools = []
    for manifest in (travel / "TravelPlanner.json", review / "ClaudeCodeReview.json", scientific):
        tools.extend(d["name"] for task in json.loads(manifest.read_text())["tasks"] for d in task["tool_definitions"])
    tools = sorted(set(tools) | {"repo_git_select_repository"})
    from atsuite.tracing.storage import read_text
    from benchmarks.DataSciBench.langgraph_agent.trace_recorder import export_atg

    (benchmark / "trace").mkdir(exist_ok=True)
    export_atg(
        [json.loads(line) for line in read_text(capture / "events.jsonl").splitlines()],
        benchmark / "trace/pilot.atg.json",
    )
    base = {
        "trace_file": "./trace/pilot.atg.json",
        "nodes": [{"name": "Tools", "dir": "./nodes/Tools", "trace_names": [{"name": n, "tool": n} for n in tools]}],
        "pipeline": {},
    }
    paths = {}
    for provider in ("aws_lambda", "aws_agentcore", "ali_fc", "ali_agentrun"):
        name = "E3" + "".join(x.title() for x in provider.split("_"))
        resources = {
            "cpu": 2,
            "memory": 3008 if provider == "aws_lambda" else 3072,
            "disk": 10240 if provider.startswith("ali_") else 2048,
            "timeout": 240,
        }
        build = {"python_version": "3.12.12"}
        if provider.startswith("ali_"):
            build["base_image"] = (
                "python:3.12.12-slim@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c"
            )
        value = {
            **base,
            "pipeline": {
                "faas": {
                    "build": build,
                    "units": [{"name": name, "node": "Tools", "trace_names": tools, "deploy": resources}],
                },
                "mcp_serverless": {
                    "build": build,
                    "servers": [{"name": name, "nodes": ["Tools"], "deploy": resources}],
                },
            },
        }
        path = benchmark / "config" / f"{provider}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")
        paths[provider] = str(path)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--travel", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--scientific", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument(
        "--benchmark", type=Path, help="Separate prepared benchmark directory; preserves the earlier Pilot payload"
    )
    a = p.parse_args()
    print(json.dumps(prepare(a.travel, a.review, a.scientific, a.capture, a.benchmark)))


if __name__ == "__main__":
    main()
