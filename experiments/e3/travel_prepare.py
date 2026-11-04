"""Select the first official validation query per difficulty before provider profiling."""

import argparse
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

import requests

from experiments.e3.prepare import definitions
from experiments.e3.travel_eval import REVISION


def expand(prepared: Path, output: Path, indices: list[int]) -> dict:
    """Select declared rows from the already pinned official snapshot without downloading again."""
    snapshot = prepared / "source-rows.json"
    lock = json.loads(Path(__file__).with_name("travel_source.lock.json").read_text())
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != lock["source_rows_sha256"]:
        raise ValueError("Official validation snapshot changed")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Declare unique official row indices")
    original = json.loads((prepared / "TravelPlanner.json").read_text())
    entries = {e["row_idx"]: e["row"] for e in json.loads(snapshot.read_text())["rows"]}
    if not set(indices) <= entries.keys():
        raise ValueError("Unknown official row index")
    output.mkdir(parents=True, exist_ok=False)
    tasks, queries = [], []
    for index in indices:
        row = entries[index]
        query = output / f"query-{index}.json"
        query.write_text(json.dumps(row, ensure_ascii=False) + "\n")
        queries.append(str(query.resolve()))
        base = original["tasks"][0]
        command = list(base["verification"]["command"])
        command[command.index("--query") + 1] = str(query.resolve())
        tasks.append(
            {
                **base,
                "task_id": f"validation-{index}",
                "prompt": row["query"],
                "prompt_sha256": hashlib.sha256(row["query"].encode()).hexdigest(),
                "source": {**base["source"], "row": index, "level": row["level"]},
                "verification": {"command": command},
            }
        )
    manifest = {"tasks": tasks, "assets": sorted(set(original["assets"] + queries + [str(snapshot.resolve())]))}
    (output / "TravelPlanner.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {"tasks": len(tasks), "indices": indices}


def prepare(upstream: Path, output: Path, evaluator_python: Path) -> dict:
    root = upstream.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = root / "database"
    copies = {
        "flights/clean_Flights_2022.csv": "flights.csv",
        "accommodations/clean_accommodations_2022.csv": "accommodations.csv",
        "restaurants/clean_restaurant_2022.csv": "clean_restaurant_2022.csv",
        "attractions/attractions.csv": "attractions.csv",
        "googleDistanceMatrix/distance.csv": "distance.csv",
        "background/citySet_with_states.txt": "citySet_with_states.txt",
    }
    for source, target in copies.items():
        shutil.copyfile(data / source, output / target)
    response = requests.get(
        "https://datasets-server.huggingface.co/rows",
        params={
            "dataset": "osunlp/TravelPlanner",
            "config": "validation",
            "split": "validation",
            "offset": 0,
            "length": 100,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    for offset in range(100, payload["num_rows_total"], 100):
        more = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": "osunlp/TravelPlanner",
                "config": "validation",
                "split": "validation",
                "offset": offset,
                "length": 100,
            },
            timeout=60,
        )
        more.raise_for_status()
        payload["rows"].extend(more.json()["rows"])
    (output / "source-rows.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    lock = json.loads(Path(__file__).with_name("travel_source.lock.json").read_text())
    if hashlib.sha256((output / "source-rows.json").read_bytes()).hexdigest() != lock["source_rows_sha256"]:
        raise ValueError("Official validation snapshot changed; retain the sealed dataset version")
    for name, expected in lock["database"].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Official offline database checksum mismatch: " + name)
    selected = {}
    for entry in payload["rows"]:
        selected.setdefault(entry["row"]["level"], entry)
    if set(selected) != {"easy", "medium", "hard"}:
        raise ValueError("Official validation rows did not cover every difficulty")
    previous = Path.cwd()
    os.chdir(output)
    try:
        tools = definitions(
            [
                f"benchmarks.TravelPlanner.nodes.{n}.implementation"
                for n in (
                    "flights",
                    "accommodations",
                    "restaurants",
                    "attractions",
                    "cities",
                    "google_distance_matrix",
                    "notebook",
                )
            ]
        )
    finally:
        os.chdir(previous)
    tools = [
        d
        for d in tools
        if d["name"]
        in {
            "flights_run_for_annotation",
            "accommodations_run_for_annotation",
            "restaurants_run_for_annotation",
            "attractions_run_for_annotation",
            "cities_run",
            "googledistancematrix_run",
            "notebook_write",
            "notebook_read",
            "notebook_list",
        }
    ]
    prompt_source = ast.parse((root / "agents/prompts.py").read_text())
    planner_instruction = next(
        ast.literal_eval(node.value)
        for node in prompt_source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PLANNER_INSTRUCTION" for t in node.targets)
    )
    planner_instruction = planner_instruction.split("Given information:")[0]
    tasks = []
    for level, entry in selected.items():
        row = entry["row"]
        query = output / f"query-{level}.json"
        query.write_text(json.dumps(row, ensure_ascii=False) + "\n")
        tasks.append(
            {
                "task_id": f"validation-{entry['row_idx']}",
                "prompt": row["query"],
                "prompt_sha256": hashlib.sha256(row["query"].encode()).hexdigest(),
                "readiness": {"inputs": {"files": []}},
                "tool_definitions": tools,
                "planner_instruction": planner_instruction,
                "source": {
                    "dataset": "osunlp/TravelPlanner",
                    "split": "validation",
                    "row": entry["row_idx"],
                    "level": level,
                    "upstream_revision": REVISION,
                },
                "verification": {
                    "command": [
                        str(evaluator_python.absolute()),
                        "-m",
                        "experiments.e3.travel_eval",
                        "--upstream",
                        str(root),
                        "--query",
                        str(query),
                    ]
                },
            }
        )
    (output / "TravelPlanner.json").write_text(
        json.dumps(
            {
                "tasks": tasks,
                "assets": [str(p.resolve()) for p in output.iterdir() if p.is_file()]
                + [str((data / name).resolve()) for name in copies]
                + [str(p.resolve()) for d in ("evaluation", "tools", "utils") for p in (root / d).rglob("*.py")],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    lock = {str((output / p).resolve()): hashlib.sha256((output / p).read_bytes()).hexdigest() for p in copies.values()}
    (output / "database.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return {"tasks": len(tasks), "manifest": str(output / "TravelPlanner.json"), "database_sha256": lock}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-python", type=Path)
    args = parser.parse_args()
    if args.prepared:
        if not args.indices or args.upstream or args.evaluator_python:
            parser.error("--prepared requires --indices and reuses the original evaluator")
        result = expand(args.prepared, args.output, args.indices)
    else:
        if not args.upstream or not args.evaluator_python or args.indices:
            parser.error("Initial preparation requires --upstream and --evaluator-python")
        result = prepare(args.upstream, args.output, args.evaluator_python)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
