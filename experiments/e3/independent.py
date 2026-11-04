"""Prepare and seal one independently captured request per original E3 arrival."""

import argparse
import asyncio
import copy
import hashlib
import json
import random
from collections import Counter
from decimal import Decimal
from pathlib import Path

from atsuite.tracing.storage import read_text
from experiments.e3.balance import balanced_counts
from experiments.e3.corpus import capture
from experiments.e3.manifest import WORKLOADS, digest, seal

ADDITIONS = dict(zip(WORKLOADS, (48, 47, 47, 47)))
BALANCE_TOLERANCE = Decimal("0.05")


def write(path: Path, value: object) -> None:
    """Write an auditable JSON artifact without overwriting an existing one."""
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def historical_source(directory: Path) -> dict:
    """Verify reused evidence while allowing a new runner revision for the new experiment."""
    source = json.loads((directory / "manifest.json").read_text())
    if digest(directory / "manifest.json") != json.loads((directory / "seal.json").read_text())["manifest_sha256"]:
        raise ValueError("Historical manifest changed")
    checked = set()
    for task in source["tasks"]:
        manifest = Path(task["manifest"])
        document = json.loads(manifest.read_text())
        paths = [manifest, *map(Path, task["inputs"]), *map(Path, document.get("assets", []))]
        paths.extend(p for p in Path(task["capture"]).rglob("*") if p.is_file())
        for path in paths:
            if path.resolve() in checked:
                continue
            if source["files"].get(str(path.resolve())) != digest(path):
                raise ValueError("Reused input or Capture changed: " + str(path))
            checked.add(path.resolve())
    return source


def allocate(source: dict, pairs: list[dict]) -> tuple[dict, dict]:
    """Keep 24 originals; distribute 189 additions evenly with predeclared Ali cost balance."""
    tasks = {t["id"]: t for t in source["tasks"]}
    if len(source["requests"]) != 213 or Counter(t["workload"] for t in tasks.values()) != dict.fromkeys(WORKLOADS, 6):
        raise ValueError("Expected 24 original tasks, six per workload, and 213 arrivals")
    if len(pairs) != 24 or {p["task"] for p in pairs} != set(tasks):
        raise ValueError("Planning matrix must cover the 24 original tasks exactly")
    initial = {}
    for workload, added in ADDITIONS.items():
        ids = [t for t, d in tasks.items() if d["workload"] == workload]
        total = added + len(ids)
        initial.update({t: total // len(ids) + int(i < total % len(ids)) for i, t in enumerate(ids)})
    for pair in pairs:
        task = tasks[pair["task"]]
        if pair["workload"] != task["workload"] or pair["task_id"] != task["task_id"]:
            raise ValueError("Planning matrix identity mismatch")
    counts = balanced_counts(pairs, initial)
    fc = sum(counts[p["task"]] * Decimal(str(p["fc_cost_cny"])) for p in pairs)
    mcp = sum(counts[p["task"]] * Decimal(str(p["mcp_cost_cny"])) for p in pairs)
    gap = abs(fc - mcp) / max(fc, mcp)
    if gap > BALANCE_TOLERANCE:
        raise ValueError("Planning balance exceeds the predeclared five-percent tolerance")
    return counts, {
        "basis": "historical Ali family-cost planning estimate; new request costs are unmeasured",
        "fc_cny": str(fc),
        "mcp_cny": str(mcp),
        "relative_gap": str(gap),
        "tolerance": str(BALANCE_TOLERANCE),
        "aws_balance": "unmeasured",
    }


def prepare(source_dir: Path, matrix: Path, output: Path) -> dict:
    """Materialize 189 derived requests, preserving originals and exact original arrival times."""
    from experiments.e3.independent_inputs import derive

    source = historical_source(source_dir)
    counts, estimate = allocate(source, json.loads(matrix.read_text())["pairs"])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for parent in source["tasks"]:
        document = json.loads(Path(parent["manifest"]).read_text())
        task = next(t for t in document["tasks"] if t["task_id"] == parent["task_id"])
        start = next(
            json.loads(line)
            for line in read_text(Path(parent["capture"]) / "events.jsonl").splitlines()
            if json.loads(line)["kind"] == "run_start"
        )
        agent = {k: v for k, v in start["config"].items() if k != "tool"}
        for ordinal in range(counts[parent["id"]]):
            identity = parent["id"] if ordinal == 0 else parent["id"] + f"-request-{ordinal:02d}"
            folder = output / "requests" / identity
            folder.mkdir(parents=True)
            if ordinal == 0:
                definition, assets = copy.deepcopy(task), document.get("assets", [])
                retained_capture = parent["capture"]
            else:
                definition, assets = derive(parent, document, task, ordinal, folder)
                retained_capture = None
            write(folder / "task.json", {"tasks": [definition], "assets": assets})
            write(folder / "agent.json", agent)
            entries.append(
                {
                    "id": identity,
                    "family_id": parent["id"],
                    "workload": parent["workload"],
                    "task_id": definition["task_id"],
                    "manifest": str(folder / "task.json"),
                    "inputs": parent["inputs"],
                    "agent_config": str(folder / "agent.json"),
                    "capture": retained_capture,
                    "retained": ordinal == 0,
                    "input_origin": "original" if ordinal == 0 else "derived request; not an observed user submission",
                }
            )
    random.Random(source["seed"]).shuffle(entries)
    requests = []
    for arrival, entry in zip(source["requests"], entries):
        requests.append(
            {
                **{k: v for k, v in arrival.items() if k not in {"task", "workload", "phase"}},
                "task": entry["id"],
                "workload": entry["workload"],
            }
        )
    assert Counter(e["workload"] for e in entries if not e["retained"]) == ADDITIONS
    value = {
        "schema": 1,
        "seed": source["seed"],
        "duration_ms": source["duration_ms"],
        "source_manifest": str((source_dir / "manifest.json").resolve()),
        "source_manifest_sha256": digest(source_dir / "manifest.json"),
        "planning_matrix": str(matrix.resolve()),
        "planning_matrix_sha256": digest(matrix),
        "planning_balance": estimate,
        "balance_tolerance": str(BALANCE_TOLERANCE),
        "requests": requests,
        "tasks": entries,
    }
    write(output / "plan.json", value)
    seal_preparation(output)
    return {
        "requests": len(entries),
        "retained_captures": 24,
        "pending_captures": 189,
        "additions": ADDITIONS,
        "planning_balance": estimate,
    }


def seal_preparation(directory: Path) -> None:
    """Freeze prepared inputs once and export manifests for the existing runtime packager."""
    plan = json.loads((directory / "plan.json").read_text())
    stage_scientific(directory, plan)
    paths, documents = set(), {}
    prompts = set()
    for entry in plan["tasks"]:
        paths.update(Path(p) for p in (entry["manifest"], entry["agent_config"], *entry["inputs"]))
        document = json.loads(Path(entry["manifest"]).read_text())
        paths.update(map(Path, document.get("assets", [])))
        task = document["tasks"][0]
        if task["prompt_sha256"] in prompts:
            raise ValueError("Independent requests require distinct prepared prompts")
        prompts.add(task["prompt_sha256"])
        if entry["retained"]:
            paths.update(p for p in Path(entry["capture"]).rglob("*") if p.is_file())
        if entry["workload"] in {"ScientificComputation", "TravelPlanner"}:
            group = documents.setdefault(entry["workload"], {"tasks": [], "assets": []})
            group["tasks"].append(task)
            group["assets"].extend(document.get("assets", []))
    exports = directory / "runtime-manifests"
    exports.mkdir(exist_ok=True)
    for workload, document in documents.items():
        document["assets"] = sorted(set(document["assets"]))
        path = exports / (workload + ".json")
        if path.exists():
            if json.loads(path.read_text()) != document:
                raise ValueError("Runtime input manifest changed")
        else:
            write(path, document)
        paths.add(path)
    paths.update([Path(plan["source_manifest"]), Path(plan["planning_matrix"])])
    write(
        directory / "preparation-seal.json",
        {
            "plan_sha256": digest(directory / "plan.json"),
            "files": {str(p.resolve()): digest(p) for p in sorted(paths)},
        },
    )


def stage_scientific(directory: Path, plan: dict) -> Path:
    """Expose immutable scientific uploads in the existing Tool's flat input directory."""
    destination = directory / "runtime-inputs" / "scientific"
    destination.mkdir(parents=True, exist_ok=True)
    for entry in plan["tasks"]:
        if entry["workload"] != "ScientificComputation":
            continue
        document = json.loads(Path(entry["manifest"]).read_text())
        for record in document["tasks"][0]["readiness"]["inputs"]["files"]:
            source = next(Path(p) for p in document["assets"] if Path(p).name == record["name"])
            target = destination / source.name
            if not target.exists():
                target.hardlink_to(source)
            if digest(target) != record["sha256"]:
                raise ValueError("Scientific upload name collision or changed input")
    return destination.resolve()


def approve_datasci(policy: dict, task: dict, events: list[dict]) -> None:
    """Merge exact captured programs into the existing frozen-program admission format."""
    preparation = next(e["args"] for e in events if e["kind"] == "tool" and e.get("tool_name") == "prepare_workspace")
    identity = task["task_id"]
    approved = policy.setdefault(
        identity, {"task_id": identity, "dataset_refs": preparation["dataset_refs"], "analyses": []}
    )
    if preparation["task_id"] != identity or approved["dataset_refs"] != preparation["dataset_refs"]:
        raise ValueError("Conflicting DataSci dataset versions under the same backend identity")
    for event in events:
        if event["kind"] != "tool" or event.get("tool_name") != "execute_analysis":
            continue
        args = event["args"]
        program = {k: args[k] for k in ("step_id", "outputs", "snapshot", "seed")}
        program["code_sha256"] = hashlib.sha256(args["code"].encode()).hexdigest()
        if program not in approved["analyses"]:
            approved["analyses"].append(program)


def prepared(directory: Path) -> dict:
    """Refuse modified request definitions or inputs after predeclaring the corpus."""
    lock = json.loads((directory / "preparation-seal.json").read_text())
    if digest(directory / "plan.json") != lock["plan_sha256"]:
        raise ValueError("Independent plan changed")
    if any(digest(Path(p)) != checksum for p, checksum in lock["files"].items()):
        raise ValueError("Independent request definition or input changed")
    return json.loads((directory / "plan.json").read_text())


async def capture_pending(directory: Path, configs: dict, concurrency: int = 2) -> dict:
    """Capture each new request through the existing live harness; retain failed attempts."""
    plan = prepared(directory)
    slots = asyncio.Semaphore(concurrency)
    if concurrency < 1:
        raise ValueError("Positive capture concurrency required")

    async def one(task):
        if task["retained"]:
            return
        output = directory / "captures" / task["id"]
        if output.exists():
            receipts = json.loads((output / "receipts.json").read_text())
            if len(receipts) == 1 and receipts[0]["capture"]:
                return
            raise ValueError("Incomplete prior Capture retained; inspect it before retrying: " + str(output))
        async with slots:
            agent = json.loads(Path(task["agent_config"]).read_text())
            campaign = configs[task["workload"]]
            agent["model"].update(campaign.get("model", {}))
            for key in ("max_iterations", "max_parallel", "session_timeout_s"):
                if key in campaign:
                    agent[key] = campaign[key]
            agent["tool"] = campaign["tool"]
            await capture(task["workload"], Path(task["manifest"]), [task["task_id"]], agent, output, concurrency=1)

    await asyncio.gather(*(one(t) for t in plan["tasks"]))
    result = {"retained": 24, "new_successes": 0, "new_failed_attempts": 0}
    for task in plan["tasks"]:
        if task["retained"]:
            continue
        rows = json.loads((directory / "captures" / task["id"] / "receipts.json").read_text())
        result["new_successes"] += int(bool(rows[0]["capture"]))
        result["new_failed_attempts"] += sum(not a["success"] for a in rows[0]["attempts"])
    return result


def seal_requests(directory: Path, output: Path) -> dict:
    """Seal only a complete corpus with 213 independently generated Capture identities."""
    plan = prepared(directory)
    tasks, artifacts, identities = [], [directory / "plan.json", directory / "preparation-seal.json"], set()
    local = directory / "local-capture"
    artifacts.extend(
        p for p in local.rglob("*")
        if p.is_file() and p.name != "active.lock"
        and not {"travel-inputs", "datasci-image", "scratch", "state", "__pycache__"}.intersection(p.relative_to(local).parts)
    )
    approved_datasci = {}
    for entry in plan["tasks"]:
        task = copy.deepcopy(entry)
        if not task["retained"]:
            receipt_path = directory / "captures" / task["id"] / "receipts.json"
            if not receipt_path.exists():
                raise ValueError("Independent live Capture missing: " + task["id"])
            receipt = json.loads(receipt_path.read_text())[0]
            if not receipt["capture"]:
                raise ValueError("Independent live Capture missing: " + task["id"])
            task["capture"] = receipt["capture"]
            artifacts.extend(
                [
                    receipt_path,
                    *[p for a in receipt["attempts"] for p in Path(a["directory"]).rglob("*") if p.is_file()],
                ]
            )
        events = [json.loads(line) for line in read_text(Path(task["capture"]) / "events.jsonl").splitlines()]
        start = next(e for e in events if e["kind"] == "run_start")
        live_start = start
        if start.get("source_capture"):
            live_events = [
                json.loads(line) for line in read_text(Path(start["source_capture"]) / "events.jsonl").splitlines()
            ]
            live_start = next(e for e in live_events if e["kind"] == "run_start")
        identity = live_start.get("request_id")
        if not identity or identity in identities:
            raise ValueError("Capture identity reused or missing: " + task["id"])
        identities.add(identity)
        if task.get("workload") == "DataSciBench":
            approve_datasci(approved_datasci, task, events)
        agent_path = directory / "captures" / (task["id"] + "-replay-agent.json")
        agent_path.parent.mkdir(exist_ok=True)
        actual_agent = {k: v for k, v in start["config"].items() if k != "tool"}
        if agent_path.exists():
            if json.loads(agent_path.read_text()) != actual_agent:
                raise ValueError("Previously sealed agent config changed")
        else:
            write(agent_path, actual_agent)
        task["agent_config"] = str(agent_path.resolve())
        tasks.append(task)
    policy_path = directory / "captures" / "approved_trace.json"
    policy = {"tasks": approved_datasci}
    if policy_path.exists():
        if json.loads(policy_path.read_text()) != policy:
            raise ValueError("Previously exported frozen-program policy changed")
    else:
        write(policy_path, policy)
    artifacts.append(policy_path)
    return seal(
        tasks,
        plan["requests"],
        [{"start_ms": 0, "weights": dict.fromkeys(WORKLOADS, 1)}],
        plan["seed"],
        plan["duration_ms"],
        output,
        artifacts=artifacts,
        assignments=plan["requests"],
    )


def cost_balance(plan: dict, rows: list[dict]) -> dict:
    """Compare complete paired native ATSuite prices; never rebalance captured requests."""
    expected = {r["arrival_id"]: r for r in plan["requests"]}
    indexed = {}
    for row in rows:
        key = (row["pool"], row["request_id"])
        if key in indexed or row["request_id"] not in expected:
            raise ValueError("Duplicate or unexpected priced request")
        request = expected[row["request_id"]]
        if row["task"] != request["task"] or row["workload"] != request["workload"]:
            raise ValueError("Priced task does not match the independent request assignment")
        if row["pool"] not in {"ali_fc", "ali_agentrun", "aws_lambda", "aws_agentcore"}:
            raise ValueError("Unknown priced provider")
        indexed[key] = row
    result = {}
    for name, left, right, currency in (
        ("ali", "ali_fc", "ali_agentrun", "CNY"),
        ("aws", "aws_lambda", "aws_agentcore", "USD"),
    ):
        groups = [[indexed.get((p, identity)) for identity in sorted(expected)] for p in (left, right)]
        totals = []
        for group in groups:
            if any(r is None or not r["success"] or r["priced_cost"]["total"] is None for r in group):
                totals.append(None)
                continue
            if any(r["priced_cost"]["currency"] != currency for r in group):
                raise ValueError("Pairwise balance requires matching native currencies")
            values = [Decimal(str(r["priced_cost"]["total"])) for r in group]
            if any(not v.is_finite() or v < 0 for v in values):
                raise ValueError("Invalid native request cost")
            totals.append(sum(values))
        gap = None if None in totals else abs(totals[0] - totals[1]) / max(*totals, Decimal("1e-30"))
        result[name] = {
            "currency": currency,
            "faas_total": None if totals[0] is None else str(totals[0]),
            "mcp_total": None if totals[1] is None else str(totals[1]),
            "relative_gap": None if gap is None else str(gap),
            "within_tolerance": None if gap is None else gap <= Decimal(plan["balance_tolerance"]),
        }
    return result


def main() -> None:
    """Prepare requests, capture them, seal Replay inputs, or check measured cost balance."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--matrix", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--directory", type=Path, required=True)
    capture_parser.add_argument("--configs", type=Path, required=True)
    capture_parser.add_argument("--concurrency", type=int, default=2)
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("--directory", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    balance_parser = commands.add_parser("balance")
    balance_parser.add_argument("--directory", type=Path, required=True)
    balance_parser.add_argument("--prices", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.matrix, args.output)
    elif args.command == "capture":
        from dotenv import load_dotenv

        load_dotenv(".env")
        result = asyncio.run(capture_pending(args.directory, json.loads(args.configs.read_text()), args.concurrency))
    elif args.command == "seal":
        sealed = seal_requests(args.directory, args.output)
        result = {"requests": len(sealed["requests"]), "independent_captures": len(sealed["tasks"])}
    else:
        result = cost_balance(
            prepared(args.directory), [json.loads(line) for line in args.prices.read_text().splitlines()]
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "capture" and result["new_successes"] != 189:
        raise SystemExit("Some independent requests lack valid Capture; all failed attempts are retained")


if __name__ == "__main__":
    main()
