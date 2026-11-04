"""Run Wabber locally; account each choice with its matching measured cloud cost."""

import argparse
import ast
import copy
import csv
import hashlib
import heapq
import json
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

from atsuite.provider_model import ProviderModel
from atsuite.routing import OnlineRouter
from atsuite.tracing.storage import read_text
from atsuite.workload import WorkloadModel
from atsuite.workload.context import DATASCI_TOOLS, describe_request
from atsuite.workload.descriptor import compatibility_key, fingerprint
from experiments.exp1.runner.arrival_scheduler import ms_ns

PROVIDERS = ("ali_fc", "ali_agentrun")
ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict:
    """Read an existing plain or compressed experiment artifact."""
    return json.loads(read_text(path))


def identity(workload: str, task: dict, cloud: Path) -> dict:
    """Bind arrival capabilities to the sources preserved with the measured deployment."""
    archived = cloud / "runner-sources-verified"
    harness_files = [archived / "experiments/exp1/capture/capture_agent_traces.py",
                     archived / f"benchmarks/{workload}/langgraph_agent/graph.py"]
    if workload != "DataSciBench":
        harness_files.append(archived / "experiments/e3/agent.py")
    if workload == "DataSciBench":
        source = cloud / "images/datasci/implementation.py"
        functions = {n.name: n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)}
        definitions = [{"name": name, "python_parameters": ast.unparse(functions[name].args),
                        "returns": ast.unparse(functions[name].returns)} for name in DATASCI_TOOLS]
        files = list((cloud / "images/datasci").glob("*.py"))
    elif workload == "ScientificComputation":
        definitions = task["tool_definitions"]
        files = [cloud / "images/e3/ScientificComputation.py"]
    else:
        definitions = task["tool_definitions"]
        files = list((archived / f"benchmarks/{workload}/nodes").rglob("*.py"))
    hashes = lambda paths: {str(p.relative_to(cloud)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    tool_sources, harness_sources = hashes(files), hashes(harness_files)
    return {
        "tools": [{"name": d["name"], "definition": d,
                   "implementation": {"id": f"benchmarks/{workload}/tools", "revision": fingerprint(tool_sources)}}
                  for d in definitions],
        "harness": {"id": f"benchmarks/{workload}/langgraph_agent", "revision": fingerprint(harness_sources)},
        "evidence": {**tool_sources, **harness_sources},
    }


def load_inputs(cloud: Path, corpus: Path) -> tuple[dict, dict, dict, dict]:
    """Validate exact request/provider joins before routing; costs never enter descriptors."""
    final = read_json(cloud / "FINAL.json")
    evidence = Path(final["final_evidence"])
    plan = read_json(corpus / "capture-index.json")
    captures = {r["id"]: r for r in plan["captures"]}
    config = read_json(cloud / "config.json")
    expected = {r["arrival_id"]: r for r in plan["requests"]}
    outcomes = {}
    for provider in PROVIDERS:
        for line in (cloud / f"formal-repaired/{provider}/requests.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (row["request_id"], provider)
            if key in outcomes or row["pool"] != provider:
                raise ValueError("Duplicate or mismatched baseline outcome")
            outcomes[key] = {k: row[k] for k in ("task", "workload", "success", "latency_s", "directory")}
    seen = set()
    for line in (evidence / "priced/request-usage.jsonl").read_text().splitlines():
        row = json.loads(line, parse_float=Decimal)
        key = (row["request_id"], row["pool"])
        cost = row["priced_cost"]
        if key in seen or key not in outcomes or cost["currency"] != "CNY" or cost["total"] is None:
            raise ValueError("Missing, duplicate or unpaired native price")
        amount = Decimal(cost["total"])
        if not amount.is_finite() or amount < 0 or cost["missing"]:
            raise ValueError("Native cost must be complete, finite and nonnegative")
        request = expected[key[0]]
        if (row["task"], row["workload"]) != (request["task"], request["workload"]):
            raise ValueError("Native cost belongs to a different logical request")
        outcomes[key].update(cost=amount, cost_components=cost["lines"], cost_evidence=str(evidence))
        seen.add(key)
    if seen != {(rid, provider) for rid in expected for provider in PROVIDERS} or set(outcomes) != seen:
        raise ValueError("Every request needs exactly one price and outcome per provider")
    descriptors, identities = {}, {}
    for request in plan["requests"]:
        entry = captures[request["task"]]
        task = next(t for t in read_json(Path(entry["task_manifest"]))["tasks"] if t["task_id"] == entry["task_id"])
        agent = copy.deepcopy(entry["actual_agent_config"])
        for field in ("planner_instruction", "validation_contract"):
            if field in agent:
                task[field] = agent[field]
        pinned = identity(entry["workload"], task, cloud)
        identities[entry["workload"]] = pinned
        agent.update(prediction_identity={k: pinned[k] for k in ("tools", "harness")}, tool={"timeout": 240})
        descriptor = describe_request(task, agent, entry["workload"], planning_mode="replay")
        descriptor["control"]["calibration"] = entry["workload"] == "DataSciBench"
        if compatibility_key(descriptor) is None:
            raise ValueError("Arrival lacks an auditable tool/Harness identity")
        descriptors[request["arrival_id"]] = descriptor
    return plan, config, outcomes, {"descriptors": descriptors, "identities": identities, "evidence": str(evidence)}


def build_router(config: dict, retrieval: dict, descriptors: dict) -> OnlineRouter:
    """Enable the current contextual policy over the two measured, fixed Ali pools."""
    tariffs = read_json(ROOT / "experiments/exp1/configs/stage6-tariffs.json")
    pools = {}
    for provider in PROVIDERS:
        actual = config["native_elasticity"][provider]
        resources = {k: actual["e3"][k] for k in ("cpu", "memory_mb", "disk_mb")}
        for item in actual.values():
            if any(item[k] != resources[k] for k in resources):
                raise ValueError("This replay requires the measured common resources")
        card = copy.deepcopy(tariffs[provider])
        card.update(free_disk_gb=10, session_tail_s=0)
        pools[provider] = {"provider": provider, "environment": fingerprint(actual), "resources": resources,
                           "instances": None, "sessions_per_instance": 1 if provider == "ali_agentrun" else None,
                           "invocations_per_instance": 1 if provider == "ali_fc" else None, "billing": card}
    template = read_json(ROOT / "results/e3-two-arm/initial-final/config.json")["routing"]
    routing = {k: copy.deepcopy(template[k]) for k in ("default_pool", "cost_weight", "latency_weight", "exploration")}
    tools = sorted({tool for descriptor in descriptors.values() for tool in descriptor["tools"]})
    routing["execution_bounds"] = {p: {"timeout_s": 900, "currency": "CNY", "tools": tools,
        "evidence": "recorded native deployment and preflight; local replay only"} for p in PROVIDERS}
    router = OnlineRouter(WorkloadModel(retrieval_config=retrieval), ProviderModel(minimum_samples=3),
                          pools, dict.fromkeys(PROVIDERS, 64), enable_context_routing=True, **routing)
    return router


def replay(plan: dict, descriptors: dict, outcomes: dict, router, trace_loader, output: Path) -> list[dict]:
    """Process virtual arrivals before tied completions; reveal only selected feedback."""
    output.mkdir(parents=True, exist_ok=False)
    events, rows = [], []
    for index, request in enumerate(plan["requests"], 1):
        heapq.heappush(events, (1_000_000_000 + ms_ns(request["relative_arrival_ms"]), 0, index, request))
    while events:
        stamp, event, index, item = heapq.heappop(events)
        if event == 0:
            rid = item["arrival_id"]
            decision = router.decide(rid, None, len(plan["requests"]) / (plan["duration_ms"] / 1000),
                                     as_of_ns=stamp, arrival_ns=stamp, descriptor=descriptors[rid])
            selected = decision["selected"]
            if selected not in PROVIDERS:
                raise RuntimeError("No admissible pool; local replay stopped without inventing zero cost")
            chosen = outcomes[(rid, selected)]
            overhead = decision["prediction_overhead"]
            completed = stamp + round((chosen["latency_s"] + overhead["total_latency_s"]) * 1e9)
            row = {"request_index": index, "arrival_id": rid, "task": item["task"], "workload": item["workload"],
                   "selected_provider": selected, "cost_cny": str(chosen["cost"]),
                   "fc_cost_cny": str(outcomes[(rid, "ali_fc")]["cost"]),
                   "agentrun_cost_cny": str(outcomes[(rid, "ali_agentrun")]["cost"]),
                   "routing_model_cost_cny": str(Decimal(str(overhead["cost"]))),
                   "routing_cost_complete": overhead.get("cost_complete", True),
                   "routing_latency_s": overhead["total_latency_s"], "selected_measured_latency_s": chosen["latency_s"],
                   "virtual_arrival_ns": stamp, "virtual_completion_ns": completed,
                   "success": chosen["success"], "decision_reason": decision["reason"],
                   "explored": decision["exploration"]["explored"],
                   "history_samples": decision["workload"]["sample_count"],
                   "retrieval_basis": decision["workload"]["match_basis"],
                   "retrieval_error": decision["workload"]["retrieval"].get("error"),
                   "low_confidence": decision["workload"]["low_confidence"],
                   "missing_context": json.dumps(decision["workload"]["missing_context"]),
                   "selected_trace": chosen["directory"], "price_evidence": chosen.get("cost_evidence", "test")}
            rows.append(row)
            compact = {"request_index": index, "request_id": rid, "decision_ns": stamp, "selected": selected,
                       "reason": decision["reason"], "exploration": decision["exploration"],
                       "retrieval": decision["workload"]["retrieval"],
                       "history_ids": [p["request_id"] for p in decision["workload"]["joint_profiles"]],
                       "candidates": [{k: c[k] for k in ("key", "flags", "prediction_flags", "objective_cost", "objective_latency")}
                                      for c in decision["candidates"]]}
            with (output / "decisions.jsonl").open("a") as stream:
                stream.write(json.dumps(compact) + "\n")
            heapq.heappush(events, (completed, 1, index, row))
            if index % 10 == 0:
                print(json.dumps({"routed": index, "selected": dict(Counter(r["selected_provider"] for r in rows)),
                                  "known_feedback": len(router.completed)}), flush=True)
        else:
            row = item
            selected = row["selected_provider"]
            chosen = outcomes[(row["arrival_id"], selected)]
            trace = copy.deepcopy(trace_loader(chosen))
            trace.update(request_id=row["arrival_id"], arrival_ns=row["virtual_arrival_ns"],
                         completed_ns=stamp, request_context=descriptors[row["arrival_id"]])
            starts = [s["start_ns"] for s in trace["spans"] if s.get("start_ns") is not None]
            shift = row["virtual_arrival_ns"] + round(row["routing_latency_s"] * 1e9) - min(starts)
            for span in trace["spans"]:
                for field in ("start_ns", "end_ns"):
                    if span.get(field) is not None:
                        span[field] += shift
            observation = router.complete(row["arrival_id"], trace, available_ns=stamp,
                                          actual_cost=float(chosen["cost"]), cost_basis="attributable_execution_cost")
            observation = copy.deepcopy(observation)
            components = {}
            for line in chosen.get("cost_components", []):
                components[line["category"]] = components.get(line["category"], 0) + float(line["amount"])
            observation["cost"] = {"basis": "estimated", "currency": "CNY", "scope": "matched_native_usage",
                                   "complete": True, "components": components or {"request": float(chosen["cost"])},
                                   "evidence": [chosen.get("cost_evidence", "test")]}
            observation["available_ns"] = stamp + 1
            router.provider.update(observation)
            with (output / "completions.jsonl").open("a") as stream:
                stream.write(json.dumps({"arrival_id": row["arrival_id"], "provider": selected,
                    "available_ns": stamp + 1, "cost_cny": row["cost_cny"], "success": row["success"]}) + "\n")
    totals = [Decimal(0)] * 4
    for row in rows:
        for j, field in enumerate(("cost_cny", "fc_cost_cny", "agentrun_cost_cny", "routing_model_cost_cny")):
            totals[j] += Decimal(row[field])
        row.update(cumulative_cost_cny=str(totals[0]), cumulative_fc_cost_cny=str(totals[1]),
                   cumulative_agentrun_cost_cny=str(totals[2]), cumulative_routing_model_cost_cny=str(totals[3]),
                   cumulative_with_routing_cny=str(totals[0] + totals[3]))
    with (output / "requests.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    """Run the configured retrieval services but make no cloud Tool invocations."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", type=Path, default=ROOT / "results/e3-independent-ali")
    parser.add_argument("--corpus", type=Path, default=ROOT / "results/e3-independent")
    parser.add_argument("--retrieval", type=Path, default=ROOT / "experiments/exp1/configs/retrieval-dashscope.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan, config, outcomes, context = load_inputs(args.cloud, args.corpus)
    retrieval = read_json(args.retrieval)
    router = build_router(config, retrieval, context["descriptors"])
    assert not router.workload.samples and not router.provider.samples
    started = time.perf_counter()
    rows = replay(plan, context["descriptors"], outcomes, router,
                  lambda o: read_json(Path(o["directory"]) / "trace.json"), args.output)
    assert len(rows) == len(router.completed) == 213 and not router.active
    assert all(Decimal(r["cost_cny"]) == outcomes[(r["arrival_id"], r["selected_provider"])]["cost"] for r in rows)
    summary = {"requests": len(rows), "successes_from_selected_baselines": sum(r["success"] for r in rows),
               "selected_providers": dict(Counter(r["selected_provider"] for r in rows)),
               "cumulative_tool_cost_cny": rows[-1]["cumulative_cost_cny"],
               "fixed_fc_cny": rows[-1]["cumulative_fc_cost_cny"],
               "fixed_agentrun_cny": rows[-1]["cumulative_agentrun_cost_cny"],
               "routing_model_cost_cny": rows[-1]["cumulative_routing_model_cost_cny"],
               "cost_including_routing_cny": rows[-1]["cumulative_with_routing_cny"],
               "routing_cost_complete": all(r["routing_cost_complete"] for r in rows),
               "decision_reasons": dict(Counter(r["decision_reason"] for r in rows)),
               "retrieval_modes": dict(Counter(r["retrieval_basis"] for r in rows)),
               "retrieval_errors": dict(Counter(r["retrieval_error"] for r in rows if r["retrieval_error"])),
               "low_confidence_requests": sum(r["low_confidence"] for r in rows),
               "elapsed_local_s": time.perf_counter() - started, "initial_history": 0,
               "cloud_tool_invocations": 0, "context_routing_enabled": True,
               "retrieval_config": retrieval, "pools": router.pools,
               "policy": {"cost_weight": router.cost_weight, "latency_weight": router.latency_weight,
                          "exploration": router.exploration.config},
               "source_native_evidence": context["evidence"],
               "limits": ["Local causal replay of fixed-arm outcomes; no new routed cloud execution.",
                          "Chosen fixed-arm latency schedules feedback; dynamic-load effects are not measured.",
                          "Planner model revision is not used; sample and residual calibration gates remain active.",
                          "Failed/partial source replays retain their measured cost and status."]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "contexts.json").write_text(json.dumps(context, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
