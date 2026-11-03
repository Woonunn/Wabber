"""Attribute measured usage to sealed formal requests, then apply undiscounted tariffs."""

import argparse
import csv
from datetime import datetime
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path

from experiments.exp1.data.prepare_e1b import ROOT, sha


def number(value) -> Decimal:
    """Reject missing, negative and non-finite metering rather than treating it as zero."""
    if value is None or isinstance(value, bool):
        raise ValueError("Missing or invalid metering")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("Invalid metering")
    return result


def lambda_cost(billed_ms, memory_mb, disk_mb, rates) -> dict:
    """Provider billed duration already includes billable initialization."""
    seconds = number(billed_ms) / 1000
    gb_seconds = seconds * number(memory_mb) / 1024
    disk_seconds = seconds * max(Decimal(0), number(disk_mb) - rates["included_ephemeral_mb"]) / 1024
    parts = {"compute": gb_seconds * number(rates["gb_second"]),
             "requests": number(rates["request"]),
             "ephemeral_storage": disk_seconds * number(rates["ephemeral_gb_second"])}
    return {"gb_seconds": gb_seconds, "ephemeral_gb_seconds": disk_seconds,
            "cost": sum(parts.values()), "parts": parts}


def agentcore_usage(events, selected_sessions: set[str]) -> dict:
    """Keep complete returned Session streams, including post-window usage; deduplicate once."""
    sessions = {}
    seen = {}
    for outer in events:
        event = json.loads(outer["message"], parse_float=Decimal)
        sid = event.get("attributes", {}).get("session.id")
        if sid not in selected_sessions:
            continue
        identity = outer.get("eventId") or hashlib.sha256(outer["message"].encode()).hexdigest()
        if identity in seen:
            if seen[identity] != outer["message"]:
                raise ValueError("Conflicting usage events")
            continue
        seen[identity] = outer["message"]
        metrics = event["metrics"]
        stamp = int(event["event_timestamp"])
        row = sessions.setdefault(sid, {"vcpu_hours": Decimal(0), "memory_gb_hours": Decimal(0),
            "reported_seconds": Decimal(0), "samples": 0, "first_epoch_ms": stamp, "last_epoch_ms": stamp})
        row["vcpu_hours"] += number(metrics["agent.runtime.vcpu.hours.used"])
        row["memory_gb_hours"] += number(metrics["agent.runtime.memory.gb_hours.used"])
        row["reported_seconds"] += number(event["attributes"]["time_elapsed_seconds"])
        row["samples"] += 1
        row["first_epoch_ms"] = min(row["first_epoch_ms"], stamp)
        row["last_epoch_ms"] = max(row["last_epoch_ms"], stamp)
    if set(sessions) != selected_sessions:
        raise ValueError("Missing formal AgentCore Session usage")
    return sessions


def fc_hour_bounds(hours, formal_intervals, other_intervals, cu_price, observed_minimum=None) -> dict:
    """Assign exclusive whole-hour metering; retain mixed hours as bounds, never averages."""
    lower = upper = Decimal(0)
    details = []
    for hour in hours:
        start = datetime.fromisoformat(hour["StartTime"].replace("Z", "+00:00")).timestamp() * 1000
        end = datetime.fromisoformat(hour["EndTime"].replace("Z", "+00:00")).timestamp() * 1000
        overlaps = lambda spans: any(a < end and b > start for a, b in spans)
        formal, other = overlaps(formal_intervals), overlaps(other_intervals)
        cost = number(hour["CUUsage"]) * number(cu_price)
        assignment = "mixed" if formal and other else "formal" if formal else "nonformal"
        observed = (observed_minimum or {}).get(hour["StartTime"], {})
        formal_min = number(observed.get("formal_cu", 0)) * number(cu_price)
        other_min = number(observed.get("nonformal_cu", 0)) * number(cu_price)
        if formal_min + other_min > cost:
            raise ValueError("Observed execution exceeds the provider's metered CU")
        minimum = cost if formal and not other else formal_min if formal else Decimal(0)
        maximum = cost - other_min if formal else Decimal(0)
        lower += minimum
        upper += maximum
        details.append({**hour, "assignment": assignment, "standard_cost": cost,
                        "formal_minimum": minimum, "formal_maximum": maximum,
                        "observed_execution_minimum": observed})
    return {"minimum": lower, "maximum": upper,
            "exact": lower if lower == upper else None, "hours": details}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n")


def load_gzip(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def calculate(evidence: Path, output: Path, tariff: Path) -> dict:
    """Generate request ledgers and component gaps without editing any original evidence."""
    if output.exists():
        raise FileExistsError("Preserve previous cost results")
    index = json.loads((evidence / "request-index.json").read_text())
    rates = json.loads(tariff.read_text())
    if (rates["basis"] != "standard_public_rates" or rates["account_free_tier_applied"]
            or rates["account_credits_or_discounts_applied"]):
        raise ValueError("This experiment requires undiscounted public tariffs")
    for path, digest in json.loads((evidence / "trace-sources.json").read_text()).items():
        if sha(ROOT / path) != digest:
            raise ValueError("Original trace changed")
    ledger, providers = [], {}
    for provider, data in index.items():
        run = ROOT / data["run"]
        if json.loads((run / "run.json").read_text()) != data["info"]:
            raise ValueError("Run identity changed")
        with (run / "planned_requests.csv").open() as handle:
            planned = list(csv.DictReader(handle))
        if rates["region"] != data["info"]["deployment"]["actual"]["region"]:
            raise ValueError("Tariff region differs from the deployment")
        if any(r["start_utc"][:10] != rates["usage_date"] for r in data["requests"]):
            raise ValueError("Tariff usage date differs from the experiment")
        if [(r["arrival_id"], r["task_id"]) for r in planned] != [
                (r["arrival_id"], r["task_id"]) for r in data["requests"]]:
            raise ValueError("Formal request assignments changed")
        calls = [c for r in data["requests"] for c in r["calls"]]
        if len(data["requests"]) != 764 or len(calls) != 3404:
            raise ValueError("Unexpected formal workload size")
        if len({c["call_id"] for c in calls}) != len(calls):
            raise ValueError("Duplicate Tool call ownership")
        if provider != "aws_agentcore" and (any(not c["provider_request_id"] for c in calls)
                or len({c["provider_request_id"] for c in calls}) != len(calls)):
            raise ValueError("Native request IDs cannot be allocated more than once")
        transfers = {}
        for field in ("load_bytes", "save_bytes", "branch_read_bytes", "branch_write_bytes"):
            values = [(c.get("state_sync") or {}).get(field) for c in calls]
            transfers[field] = {"known_bytes": sum(number(v) for v in values if v is not None),
                "covered_calls": sum(v is not None for v in values),
                "positive_transfer_calls": sum(v is not None and number(v) > 0 for v in values)}
        provider_record = {"requests": 764, "tool_calls": len(calls),
            "currency": "USD" if provider.startswith("aws") else "CNY",
            "cost_basis": "standard_public_rates", "total_infrastructure_cost": None,
            "missing_components": ["state_storage_capacity_and_retention", "state_storage_rpc_inventory",
                "log_ingestion_and_retention", "registry_storage_and_distribution", "network_egress", "control_host"],
            "sdk_transport_observations": transfers,
            "transport_note": "Completed transfer bytes only, not vendor RPC inventory. MCP branch transfers are local files, not object-storage traffic."}
        if provider == "aws_lambda":
            from atsuite.aws.CloudWatch import AWSCloudWatch

            ref = json.loads((run / "provider_collection_status.json").read_text())
            if sha(ROOT / ref["path"]) != ref["sha256"]:
                raise ValueError("Original Lambda metering evidence changed")
            vendor = json.loads((ROOT / ref["path"]).read_text())
            metering = {m["provider_request_id"]: m["fields"] for m in vendor["metrics"]}
            parser = AWSCloudWatch.__new__(AWSCloudWatch)
            if data["info"]["deployment"]["actual"]["architectures"] != [rates[provider]["architecture"]]:
                raise ValueError("Lambda tariff architecture mismatch")
            for row in json.loads((evidence / "lambda-recovered.json").read_text())["rows"]:
                report = [e for e in row["events"] if e["message"].startswith("REPORT")]
                if len(report) != 1:
                    raise ValueError("Ambiguous Lambda REPORT")
                metering[row["request_id"]] = parser._parse_lambda_logs(report)
            total = gb_seconds = disk_seconds = Decimal(0)
            parts = {name: Decimal(0) for name in ("compute", "requests", "ephemeral_storage")}
            for request in data["requests"]:
                cost = Decimal(0)
                for call in request["calls"]:
                    m = metering[call["provider_request_id"]]
                    if number(m.get("memory_limit_mb")) != number(data["info"]["deployment"]["actual"]["memory_mb"]):
                        raise ValueError("Lambda metered memory differs from the sealed deployment")
                    charge = lambda_cost(m.get("billed_duration_ms"), m.get("memory_limit_mb"),
                        data["info"]["deployment"]["actual"]["ephemeral_storage"]["Size"], rates[provider])
                    cost += charge["cost"]
                    gb_seconds += charge["gb_seconds"]
                    disk_seconds += charge["ephemeral_gb_seconds"]
                    for name, value in charge["parts"].items():
                        parts[name] += value
                    call["billable_usage"] = {"billed_ms": m["billed_duration_ms"], "memory_mb": m["memory_limit_mb"],
                        "gb_seconds": charge["gb_seconds"], "ephemeral_gb_seconds": charge["ephemeral_gb_seconds"]}
                    call["runtime_standard_cost"] = charge["cost"]
                total += cost
                ledger.append({"provider": provider, **request, "runtime_standard_cost": cost})
            provider_record.update(runtime_standard_cost=total, runtime_parts=parts, runtime_metering_coverage="3404/3404 calls",
                runtime_usage={"gb_seconds": gb_seconds, "ephemeral_gb_seconds": disk_seconds, "invocations": len(calls)})
        elif provider == "aws_agentcore":
            receipt = json.loads((evidence / "agentcore-usage-receipt.json").read_text())
            if not receipt["pagination_complete"] or sha(evidence / "agentcore-usage.jsonl.gz") != receipt["sha256"]:
                raise ValueError("AgentCore metering collection incomplete")
            sids = {r["session_id"] for r in data["requests"]}
            if None in sids or len(sids) != 764:
                raise ValueError("AgentCore Sessions are not exclusive")
            latest_start = max(datetime.fromisoformat(r["start_utc"].replace("Z", "+00:00")).timestamp() for r in data["requests"])
            lifetime = data["info"]["deployment"]["actual"]["max_lifetime_s"]
            if receipt["end_ms"] < (latest_start + lifetime) * 1000:
                raise ValueError("Usage query truncates possible Session tail")
            with gzip.open(evidence / "agentcore-usage.jsonl.gz", "rt") as handle:
                usage = agentcore_usage((json.loads(line) for line in handle), sids)
            total = Decimal(0)
            for request in data["requests"]:
                row = usage[request["session_id"]]
                charge = row["vcpu_hours"] * number(rates[provider]["vcpu_hour"]) + row["memory_gb_hours"] * number(rates[provider]["memory_gb_hour"])
                total += charge
                ledger.append({"provider": provider, **request, "runtime_usage": row, "runtime_standard_cost": charge})
            provider_record.update(runtime_standard_cost=total, runtime_metering_coverage="764/764 Sessions",
                runtime_usage={key: sum(v[key] for v in usage.values()) for key in ("vcpu_hours", "memory_gb_hours", "reported_seconds", "samples")},
                runtime_parts={"cpu": sum(v["vcpu_hours"] for v in usage.values()) * number(rates[provider]["vcpu_hour"]),
                               "memory": sum(v["memory_gb_hours"] for v in usage.values()) * number(rates[provider]["memory_gb_hour"])},
                tail_policy="All returned meter events belonging to selected Sessions, including after the 40-minute arrival window; no synthetic timeout multiplied into cost")
        else:
            raw = load_gzip(evidence / f"{provider}-requests-raw-v2.json.gz")
            life = load_gzip(evidence / f"{provider}-lifecycle-raw-v2.json.gz")
            if not raw["pagination_complete"] or not life["pagination_complete"]:
                raise ValueError("FC collection incomplete")
            inventory = load_gzip(evidence / f"{provider}-requests.json.gz")["response"]["body"]
            if len(raw["rows"]) != len(inventory):
                raise ValueError("FC raw pagination differs from independent SQL inventory")
            all_calls = {r["requestId"]: r for r in raw["rows"]}
            if len(all_calls) != len(raw["rows"]):
                raise ValueError("Duplicate platform request records")
            selected_calls, instance_owners = set(), {}
            for request in data["requests"]:
                for call in request["calls"]:
                    selected_calls.add(call["provider_request_id"])
                    iid = all_calls[call["provider_request_id"]]["instanceID"]
                    if provider == "ali_agentrun" and iid in instance_owners and instance_owners[iid] != request["uid"]:
                        raise ValueError("MCP instance shared across formal requests")
                    instance_owners[iid] = request["uid"]
            lifetime = {}
            for row in life["rows"]:
                item = lifetime.setdefault(row["instanceID"], {})
                for event in json.loads(row["events"]):
                    if event["type"] == "fc:Instance:Creating":
                        item["start"] = event["time"]
                    if event["type"] == "fc:Instance:DestroySuccess":
                        item["end"] = event["time"]
            formal_intervals, other_intervals = [], []
            if provider == "ali_agentrun":
                if any(set(item) != {"start", "end"} for item in lifetime.values()):
                    raise ValueError("Incomplete MCP instance lifecycle")
                for iid, item in lifetime.items():
                    (formal_intervals if iid in instance_owners else other_intervals).append((item["start"], item["end"]))
                selected_calls = {key for key, call in all_calls.items() if call["instanceID"] in instance_owners}
            else:
                for key, call in all_calls.items():
                    start = float(call.get("runtimeInitializationStartTimestamp", call["invocationStartTimestamp"]))
                    end = float(call["invocationStartTimestamp"]) + float(call["durationMs"])
                    (formal_intervals if key in selected_calls else other_intervals).append((start, end))
            oms = json.loads((evidence / "ali-FC-hour-metering.json").read_text())["body"]
            if not oms["Success"] or oms["Data"].get("Marker") or oms["Data"].get("NextToken"):
                raise ValueError("Incomplete hourly FC metering")
            name = data["info"]["deployment"]["actual"]["runtime_name"]
            hours = [row for row in oms["Data"]["OmsData"] if row["FunctionName"] == name]
            observed = {}
            if provider == "ali_fc":
                actual = data["info"]["deployment"]["actual"]
                if actual["instance_concurrency"] != 1:
                    raise ValueError("Execution minima require serial requests within each FC instance")
                for hour in hours:
                    a = datetime.fromisoformat(hour["StartTime"].replace("Z", "+00:00")).timestamp() * 1000
                    b = datetime.fromisoformat(hour["EndTime"].replace("Z", "+00:00")).timestamp() * 1000
                    minimum = {"formal_cu": Decimal(0), "nonformal_cu": Decimal(0)}
                    for key, call in all_calls.items():
                        start = float(call["invocationStartTimestamp"])
                        end = start + float(call["durationMs"])
                        gateway = float(call["invokeFunctionStartTimestamp"])
                        if a <= gateway <= start and end < b:
                            # Only fully contained blocking calls; omit init, rounding and other overheads.
                            seconds = Decimal(int(number(call["durationMs"]))) / 1000
                            unit = number(actual["cpu"]) + number(call["memoryMB"]) / 1024 * number(rates["ali_fc"]["memory_gb_second_cu"])
                            target = "formal_cu" if key in selected_calls else "nonformal_cu"
                            minimum[target] += seconds * unit
                    observed[hour["StartTime"]] = minimum
            bounds = fc_hour_bounds(hours, formal_intervals, other_intervals, rates["ali_fc"]["cu"], observed)
            protocol = len(selected_calls) - len(calls)
            for request in data["requests"]:
                owned = [all_calls[c["provider_request_id"]] for c in request["calls"]]
                if provider == "ali_agentrun":
                    owned = [c for c in all_calls.values() if instance_owners.get(c["instanceID"]) == request["uid"]]
                ledger.append({"provider": provider, **request, "platform_request_ids": [c["requestId"] for c in owned],
                    "instance_ids": sorted({c["instanceID"] for c in owned}),
                    "runtime_standard_cost": None, "reason": "Per-call or per-Session CU not exposed; mixed hourly meter cannot be split exactly"})
            provider_record.update(runtime_standard_cost=bounds["exact"], runtime_cost_bounds=bounds,
                runtime_metering_coverage="All formal platform request IDs and hourly CU available; mixed-hour allocation unresolved",
                platform_calls=len(selected_calls), protocol_calls=protocol, distinct_instances=len(instance_owners),
                metering_warning="MCP HTTP response duration is not billable instance active duration. Lifecycle duration is not substituted for billed CPU time.")
        if provider in {"aws_agentcore", "ali_agentrun"}:
            provider_record["missing_components"] = [x for x in provider_record["missing_components"] if not x.startswith("state_storage")]
            provider_record["canonical_object_storage_cost"] = "0"
            provider_record["canonical_object_storage_basis"] = "Resident MCP state and local branches; audited SDK transport counters report no canonical object-storage transfers"
        providers[provider] = provider_record
    output.mkdir(parents=True)
    write(output / "requests.json", ledger)
    with (output / "requests.csv").open("w", newline="") as handle:
        fields = ["provider", "arrival_id", "task_id", "uid", "session_id", "tool_calls", "platform_calls", "runtime_standard_cost", "currency"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ledger:
            writer.writerow({**{k: row.get(k) for k in fields[:5]},
                "tool_calls": len(row["calls"]),
                "platform_calls": len(row["platform_request_ids"]) if "platform_request_ids" in row else len(row["calls"]) if row["provider"] == "aws_lambda" else None,
                "runtime_standard_cost": row["runtime_standard_cost"],
                "currency": providers[row["provider"]]["currency"]})
    fx = json.loads((evidence / "fx.json").read_text())
    for p, row in providers.items():
        multiplier = number(fx["usd_to_cny"]) if row["currency"] == "USD" else Decimal(1)
        row["runtime_standard_cost_cny"] = row["runtime_standard_cost"] * multiplier if row["runtime_standard_cost"] is not None else None
    result = {"basis": "standard_public_rates", "free_tier_applied": False,
        "formal_requests": len(ledger), "providers": providers, "fx": fx,
        "complete_total_infrastructure_cost": False, "winner": None,
        "rate_file": str(tariff), "rate_sha256": sha(tariff),
        "interpretation": "Exact metered runtime subtotals and rigorous mixed-hour bounds only. They are not full infrastructure totals or invoice allocations."}
    write(output / "summary.json", result)
    coverage = []
    for provider, row in providers.items():
        coverage.append({"provider": provider, "component": "runtime",
            "status": "metered_exact" if row["runtime_standard_cost"] is not None else "bounded_mixed_hour",
            "standard_cost": row["runtime_standard_cost"], "currency": row["currency"]})
        coverage.extend({"provider": provider, "component": name, "status": "unknown",
                         "standard_cost": None, "currency": row["currency"]} for name in row["missing_components"])
        if "canonical_object_storage_cost" in row:
            coverage.append({"provider": provider, "component": "canonical_object_storage", "status": "not_used",
                             "standard_cost": "0", "currency": row["currency"]})
    with (output / "component-coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["provider", "component", "status", "standard_cost", "currency"])
        writer.writeheader()
        writer.writerows(coverage)
    write(output / "calculation-provenance.json", {"code": str(Path(__file__).relative_to(ROOT)),
        "code_sha256": sha(Path(__file__)), "rates_sha256": sha(tariff),
        "request_index_sha256": sha(evidence / "request-index.json"),
        "trace_sources_sha256": sha(evidence / "trace-sources.json")})
    write(output / "evidence-hashes.json", {str(p.relative_to(evidence)): sha(p) for p in evidence.iterdir() if p.is_file()})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tariffs", type=Path, default=ROOT / "experiments/exp1/configs/stage8-standard-rates.json")
    args = parser.parse_args()
    result = calculate(args.evidence.resolve(), args.output.resolve(), args.tariffs.resolve())
    print(json.dumps({p: {k: v for k, v in row.items() if k in ("runtime_standard_cost", "currency", "runtime_standard_cost_cny")} for p, row in result["providers"].items()}, default=str, indent=2))
