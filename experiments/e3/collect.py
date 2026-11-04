"""Retain native usage and vendor billing alongside the E3 pilot, without invented costs."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv



def collect(deployments: Path, output: Path, start_s: float, end_s: float, *,
            providers: tuple[str, ...] = ("aws_lambda", "aws_agentcore", "ali_fc", "ali_agentrun")) -> dict:
    """CloudWatch/SLS are measured usage evidence; daily bills remain unallocated."""
    output.mkdir(parents=True, exist_ok=False)
    receipts = {}
    if not providers or len(set(providers)) != len(providers) or not set(providers) <= {
        "aws_lambda", "aws_agentcore", "ali_fc", "ali_agentrun"
    }:
        raise ValueError("Choose unique supported providers")
    for provider in providers:
        actual = json.loads((deployments / f"deployed-{provider}.json").read_text())["actual"]
        path = output / (provider + "-usage.json")
        try:
            if provider.startswith("aws_"):
                import boto3

                group = (
                    "/aws/lambda/" + actual["runtime_name"]
                    if provider == "aws_lambda"
                    else actual["usage_delivery"]["log_group"]
                )
                client = boto3.client("logs", region_name=actual["region"])
                events = []
                token = None
                while True:
                    page = client.filter_log_events(
                        logGroupName=group,
                        startTime=int((start_s - 60) * 1000),
                        endTime=int((end_s + 60) * 1000),
                        **({"nextToken": token} if token else {}),
                    )
                    events.extend(page.get("events", []))
                    next_token = page.get("nextToken")
                    if not next_token or next_token == token:
                        break
                    token = next_token
                payload = {
                    "basis": "provider_metered_usage",
                    "provider": provider,
                    "log_group": group,
                    "events": events,
                    "start_s": start_s,
                    "end_s": end_s,
                    "scope": "runtime log window; join by provider request/session ID before attribution",
                }
            else:
                from alibabacloud_sls20201230.models import GetLogsRequest
                from alibabacloud_tea_util.models import RuntimeOptions

                from atsuite.ali.sls import AliSLS

                sls = AliSLS(
                    project=os.environ.get("ATSUITE_ALI_SLS_PROJECT", "atsuite"),
                    location=os.environ.get("ATSUITE_ALI_SLS_LOCATION", "us-east-1"),
                )
                response = sls.client.get_logs_with_options(
                    sls.project,
                    actual["runtime_name"],
                    GetLogsRequest(
                        from_=int(start_s) - 60, to=int(end_s) + 60, query="* | SELECT * FROM log LIMIT 100000"
                    ),
                    {},
                    RuntimeOptions(),
                )
                events = response.body or []
                if len(events) >= 100000:
                    raise ValueError("SLS usage response may be truncated")
                payload = {
                    "basis": "provider_metered_usage",
                    "provider": provider,
                    "events": events,
                    "start_s": start_s,
                    "end_s": end_s,
                    "scope": "runtime log window; join by request ID before attribution",
                }
            path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
            receipts[provider] = {"path": str(path), "events": len(events)}
        except Exception as exc:
            receipts[provider] = {"error": type(exc).__name__, "message": str(exc)}
    (output / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    return receipts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deployments", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-s", type=float, required=True)
    p.add_argument("--end-s", type=float, required=True)
    a = p.parse_args()
    load_dotenv(".env")
    print(json.dumps(collect(a.deployments, a.output, a.start_s, a.end_s)))


if __name__ == "__main__":
    main()
