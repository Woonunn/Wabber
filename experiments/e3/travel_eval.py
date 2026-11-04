"""JSON stdin/stdout bridge to a pinned, unmodified official TravelPlanner checkout."""

import argparse
import ast
import importlib
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

REVISION = "e52c87f4ac348a3410c46dc3553c519db5ec5e23"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    args = parser.parse_args()
    root = args.upstream.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION or subprocess.check_output(
        ["git", "-C", str(root), "diff", "HEAD", "--", "evaluation", "utils", "tools"]
    ):
        raise ValueError("Official evaluator revision mismatch or modified source")
    query = json.loads(args.query.read_text())
    if isinstance(query["local_constraint"], str):
        query["local_constraint"] = ast.literal_eval(query["local_constraint"])
    answer = json.load(sys.stdin)
    plan = answer.get("plan")
    fields = {"days", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"}
    if (
        not isinstance(plan, list)
        or len(plan) != query["days"]
        or any(not isinstance(day, dict) or not fields <= day.keys() for day in plan)
    ):
        print(json.dumps({"ok": False, "basis": "official itinerary format", "checks": {}}))
        return
    os.chdir(root / "evaluation")
    sys.path[:0] = [str(root), str(root / "evaluation")]
    with redirect_stdout(sys.stderr):
        common = importlib.import_module("commonsense_constraint").evaluation(query, plan)
        hard = importlib.import_module("hard_constraint").evaluation(query, plan)
    checks = {**common, **hard}
    print(
        json.dumps(
            {
                "ok": all(bool(v[0]) for v in common.values())
                and all(v[0] is None or bool(v[0]) for v in hard.values()),
                "basis": "official TravelPlanner commonsense and hard constraints; " + revision,
                "checks": checks,
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
