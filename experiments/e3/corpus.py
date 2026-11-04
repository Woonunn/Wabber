"""Capture a predeclared task list with bounded attempts, retaining every failure."""

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from experiments.exp1.capture.capture_agent_traces import run


async def capture(
    workload: str, manifest: Path, tasks: list[str], config: dict, output: Path, concurrency: int = 2,
    *, before_attempt: Callable[[int], Awaitable[None]] | None = None,
) -> list:
    """Select the first independently valid live Capture, never feeding validation back to the model."""
    declared = {t["task_id"] for t in json.loads(manifest.read_text())["tasks"]}
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= declared or concurrency < 1:
        raise ValueError("Unique declared tasks and positive concurrency required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    slots = asyncio.Semaphore(concurrency)
    receipts = []

    async def one(task):
        async with slots:
            receipt = {
                "workload": workload,
                "task_id": task,
                "manifest": str(manifest.resolve()),
                "attempts": [],
                "capture": None,
            }
            for attempt in range(3):
                directory = output / task / f"attempt-{attempt}"
                current = {**config, "model": {**config["model"], "seed": config["model"]["seed"] + attempt}}
                row = {"directory": str(directory.resolve()), "success": False}
                try:
                    if before_attempt is not None:
                        await before_attempt(attempt)
                    await run(task, current, directory, manifest=manifest, workload=workload)
                    row["success"] = True
                    receipt["capture"] = str(directory.resolve())
                except Exception as exc:
                    row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                receipt["attempts"].append(row)
                if row["success"]:
                    break
            receipts.append(receipt)
            (output / "receipts.json").write_text(
                json.dumps(sorted(receipts, key=lambda r: r["task_id"]), indent=2) + "\n"
            )
            print(
                json.dumps({"task": task, "success": bool(receipt["capture"]), "attempts": len(receipt["attempts"])}),
                flush=True,
            )

    await asyncio.gather(*(one(t) for t in tasks))
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(".env")
    rows = asyncio.run(
        capture(
            args.workload, args.manifest, args.tasks, json.loads(args.config.read_text()), args.output, args.concurrency
        )
    )
    if any(not r["capture"] for r in rows):
        raise SystemExit(
            "Some declared tasks exhausted their capture attempts; retain failures and do not replace tasks"
        )


if __name__ == "__main__":
    main()
