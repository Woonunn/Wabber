"""Feed only completed formal requests' available native prices back into the online model."""

import asyncio
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

from experiments.e3.collect import collect
from experiments.e3.evidence import reconcile


class NativeCostFeedback:
    """Poll existing native collectors; keep missing prices unknown until delivery."""

    def __init__(self, router, config: dict, output: Path):
        self.router, self.config, self.output = router, config, output
        self.imported = {}
        self.failed = None
        self.snapshots = output / "cost-feedback"
        self.snapshots.mkdir()
        for provider, kinds in config["native_elasticity"].items():
            for kind, actual in kinds.items():
                directory = self.snapshots / "deployments" / kind
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"deployed-{provider}.json").write_text(json.dumps({"actual": actual}) + "\n")

    def refresh(self) -> None:
        """Collect a bounded completed prefix and import available complete native prices."""
        path = self.output / "requests.jsonl"
        if not path.exists():
            return
        rows = [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith("\n")]
        if not rows:
            return
        snapshot = self.snapshots / str(time.time_ns())
        snapshot.mkdir()
        run = snapshot / "run"
        run.mkdir()
        (run / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        for kind in ("datasci", "e3"):
            group = [r for r in rows if (r["workload"] == "DataSciBench") == (kind == "datasci")]
            if group:
                collect(
                    self.snapshots / "deployments" / kind,
                    snapshot / "usage" / kind,
                    min(r["started_epoch"] for r in group),
                    max(r["completed_epoch"] for r in group),
                    providers=tuple(sorted({r["pool"] for r in group})),
                )
        reconcile(run, snapshot / "usage", snapshot / "priced", self.config)
        priced = [json.loads(line) for line in (snapshot / "priced" / "request-usage.jsonl").read_text().splitlines()]
        self.ingest(priced, str(snapshot))

    def ingest(self, rows: list[dict], evidence: str) -> None:
        """Enrich existing completed observations; never import unseen counterfactual outcomes."""
        for row in rows:
            identity = row["request_id"]
            cost = row["priced_cost"]
            if self.imported.get(identity) == cost or identity not in self.router.completed or cost["total"] is None:
                continue
            if self.router.decisions[identity]["selected"] != row["pool"]:
                raise ValueError("Native feedback provider differs from the selected provider")
            with self.router.provider.lock:
                matches = [
                    copy.deepcopy(history[-1])
                    for history in self.router.provider.samples.values()
                    if history[-1]["id"] == identity and history[-1]["pool"]["provider"] == row["pool"]
                ]
            if len(matches) != 1:
                raise ValueError("Native feedback must match exactly one completed observation")
            observation = matches[0]
            components = defaultdict(float)
            for line in cost["lines"]:
                components[line["category"]] += line["amount"]
            observation["cost"] = {
                "basis": "estimated",
                "currency": cost["currency"],
                "scope": "atsuite_runtime_and_traced_state",
                "complete": True,
                "components": dict(components),
                "evidence": [evidence],
            }
            observation["available_ns"] = time.time_ns()
            self.router.provider.update(observation)
            self.imported[identity] = copy.deepcopy(cost)

    async def watch(self) -> None:
        """Import delayed usage during the formal stream without blocking its event loop."""
        while True:
            await asyncio.sleep(30)
            job = asyncio.create_task(asyncio.to_thread(self.refresh))
            try:
                await asyncio.shield(job)
            except asyncio.CancelledError:
                await job
                raise
            except Exception as exc:
                self.failed = {"error": type(exc).__name__, "message": str(exc), "observed_ns": time.time_ns()}
                with (self.snapshots / "errors.jsonl").open("a") as stream:
                    stream.write(json.dumps(self.failed) + "\n")
