import json
import tempfile
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path

from experiments.e3.balance import TOLERANCE_CNY, balanced_counts, build
from experiments.e3.manifest import WORKLOADS, digest, load


class CostBalanceTests(unittest.TestCase):
    def test_integer_counts_balance_opposite_penalties(self):
        pairs = [
            {
                "task": "fc",
                "workload": "same",
                "fc_cost_cny": ".001",
                "mcp_cost_cny": ".003",
            },
            {
                "task": "mcp",
                "workload": "same",
                "fc_cost_cny": ".004",
                "mcp_cost_cny": ".001",
            },
        ]
        result = balanced_counts(pairs, {"fc": 5, "mcp": 5})
        self.assertEqual(result, {"fc": 6, "mcp": 4})
        self.assertEqual(6 * Decimal(".002"), 4 * Decimal(".003"))

    def test_invalid_measurements_and_unreachable_workload_quotas_fail(self):
        first = {
            "task": "a",
            "workload": "A",
            "fc_cost_cny": ".001",
            "mcp_cost_cny": ".002",
        }
        second = {
            "task": "b",
            "workload": "B",
            "fc_cost_cny": ".003",
            "mcp_cost_cny": ".001",
        }
        counts = {"a": 1, "b": 1}
        for pairs, original in [
            ([first, first], counts),
            ([first, second], {"a": 0, "b": 2}),
            ([first, second], {"a": True, "b": 1}),
            ([{**first, "fc_cost_cny": "NaN"}, second], counts),
            ([{**first, "fc_cost_cny": "-1"}, second], counts),
            ([first, {**second, "fc_cost_cny": "0"}], counts),
            ([first, second], counts),
        ]:
            with (
                self.subTest(pairs=pairs, original=original),
                self.assertRaises(ValueError),
            ):
                balanced_counts(pairs, original)

    def test_derived_manifest_preserves_arrivals_captures_and_workload_quotas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "original"
            source.mkdir()
            tasks, requests, pairs = [], [], []
            for workload in WORKLOADS:
                for mode in ["fc", "mcp"]:
                    task = workload + "-" + mode
                    tasks.append(
                        {
                            "id": task,
                            "task_id": task,
                            "workload": workload,
                            "capture": "/unchanged/capture/" + task,
                        }
                    )
                    pairs.append(
                        {
                            "task": task,
                            "task_id": task,
                            "workload": workload,
                            "fc_cost_cny": 0.001 if mode == "fc" else 0.003,
                            "mcp_cost_cny": 0.003 if mode == "fc" else 0.001,
                        }
                    )
                    for _ in range(3 if mode == "fc" else 1):
                        requests.append(
                            {
                                "arrival_id": str(len(requests)),
                                "relative_arrival_ms": str(len(requests) * 1000),
                                "original_timestamp_utc": "preserved",
                                "phase": 0,
                                "workload": workload,
                                "task": task,
                            }
                        )
            original = {
                "schema": 1,
                "seed": 9,
                "duration_ms": 600000,
                "tasks": tasks,
                "phases": [{"start_ms": 0, "weights": dict.fromkeys(WORKLOADS, 1)}],
                "requests": requests,
                "files": {},
                "source_arrivals_sha256": "unchanged",
            }
            path = source / "manifest.json"
            path.write_text(json.dumps(original))
            source_hash = digest(path)
            (source / "seal.json").write_text(
                json.dumps({"manifest_sha256": source_hash})
            )
            matrix = root / "costs.json"
            matrix.write_text(json.dumps({"pairs": pairs}))
            output = root / "balanced"
            report = build(source, matrix, output)
            result = load(output)
            repeated = root / "repeated"
            build(source, matrix, repeated)
            self.assertEqual(result["requests"], load(repeated)["requests"])
            self.assertEqual(digest(path), source_hash)
            self.assertEqual(result["tasks"], tasks)
            self.assertEqual(
                result["source_arrivals_sha256"], original["source_arrivals_sha256"]
            )
            self.assertEqual(
                Counter(r["workload"] for r in requests),
                Counter(r["workload"] for r in result["requests"]),
            )
            self.assertTrue(
                all(
                    n >= 1
                    for n in Counter(r["task"] for r in result["requests"]).values()
                )
            )
            self.assertEqual(
                [{k: v for k, v in r.items() if k != "task"} for r in requests],
                [
                    {k: v for k, v in r.items() if k != "task"}
                    for r in result["requests"]
                ],
            )
            self.assertLessEqual(
                Decimal(report["absolute_balance_gap_cny"]), TOLERANCE_CNY
            )
            estimates = report["estimated_costs"]
            self.assertEqual(
                estimates["fc_excess_on_mcp_cheaper_tasks_cny"],
                estimates["mcp_excess_on_fc_cheaper_tasks_cny"],
            )
            with self.assertRaises(FileExistsError):
                build(source, matrix, output)
            matrix.write_text("{}")
            with self.assertRaisesRegex(ValueError, "input or Capture changed"):
                load(output)


if __name__ == "__main__":
    unittest.main()
