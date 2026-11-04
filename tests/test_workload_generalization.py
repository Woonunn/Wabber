"""Domain-independent selection contracts; synthetic profiles are not cloud evidence."""

import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

from atsuite.routing.generalization import evaluate
from tests.stage6_fixtures import tariff
from tests.test_provider_model import observation, profile
from tests.workload_fixtures import describe


def add_profile(model, name, descriptor, *, available=100, state=1000, seconds=1):
    p = profile(state=state)
    p["request_id"] = name
    p["source"] = "synthetic"
    p["metrics"]["tool_elapsed_s"] = seconds
    model.update(p, available, descriptor=descriptor)


class EvaluationSamplingTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for i in range(40):
            row = observation(i)
            row.update(task_id="history", available_ns=5000)
            rows.append(row)
        held = observation(100)
        held["task_id"] = "held"
        rows.append(held)
        descriptors = {
            "history": describe(prompt="Retrieve documents", tools=["python"]),
            "held": describe(prompt="Transcode video", tools=["python"]),
        }
        labels = {"history": "retrieval", "held": "media"}
        return rows, descriptors, labels

    def evaluate(self, rows, descriptors, labels, **kwargs):
        return evaluate(
            rows,
            descriptors,
            {"aws_lambda": tariff()},
            held_family="media",
            family_labels=labels,
            history_limit=5,
            history_sampling="reservoir",
            sampling_seed=7,
            **kwargs,
        )

    def test_bounded_reservoir_is_causal_reproducible_and_not_first_arrivals(self):
        rows, descriptors, labels = self.fixture()
        report = self.evaluate(rows, descriptors, labels)
        prediction = report["predictions"][0]
        expected = sorted(
            range(40),
            key=lambda i: hashlib.sha256(
                json.dumps([7, "aws_lambda", str(i)]).encode()
            ).digest(),
        )[:5]
        self.assertEqual(
            {rid for _, rid in prediction["training_request_ids"]},
            set(map(str, expected)),
        )
        self.assertNotEqual(set(expected), set(range(5)))
        self.assertEqual(prediction["training_count"], 5)
        self.assertEqual(prediction["selected"], "aws_lambda")
        self.assertEqual(report["held_tasks"], ["held"])
        self.assertEqual(report["family_label_basis"], "independent")
        shuffled = copy.deepcopy(rows)
        random.Random(9).shuffle(shuffled)
        self.assertEqual(
            report["predictions"],
            self.evaluate(shuffled, descriptors, labels)["predictions"],
        )
        future = observation(200)
        future.update(task_id="history", available_ns=100000)
        future["profile"]["metrics"]["state_bytes"] = 10**12
        self.assertEqual(
            report["predictions"],
            self.evaluate(rows + [future], descriptors, labels)["predictions"],
        )
        embedding = self.evaluate(rows, descriptors, labels, descriptor_mode="embedding")
        self.assertEqual(
            prediction["training_request_ids"],
            embedding["predictions"][0]["training_request_ids"],
        )

    def test_independent_labels_must_cover_the_evaluation_data(self):
        rows, descriptors, labels = self.fixture()
        with self.assertRaisesRegex(ValueError, "family labels"):
            self.evaluate(rows, descriptors, {"held": "media"})
        shared = describe(prompt="Filter documents", tools=["python"])
        report = self.evaluate(rows, dict.fromkeys(descriptors, shared), labels)
        self.assertFalse(report["predictions"][0]["unknown_request"])
        self.assertEqual(report["held_tasks"], ["held"])

    def test_generic_cli_does_not_load_the_datasci_panel(self):
        from unittest.mock import patch

        from experiments.exp1.runner import stage7

        rows, descriptors, labels = self.fixture()
        labels["held"] = "media/video"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage7.write(root / "rows.json", rows)
            stage7.write(
                root / "config.json",
                {
                    "descriptors": descriptors,
                    "family_labels": labels,
                    "tariffs": {"aws_lambda": tariff()},
                    "history_limit": 5,
                },
            )
            with (
                patch.object(
                    stage7,
                    "prompt_panel",
                    side_effect=AssertionError("benchmark dependency"),
                ),
                patch(
                    "sys.argv",
                    [
                        "stage7",
                        "holdout",
                        "--rows",
                        str(root / "rows.json"),
                        "--config",
                        str(root / "config.json"),
                        "--output",
                        str(root / "out"),
                    ],
                ),
            ):
                stage7.main()
            report = stage7.load(root / "out/report.json")
            self.assertIsNone(report["panel"])
            self.assertEqual(len(report["reports"]), 4)
            for fold in report["reports"]:
                self.assertEqual(fold["family_label_basis"], "independent")
                self.assertEqual(fold["history_sampling"], "reservoir")
                self.assertTrue((root / "out" / fold["prediction_file"]).is_file())

    def test_observation_revision_does_not_count_as_an_extra_draw(self):
        rows, descriptors, labels = self.fixture()
        report = self.evaluate(rows, descriptors, labels)
        selected = report["predictions"][0]["training_request_ids"][0][1]
        revision = copy.deepcopy(next(r for r in rows if r["id"] == selected))
        revision["available_ns"] = 6000
        revision["measurements"]["latency_s"] = 3
        updated = self.evaluate(rows + [revision], descriptors, labels)
        self.assertEqual(
            report["predictions"][0]["training_request_ids"],
            updated["predictions"][0]["training_request_ids"],
        )
        conflicting = copy.deepcopy(revision)
        conflicting["available_ns"] = 7000
        conflicting["profile"]["metrics"]["state_bytes"] *= 2
        with self.assertRaisesRegex(ValueError, "Revision changes"):
            self.evaluate(rows + [revision, conflicting], descriptors, labels)


if __name__ == "__main__":
    unittest.main()
