"""Check paired cost accumulation and rendering with temporary synthetic fixtures only."""

import csv
from pathlib import Path
import tempfile
import unittest

from experiments.exp1.analysis.plot_cumulative_cost import COLUMNS, SERIES, load_costs, plot


class CumulativeCostPlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "costs.csv"

    def write_rows(self, rows, header=COLUMNS):
        with self.source.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def test_accumulates_each_strategy_without_sorting_by_cost(self):
        self.write_rows([[1, 0, 0.2, 0.3, 0.4, 0.5], [2, 0.1, 0, 0.3, 0.2, 0.6]])
        requests, costs = load_costs(self.source)
        self.assertEqual(requests, [0, 1, 2])
        for key, expected in zip(SERIES, [[0, 0, 0.1], [0, 0.2, 0.2], [0, 0.3, 0.6], [0, 0.4, 0.6], [0, 0.5, 1.1]]):
            for actual, value in zip(costs[key], expected):
                self.assertAlmostEqual(actual, value)

    def test_rejects_missing_negative_and_nonfinite_costs(self):
        for value in ["", "null", "NaN", "inf", "-inf", "-0.001"]:
            with self.subTest(value=value):
                self.write_rows([[1, 0, 0, value, 0, 0]])
                with self.assertRaises(ValueError):
                    load_costs(self.source)
        self.write_rows([[1, 1e308, 0, 0, 0, 0], [2, 1e308, 0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            load_costs(self.source)

    def test_rejects_empty_incomplete_or_unpaired_input(self):
        for rows in [[], [[1, 0, 0, 0, 0]], [[2, 0, 0, 0, 0, 0]],
                     [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
                     [[1, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0]]]:
            with self.subTest(rows=rows):
                self.write_rows(rows)
                with self.assertRaises(ValueError):
                    load_costs(self.source)
        self.write_rows([[1, 0, 0, 0, 0, 0]], header=COLUMNS[:-1])
        with self.assertRaises(ValueError):
            load_costs(self.source)

    def test_renders_all_formats_at_zero_and_small_usd_scales_and_preserves_outputs(self):
        for cost in [0, 1e-7]:
            with self.subTest(cost=cost):
                self.write_rows([[1, cost, cost, cost, cost, cost]])
                output = self.root / str(cost) / "cumulative_cost"
                paths = plot(self.source, output)
                self.assertEqual({path.suffix for path in paths}, {".pdf", ".svg", ".png"})
                for path in paths:
                    self.assertGreater(path.stat().st_size, 500)
                svg = output.with_suffix(".svg").read_text()
                for label, *_ in SERIES.values():
                    self.assertIn(label, svg)
                before = [path.read_bytes() for path in paths]
                with self.assertRaises(FileExistsError):
                    plot(self.source, output)
                self.assertEqual(before, [path.read_bytes() for path in paths])


if __name__ == "__main__":
    unittest.main()
