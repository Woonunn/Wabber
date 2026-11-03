"""Clock annotations vary legitimately; computed data and rendering must not."""

from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("matplotlib", "joblib", "pandas")),
                     "Run in the pinned Mix image")
class ClockAnnotationTests(unittest.TestCase):
    def setUp(self):
        nodes = Path(__file__).resolve().parents[1] / "benchmarks/DataSciBench/nodes"
        imports = patch.object(sys, "path", [str(nodes / "AgentTools"), str(nodes / "DataSci"), *sys.path])
        imports.start()
        self.addCleanup(imports.stop)

    def test_timing_statistics_are_recomputed_from_actual_observations(self):
        import numpy as np
        from bcb_checks import timing_signature

        evidence = {"samples": 2, "delay": 0.1, "timestamps": [1.0, 1.12, 1.2, 1.35], "sleeps": [0.1, 0.1]}
        elapsed = np.array(evidence["timestamps"][1::2]) - np.array(evidence["timestamps"][::2])
        result = (float(np.mean(elapsed)), float(np.std(elapsed)))
        signature = timing_signature(result, evidence)
        with self.assertRaisesRegex(ValueError, "statistics differ"):
            timing_signature((result[0] + 0.1, result[1]), evidence)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            timing_signature(result, {**evidence, "sleeps": []})
        other = {**evidence, "timestamps": [2.0, 2.2, 2.4, 2.7]}
        elapsed = np.array(other["timestamps"][1::2]) - np.array(other["timestamps"][::2])
        self.assertEqual(signature, timing_signature((float(np.mean(elapsed)), float(np.std(elapsed))), other))

    def test_clock_validation_preserves_pixels_and_original_annotation(self):
        import matplotlib.pyplot as plt
        from bcb_checks import return_signature

        fig, ax = plt.subplots()
        line, = ax.plot([1, 2, 3])
        annotation = ax.annotate("2026-11-01 12:00:00", (0.5, 0.95))
        start = datetime(2026, 11, 1, 12).timestamp()
        original = return_signature(ax)
        signature = return_signature(ax, clock_window=[start, start + 0.5])
        self.assertEqual(annotation.get_text(), "2026-11-01 12:00:00")
        annotation.set_text("2026-11-01 12:01:00")
        self.assertNotEqual(original, return_signature(ax))
        self.assertEqual(signature, return_signature(ax, clock_window=[start + 60, start + 60.5]))
        with self.assertRaisesRegex(ValueError, "actual invocation time"):
            return_signature(ax, clock_window=[start, start + 0.5])
        line.set_ydata([1, 2, 4])
        self.assertNotEqual(signature, return_signature(ax, clock_window=[start + 60, start + 60.5]))
        annotation.remove()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            return_signature(ax, clock_window=[start, start + 0.5])
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
