"""Replay causality checks; synthetic test traces never count as cloud evidence."""

import unittest

from experiments.exp1.runner.stage8_empirical import replay, summarize
from tests import test_stage8
from tests.test_workload import trace


class EmpiricalReplayTests(unittest.TestCase):
    def arrivals(self, descriptor):
        return [{"arrival_id": str(i), "at_ns": at, "descriptor": descriptor,
                 "segment": 0, "first_prompt": i == 0, "first_family": i == 0}
                for i, at in enumerate([10**9, 2 * 10**9, 3 * 10**9 + 1])]

    def test_selected_only_and_completion_ties_are_not_visible(self):
        router, descriptor = test_stage8.ColdStartTests().router(epsilon=0)
        calls = []

        def outcome(arrival_id, selected):
            calls.append((arrival_id, selected))
            self.assertEqual(selected, "default")
            return trace("original", arrival=0, end=10**9), 1

        rows = replay(router, self.arrivals(descriptor), outcome)
        self.assertEqual([r["visible_feedback"] for r in rows], [0, 0, 2])
        self.assertEqual(calls, [(str(i), "default") for i in range(3)])
        self.assertEqual(len(router.provider.samples), 3)
        self.assertTrue(all(k[1] == "default" for k in router.provider.samples))
        self.assertTrue(all(d["task_id"] is None for d in router.decisions.values()))
        self.assertTrue(all(v["basis"] == "execution_cap" for v in router.feedback.values()))
        self.assertIsNone(summarize(rows)["net_savings"])

    def test_future_outcome_does_not_change_earlier_decisions(self):
        results = []
        for final_duration in (1, 500):
            router, descriptor = test_stage8.ColdStartTests().router()

            def outcome(arrival_id, selected):
                return trace("original", arrival=0, end=10**9), final_duration if arrival_id == "2" else 1

            rows = replay(router, self.arrivals(descriptor), outcome)
            results.append([(r["selected"], r["reason"], r["visible_feedback"], r["candidates"]) for r in rows])
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
