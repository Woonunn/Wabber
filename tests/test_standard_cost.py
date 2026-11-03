"""Cost attribution must preserve unknowns, Session tails and unambiguous ownership."""
from decimal import Decimal
import json
import unittest

from experiments.exp1.analysis.standard_cost import agentcore_usage, fc_hour_bounds, lambda_cost, number


class StandardCostTests(unittest.TestCase):
    def test_lambda_zero_invoice_does_not_remove_measured_standard_charge(self):
        rates = {"gb_second": "0.0000166667", "request": "0.0000002",
                 "ephemeral_gb_second": "0.0000000309", "included_ephemeral_mb": 512}
        result = lambda_cost(1000, 1024, 1536, rates)
        self.assertEqual(result["cost"], Decimal("0.0000168976"))
        self.assertEqual(result["gb_seconds"], 1)
        self.assertEqual(result["ephemeral_gb_seconds"], 1)
        with self.assertRaises(ValueError):
            lambda_cost(None, 1024, 512, rates)

    def test_usage_rejects_unknown_nonfinite_and_negative(self):
        for value in [None, True, "NaN", "Infinity", -1]:
            with self.assertRaises(ValueError):
                number(value)

    def test_session_tail_is_included_and_duplicate_records_are_not_double_charged(self):
        def event(sid, stamp, cpu):
            return {"eventId": f"{sid}:{stamp}", "message": json.dumps({"event_timestamp": stamp,
                "attributes": {"session.id": sid, "time_elapsed_seconds": 1},
                "metrics": {"agent.runtime.vcpu.hours.used": cpu,
                            "agent.runtime.memory.gb_hours.used": 0.2}})}
        a, tail, pilot = event("formal", 0, 0.1), event("formal", 3000000, 0.2), event("pilot", 0, 9)
        result = agentcore_usage([a, a, tail, pilot], {"formal"})["formal"]
        self.assertEqual(result["vcpu_hours"], Decimal("0.3"))
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["last_epoch_ms"], 3000000)
        with self.assertRaisesRegex(ValueError, "Missing formal"):
            agentcore_usage([a], {"formal", "missing"})
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            agentcore_usage([a, {**event("formal", 0, 2), "eventId": a["eventId"]}], {"formal"})

    def test_mixed_hour_is_an_interval_not_a_proportional_allocation(self):
        hours = [{"StartTime": "1970-01-01T00:00:00Z", "EndTime": "1970-01-01T01:00:00Z", "CUUsage": "100"},
                 {"StartTime": "1970-01-01T01:00:00Z", "EndTime": "1970-01-01T02:00:00Z", "CUUsage": "50"}]
        result = fc_hour_bounds(hours, [(1800000, 5400000)], [(0, 1800000)], "0.1")
        self.assertIsNone(result["exact"])
        self.assertEqual((result["minimum"], result["maximum"]), (Decimal(5), Decimal(15)))
        # A finished Pilot cannot contaminate the next half-open meter bucket.
        result = fc_hour_bounds(hours[1:], [(3600000, 5400000)], [(0, 3600000)], "0.1")
        self.assertEqual(result["exact"], Decimal(5))

    def test_observed_nonformal_execution_tightens_only_the_bound(self):
        hours = [{"StartTime": "1970-01-01T00:00:00Z", "EndTime": "1970-01-01T01:00:00Z", "CUUsage": "100"}]
        observed = {hours[0]["StartTime"]: {"formal_cu": 40, "nonformal_cu": 30}}
        result = fc_hour_bounds(hours, [(0, 100)], [(200, 300)], "0.1", observed)
        self.assertEqual((result["minimum"], result["maximum"]), (Decimal(4), Decimal(7)))
        self.assertIsNone(result["exact"])
        observed[hours[0]["StartTime"]]["nonformal_cu"] = 80
        with self.assertRaisesRegex(ValueError, "exceeds"):
            fc_hour_bounds(hours, [(0, 100)], [(200, 300)], "0.1", observed)


if __name__ == "__main__":
    unittest.main()
