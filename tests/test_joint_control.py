"""Control safety and accounting checks; fixtures are not cloud measurements."""

import unittest
from unittest.mock import Mock

from atsuite.provider_model import ProviderModel
from atsuite.routing import (
    Capacity,
    JointController,
    OnlineRouter,
    ScalingAdapter,
    candidate_plans,
)
from atsuite.routing.cost import forecast_cost, total_cost
from atsuite.workload import WorkloadModel
from tests.test_provider_model import observation, pool
from tests.test_workload import trace
from tests.stage6_fixtures import tariff, bounds

FX = {
    "currency": "USD",
    "rates": {"USD": 1, "CNY": 0.15},
    "date": "2026-10-29",
    "source": "synthetic fixture",
}


def price(retention=0.01):
    return {
        "currency": "USD",
        "complete": True,
        "scope": "total_infrastructure",
        "exclusive_components": ["execution", "retention", "scaling", "storage"],
        "evidence": ["synthetic only"],
        "per_request": {"task": 0.001},
        "retention_per_s": retention,
        "session_per_s": 0.001,
        "storage_per_s": 0.0001,
        "per_change": 0.001,
    }


class DelayedAdapter:
    resource = "synthetic-test-resource"
    control = "reserved_concurrency"

    def __init__(self, target=1):
        self.target = self.effective = target
        self.calls = []
        self.fail = False

    def read(self, now_ns=None):
        if self.fail:
            raise RuntimeError("unavailable")
        return Capacity(
            now_ns or 1,
            self.control,
            self.target,
            self.effective,
            concurrency_quota=self.effective,
            status="READY" if self.target == self.effective else "IN_PROGRESS",
        )

    def apply(self, target, *, active_sessions):
        if target < self.target and active_sessions:
            raise ValueError("active sessions")
        self.calls.append(target)
        self.target = target


def controller(*, start=1, retention=None, policy="joint", weights=None):
    model, workload = ProviderModel(), WorkloadModel()
    levels = {}
    for target in (1, 2):
        config = pool(instances=None)
        config["billing"] = tariff(floor=(retention or {}).get(target, 0.01 * target))
        config["resources"]["capacity_control"] = {
            "control": "reserved_concurrency",
            "target": target,
        }
        levels[target] = {
            "target": target,
            "pool": config,
            "session_limit": target * 3,
            "scale_delay_s": 1,
            "evidence": ["fixture"],
            "cost": price((retention or {}).get(target, 0.01 * target)),
        }
        for i in range(100):
            row = observation(i)
            row.update(id=f"{target}-{i}", task_id="task", pool=config)
            row["load_rps"] = i / 10
            model.update(row)
    for i in range(3):
        p = observation()["profile"]
        p["request_id"] = f"reference-{i}"
        workload.update(p, 10000)
    router = OnlineRouter(
        workload,
        model,
        {"aws": levels[start]["pool"]},
        {"aws": start * 3},
        default_pool="aws",
        execution_bounds={"aws": bounds()},
    )
    adapter = DelayedAdapter(start)
    control = JointController(
        router,
        {"aws": adapter},
        {"aws": levels},
        candidate_plans({"aws": levels}, weights or [{"aws": 1}]),
        FX,
        period_s=1,
        cooldown_s=2,
        horizon_s=10,
        policy=policy,
    )
    return control, adapter


class JointControlTests(unittest.TestCase):
    def test_joint_search_prices_retention_and_uses_real_capacity(self):
        control, adapter = controller(retention={1: 10, 2: 0})
        result = control.tick({"task": 0.5}, now_ns=20000)
        self.assertEqual(result["selected"]["targets"], {"aws": 2})
        self.assertEqual(adapter.calls, [2])
        self.assertIsNone(control.router.limits["aws"])
        self.assertIsNone(
            control.decide("pending", "task", 0.5, as_of_ns=21000)["selected"]
        )
        adapter.effective = 2
        control.tick({"task": 0.5}, now_ns=50000)
        self.assertEqual(
            control.decide("ready", "task", 0.5, as_of_ns=51000)["selected"], "aws"
        )
        self.assertFalse(control.pending)

    def test_drain_all_sessions_before_shrink(self):
        control, adapter = controller(start=2, policy="independent")
        admitted = control.router.decide("active", "task", 0.5, as_of_ns=20000)
        self.assertEqual(admitted["selected"], "aws")
        control.tick({"task": 0.5}, now_ns=30000)
        self.assertEqual(control.draining, {"aws": 1})
        self.assertFalse(adapter.calls)
        self.assertEqual(control.router.limits["aws"], 0)
        control.complete(
            "active", trace("active", arrival=20000, end=40000), available_ns=40000
        )
        control.tick({"task": 0.5}, now_ns=50000)
        self.assertEqual(adapter.calls, [1])

    def test_pool_snapshot_and_transition_are_not_stable_evidence(self):
        control, _ = controller()
        decision = control.router.decide("active", "task", 0.5, as_of_ns=20000)
        control.router.set_capacity("aws", control.levels["aws"][2]["pool"], 6)
        row = control.complete(
            "active", trace("active", arrival=20000, end=30000), available_ns=30000
        )
        self.assertEqual(row["pool"], decision["pool_snapshot"]["aws"])
        self.assertTrue(row["capacity_transition"])
        revision = {**row, "available_ns": 40000, "capacity_transition": False}
        with self.assertRaisesRegex(ValueError, "capacity_transition"):
            control.router.provider.update(revision)
        decision = control.router.decide("new", "task", 0.5, as_of_ns=40000)
        self.assertEqual(decision["candidates"][0]["prediction"]["sample_count"], 100)

    def test_missing_cost_unknown_task_read_failure_and_stale_policy(self):
        control, adapter = controller()
        self.assertIsNotNone(control.tick({"unknown": 0.5}, now_ns=20000)["selected"])
        result = control.tick({"task": 0.5}, now_ns=2_000_020_000)
        self.assertIsNotNone(result["selected"])
        self.assertIsNone(
            control.decide("stale", "task", 0.5, as_of_ns=5_000_020_000)["selected"]
        )
        adapter.fail = True
        self.assertEqual(
            control.tick({"task": 0.5}, now_ns=6_000_020_000)["reason"],
            "capacity_readback_failed",
        )
        self.assertIsNone(control.router.limits["aws"])
        fresh, _ = controller()
        for level in fresh.levels["aws"].values():
            level["pool"].pop("billing")
        self.assertIsNone(fresh.tick({"task": 0.5}, now_ns=20000)["selected"])

    def test_failed_teardown_retains_occupancy(self):
        control, adapter = controller(start=2)
        control.router.decide("active", "task", 0.5, as_of_ns=20000)
        control.hold_release("active")
        control.complete(
            "active", trace("active", arrival=20000, end=30000), available_ns=30000
        )
        control.tick({"task": 0.5}, now_ns=40000)
        self.assertIn("active", control.active)
        self.assertFalse(adapter.calls)

    def test_invalidated_drain_does_not_apply_an_old_scale_in(self):
        control, adapter = controller(start=2, policy="independent")
        control.router.decide("active", "task", 0.5, as_of_ns=20000)
        control.tick({"task": 0.5}, now_ns=30000)
        self.assertIn("aws", control.draining)
        control.tick({"unmeasured-task": 1}, now_ns=2_000_030_000)
        self.assertFalse(control.draining)
        control.complete(
            "active",
            trace("active", arrival=20000, end=2_000_040_000),
            available_ns=2_000_040_000,
        )
        control.tick({"unmeasured-task": 1}, now_ns=2_000_050_000)
        self.assertFalse(adapter.calls)

    def test_boundary_validation_and_grid_bound(self):
        with self.assertRaises(ValueError):
            candidate_plans({"a": range(300)}, [{"a": 1}])
        with self.assertRaises(ValueError):
            Capacity(0, "reserved", True, 1)
        control, _ = controller()
        with self.assertRaises(ValueError):
            control.tick({"task": float("nan")})
        with self.assertRaises(ValueError):
            control.tick({"task": 0.5}, utilization={"aws": 2})


class AdapterTests(unittest.TestCase):
    def test_completed_mcp_session_keeps_its_remote_residency_reservation(self):
        import copy

        base, _ = controller()
        level = copy.deepcopy(base.levels["aws"][1])
        level["pool"]["provider"] = "ali_agentrun"
        level["pool"]["billing"] = tariff("ali_agentrun")
        level["pool"]["resources"]["session_affinity_config"] = (
            '{"sessionTTLInSeconds":3}'
        )
        level["pool"]["resources"]["capacity_control"]["control"] = "min_instances"
        model = ProviderModel()
        for i in range(100):
            row = observation(i)
            row.update(pool=level["pool"], task_id="task")
            model.update(row)
        router = OnlineRouter(
            base.router.workload,
            model,
            {"ali": level["pool"]},
            {"ali": 3},
            default_pool="ali",
            execution_bounds={"ali": bounds()},
        )
        adapter = DelayedAdapter()
        adapter.control = "min_instances"
        control = JointController(
            router,
            {"ali": adapter},
            {"ali": {1: level}},
            [{"targets": {"ali": 1}, "weights": {"ali": 1}}],
            FX,
        )
        router.decide("mcp", "task", 0.5, as_of_ns=20000)
        control.complete(
            "mcp", trace("mcp", arrival=20000, end=30000), available_ns=30000
        )
        self.assertIn("mcp", router.completed)
        self.assertEqual(control._occupied("ali"), 1)
        control.expire_retained(3_000_029_999)
        self.assertEqual(control._occupied("ali"), 1)
        control.expire_retained(3_000_030_000)
        self.assertEqual(control._occupied("ali"), 0)
        router.decide("closed", "task", 0.5, as_of_ns=3_000_040_000)
        completed = {**trace("closed", arrival=3_000_040_000, end=3_000_050_000), "sessions_closed": True}
        control.complete("closed", completed, available_ns=3_000_050_000)
        self.assertEqual(control._occupied("ali"), 0)
        self.assertNotIn("closed", control.retained_until)

    def test_lambda_quota_is_not_instances_and_shrink_is_guarded(self):
        client = Mock()
        client.get_function_configuration.return_value = {
            "State": "Active",
            "LastUpdateStatus": "Successful",
        }
        client.get_function_concurrency.return_value = {
            "ReservedConcurrentExecutions": 3
        }
        adapter = ScalingAdapter("aws_lambda", client, "fn")
        state = adapter.read(10)
        self.assertEqual(state.effective, 3)
        self.assertIsNone(state.actual_instances)
        self.assertIsNone(state.provisioned)
        with self.assertRaises(ValueError):
            adapter.apply(1, active_sessions=1)
        adapter.apply(4, active_sessions=1)
        client.put_function_concurrency.assert_called_once_with(
            FunctionName="fn", ReservedConcurrentExecutions=4
        )

    def test_lambda_provisioning_waits_for_ready(self):
        client = Mock()
        client.get_function_configuration.return_value = {
            "State": "Active",
            "LastUpdateStatus": "Successful",
        }
        client.get_function_concurrency.return_value = {
            "ReservedConcurrentExecutions": 10
        }
        client.get_provisioned_concurrency_config.return_value = {
            "RequestedProvisionedConcurrentExecutions": 5,
            "AllocatedProvisionedConcurrentExecutions": 2,
            "AvailableProvisionedConcurrentExecutions": 1,
            "Status": "IN_PROGRESS",
        }
        state = ScalingAdapter("aws_lambda", client, "fn", qualifier="v1").read(10)
        self.assertEqual((state.target, state.provisioned, state.available), (5, 2, 1))
        self.assertIsNone(state.effective)

    def test_fc_calls_installed_sdk_and_preserves_session_configuration(self):
        from alibabacloud_fc20230330 import models

        client = Mock()
        client.get_scaling_config.return_value.body = models.ScalingConfigStatus(
            min_instances=2, current_instances=2, target_instances=2
        )
        client.get_function.return_value.body = models.Function(
            state="Active", last_update_status="Successful"
        )
        client.get_concurrency_config.return_value.body = models.ConcurrencyConfig(
            reserved_concurrency=64
        )
        adapter = ScalingAdapter("ali_agentrun", client, "fn")
        self.assertEqual(adapter.read(10).effective, 2)
        with self.assertRaises(ValueError):
            adapter.apply(1, active_sessions=1)
        adapter.apply(3, active_sessions=1)
        request = client.put_scaling_config.call_args.args[1]
        self.assertEqual(request.body.min_instances, 3)
        self.assertFalse(client.update_function.called)

    def test_agentcore_is_managed_and_not_mutated(self):
        client = Mock()
        client.get_agent_runtime.return_value = {"status": "READY"}
        adapter = ScalingAdapter("aws_agentcore", client, "runtime")
        self.assertIsNone(adapter.read(10).actual_instances)
        with self.assertRaises(ValueError):
            adapter.apply(1, active_sessions=0)
        self.assertFalse(client.update_agent_runtime.called)


class CostTests(unittest.TestCase):
    def test_shared_charges_once_and_explicit_fx(self):
        result = forecast_cost(
            price(),
            requests={"task": 10},
            horizon_s=10,
            retained_s=20,
            changes=1,
            fx=FX,
        )
        self.assertAlmostEqual(result["total"], 0.132)
        cny = {**price(), "currency": "CNY"}
        converted = forecast_cost(
            cny, requests={"task": 10}, horizon_s=10, retained_s=20, changes=1, fx=FX
        )
        self.assertAlmostEqual(converted["total"], result["total"] * 0.15)
        self.assertEqual(converted["basis"], "predicted")

    def test_bills_deduplicate_and_reject_overlaps(self):
        lines = [
            {
                "provider": "aws_lambda",
                "charge_id": key,
                "resource": "pool",
                "start_ns": 0,
                "end_ns": 10,
                "component": key,
                "currency": "USD",
                "amount": 1,
                "basis": "estimated",
                "complete": True,
                "attribution": "exclusive",
                "evidence": ["fixture"],
            }
            for key in ("execution", "retention", "storage", "scaling")
        ]
        self.assertEqual(total_cost(lines + lines, FX)["total"], 4)
        with self.assertRaises(ValueError):
            total_cost(lines + [{**lines[0], "charge_id": "other"}], FX)
        with self.assertRaises(ValueError):
            total_cost(
                lines
                + [{**lines[0], "charge_id": "other", "start_ns": 5, "end_ns": 15}],
                FX,
            )
        self.assertIsNone(total_cost(lines[:1], FX)["total"])
        short = [dict(row) for row in lines]
        short[-1]["end_ns"] = 5
        self.assertIsNone(total_cost(short, FX)["total"])


class SimulationTests(unittest.TestCase):
    def surface(self, config):
        from atsuite.routing.simulation import ServiceSurface

        samples = []
        for run in ("train", "held"):
            rows = [
                {
                    "id": f"{run}-{i}",
                    "request_id": f"{run}-{i}",
                    "source_run": run,
                    "pool": config,
                    "task_id": "task",
                    "tool": "python",
                    "load_rps": 0.5,
                    "concurrent_sessions": 1,
                    "service_s": 0.1 + (i % 3) * 0.1,
                    "cold_start_s": 0.05,
                    "status": "success",
                    "queue_excluded": True,
                    "evidence": ["synthetic fixture"],
                }
                for i in range(20)
            ]
            samples.append(rows)
        return ServiceSurface(*samples), samples

    def test_surface_requires_independent_run_and_known_configuration(self):
        import random

        from atsuite.routing.simulation import ServiceSurface

        surface, samples = self.surface(pool())
        self.assertTrue(surface.validation[0]["passed"])
        with self.assertRaisesRegex(ValueError, "leaks"):
            ServiceSurface(samples[0], samples[0])
        with self.assertRaises(ValueError):
            surface.sample(
                pool(instances=3), "task", "python", 0.5, 1, random.Random(1)
            )
        with self.assertRaises(ValueError):
            surface.sample(pool(), "task", "python", 0.5, 2, random.Random(1))

    def test_seeded_tool_sampling_and_all_offered_requests_reported(self):
        from atsuite.routing.simulation import SimulatedAdapter, simulate

        def run_once():
            control, _ = controller()
            control.observation_s = 2
            for level in control.levels["aws"].values():
                level["tool_slots"] = 1
            control.adapters = {
                "aws": SimulatedAdapter("reserved_concurrency", 1, {1: 1, 2: 1})
            }
            surface, _ = self.surface(control.levels["aws"][1]["pool"])
            workflows = {
                "task": {
                    "nodes": [
                        {"id": "wait", "kind": "llm", "name": "model", "wait_s": 0.1},
                        {"id": "tool", "kind": "tool", "name": "python"},
                    ],
                    "edges": [["wait", "tool"]],
                }
            }
            rows = [
                {"id": f"sim-{i}", "task_id": "task", "at_s": 1 + i * 2.1}
                for i in range(4)
            ]
            return simulate(
                control, rows, workflows, surface, seed=7, end_s=10, start_ns=10**9
            )

        one, two = run_once(), run_once()
        self.assertEqual(one, two)
        self.assertEqual(one["offered"], 4)
        self.assertGreater(one["success"], 0)
        self.assertTrue(one["valid"])
        self.assertEqual(one["cost"]["basis"], "estimated")
        self.assertIsNone(one["cost"]["actual"])
        self.assertLess(one["p95_s"], 2)  # Historical request latency was 2s.


class IntegrationBoundaryTests(unittest.TestCase):
    def test_queue_time_is_part_of_admitted_latency(self):
        control, _ = controller()
        decision = control.router.decide(
            "queued", "task", 0.5, as_of_ns=30000, arrival_ns=20000
        )
        self.assertEqual(decision["decision_ns"], 30000)
        row = control.complete(
            "queued", trace("queued", arrival=20000, end=40000), available_ns=40000
        )
        self.assertEqual(row["measurements"]["latency_s"], 0.00002)
        control.observe_arrival("rejected-in-harness", "task", 50000)
        self.assertGreater(control.observed_rates(now_ns=50001)["global"], 0)
        late = control.router.decide(
            "expired", "task", 0.5, as_of_ns=121_000_000_001, arrival_ns=1
        )
        self.assertEqual(late["selected"], "aws")
        self.assertNotIn("slo_budget_exhausted", late["candidates"][0]["flags"])

    def test_joint_plan_can_disable_an_idle_pool_and_change_routing(self):
        import copy

        base, _ = controller()
        levels = copy.deepcopy(base.levels)
        off = copy.deepcopy(levels["aws"][1])
        off.update(target=0, session_limit=0, cost=price(0))
        off["pool"]["resources"]["capacity_control"]["target"] = 0
        off["pool"]["billing"] = tariff()
        levels["aws"][0] = off
        levels["ali"] = {}
        for target in (1, 2):
            level = copy.deepcopy(levels["aws"][target])
            level["pool"]["provider"] = "ali_fc"
            level["pool"]["billing"] = tariff("ali_fc", floor=0.001)
            level["cost"].update(currency="CNY", retention_per_s=0.001)
            levels["ali"][target] = level
            for i in range(100):
                row = observation(i, provider="ali_fc")
                row.update(id=f"ali-{target}-{i}", task_id="task", pool=level["pool"])
                base.router.provider.update(row)
        pools = {k: v[1]["pool"] for k, v in levels.items()}
        router = OnlineRouter(base.router.workload, base.router.provider, pools, {})
        adapters = {k: DelayedAdapter() for k in pools}
        plans = candidate_plans(levels, [{"aws": 1, "ali": 0}, {"aws": 0, "ali": 1}])
        joint = JointController(router, adapters, levels, plans, FX, horizon_s=30)
        result = joint.tick({"task": 0.5}, now_ns=20000)
        self.assertEqual(result["selected"]["weights"], {"aws": 0, "ali": 1})
        self.assertEqual(result["selected"]["targets"]["aws"], 0)
        self.assertEqual(adapters["aws"].calls, [0])

    def test_sealed_atg_discards_historical_tool_durations(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from atsuite.routing.simulation import load_sealed_workload
        from experiments.exp1.data.prepare_e1b import ROOT

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp)
            atg = {
                "nodes": [
                    {
                        "id": 0,
                        "name": "model",
                        "type": "llm",
                        "time": 500,
                        "edge_to": [{"id": 1}, {"id": 2}],
                    },
                    {
                        "id": 1,
                        "name": "python",
                        "type": "tool_use",
                        "time": 99999,
                        "edge_to": [],
                    },
                    {
                        "id": 2,
                        "name": "python",
                        "type": "tool_use",
                        "time": 88888,
                        "edge_to": [],
                    },
                ]
            }
            (path / "trace.atg.json").write_text(json.dumps(atg))
            (path / "manifest.json").write_text("{}")
            manifest = {
                "config": {"window_duration_ms": 2000},
                "tasks": {"task": {"capture": str(path), "capture_files": {}}},
                "request_table_sha256": "fixture",
            }
            rows = [
                {"arrival_id": "one", "task_id": "task", "relative_arrival_ms": "1500"}
            ]
            with patch(
                "experiments.exp1.data.prepare_e1b.load", return_value=(manifest, rows)
            ):
                arrivals, workflows, _ = load_sealed_workload(path)
            self.assertEqual(arrivals[0]["at_s"], 1.5)
            self.assertEqual(workflows["task"]["edges"], [["0", "1"], ["0", "2"]])
            self.assertEqual(
                [n["wait_s"] for n in workflows["task"]["nodes"]], [0.5, 0, 0]
            )


class CloudBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def test_simulation_identity_binds_capacity_and_cost_inputs(self):
        import json
        import tempfile
        from pathlib import Path

        from atsuite.routing.experiment import control_identity

        with tempfile.TemporaryDirectory() as tmp:
            config = json.loads(
                write_simulation_fixture(Path(tmp) / "inputs").read_text()
            )
            identity = control_identity(config)
            config["pools"]["aws"]["levels"][0]["cost"]["retention_per_s"] *= 2
            self.assertNotEqual(identity, control_identity(config))

    async def test_no_feasible_preview_makes_no_mutation_or_budget_reservation(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from experiments.exp1.runner import run_joint

        control, adapter = controller()
        control.tick = Mock(return_value={"selected": None})
        control.snapshots = {"aws": adapter.read(1)}
        config = {"pools": {"aws": {"initial": 1}}, "initial_task_rates": {"task": 0.5}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(
                    run_joint, "validate_inputs", return_value=({}, [], [], {}, [])
                ),
                patch.object(
                    run_joint, "cloud_adapters", return_value={"aws": adapter}
                ),
                patch.object(run_joint, "build_controller", return_value=control),
                patch.object(run_joint, "experiment_identity", return_value="fixture"),
                patch.object(run_joint, "sha", return_value="fixture"),
                patch.object(run_joint, "reserve_budget") as budget,
            ):
                result = await run_joint.execute(
                    config, root, root, root, root, root / "output", pilot=True
                )
            self.assertEqual(result["status"], "blocked_no_trusted_feasible_plan")
            self.assertFalse(adapter.calls)
            budget.assert_not_called()


def write_simulation_fixture(directory):
    """Write a reproducible, explicitly synthetic CLI smoke fixture (no cloud claims)."""
    import json
    from pathlib import Path

    path = Path(directory).resolve()
    path.mkdir(parents=True, exist_ok=False)
    control, _ = controller()
    control.router.provider.save(path / "provider.json")
    control.router.workload.save(path / "workload.json")
    for level in control.levels["aws"].values():
        level["tool_slots"] = 1
    _, samples = SimulationTests().surface(control.levels["aws"][1]["pool"])
    values = {
        "training": samples[0],
        "validation": samples[1],
        "workflows": {
            "task": {
                "nodes": [
                    {"id": "wait", "kind": "llm", "name": "model", "wait_s": 0.1},
                    {"id": "tool", "kind": "tool", "name": "python"},
                ],
                "edges": [["wait", "tool"]],
            }
        },
        "arrivals": [
            {"id": f"sim-{i}", "task_id": "task", "at_s": 1 + i * 2.1} for i in range(4)
        ],
    }
    for key, value in values.items():
        (path / (key + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    config = {k: str(path / (k + ".json")) for k in values}
    config.update(
        provider_model=str(path / "provider.json"),
        workload_model=str(path / "workload.json"),
        pools={
            "aws": {
                "control": "reserved_concurrency",
                "initial": 1,
                "levels": list(control.levels["aws"].values()),
            }
        },
        weights=[{"aws": 1}],
        fx=FX,
        control={"period_s": 1, "horizon_s": 10, "cooldown_s": 2, "observation_s": 2},
        seed=7,
        start_ns=10**9,
        end_s=10,
    )
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return path / "config.json"


if __name__ == "__main__":
    unittest.main()
