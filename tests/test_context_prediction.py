"""Strict context prediction contracts; fake model outputs are not semantic accuracy evidence."""

import copy
import tempfile
import unittest
from pathlib import Path

from atsuite.provider_model import ProviderModel
from atsuite.workload import WorkloadModel
from atsuite.workload.descriptor import (
    LEGACY_FIELDS,
    compatibility_key,
    compatible,
    family_key,
)
from atsuite.workload.descriptor import describe as raw_describe
from atsuite.workload.migrate import migrate
from atsuite.workload.retrieval import ContextRetriever
from tests.test_provider_model import observation, pool, profile
from tests.test_workload import trace
from tests.workload_fixtures import describe, stable_prediction


class Backend:
    revision = "test-reranker-v1"
    currency = "USD"

    def __init__(self, accepted=()):
        self.accepted = set(accepted)
        self.seen = []
        self.cost = 0
        self.embed_inputs = []

    def embed(self, texts):
        self.embed_inputs.extend(texts)
        self.cost += 0.001
        # Equal embeddings deliberately cannot distinguish negation or execution.
        return [[1, -1] for _ in texts]

    def rerank(self, query, candidates):
        self.seen.extend(candidates)
        self.cost += 0.002
        return [
            {
                "id": p["id"],
                "usable": p["id"] in self.accepted,
                "utility": 0.9,
                "reason": "fixture execution relevance",
            }
            for p in candidates
        ]


def add(model, rid, context, *, seconds=1, state=100, idle=2, available=100):
    p = profile(state=state, idle=idle)
    p.update(request_id=rid, source="synthetic")
    p["metrics"]["tool_elapsed_s"] = seconds
    model.update(p, available, descriptor=context)
    return p


class ContextPredictionTests(unittest.TestCase):
    def context(self, **kwargs):
        return describe(
            prompt="Read the header only; do not train", tools=["python"], **kwargs
        )

    def test_full_set_harness_revision_and_schema_are_hard_gates(self):
        query = self.context()
        changed = []
        for field in ("revision", "id"):
            d = copy.deepcopy(query)
            d["harness"][field] += "other"
            changed.append(d)
            d = copy.deepcopy(query)
            d["tool_definitions"][0]["implementation"][field] += "other"
            changed.append(d)
        d = copy.deepcopy(query)
        d["tool_definitions"][0]["definition"]["parameters"] = {"new": True}
        changed.append(d)
        changed += [
            describe(prompt=query["prompt"], tools=["other"]),
            describe(prompt=query["prompt"], tools=["python", "other"]),
            raw_describe(prompt=query["prompt"], tools=["python"]),
        ]
        backend = Backend(["good"])
        model = WorkloadModel(retriever=ContextRetriever(backend))
        for i, d in enumerate(changed):
            self.assertFalse(compatible(query, d))
            add(model, "bad" + str(i), d)
        add(model, "good", query)
        result = model.predict(101, descriptor=query)
        self.assertEqual([p["request_id"] for p in result["joint_profiles"]], ["good"])
        self.assertEqual({p["id"] for p in backend.seen}, {"good"})
        self.assertEqual(len(backend.embed_inputs), 1)
        self.assertEqual(result["eligible_success_count"], 1)

    def test_compatibility_does_not_establish_behavior_similarity(self):
        files = [{"name": "same.csv", "bytes": 1000, "sha256": "same-file"}]
        query = self.context(files=files)
        opposite = describe(
            prompt="Read all data and train the model", tools=["python"], files=files
        )
        paraphrase = describe(
            prompt="Inspect just the column names", tools=["python"], files=files
        )
        backend = Backend(["paraphrase"])
        model = WorkloadModel(retriever=ContextRetriever(backend))
        add(model, "opposite", opposite, seconds=100, state=10**9)
        add(model, "paraphrase", paraphrase, seconds=1, state=100)
        prediction = model.predict(101, descriptor=query)
        self.assertEqual(prediction["metrics"]["tool_elapsed_s"]["mean"], 1)
        self.assertEqual(len(backend.seen), 2)
        self.assertFalse(prediction["uncertainty"]["calibrated"])
        self.assertTrue(prediction["low_confidence"])

    def test_scale_model_and_policy_remain_visible_and_invalidate_exact_cache(self):
        query = self.context(
            files=[{"name": "x.csv", "bytes": None, "usage": "uncertain"}]
        )
        for field, value in (
            ("model", {"name": "new"}),
            ("control", {"max_iterations": 100}),
            ("session", {"messages": ["use every file"]}),
        ):
            other = {**query, field: value}
            self.assertNotEqual(family_key(query), family_key(other))
            model = WorkloadModel()
            add(model, "old", query)
            self.assertEqual(model.predict(101, descriptor=other)["sample_count"], 0)
        backend = Backend(["huge"])
        model = WorkloadModel(retriever=ContextRetriever(backend))
        add(model, "huge", self.context(files=[{"name": "x.csv", "bytes": 10**10}]))
        tiny = self.context(files=[{"name": "x.csv", "bytes": 10}])
        self.assertEqual(model.predict(101, descriptor=tiny)["sample_count"], 0)

    def test_no_context_no_history_and_no_cross_group_prior(self):
        d = self.context()
        p = profile()
        p.update(
            descriptor=describe(prompt=d["prompt"], tools=["different"]),
            available_ns=90,
        )
        model = WorkloadModel(initial_profiles=[p])
        add(model, "no-identity", raw_describe(prompt=d["prompt"], tools=["python"]))
        for query in (None, d):
            r = model.predict(101, descriptor=query)
            self.assertEqual(r["sample_count"], 0)
            self.assertTrue(r["unknown_request"])
        matching = profile()
        matching.update(descriptor=d, available_ns=100)
        model = WorkloadModel(initial_profiles=[matching])
        self.assertEqual(model.predict(100, descriptor=d)["sample_count"], 0)
        self.assertEqual(model.predict(101, descriptor=d)["sample_count"], 1)

    def test_failed_reranker_and_missing_keys_fail_closed(self):
        backend = Backend()
        backend.rerank = lambda q, p: [
            {"id": "injected", "usable": True, "utility": 1, "reason": "bad"}
        ]
        model = WorkloadModel(retriever=ContextRetriever(backend))
        d = self.context()
        add(model, "good", d)
        result = model.predict(101, descriptor=d)
        self.assertEqual(result["sample_count"], 0)
        self.assertEqual(result["retrieval"]["error"], "ValueError")
        self.assertGreater(result["retrieval"]["cost"], 0)

    def test_joint_scenarios_do_not_cross_product_marginals(self):
        d = self.context()
        model = WorkloadModel()
        add(model, "cpu", d, seconds=100, state=1, idle=1)
        add(model, "state", d, seconds=1, state=100, idle=100)
        result = model.predict(101, descriptor=d)
        self.assertEqual(result["scenario_weights"], [0.5, 0.5])
        self.assertEqual(
            {
                (
                    p["metrics"]["tool_elapsed_s"],
                    p["metrics"]["state_bytes"],
                    p["metrics"]["llm_idle_s"],
                )
                for p in result["joint_profiles"]
            },
            {(100, 1, 1), (1, 100, 100)},
        )

    def test_time_causality_and_cache_partition(self):
        d = self.context()
        backend = Backend(["past", "future", "other"])
        model = WorkloadModel(retriever=ContextRetriever(backend))
        add(model, "past", d, available=100)
        add(model, "future", d, available=200)
        a = model.predict(150, descriptor=d)
        self.assertEqual({p["id"] for p in backend.seen}, {"past"})
        other = copy.deepcopy(d)
        other["harness"]["revision"] = "new"
        add(model, "other", other, available=100)
        model.predict(150, descriptor=other)
        self.assertEqual(len(model.retriever.cache), 2)
        self.assertEqual(a["sample_count"], 1)

    def test_calibration_uses_frozen_predictions_before_feedback(self):
        d = self.context()
        model = WorkloadModel()
        add(model, "seed", d, available=100)
        for i in range(12):
            stamp = 200 + i * 10
            before = model.begin("feedback" + str(i), "s", stamp, descriptor=d)
            self.assertLessEqual(
                before["uncertainty"]["intervals"]["state_bytes"]["n"], i
            )
            add(model, "feedback" + str(i), d, state=100 + i, available=stamp + 1)
        pred = model.predict(400, descriptor=d)
        self.assertTrue(pred["uncertainty"]["calibrated"])
        self.assertFalse(pred["low_confidence"])
        self.assertEqual(pred["uncertainty"]["intervals"]["state_bytes"]["n"], 12)
        changed = {**d, "model": {"name": "new-model"}}
        self.assertFalse(
            model.predict(400, descriptor=changed)["uncertainty"]["calibrated"]
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snapshot.json"
            model.save(path)
            self.assertEqual(
                stable_prediction(pred),
                stable_prediction(WorkloadModel.load(path).predict(400, descriptor=d)),
            )

    def test_completion_cannot_replace_frozen_context(self):
        d = self.context()
        model = WorkloadModel()
        model.begin("r", "s", 1, descriptor=d)
        with self.assertRaises(ValueError):
            add(
                model,
                "r",
                describe(prompt="different", tools=["python"]),
                available=100,
            )
        p = profile()
        p.update(request_id="r", descriptor=None)
        saved = model.update(p, 100)
        self.assertEqual(saved["descriptor"], d)

    def test_legacy_migration_requires_original_context_not_labels(self):
        d = self.context()
        p = profile()
        p["descriptor"] = {k: d.get(k) for k in LEGACY_FIELDS}
        p["descriptor"]["semantic"] = {"primary": "old-label"}
        snapshot = {
            "schema_version": 1,
            "samples": [{"available_ns": 100, "profile": p}],
        }
        self.assertEqual(migrate(snapshot, {})["migration"]["quarantined"], 1)
        arrivals = {
            p["request_id"]: {
                "context": d,
                "visible_ns": 1,
                "evidence": ["raw prompt and deployment lock"],
            }
        }
        result = migrate(snapshot, arrivals)
        self.assertEqual(result["migration"]["migrated"], 1)
        self.assertTrue(
            compatibility_key(result["samples"][0]["profile"]["descriptor"])
        )
        arrivals[p["request_id"]]["visible_ns"] = 2
        with self.assertRaises(ValueError):
            migrate(snapshot, arrivals)

    def test_provider_and_exploration_cannot_use_incompatible_requests(self):
        from atsuite.routing.exploration import Exploration

        model = ProviderModel()
        d = self.context()
        for i in range(3):
            row = observation(i)
            row["profile"]["descriptor"] = {
                **d,
                "harness": {"id": "other", "revision": "v1"},
            }
            model.update(row)
        p = profile()
        p["descriptor"] = d
        self.assertEqual(
            model.predict(p, [pool()], 0.5, as_of_ns=1000)["pools"][0]["sample_count"],
            0,
        )
        self.assertEqual(
            Exploration({}).evidence(model, pool(), d, 1000)["attempts"], 0
        )

    def test_observations_preserve_unknown_and_llm_overlap(self):
        t = trace()
        t["request_context"] = self.context()
        llm = next(s for s in t["spans"] if s["kind"] == "llm")
        t["spans"].append({**llm, "id": "retry", "metadata": {"retry_of": llm["id"]}})
        from atsuite.workload import extract_profile

        p = extract_profile(t)
        self.assertEqual(p["metrics"]["llm_idle_s"], 4)
        self.assertEqual(len(p["observations"]["llm_calls"]), 2)
        self.assertEqual(p["metrics"]["state_bytes"], 8000)
        self.assertIsNone(p["metrics"]["state_transfer_bytes"])
        self.assertEqual(p["descriptor"], t["request_context"])


class RoutingFeedbackTests(unittest.TestCase):
    def test_new_predictor_defaults_to_shadow_until_explicit_activation(self):
        from tests.test_stage7 import router_fixture

        router, d = router_fixture(epsilon=0)
        router.default_pool = "expensive"
        router.enable_context_routing = False
        active = router.decide("safe", descriptor=d, as_of_ns=1000, load_rps=0.5)
        self.assertEqual(active["selected"], "expensive")
        proposal = router.decide(
            "proposal", descriptor=d, as_of_ns=1000, shadow=True, load_rps=0.5
        )
        self.assertEqual(proposal["selected"], "known")
        self.assertNotIn("proposal", router.active)
        router.enable_context_routing = True
        self.assertEqual(
            router.decide("activated", descriptor=d, as_of_ns=1000, load_rps=0.5)[
                "selected"
            ],
            "known",
        )

    def test_prediction_cost_is_in_budget_and_objective(self):
        from tests.test_stage7 import router_fixture

        router, d = router_fixture(epsilon=0, budget=1.002, exploration_budget=0)
        router.workload.retriever = ContextRetriever(Backend(["0", "1", "2"]))
        decision = router.decide("costly-retrieval", descriptor=d, as_of_ns=1000)
        self.assertAlmostEqual(decision["prediction_overhead"]["cost"], 0.003)
        self.assertAlmostEqual(router.spent, 0.003)
        self.assertIsNone(decision["selected"])
        self.assertTrue(
            all("budget_exhausted" in c["flags"] for c in decision["candidates"])
        )
        for c in decision["candidates"]:
            if c["objective_cost"] is not None:
                self.assertGreaterEqual(c["objective_cost"], 0.003)

    def test_request_id_cache_cannot_reuse_an_incompatible_context(self):
        from tests.test_stage7 import router_fixture

        router, d = router_fixture(epsilon=0)
        router.decide("cached", descriptor=d, as_of_ns=1000)
        changed = copy.deepcopy(d)
        changed["harness"]["revision"] = "another"
        with self.assertRaises(ValueError):
            router.decide("cached", descriptor=changed, as_of_ns=1000)


class ReplayValidationTests(unittest.TestCase):
    def records(self):
        from tests.stage6_fixtures import bounds, tariff

        context = describe(prompt="Inspect the column names", tools=["python"])
        target = pool()
        rows = []
        for i in range(15):
            row = observation(i)
            row["profile"].update(
                request_id=str(i),
                arrival_ns=row["arrival_ns"],
                completed_ns=row["completed_ns"],
                descriptor=context,
            )
            row["outcomes"] = {"only": {"cost": 1, "latency_s": 2, "status": "success"}}
            rows.append(row)
        config = {
            "pools": {"only": target},
            "tariffs": {"aws_lambda": tariff()},
            "default_pool": "only",
            "execution_bounds": {"only": bounds()},
            "currency": "USD",
        }
        return rows, config

    def test_replay_reports_behavior_calibration_and_net_matched_results(self):
        from atsuite.workload.validation import evaluate

        rows, config = self.records()
        report = evaluate(rows, config)
        self.assertEqual(report["matched_requests"], 15)
        self.assertEqual(report["best_fixed_pool"], "only")
        self.assertEqual(
            report["best_fixed_objective"], report["matched_oracle_objective"]
        )
        self.assertIsNone(report["online_routing_results"])
        pred = report["methods"]["context"]["predictions"]
        self.assertEqual(pred["0"]["sample_count"], 0)
        self.assertTrue(pred["0"]["cold_start"])
        self.assertEqual(pred["1"]["sample_count"], 1)
        self.assertIsNotNone(pred["14"]["interval_covered"]["state_bytes"])
        self.assertEqual(
            report["methods"]["context"]["matched_replay"]["scored_requests"], 15
        )

    def test_no_matched_outcomes_means_no_fabricated_routing_gain(self):
        from atsuite.workload.validation import evaluate

        rows, config = self.records()
        for r in rows:
            r.pop("outcomes")
        report = evaluate(rows, config)
        self.assertEqual(report["matched_requests"], 0)
        self.assertIsNone(report["best_fixed_objective"])
        self.assertIsNone(report["methods"]["context"]["matched_replay"]["cost"])

    def test_supervision_never_pairs_future_or_incompatible_executions(self):
        from atsuite.workload.validation import supervision_pairs

        rows, _ = self.records()
        rows[1]["profile"]["descriptor"] = describe(
            prompt="same words", tools=["other"]
        )
        pairs = supervision_pairs(rows[:3])
        self.assertEqual(
            [(p["query_id"], p["history_id"]) for p in pairs], [("2", "0")]
        )
        self.assertEqual(pairs[0]["behavior_distance"], 0)


class BackendContractTests(unittest.TestCase):
    def test_sdk_adapter_accounts_tokens_without_calling_a_service(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from atsuite.workload.retrieval import ContextBackend

        config = {
            "base_url": "https://context.invalid/v1",
            "api_key_env": "TEST_CONTEXT_KEY",
            "embedding_model": "pinned-embedding",
            "rerank_model": "pinned-rerank",
            "revision": "locked-v1",
            "currency": "USD",
            "rates": {"embedding": 1, "input": 2, "output": 3},
        }
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(
            usage=SimpleNamespace(total_tokens=100),
            data=[SimpleNamespace(index=0, embedding=[1.0, -1.0])],
        )
        client.chat.completions.create.return_value = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"candidates": []}'),
                )
            ],
        )
        with (
            patch.dict(os.environ, {"TEST_CONTEXT_KEY": "test-only"}),
            patch("openai.OpenAI", return_value=client),
        ):
            backend = ContextBackend(config)
            self.assertEqual(backend.embed(["context"]), [[1.0, -1.0]])
            self.assertEqual(backend.rerank({}, []), [])
            self.assertAlmostEqual(backend.cost, 0.00033)
            self.assertNotIn("test-only", str(backend.config))

    def test_provider_rejects_mixed_compatibility_mixtures(self):
        p, other = profile(), profile()
        other["descriptor"] = describe(tools=["different"])
        with self.assertRaises(ValueError):
            ProviderModel().predict([p, other], [pool()], 0.5)

    def test_tool_order_does_not_change_identity(self):
        left = describe(tools=["python", "reader"])
        right = describe(tools=["reader", "python"])
        self.assertEqual(compatibility_key(left), compatibility_key(right))
