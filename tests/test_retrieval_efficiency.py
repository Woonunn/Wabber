"""Skip unusable candidates and remove only verified shared BGE payload fields."""

import copy
import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from atsuite.workload.retrieval import ContextBackend, ContextRetriever
from tests.test_context_prediction import Backend
from tests.test_provider_model import profile
from tests.workload_fixtures import describe


class RetrievalEfficiencyTests(unittest.TestCase):
    def backend(self):
        config = json.loads(
            (Path(__file__).resolve().parents[1]
             / "experiments/exp1/configs/retrieval-siliconflow.json").read_text()
        )
        client = Mock()
        client.post.return_value = {
            "meta": {"tokens": {"input_tokens": 10}},
            "results": [{"index": 0, "relevance_score": 0.8}],
        }
        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-only"}), patch(
            "openai.OpenAI", return_value=client
        ):
            return ContextBackend(config), client

    def test_bge_keeps_variable_context_behavior_and_original_inputs(self):
        backend, client = self.backend()
        query = describe(
            prompt="Do not train; read only the header", tools=["python"],
            files=[{"name": "x.csv", "bytes": None, "usage": "uncertain"}],
            model={"name": "planner"}, control={"max_iterations": 7},
            session={"messages": ["read only"]},
        )
        other = {**query, "prompt": "Train on all rows",
                 "control": {"max_iterations": 8}}
        candidates = [{"id": "a", "context": other,
                       "behavior": {"metrics": {"state_bytes": 100}, "workflow": {}}}]
        original = copy.deepcopy((query, candidates))
        backend.rerank(query, candidates)
        body = client.post.call_args.kwargs["body"]
        shared = {"tool_definitions", "harness"}
        self.assertEqual(json.loads(body["query"]),
                         {k: v for k, v in query.items() if k not in shared})
        document = json.loads(body["documents"][0])
        self.assertEqual(document["context"],
                         {k: v for k, v in other.items() if k not in shared})
        self.assertEqual(document["behavior"], candidates[0]["behavior"])
        self.assertEqual((query, candidates), original)

    def test_bge_preserves_full_payload_without_verified_shared_identity(self):
        backend, client = self.backend()
        query = describe(prompt="header", tools=["python"])
        for other in [
            describe(prompt="header", tools=["other"]),
            {"prompt": "no implementation identity"},
        ]:
            candidates = [{"id": "a", "context": other}]
            backend.rerank(query, candidates)
            body = client.post.call_args.kwargs["body"]
            self.assertEqual(json.loads(body["query"]), query)
            self.assertEqual(json.loads(body["documents"][0]), candidates[0])

    def samples(self, huge_count=1):
        query = describe(prompt="read", tools=["python"],
                         files=[{"name": "x.csv", "bytes": 10}])
        huge = describe(prompt="read", tools=["python"],
                        files=[{"name": "x.csv", "bytes": 10**10}])
        rows = [
            {**profile(), "request_id": f"huge-{i}", "descriptor": huge}
            for i in range(huge_count)
        ]
        rows.append({**profile(), "request_id": "good", "descriptor": query})
        return query, rows

    def test_excludes_existing_scale_rejections_before_reranking(self):
        query, rows = self.samples()
        backend = Backend(["good", "huge-0"])
        selected, evidence = ContextRetriever(backend).select(query, rows)
        self.assertEqual([p["request_id"] for p in selected], ["good"])
        self.assertEqual([c["id"] for c in backend.seen], ["good"])
        self.assertIsNone(evidence["error"])

    def test_empty_eligible_recall_skips_rerank_without_expanding_top_40(self):
        query, rows = self.samples(40)
        backend = Backend(["good"])
        selected, evidence = ContextRetriever(backend).select(query, rows)
        self.assertEqual(selected, [])
        self.assertEqual(backend.seen, [])
        self.assertEqual(backend.cost, 0.001)
        self.assertEqual(evidence["match_basis"], "cold_start")
        self.assertIsNone(evidence["error"])


    def test_native_candidates_within_limit_keep_entire_sample_without_api(self):
        query, _ = self.samples()
        for api in ("siliconflow", "dashscope"):
            for count in (1, 20):
                with self.subTest(api=api, count=count):
                    rows = [{**profile(state=i + 1), "request_id": str(i),
                             "descriptor": query} for i in range(count)]
                    backend = Backend([r["request_id"] for r in rows])
                    backend.rerank_api = api
                    selected, evidence = ContextRetriever(backend).select(query, rows)
                    self.assertEqual(selected, rows)
                    self.assertEqual(backend.seen, [])
                    self.assertEqual(backend.cost, 0.001)
                    self.assertTrue(all(
                        r["utility"] == 0 and "ranking skipped" in r["reason"]
                        for r in evidence["candidates"]
                    ))

    def test_native_candidates_over_limit_still_use_ranking(self):
        query, _ = self.samples()
        rows = [{**profile(), "request_id": str(i), "descriptor": query}
                for i in range(2)]
        backend = Backend(["0", "1"])
        backend.rerank_api = "siliconflow"
        backend.rerank = Mock(return_value=[
            {"id": "0", "usable": True, "utility": 0.1, "reason": "lower"},
            {"id": "1", "usable": True, "utility": 0.9, "reason": "higher"},
        ])
        selected, evidence = ContextRetriever(backend).select(query, rows, limit=1)
        self.assertEqual([p["request_id"] for p in selected], ["1"])
        backend.rerank.assert_called_once()
        self.assertIsNone(evidence["error"])

    def test_chat_still_judges_usability_for_one_candidate(self):
        query, rows = self.samples()
        backend = Backend()
        backend.rerank_api = "chat"
        selected, evidence = ContextRetriever(backend).select(query, rows[1:])
        self.assertEqual(selected, [])
        self.assertEqual([r["id"] for r in backend.seen], ["good"])
        self.assertFalse(evidence["candidates"][0]["usable"])

    def test_embedding_baseline_keeps_only_nearest_candidate(self):
        query, _ = self.samples()
        rows = [{**profile(), "request_id": str(i), "descriptor": query}
                for i in range(2)]
        backend = Backend(["0", "1"])
        backend.rerank_api = "siliconflow"
        selected, _ = ContextRetriever(backend, rerank=False).select(query, rows)
        self.assertEqual([p["request_id"] for p in selected], ["0"])
        self.assertEqual(backend.seen, [])


if __name__ == "__main__":
    unittest.main()
