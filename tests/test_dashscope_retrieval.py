"""Native rerank boundaries and v4 batching without paid API calls."""

import copy
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from atsuite.workload.retrieval import ContextBackend


class DashScopeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (Path(__file__).resolve().parents[1]
             / "experiments/exp1/configs/retrieval-dashscope.json").read_text()
        )
        self.client = Mock()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-only"}), patch(
            "openai.OpenAI", return_value=self.client
        ):
            self.backend = ContextBackend(self.config)

    def test_embedding_batches_and_restores_order(self):
        def response(**kwargs):
            batch = kwargs["input"]
            return SimpleNamespace(
                usage=SimpleNamespace(total_tokens=len(batch)),
                data=[SimpleNamespace(index=i, embedding=[float(text)])
                      for i, text in reversed(list(enumerate(batch)))],
            )

        self.client.embeddings.create.side_effect = response
        self.assertEqual(self.backend.embed([str(i) for i in range(23)]),
                         [[float(i)] for i in range(23)])
        self.assertEqual([len(c.kwargs["input"])
                          for c in self.client.embeddings.create.call_args_list],
                         [10, 10, 3])
        self.assertAlmostEqual(self.backend.cost, 23 * 0.5 / 1e6)
        self.assertEqual(self.backend.embed([]), [])

    def test_partial_embedding_failure_keeps_known_cost(self):
        self.client.embeddings.create.side_effect = [
            SimpleNamespace(usage=SimpleNamespace(total_tokens=20),
                            data=[SimpleNamespace(index=i, embedding=[1.0])
                                  for i in range(10)]),
            RuntimeError("second batch failed"),
        ]
        with self.assertRaises(RuntimeError):
            self.backend.embed(["text"] * 11)
        self.assertAlmostEqual(self.backend.cost, 20 * 0.5 / 1e6)

    def test_native_indices_instruction_and_cost(self):
        self.client.post.return_value = {
            "usage": {"total_tokens": 200},
            "output": {"results": [{"index": 1, "relevance_score": 0.9},
                                    {"index": 0, "relevance_score": 0.1}]},
        }
        rows = self.backend.rerank({"prompt": "do not read the file"},
                                   [{"id": "a"}, {"id": "b"}])
        self.assertEqual([r["id"] for r in rows], ["b", "a"])
        self.assertEqual([r["utility"] for r in rows], [0.9, 0.1])
        self.assertAlmostEqual(self.backend.cost, 0.0001)
        request = self.client.post.call_args
        self.assertTrue(request.args[0].endswith("/services/rerank/text-rerank/text-rerank"))
        self.assertNotIn("Return JSON", request.kwargs["body"]["parameters"]["instruct"])
        self.client.chat.completions.create.assert_not_called()
        self.assertNotIn("test-only", str(self.backend.config))

    def test_rejects_incomplete_duplicate_invalid_results_and_missing_usage(self):
        valid = {"usage": {"total_tokens": 10},
                 "output": {"results": [{"index": 0, "relevance_score": 0.8},
                                         {"index": 1, "relevance_score": 0.2}]}}
        malformed = []
        for field, value in [("index", 0), ("index", True), ("index", 5),
                             ("relevance_score", float("nan")),
                             ("relevance_score", -0.1), ("relevance_score", 1.1)]:
            data = copy.deepcopy(valid)
            data["output"]["results"][1][field] = value
            malformed.append(data)
        data = copy.deepcopy(valid)
        data["output"]["results"].pop()
        malformed.extend([data, {"output": valid["output"]}])
        for response in malformed:
            with self.subTest(response=response):
                self.client.post.return_value = response
                with self.assertRaises(ValueError):
                    self.backend.rerank({}, [{"id": "a"}, {"id": "b"}])

    def test_empty_rerank_never_calls_api(self):
        self.assertEqual(self.backend.rerank({}, []), [])
        self.client.post.assert_not_called()

    def test_sdk_parses_native_http_response(self):
        import httpx
        from openai import OpenAI

        def respond(request):
            self.assertEqual(request.url.path,
                             "/api/v1/services/rerank/text-rerank/text-rerank")
            self.assertEqual(request.headers["Authorization"], "Bearer test-only")
            return httpx.Response(200, json={
                "usage": {"total_tokens": 20},
                "output": {"results": [{"index": 0, "relevance_score": 0.8}]},
            })

        with OpenAI(api_key="test-only", base_url=self.config["base_url"],
                    http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
            self.backend.client = client
            self.assertEqual(self.backend.rerank({}, [{"id": "a"}])[0]["id"], "a")


if __name__ == "__main__":
    unittest.main()
