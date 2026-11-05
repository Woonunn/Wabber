"""SiliconFlow native response validation and SDK wiring without network calls."""

import copy
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from atsuite.workload.retrieval import ContextBackend


class SiliconFlowTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (Path(__file__).resolve().parents[1]
             / "experiments/exp1/configs/retrieval-siliconflow.json").read_text()
        )
        self.response = {
            "meta": {"tokens": {"input_tokens": 22, "output_tokens": 0}},
            "results": [{"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.1}],
        }
        self.requests = []

        def respond(request):
            self.requests.append(request)
            return httpx.Response(200, content=json.dumps(self.response),
                                  headers={"content-type": "application/json"})

        from openai import OpenAI
        self.client = OpenAI(
            api_key="test-only", base_url=self.config["base_url"],
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
            max_retries=0,
        )
        self.addCleanup(self.client.close)
        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-only"}), patch(
            "openai.OpenAI", return_value=self.client
        ):
            self.backend = ContextBackend(self.config)

    def test_native_http_mapping_and_free_cost(self):
        rows = self.backend.rerank({"prompt": "中文查询"}, [{"id": "a"}, {"id": "b"}])
        self.assertEqual([r["id"] for r in rows], ["b", "a"])
        self.assertEqual([r["utility"] for r in rows], [0.9, 0.1])
        self.assertEqual(self.backend.cost, 0)
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://api.siliconflow.cn/v1/rerank")
        self.assertEqual(request.headers["authorization"], "Bearer test-only")
        body = json.loads(request.content)
        self.assertEqual(json.loads(body["query"]), {"prompt": "中文查询"})
        self.assertEqual([json.loads(d) for d in body["documents"]], [{"id": "a"}, {"id": "b"}])
        self.assertEqual(body["model"], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(body["top_n"], 2)
        self.assertNotIn("instruction", body)
        self.assertNotIn("test-only", str(self.backend.config))

    def test_accounts_nonzero_configured_rate(self):
        self.backend.config["rates"]["input"] = 0.5
        self.backend.rerank({}, [{"id": "a"}, {"id": "b"}])
        self.assertAlmostEqual(self.backend.cost, 22 * 0.5 / 1e6)

    def test_rejects_bad_results_and_missing_usage_even_when_free(self):
        valid = copy.deepcopy(self.response)
        malformed = []
        for field, value in [
            ("index", 1), ("index", True), ("index", 2),
            ("relevance_score", float("nan")), ("relevance_score", -0.1),
            ("relevance_score", 1.1),
        ]:
            data = copy.deepcopy(valid)
            data["results"][1][field] = value
            malformed.append(data)
        malformed.extend([
            {"results": valid["results"]},
            {"meta": valid["meta"], "results": valid["results"][:1]},
            {"meta": valid["meta"], "results": None},
        ])
        for response in malformed:
            with self.subTest(response=response):
                self.response = response
                with self.assertRaises(ValueError):
                    self.backend.rerank({}, [{"id": "a"}, {"id": "b"}])

    def test_empty_candidates_do_not_call_service(self):
        self.assertEqual(self.backend.rerank({}, []), [])
        self.assertEqual(self.requests, [])

    def test_rate_limit_waits_one_minute_and_only_successful_tokens_are_charged(self):
        from openai import RateLimitError

        rejected = RateLimitError("limit", response=httpx.Response(429, request=httpx.Request("POST", "https://test/rerank")), body={})
        self.backend.config["rates"]["input"] = .5
        with patch.object(self.client, "post", side_effect=[rejected, self.response]), patch(
            "atsuite.workload.retrieval.time.sleep"
        ) as sleep:
            rows = self.backend.rerank({}, [{"id": "a"}, {"id": "b"}])
        sleep.assert_called_once_with(60)
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.backend.rate_limit_retries, 1)
        self.assertEqual(self.backend.rate_limit_wait_s, 60)
        self.assertAlmostEqual(self.backend.cost, 22 * .5 / 1e6)

    def test_repeated_rate_limits_split_candidates_without_losing_history(self):
        from openai import RateLimitError

        rejected = RateLimitError("limit", response=httpx.Response(429, request=httpx.Request("POST", "https://test/rerank")), body={})
        one = {"meta": {"tokens": {"input_tokens": 10}}, "results": [{"index": 0, "relevance_score": .8}]}
        self.backend.config["rates"]["input"] = .5
        with patch.object(self.client, "post", side_effect=[rejected, rejected, one, one]) as post, patch(
            "atsuite.workload.retrieval.time.sleep"
        ) as sleep:
            rows = self.backend.rerank({}, [{"id": "a"}, {"id": "b"}])
        self.assertEqual(sleep.call_args_list, [unittest.mock.call(60), unittest.mock.call(60)])
        self.assertEqual({r["id"] for r in rows}, {"a", "b"})
        self.assertEqual([len(c.kwargs["body"]["documents"]) for c in post.call_args_list], [2, 2, 1, 1])
        self.assertEqual(self.backend.rate_limit_retries, 2)
        self.assertAlmostEqual(self.backend.cost, 20 * .5 / 1e6)


if __name__ == "__main__":
    unittest.main()
