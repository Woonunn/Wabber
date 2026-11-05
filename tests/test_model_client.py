"""Exercise provider requests through the real OpenAI SDK without paid API calls."""
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from openai import AsyncOpenAI

from benchmarks.DataSciBench.langgraph_agent.model_client import (
    DeepSeekClient, ModelClient, OpenAIClient, OpenAICompatibleClient, create_model_client,
)
from benchmarks.DataSciBench.langgraph_agent.trace_recorder import TraceRecorder
from benchmarks.DataSciBench.langgraph_agent.trace_replayer import ReplayModel

CONFIG = {'seed': 0, 'temperature': 0, 'top_p': 1, 'max_tokens': 128, 'retries': 0}
DEFINITION = {'name': 'lookup', 'parameters': {'type': 'object', 'properties': {
    'uid': {'type': 'string'}, 'query': {'type': 'string'}}}}


class ModelClientTests(unittest.IsolatedAsyncioTestCase):
    @contextmanager
    def client(self, overrides=None, *, message=None, finish='stop', responses=None):
        self.requests = []
        payload = {'id': 'fixture-response', 'object': 'chat.completion', 'created': 0,
                   'model': 'fixture-model', 'usage': {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10},
                   'choices': [{'index': 0, 'finish_reason': finish,
                                'message': message or {'role': 'assistant', 'content': '{"ok":true}'}}]}
        def handle(request):
            self.requests.append(request)
            if responses:
                status, body = responses.pop(0)
                return httpx.Response(status, json=body)
            return httpx.Response(200, json=payload)
        def sdk(**kwargs):
            return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'deepseek-fixture', 'OPENAI_API_KEY': 'openai-fixture',
            'GPT_OSS_API_KEY': 'oss-fixture',
        }), patch('benchmarks.DataSciBench.langgraph_agent.model_client.AsyncOpenAI', side_effect=sdk):
            self.directory = Path(directory)
            self.recorder = TraceRecorder(self.directory / 'capture', 'r', 's')
            model = create_model_client(self.recorder, {**CONFIG, **(overrides or {})})
            yield model

    async def test_default_deepseek_preserves_request_and_usage(self):
        with self.client() as model:
            self.assertIsInstance(model, DeepSeekClient)
            self.assertIsInstance(model, ModelClient)
            self.assertEqual(await model.decide('PLAN', 0, []), {'ok': True})
            body = json.loads(self.requests[0].content)
            self.assertEqual(self.requests[0].url.host, 'api.deepseek.com')
            self.assertEqual(body['model'], 'deepseek-flash')
            self.assertEqual(body['extra_body'] if 'extra_body' in body else body['thinking'], {'type': 'disabled'})
            self.assertEqual((body['temperature'], body['top_p'], body['max_tokens']), (0, 1, 128))
            self.assertEqual((self.recorder.events[-1]['input_tokens'], self.recorder.events[-1]['output_tokens']), (7, 3))
            await model.aclose()
            self.assertTrue(model.client.is_closed())

    async def test_openai_does_not_send_deepseek_or_sampling_parameters_to_reasoning_model(self):
        with self.client({'provider': 'openai', 'name': 'gpt-5.4-mini', 'reasoning_effort': 'low'}) as model:
            self.assertIsInstance(model, OpenAIClient)
            await model.decide('PLAN', 0, [])
            body = json.loads(self.requests[0].content)
            self.assertEqual(str(self.requests[0].url), 'https://api.openai.com/v1/chat/completions')
            self.assertEqual(body['max_completion_tokens'], 128)
            self.assertEqual(body['reasoning_effort'], 'low')
            for key in ('thinking', 'max_tokens', 'temperature', 'top_p'):
                self.assertNotIn(key, body)
            await model.aclose()

    async def test_openai_nonreasoning_sampling(self):
        with self.client({'provider': 'openai', 'name': 'gpt-5.4-mini', 'reasoning_effort': 'none'}) as model:
            await model.decide('PLAN', 0, [])
            body = json.loads(self.requests[0].content)
            self.assertEqual((body['temperature'], body['top_p']), (0, 1))
            await model.aclose()

    async def test_gpt_oss_uses_explicit_endpoint_and_dedicated_key(self):
        with self.client({'provider': 'openai_compatible', 'name': 'openai/gpt-oss-20b',
                          'base_url': 'https://inference.example/v1', 'api_key_env': 'GPT_OSS_API_KEY',
                          'extra_body': {'provider': {'only': ['coreweave'], 'allow_fallbacks': False}}}) as model:
            self.assertIsInstance(model, OpenAICompatibleClient)
            await model.decide('PLAN', 0, [])
            body = json.loads(self.requests[0].content)
            self.assertEqual(self.requests[0].url.host, 'inference.example')
            self.assertEqual(self.requests[0].headers['authorization'], 'Bearer oss-fixture')
            self.assertEqual(body['model'], 'openai/gpt-oss-20b')
            self.assertEqual(body['provider'], {'only': ['coreweave'], 'allow_fallbacks': False})
            self.assertEqual(body['max_tokens'], 128)
            self.assertNotIn('thinking', body)
            await model.aclose()

    async def test_native_tools_and_final_answer_without_mutating_definitions(self):
        for function, arguments, expected in (
            ('lookup', {'query': 'x'}, {'tool': 'lookup', 'args': {'query': 'x'}}),
            ('submit_result', {'answer': {'ok': True}}, {'answer': {'ok': True}}),
        ):
            with self.subTest(function=function), self.client(
                {'provider': 'openai', 'name': 'gpt-5.4-mini'}, finish='tool_calls',
                message={'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': 'call_fixture', 'type': 'function', 'function': {
                        'name': function, 'arguments': json.dumps(arguments)}}]},
            ) as model:
                model.tool_definitions = [DEFINITION]
                self.assertEqual(await model.decide('ACT', 0, []), expected)
                body = json.loads(self.requests[0].content)
                self.assertEqual(body['tool_choice'], 'required')
                self.assertFalse(body['parallel_tool_calls'])
                self.assertNotIn('response_format', body)
                self.assertNotIn('uid', body['tools'][0]['function']['parameters']['properties'])
                self.assertIn('uid', DEFINITION['parameters']['properties'])
                await model.aclose()

    async def test_openai_capture_replays_without_live_api_call(self):
        with self.client({'provider': 'openai', 'name': 'gpt-5.4-mini'}) as model:
            messages = [{'role': 'user', 'content': 'Return JSON'}]
            decision = await model.decide('PLAN', 0, messages)
            self.recorder.finish({'status': 'complete'})
            replay = ReplayModel(self.directory / 'capture', TraceRecorder(self.directory / 'replay', 'r2', 's2'))
            with patch('asyncio.sleep', new_callable=AsyncMock):
                self.assertEqual(await replay.decide('PLAN', 0, messages), decision)
            replay.assert_consumed()
            self.assertEqual(len(self.requests), 1)
            await model.aclose()

    async def test_invalid_decision_and_incomplete_output_still_fail(self):
        for content, finish in (('[]', 'stop'), ('{"ok":true}', 'length'), ('not json', 'stop')):
            with self.subTest(content=content, finish=finish), self.client(
                {'provider': 'openai', 'name': 'gpt-5.4-mini'}, finish=finish,
                message={'role': 'assistant', 'content': content},
            ) as model:
                with self.assertRaises(ValueError):
                    await model.decide('PLAN', 0, [])
                self.assertEqual(self.recorder.events[-1]['kind'], 'llm_error')
                await model.aclose()

    async def test_nonretryable_error_redacts_api_key(self):
        with self.client({'provider': 'openai', 'name': 'gpt-5.4-mini', 'retries': 2},
                         responses=[(400, {'error': {'message': 'bad openai-fixture', 'type': 'invalid_request_error'}})]) as model:
            with self.assertRaises(Exception):
                await model.decide('PLAN', 0, [])
            self.assertEqual(len(self.requests), 1)
            event = self.recorder.events[-1]
            self.assertEqual(event['status_code'], 400)
            self.assertNotIn('openai-fixture', event['error'])
            await model.aclose()

    def test_abstract_and_configuration_errors_fail_before_network(self):
        with self.assertRaises(TypeError):
            ModelClient(None, CONFIG, api_key_env='OPENAI_API_KEY', base_url='https://api.openai.com/v1')
        for config, error in (
            ({'provider': 'typo'}, 'Unknown model provider'),
            ({'provider': 'openai'}, 'explicit model name'),
            ({'provider': 'openai', 'name': 'gpt-oss-20b'}, 'gpt-oss requires'),
            ({'provider': 'openai_compatible', 'name': 'gpt-oss-20b'}, 'explicit base_url'),
        ):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, error):
                create_model_client(None, {**CONFIG, **config})
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'OPENAI_API_KEY'):
            create_model_client(None, {**CONFIG, 'provider': 'openai', 'name': 'gpt-5.4-mini'})
