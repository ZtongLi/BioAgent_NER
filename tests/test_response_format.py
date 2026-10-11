import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from baseline.llm import LLMExperiment
from tool import unwrap_json_fence
from tool import call_agent


class ResponseFormatTests(unittest.TestCase):
    def test_only_whole_response_fences_are_removed(self):
        body = ' {"entities": []} '
        for tag in ('json', 'JSON', ''):
            self.assertEqual(unwrap_json_fence('```' + tag + '\n' + body + '\n```'), body)
        for raw in ('Explanation\n```json\n[]\n```', '```json\n[]\n```\nExtra',
                    '```json\n[]\n```\n```json\n[]\n```', '```python\n[]\n```',
                    '```json\n[]', '[]', None):
            self.assertEqual(unwrap_json_fence(raw), raw)
        with self.assertRaises(ValueError):
            json.loads(unwrap_json_fence('```json\n[]\n[]\n```'))

    def test_baseline_still_rejects_invalid_entities(self):
        runner = LLMExperiment.__new__(LLMExperiment)
        runner.config = {'dataset': 'ncbi', 'sleep_seconds': 0, 'retry_delay': 0}
        bad = '[{"text":"absent","type":"DISEASE","occurrence":0}]'
        good = '[{"text":"cancer","type":"DISEASE","occurrence":0}]'
        with patch('baseline.llm.call_llm', side_effect=[f'```json\n{bad}\n```',
                                                       f'```json\n{good}\n```']), \
                patch('baseline.llm.time.sleep'):
            self.assertEqual(runner._predict('cancer'), {(0, 6, 'DISEASE')})
            self.assertEqual(runner.last_attempts, 2)
            self.assertTrue(runner.last_outer_json_fence_removed)

    def test_expert_keeps_raw_and_runs_validator(self):
        raw = '```json\n{"entities": []}\n```'
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                                   message=SimpleNamespace(content=raw))], usage=None)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response)))
        config = {'_prompts': {'extract': 'test'}, 'model_name': 'mock',
                  'max_attempts': 1, 'sleep_seconds': 0}
        def validate(value):
            self.assertEqual(value, {'entities': []})
            return value['entities']
        output, trace = call_agent('extract', {}, config, client, validate)
        self.assertEqual(output, [])
        self.assertEqual(trace['attempts'][0]['raw_response'], raw)
        self.assertTrue(trace['attempts'][0]['outer_json_fence_removed'])
        def reject(value):
            raise ValueError('semantic constraint')
        output, trace = call_agent('extract', {}, config, client, reject)
        self.assertIsNone(output)
        self.assertEqual(trace['status'], 'failed')
