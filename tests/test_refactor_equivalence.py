"""Frozen requests/results and minimal terminal output; no network calls."""

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tool import calculate_metrics
from reporting import evaluate_ner
from main import main, print_summary, run_experiment
import main as pipeline


def stable_hash(value):
    def clean(item):
        if isinstance(item, dict):
            # Parser audit metadata is new; requests, raw replies, decisions and
            # entities must still match the frozen pre-refactor fixtures.
            metadata = {'elapsed_seconds', 'response_parser_version', 'outer_json_fence_removed'}
            return {key: clean(v) for key, v in item.items() if key not in metadata}
        if isinstance(item, list):
            return [clean(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ReplayClient:
    def __init__(self, replies):
        self.replies, self.calls = replies, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        item = self.replies[len(self.calls)]
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=item['raw_response']), finish_reason='stop')],
            usage=SimpleNamespace(model_dump=lambda: item['usage']) if item.get('usage') else None)


class RefactorEquivalenceTests(unittest.TestCase):
    def test_core_has_at_most_five_modules_of_200_lines(self):
        modules = list(Path(__file__).resolve().parents[1].glob('*.py'))
        self.assertLessEqual(len(modules), 5)
        for module in modules:
            with self.subTest(module=module.name):
                self.assertLessEqual(len(module.read_text().splitlines()), 200)

    def test_shared_scoring_does_not_require_the_api_sdk(self):
        # Fine-tuning servers use scoring/reporting without installing the API SDK.
        result = subprocess.run(
            [sys.executable, '-c',
             'import sys; sys.modules["openai"] = None; import tool; '
             'assert tool.calculate_metrics(1, 1, 1) == (1, 1, 1)'],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_recorded_requests_and_results_match_before_refactor(self):
        cases = json.loads((Path(__file__).parent / 'fixtures/expert_behavior.json').read_text())
        for case in cases:
            with self.subTest(case=case['name']):
                config = pipeline.prepare_config(case['config'])
                self.assertEqual(config['_prompt_hash'], case['prompt_hash'])
                client = ReplayClient(case['replies'])
                with patch('tool.time.sleep'):
                    result = pipeline.run_pipeline(case['sentence'], config, client, case['name'].split('/')[-1], {})
                self.assertEqual(len(client.calls), len(case['replies']))
                self.assertEqual(stable_hash(client.calls), case['request_hash'])
                self.assertEqual(stable_hash(result), case['result_hash'])

    def test_baseline_preserves_historical_arithmetic_and_empty_behavior(self):
        for tp, predicted, gold in [(0, 0, 0), (0, 0, 2), (0, 2, 0), (1, 3, 7),
                                    (347, 567, 632), (887, 1211, 1061), (269, 375, 420)]:
            p = tp / predicted if predicted else 0
            r = tp / gold if gold else 0
            f = 2 * p * r / (p + r) if p + r else 0
            self.assertEqual(calculate_metrics(tp, predicted, gold), (p, r, f))
        self.assertIsNone(evaluate_ner([])['f1'])

    def test_quiet_retries_preserve_attempt_logs_and_trace(self):
        from tool import call_agent
        config = pipeline.prepare_config(dict(dataset='ncbi', model_name='mock', retrieval_k=0,
                                              max_attempts=2, sleep_seconds=0, retry_delay=0))
        out, err = io.StringIO(), io.StringIO()
        logger = config['_attempt_logger'] = Mock()
        client = ReplayClient([{'raw_response': 'invalid'}, {'raw_response': '{"entities": []}'}])
        with redirect_stdout(out), redirect_stderr(err), patch('tool.time.sleep'):
            _, trace = call_agent('extract_clinical', {}, config, client, lambda data: data)
        self.assertEqual(trace['status'], 'ok')
        self.assertEqual([a['raw_response'] for a in trace['attempts']], ['invalid', '{"entities": []}'])
        self.assertEqual(logger.info.call_count, 2)
        self.assertEqual(out.getvalue() + err.getvalue(), '')

    def test_all_variants_keep_output_layout_and_compact_summary(self):
        # An empty response exercises cache sharing and every variant without an API service.
        replies = [{'raw_response': '{"entities": []}', 'usage': {'total_tokens': 10}}] * 20
        config = dict(dataset='ncbi', model_name='mock', retrieval_k=0, max_attempts=1,
                      test_file_path='bioDataset/ncbi/dev.sampled_500.json')
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'RESULT_ROOT', Path(tmp)), \
                redirect_stdout(out), redirect_stderr(err):
            results = run_experiment(config, 'all', limit=1, client=ReplayClient(replies))
            print_summary(results)
            self.assertEqual([r['variant'] for r in results], list(pipeline.VARIANTS))
            for row in results:
                folder = Path(row['result_dir'])
                for name in ['predictions.jsonl', 'run.log', 'run_config.json', 'metrics.json']:
                    self.assertTrue((folder / name).exists())
                self.assertEqual(row['paired_samples'], 1)
                self.assertIn(row['variant'], out.getvalue())
            self.assertTrue((Path(tmp) / 'summary_prf1.csv').exists())
        self.assertEqual(err.getvalue(), '')
        self.assertEqual(len(out.getvalue().splitlines()), len(pipeline.VARIANTS))
        self.assertIn('f1=', out.getvalue())
        self.assertNotIn('\033', out.getvalue())

    def test_cli_keeps_sentence_verbose_and_experiment_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'config.json'
            config_path.write_text(json.dumps({'agent_system': 'expert_agent'}))
            result = [{'variant': 'full', 'precision': 1., 'recall': 1., 'f1': 1., 'result_dir': tmp}]
            for verbose in (False, True):
                args = ['main.py', '--config', str(config_path), '--variant', 'full', '--limit', '3', '--resume']
                if verbose:
                    args.append('--verbose')
                out = io.StringIO()
                with patch('sys.argv', args), patch('main.run_experiment', return_value=result) as run, redirect_stdout(out):
                    main()
                self.assertEqual(run.call_args.args[1:], ('full', 3))
                self.assertEqual(run.call_args.kwargs, {'resume': True})
                if verbose:
                    self.assertEqual(json.loads(out.getvalue()), result)
                else:
                    self.assertEqual(len(out.getvalue().splitlines()), 1)
            out = io.StringIO()
            prediction = {'full': {'pred_entities': []}}
            with patch('sys.argv', ['main.py', '--config', str(config_path), '--sentence', 'healthy']), \
                    patch('main.predict_sentence', return_value=prediction) as predict, redirect_stdout(out):
                main()
            self.assertEqual(predict.call_args.args[0], 'healthy')
            self.assertEqual(json.loads(out.getvalue()), prediction)


if __name__ == '__main__':
    unittest.main()
