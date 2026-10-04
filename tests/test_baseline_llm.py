import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from baseline.llm import LLMExperiment, calculate_metrics, normalize_pred_entities, load_resume
from experiment import evaluate_ner


class BaselineEvaluationTests(unittest.TestCase):
    def test_invalid_attempts_retry_until_valid_empty(self):
        runner = LLMExperiment.__new__(LLMExperiment)
        runner.config = {'dataset': 'ncbi', 'sleep_seconds': 0, 'retry_delay': 0}
        replies = [TimeoutError(), None, 'broken', '{}', '[{}]',
                   '[{"text":"absent","type":"DISEASE"}]', '[]']
        with patch('baseline.llm.call_llm', side_effect=replies) as call, \
                patch('baseline.llm.time.sleep'):
            self.assertEqual(runner._predict('cancer'), set())
            self.assertEqual(call.call_count, len(replies))
            self.assertEqual(runner.last_attempts, len(replies))

    def test_exact_quotes_types_and_repeated_mentions(self):
        entity = {'text': 'cancer', 'type': 'DISEASE', 'start': 7, 'end': 13}
        self.assertEqual(normalize_pred_entities([entity, entity], 'cancer cancer', 'ncbi'),
                         {(7, 13, 'DISEASE')})
        for items in [[dict(entity, start=None)], [dict(entity, type='GENE')],
                      [entity, {}], [dict(entity, text='Cancer')]]:
            with self.assertRaises(ValueError):
                normalize_pred_entities(items, 'cancer cancer', 'ncbi')

    def test_occurrence_handles_repeated_substrings(self):
        sentence = 'ALKs express ALK and ALK'
        entity = {'text': 'ALK', 'type': 'GENE', 'occurrence': 2}
        self.assertEqual(normalize_pred_entities([entity], sentence, 'bc2gm'),
                         {(21, 24, 'GENE')})
        with self.assertRaises(ValueError):
            normalize_pred_entities([dict(entity, occurrence=3)], sentence, 'bc2gm')

    def test_resume_appends_only_missing_samples_and_restores_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = [{'sentence': 'cancer', 'entities': [{'pos': [0, 6], 'type': 'DISEASE'}]},
                    {'sentence': 'healthy', 'entities': []}]
            (root / 'data.json').write_text(json.dumps(data))
            config = {'dataset': 'ncbi', 'model_name': 'fake', 'experiment_name': 'test',
                      'max_loop': 2, 'api_keys': 'fake', 'test_file_path': str(root / 'data.json'),
                      'save_file_path': str(root / 'out.jsonl'), 'log_file_path': str(root / 'run.log')}
            cp = root / 'config.json'
            cp.write_text(json.dumps(config))
            row = {'sample_id': 0, 'fully_executed': True, 'sentence': 'cancer',
                   'evaluation_scope': 'successful_samples_only',
                   'gold_entities': [[0, 6, 'DISEASE']], 'pred_entities': [[0, 6, 'DISEASE']]}
            output = root / 'out.jsonl'
            original = json.dumps(row) + '\n'
            output.write_text(original)
            runner = LLMExperiment(cp, resume=True)
            runner.last_attempts = 1
            with patch.object(runner, '_predict', return_value=set()) as predict, \
                    patch('baseline.llm.setup_logging'), patch('baseline.llm.save_summary') as summary:
                runner.run()
                predict.assert_called_once_with('healthy')
                self.assertEqual(summary.call_args.args[1:], (1.0, 1.0, 1.0))
                predict.reset_mock()
                runner.run()
                predict.assert_not_called()
            self.assertTrue(output.read_text().startswith(original))
            self.assertEqual(len(output.read_text().splitlines()), 2)
            data[0]['sentence'] = 'changed'
            with self.assertRaises(ValueError):
                load_resume(output, data, config)
            self.assertTrue(output.read_text().startswith(original))

    def test_retry_includes_previous_response_and_exact_occurrence_table(self):
        runner = LLMExperiment.__new__(LLMExperiment)
        runner.config = {'dataset': 'bc5cdr', 'sleep_seconds': 0, 'retry_delay': 0}
        sentence = 'normal sodium and low sodium'
        invalid = json.dumps([{'text': 'sodium', 'type': 'CHEMICAL', 'occurrence': 2}])
        valid = json.dumps([{'text': 'sodium', 'type': 'CHEMICAL', 'occurrence': i} for i in (0, 1)])
        with patch('baseline.llm.call_llm', side_effect=[invalid, valid]) as call, \
                patch('baseline.llm.time.sleep'):
            self.assertEqual(runner._predict(sentence), {(7, 13, 'CHEMICAL'), (22, 28, 'CHEMICAL')})
            prompt = call.call_args.args[0]
            self.assertIn('allowed integer indices=[0, 1]', prompt)
            self.assertIn(invalid, prompt)
            self.assertIn('"occurrence": 1, "start": 22', prompt)

    def test_micro_prf_matches_expert_evaluator(self):
        gold = [{'start': 0, 'end': 6, 'type': 'DISEASE'},
                {'start': 7, 'end': 13, 'type': 'DISEASE'}]
        pred = [gold[0], {'start': 7, 'end': 12, 'type': 'DISEASE'}]
        row = {'fully_executed': True, 'gold_entities': gold, 'pred_entities': pred,
               'stages': [{'stage': 'baseline', 'status': 'ok',
                           'input_entities': [], 'output_entities': pred}]}
        metrics = evaluate_ner([row])
        self.assertEqual(calculate_metrics(1, 2, 2),
                         tuple(metrics[k] for k in ('precision', 'recall', 'f1')))
        self.assertEqual(calculate_metrics(0, 0, 2), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
