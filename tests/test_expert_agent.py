import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from expert_agent import aggregation, pipeline, voting
from expert_agent.experiment import run_experiment
from experiment import evaluate_ner


class FakeClient:
    def __init__(self, empty=False, fail_stage=None):
        self.calls = []
        self.empty = empty
        self.fail_stage = fail_stage
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        messages = kwargs['messages']
        prompt = messages[0]['content']
        payload = json.loads(messages[1]['content'])
        self.calls.append((prompt, payload))
        if 'final aggregation agent' in prompt:
            stage = 'aggregation'
            data = {'decisions': [dict(id=c['id'], action='keep', reason='explicit mention',
                                      evidence_text=c['text'], rule_id=0) for c in payload['candidates']]}
        elif 'Independently review EVERY' in prompt:
            stage = 'vote'
            data = {'decisions': [dict(id=c['id'], vote='support', reason='explicit mention',
                                      evidence_text=c['text']) for c in payload['candidates']]}
        else:
            stage = 'extract'
            data = {'entities': [] if self.empty else [dict(text='cancer', type='DISEASE', occurrence=0,
                                                           reason='explicit disease', evidence_text='cancer')]}
        if stage == self.fail_stage:
            data = {}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)),
                                                        finish_reason='stop')],
                               usage=SimpleNamespace(model_dump=lambda: {'total_tokens': 10}))


class ExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = pipeline.prepare_config(dict(dataset='ncbi', model_name='mock', retrieval_k=0,
                                                 max_attempts=1, sleep_seconds=0, retry_delay=0))

    def test_full_and_input_isolation(self):
        client = FakeClient()
        result = pipeline.run_pipeline('cancer occurs.', self.config, client)
        self.assertTrue(result['fully_executed'])
        self.assertEqual(len(client.calls), 7)
        self.assertEqual(result['vote_counts'][0]['support'], 3)
        self.assertEqual(result['pred_entities'][0]['start'], 0)
        for _, payload in client.calls[:3]:
            self.assertNotIn('candidates', payload)
        for _, payload in client.calls[3:6]:
            self.assertNotIn('ballots', payload)
            self.assertNotIn('sources', payload['candidates'][0])
        self.assertNotIn('gold', json.dumps(client.calls))

    def test_variants_share_extraction_and_ballots(self):
        client, cache = FakeClient(), {}
        for name in ('single_expert', 'candidate_union', 'vote_only', 'full', 'without_voting'):
            result = pipeline.run_pipeline('cancer occurs.', self.config, client, name, cache)
            self.assertTrue(result['fully_executed'])
        self.assertEqual(len(client.calls), 8)  # 3 extraction + 3 ballot + 2 aggregation
        self.assertEqual(client.calls[-1][1]['ballots'], {})
        self.assertEqual(client.calls[-1][1]['vote_counts'], [])

    def test_generic_repeats_have_independent_cache_slots(self):
        client = FakeClient()
        pipeline.run_pipeline('cancer occurs.', self.config, client, 'generic_ensemble', {})
        self.assertEqual(len(client.calls), 7)
        self.assertEqual(client.calls[0][0], client.calls[1][0])
        self.assertEqual(client.calls[1][0], client.calls[2][0])

    def test_without_retrieval_removes_examples_from_every_request(self):
        config = {**self.config, 'retrieval_k': 4}
        def retrieve(sentence, candidates, cfg):
            return [{'id': 'train:0', 'sentence': 'Example.', 'entities': []}] if cfg['retrieval_k'] else []
        client, cache = FakeClient(), {}
        with patch('expert_agent.pipeline.retrieve_examples', side_effect=retrieve):
            pipeline.run_pipeline('cancer occurs.', config, client, 'full', cache)
            pipeline.run_pipeline('cancer occurs.', config, client, 'without_retrieval', cache)
        self.assertEqual(len(client.calls), 14)
        self.assertTrue(all(p['training_examples'] for _, p in client.calls[:7]))
        self.assertTrue(all(p['training_examples'] == [] for _, p in client.calls[7:]))

    def test_empty_pool(self):
        client = FakeClient(empty=True)
        result = pipeline.run_pipeline('No target.', self.config, client)
        self.assertTrue(result['fully_executed'])
        self.assertEqual(result['pred_entities'], [])
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(result['stages'][-1]['status'], 'skipped_empty')

    def test_failures_are_not_abstentions_or_predictions(self):
        for stage in ('extract', 'vote', 'aggregation'):
            with self.subTest(stage=stage):
                client = FakeClient(fail_stage=stage)
                result = pipeline.run_pipeline('cancer occurs.', self.config, client)
                self.assertFalse(result['fully_executed'])
                self.assertEqual(result['pred_entities'], [])
                metrics = evaluate_ner([{**result, 'gold_entities': [{'pos': [0, 6], 'type': 'DISEASE'}]}])
                self.assertEqual(metrics['evaluated_samples'], 0)
                self.assertEqual(metrics['false_negative'], 0)

    def test_three_exhausted_attempts_are_excluded_but_valid_empty_is_scored(self):
        config = {**self.config, 'max_attempts': 3}
        failed = pipeline.run_pipeline('cancer occurs.', config, FakeClient(fail_stage='extract'))
        self.assertEqual(len(failed['stages'][0]['attempts']), 3)
        valid_empty = pipeline.run_pipeline('cancer occurs.', config, FakeClient(empty=True))
        gold = [{'pos': [0, 6], 'type': 'DISEASE'}]
        metrics = evaluate_ner([{**row, 'gold_entities': gold} for row in (failed, valid_empty)])
        self.assertEqual(metrics['num_samples'], 2)
        self.assertEqual(metrics['failed_samples'], 1)
        self.assertEqual(metrics['evaluated_samples'], 1)
        self.assertEqual(metrics['false_negative'], 1)

    def test_success_after_retry_is_evaluated_only_once(self):
        class Recovers(FakeClient):
            def create(self, **kwargs):
                self.fail_stage = 'extract' if not self.calls else None
                return super().create(**kwargs)
        result = pipeline.run_pipeline('cancer occurs.', {**self.config, 'max_attempts': 3}, Recovers())
        self.assertTrue(result['fully_executed'])
        self.assertEqual(len(result['stages'][0]['attempts']), 2)
        metrics = evaluate_ner([{**result, 'gold_entities': [{'pos': [0, 6], 'type': 'DISEASE'}]}])
        self.assertEqual(metrics['evaluated_samples'], 1)
        self.assertEqual(metrics['true_positive'], 1)
        self.assertEqual(metrics['false_positive'], 0)
        self.assertEqual(metrics['false_negative'], 0)

    def test_unlimited_retry_survives_more_than_three_invalid_responses(self):
        class RecoversLate(FakeClient):
            def create(self, **kwargs):
                self.fail_stage = 'extract' if len(self.calls) < 4 else None
                return super().create(**kwargs)
        result = pipeline.run_pipeline('cancer occurs.', {**self.config, 'max_attempts': None}, RecoversLate())
        self.assertTrue(result['fully_executed'])
        self.assertEqual(len(result['stages'][0]['attempts']), 5)
        metrics = evaluate_ner([{**result, 'gold_entities': [{'pos': [0, 6], 'type': 'DISEASE'}]}])
        self.assertEqual(metrics['evaluated_samples'], 1)
        self.assertEqual(metrics['true_positive'], 1)

    def test_unlimited_http_retries_and_manual_cancellation(self):
        import httpx
        from openai import APIStatusError
        from tool import call_agent
        client = FakeClient()
        error = APIStatusError('test', response=httpx.Response(401, request=httpx.Request('POST', 'https://example.test')),
                               body=None)
        response = client.create(messages=[{'content': 'extract'}, {'content': '{}'}])
        cfg = {**self.config, 'max_attempts': None}
        with patch.object(client.chat.completions, 'create', side_effect=[error] * 4 + [response]):
            output, trace = call_agent('extract_clinical', {}, cfg, client, lambda data: data)
        self.assertEqual(trace['status'], 'ok')
        self.assertFalse(trace['fatal'])
        self.assertEqual(len(trace['attempts']), 5)
        with patch.object(client.chat.completions, 'create', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                call_agent('extract_clinical', {}, cfg, client, lambda data: data)

    def test_votes_abstention_majority_and_missing_ballots(self):
        candidates = [{'id': 0}]
        def ballot(v):
            return [{'id': 0, 'vote': v}]
        for votes, status in [(('support', 'support', 'abstain'), 'accepted'),
                              (('support', 'oppose', 'abstain'), 'unresolved'),
                              (('oppose', 'oppose', 'support'), 'rejected')]:
            self.assertEqual(voting.tally(candidates, dict(zip('abc', map(ballot, votes))))[0]['status'], status)
        with self.assertRaises(ValueError):
            voting.tally(candidates, {'a': ballot('support')})
        with self.assertRaises(ValueError):
            voting.tally(candidates, {'a': ballot('support'), 'b': [], 'c': ballot('support')})

    def test_overlap_and_majority_constraints(self):
        cs = [dict(id=0, start=0, end=10), dict(id=1, start=0, end=6), dict(id=2, start=11, end=17)]
        pairs = voting.conflict_pairs(cs, 'forbid')
        self.assertEqual(pairs, [[0, 1]])
        self.assertEqual(voting.conflict_pairs(cs, 'allow_nested'), [])
        counts = [dict(id=i, status='accepted') for i in range(3)]
        self.assertEqual(voting.vote_selection(counts, pairs), {2})
        self.assertEqual(voting.constraints(counts, pairs), {2: 'keep'})
        counts[1]['status'] = 'rejected'
        self.assertEqual(voting.constraints(counts, pairs), {0: 'keep', 1: 'drop', 2: 'keep'})

    def test_aggregation_rejects_overrides_conflicts_and_new_ids(self):
        payload = dict(sentence='cancer', candidates=[dict(id=0, text='cancer')],
                       training_examples=[], locked_decisions={0: 'keep'}, conflict_pairs=[])
        def validate_with(data):
            def call(stage, request, config, client, validate):
                return validate(data)
            with patch('expert_agent.aggregation.call_agent', side_effect=call):
                return aggregation.run(payload, self.config, None)
        decision = dict(id=0, action='drop', reason='test', evidence_text='cancer', rule_id=0)
        with self.assertRaises(ValueError):
            validate_with({'decisions': [decision]})
        decision.update(id=1, action='keep')
        with self.assertRaises(ValueError):
            validate_with({'decisions': [decision]})
        payload.update(candidates=[dict(id=0), dict(id=1)], locked_decisions={}, conflict_pairs=[[0, 1]])
        with self.assertRaises(ValueError):
            validate_with({'decisions': [{**decision, 'id': i} for i in range(2)]})

    def test_repeated_mentions_and_alternative_types_are_distinct(self):
        from tool import entity_list, tokenize
        sentence = 'cancer and cancer'
        entities = entity_list({'entities': [dict(text='cancer', type='DISEASE', occurrence=i) for i in [0, 1]]},
                               sentence, tokenize(sentence), ['DISEASE'])
        self.assertEqual([e['start'] for e in entities], [0, 11])
        pairs = voting.conflict_pairs([dict(id=0, start=0, end=6), dict(id=1, start=0, end=6)], 'allow_nested')
        self.assertEqual(pairs, [[0, 1]])

    def test_runner_writes_isolated_results_and_redacts_config(self):
        config = dict(dataset='ncbi', model_name='mock', retrieval_k=0, max_attempts=1,
                      test_file_path='bioDataset/ncbi/dev.sampled_500.json', api_keys='secret-test-key')
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'RESULT_ROOT', Path(tmp)):
            results = run_experiment(config, 'compare', limit=1, client=FakeClient(empty=True))
            self.assertEqual(len(results), 2)
            self.assertTrue((Path(tmp) / 'summary_prf1.csv').exists())
            for row in results:
                directory = Path(row['result_dir'])
                self.assertIn('expert_v1', (directory / 'run_config.json').read_text())
                self.assertNotIn('secret-test-key', (directory / 'run_config.json').read_text())
                self.assertEqual(row['paired_samples'], 1)
                self.assertEqual(row['failed_samples'], 0)
                self.assertTrue((directory / 'predictions.jsonl').exists())
                self.assertTrue((directory / 'run.log').exists())

    def test_three_dataset_configs_and_legacy_backend(self):
        from tool import prepare_config
        for dataset in ('bc2gm', 'bc5cdr', 'ncbi'):
            raw = dict(dataset=dataset, model_name='mock', retrieval_k=0)
            new = pipeline.prepare_config(raw)
            self.assertEqual(len(new['_prompts']), 13)
            old = prepare_config(raw)
            self.assertEqual(old['system_version'], 'unified_v2')
            self.assertEqual(set(old['_prompts']), {'extraction', 'discovery', 'boundary', 'verification'})

    def test_legacy_runner_still_uses_original_layout(self):
        from experiment import run_experiment as legacy_run
        config = dict(agent_system='ner_agent', dataset='ncbi', model_name='mock', retrieval_k=0,
                      max_attempts=1, test_file_path='bioDataset/ncbi/dev.sampled_500.json')
        with tempfile.TemporaryDirectory() as tmp, patch('experiment.RESULT_ROOT', Path(tmp)):
            rows = legacy_run(config, 'compare', limit=1, client=FakeClient(empty=True))
            self.assertEqual([row['variant'] for row in rows], ['extract_only', 'full'])
            self.assertTrue(all(row['system_version'] == 'unified_v2' for row in rows))
            self.assertTrue(all(row['paired_samples'] == 1 for row in rows))


if __name__ == '__main__':
    unittest.main()
