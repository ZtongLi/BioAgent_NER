import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import run_experiment
import main as pipeline
from main import read_checkpoint
from test_refactor_equivalence import ReplayClient


class CheckpointTests(unittest.TestCase):
    def test_interrupted_full_matches_continuous_and_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(dataset="ncbi", model_name="mock", retrieval_k=0,
                          max_attempts=1, result_root=str(root / "runs"),
                          summary_csv=str(root / "summary.csv"),
                          test_file_path="bioDataset/ncbi/dev.sampled_500.json")
            def client():
                return ReplayClient([{"raw_response": '{"entities": []}', "usage": {}}] * 40)
            original = pipeline.run_pipeline
            calls = []
            def interrupted(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return original(*args, **kwargs)
            with patch.object(pipeline, "run_pipeline", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    run_experiment(config, limit=3, client=client(), resume=True)
            saved = next((root / "runs").glob("*/*/full/*/predictions.jsonl"))
            prefix = saved.read_bytes()
            self.assertEqual(len(prefix.splitlines()), 1)
            with patch.object(pipeline, "run_pipeline", wraps=original) as run:
                resumed = run_experiment(config, limit=3, client=client(), resume=True)[0]
                self.assertEqual(run.call_count, 2)
            self.assertTrue(saved.read_bytes().startswith(prefix))
            self.assertEqual(Path(resumed["result_dir"]), saved.parent)
            with patch("main.make_client") as api, patch.object(pipeline, "run_pipeline") as run:
                run_experiment(config, limit=3, resume=True)
                api.assert_not_called()
                run.assert_not_called()
            with (root / "summary.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 1)
            before = saved.read_bytes()
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                run_experiment({**config, "temperature": .5}, limit=3, client=client(), resume=True)
            self.assertEqual(saved.read_bytes(), before)
            continuous = run_experiment({**config, "result_root": str(root / "continuous")},
                                        limit=3, client=client())[0]
            for key in ("precision", "recall", "f1", "true_positive", "false_positive", "false_negative", "api_calls"):
                self.assertEqual(resumed[key], continuous[key])

    def test_resume_rejects_historical_shards_without_changing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(dataset="ncbi", model_name="mock", retrieval_k=0,
                          max_attempts=1, result_root=str(root / "runs"),
                          summary_csv=str(root / "summary.csv"),
                          test_file_path="bioDataset/ncbi/dev.sampled_500.json")
            client = ReplayClient([{"raw_response": '{"entities": []}'}] * 3)
            result = run_experiment(config, limit=1, client=client)[0]
            directory = Path(result["result_dir"])
            shards = directory / "shards"
            shards.mkdir()
            (shards / "manifest.json").write_text('{}')
            paths = [directory / name for name in ("predictions.jsonl", "run_config.json", "metrics.json", "run.log")]
            paths += [shards / "manifest.json", root / "summary.csv"]
            before = {path: path.read_bytes() for path in paths}
            with patch("main.make_client") as api, patch.object(pipeline, "run_pipeline") as run:
                with self.assertRaisesRegex(RuntimeError, "This run is sharded"):
                    run_experiment(config, limit=1, resume=True)
                api.assert_not_called()
                run.assert_not_called()
            self.assertEqual({path: path.read_bytes() for path in paths}, before)

    def test_partial_tail_is_recoverable_but_interior_damage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            data = [{"sentence": "healthy", "entities": []}]
            row = {"sample_id": 0, **data[0], "gold_entities": [], "pred_entities": [],
                   "fully_executed": True, "stages": [{"status": "ok"}]}
            prefix = (json.dumps(row) + "\n").encode()
            path = directory / "predictions.jsonl"
            path.write_bytes(prefix + b'{"sample_id":')
            records, raw, retained = read_checkpoint(directory, data)
            self.assertEqual(len(records), 1)
            self.assertEqual(retained, prefix)
            self.assertEqual(path.read_bytes(), raw)
            path.write_bytes(b'broken\n' + prefix)
            with self.assertRaises(ValueError):
                read_checkpoint(directory, data)
