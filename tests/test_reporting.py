import csv
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from reporting import FIELDS, write_report
from baseline.llm import LLMExperiment
from unittest.mock import patch
import json


class TestReporting(unittest.TestCase):
    def test_full_uses_configured_result_and_summary_paths(self):
        from main import run_experiment
        from test_refactor_equivalence import ReplayClient
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(dataset="ncbi", model_name="mock", retrieval_k=0,
                          max_attempts=1, result_root=str(root / "runs"),
                          summary_csv=str(root / "summary.csv"),
                          test_file_path="bioDataset/ncbi/dev.sampled_500.json")
            replies = [{"raw_response": '{"entities": []}', "usage": {}}] * 20
            result = run_experiment(config, "full", limit=1, client=ReplayClient(replies))[0]
            self.assertTrue(Path(result["result_dir"]).is_relative_to(root / "runs"))
            self.assertFalse((root / "runs" / "summary_prf1.csv").exists())
            with (root / "summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["variant"], "full")
            self.assertEqual(float(rows[0]["f1"]), result["f1"])

    def test_parallel_reports_and_reexport(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "test.json"
            source.write_text("[]")
            config = {"summary_csv": str(root / "summary.csv"), "dataset": "ncbi",
                      "model_name": "mock", "test_file_path": str(source)}
            metrics = {"precision": 1, "recall": .5, "f1": 2 / 3}
            processes = [multiprocessing.Process(target=write_report, args=(
                config, "full", metrics, root / f"{i}.jsonl", root / f"{i}.log"))
                for i in range(8)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(15)
                self.assertEqual(process.exitcode, 0)
            write_report(config, "full", metrics, root / "0.jsonl", root / "0.log")
            with (root / "summary.csv").open() as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, FIELDS)
                self.assertEqual(len(list(reader)), 8)

    def test_baseline_reports_valid_empty_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "test.json"
            source.write_text(json.dumps([{"sentence": "healthy", "entities": []}]))
            config = {"summary_csv": str(root / "summary.csv"), "dataset": "ncbi",
                      "model_name": "mock", "test_file_path": str(source),
                      "experiment_name": "test", "api_keys": "mock", "max_loop": 500,
                      "save_file_path": str(root / "out.jsonl"),
                      "log_file_path": str(root / "run.log")}
            cp = root / "config.json"
            cp.write_text(json.dumps(config))
            runner = LLMExperiment(cp, resume=True)
            runner.last_attempts = 1
            with patch.object(runner, "_predict", return_value=set()) as predict, \
                    patch("baseline.llm.setup_logging"):
                runner.run()
                runner.run()
                self.assertEqual(predict.call_count, 1)
            with (root / "summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["variant"], "llm_baseline")
            self.assertEqual(rows[0]["evaluated_samples"], "1")
            self.assertEqual(float(rows[0]["f1"]), 0)
