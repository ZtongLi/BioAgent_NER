"""Offline tests: synthetic tokenizer/weights only, no pretrained downloads."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

from baseline.finetune import bert_ner as runner
from baseline.finetune.utils import (character_bio, decode_bio, encode_sentence,
                                    evaluation_record, label_schema, merge_windows)
from reporting import evaluate_ner


def tokenizer():
    vocab = {s: i for i, s in enumerate(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "ALT", "and", "Hb", "x", "α"])}
    backend = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                  pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
                                  mask_token="[MASK]", model_max_length=512)


class SupervisedTests(unittest.TestCase):
    def test_overlap_global_bio_and_character_scoring(self):
        text = "α ALT and Hb x ALT and Hb x ALT and Hb"
        tok = tokenizer()
        offsets, windows = encode_sentence(tok, text, 7, 2)
        labels = label_schema("bc2gm")
        # Span crosses a window boundary, plus a repeated text at the end.
        gold = [{"name": "and Hb x ALT", "pos": [6, 18], "type": "GENE"},
                {"name": "Hb", "pos": [36, 38], "type": "GENE"}]
        tags = character_bio(offsets, gold, labels)
        scores = []
        for w in windows:
            logits = torch.zeros(len(w["indices"]), len(labels))
            for j, i in enumerate(w["indices"]):
                logits[j, tags[i] if i >= 0 else 0] = 9
            scores.append(logits)
        merged = merge_windows(windows, scores, len(offsets))
        pred = decode_bio(text, offsets, merged, labels)
        self.assertEqual([(p["start"], p["end"]) for p in pred], [(6, 18), (36, 38)])
        for p in pred:
            self.assertEqual(p["text"], text[p["start"]:p["end"]])
        row = evaluation_record(0, {"sentence": text, "entities": gold}, pred, "bc2gm", "BioBERT")
        self.assertEqual(evaluate_ner([row])["f1"], 1)
        row["pred_entities"] = []
        row["stages"][0]["output_entities"] = []
        self.assertEqual(evaluate_ner([row])["false_negative"], 2)
        row["fully_executed"] = False
        self.assertEqual(evaluate_ner([row])["evaluated_samples"], 0)

    def test_alignment_failure_and_overlap_loss_ownership(self):
        labels = label_schema("bc2gm")
        with self.assertRaisesRegex(ValueError, "boundary"):
            character_bio([(0, 3)], [{"pos": [1, 3], "type": "GENE"}], labels)
        data = [{"sentence": "ALT and Hb x ALT and Hb x ALT", "entities": []}]
        prepared = runner.prepare_samples(data, tokenizer(), {"max_length": 6, "stride": 2}, labels, True)
        offsets, windows = prepared[0]
        active = [i for w in windows for i, label in zip(w["indices"], w["inputs"]["labels"]) if label != -100]
        self.assertEqual(sorted(active), list(range(len(offsets))))
        with self.assertRaises(ValueError):
            encode_sentence(tokenizer(), data[0]["sentence"], 6, 4)

    def test_offline_training_best_checkpoint_and_report(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pre = root / "pretrained"
            tok = tokenizer()
            tok.save_pretrained(pre)
            BertForMaskedLM(BertConfig(vocab_size=len(tok), hidden_size=16, num_hidden_layers=1,
                                      num_attention_heads=2, intermediate_size=24,
                                      max_position_embeddings=32)).save_pretrained(pre)
            sample = {"sentence": "ALT and Hb", "entities": [
                {"name": "ALT", "pos": [0, 3], "type": "GENE"}]}
            for split in ("train", "dev", "test"):
                (root / f"{split}.json").write_text(json.dumps([sample, {"sentence": "x x", "entities": []}]))
            cfg = {"dataset": "bc2gm", "model_name": "BioBERT", "pretrained_model_path": str(pre),
                   **{f"{s}_file_path": str(root / f"{s}.json") for s in ("train", "dev", "test")},
                   "checkpoint_dir": str(root / "best"), "prediction_path": str(root / "result/predictions.jsonl"),
                   "summary_csv": str(root / "summary.csv"), "seed": 42, "max_length": 8, "stride": 2,
                   "learning_rate": 2e-5, "epochs": 2, "batch_size": 2}
            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            calls = []
            real_predict = runner.predict
            def tracked(*args):
                calls.append(args[-1])
                return real_predict(*args)
            with patch.object(runner, "predict", side_effect=tracked):
                metrics = runner.run(path)
            self.assertEqual(calls, ["dev", "dev", "test"])
            self.assertEqual(metrics["evaluated_samples"], 2)
            records = [json.loads(line) for line in Path(cfg["prediction_path"]).read_text().splitlines()]
            self.assertEqual(evaluate_ner(records), metrics)
            history = json.loads((root / "best/training_history.json").read_text())
            info = json.loads((root / "best/run_config.json").read_text())
            expected_epoch = max(history, key=lambda x: x["dev"]["f1"])["epoch"]
            self.assertEqual(info["best_epoch"], expected_epoch)
            with (root / "summary.csv").open() as f:
                report = list(csv.DictReader(f))
            self.assertEqual(report[0]["variant"], "supervised_baseline")
            self.assertEqual(float(report[0]["f1"]), metrics["f1"])
            with self.assertRaises(FileExistsError):
                runner.run(path)


if __name__ == "__main__":
    unittest.main()
