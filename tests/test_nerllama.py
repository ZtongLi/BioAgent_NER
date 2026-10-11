"""Offline NER protocol tests; CPU smoke test substitutes the CUDA QLoRA loader."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import DataCollatorForTokenClassification, LlamaConfig, LlamaForCausalLM

from baseline.finetune import nerllama as ner
from baseline.finetune.utils import evaluation_record, finish_run
from reporting import evaluate_ner
from test_supervised_ner import tokenizer


def chat_tokenizer():
    tok = tokenizer()
    tok.bos_token = "[CLS]"
    tok.eos_token = "[SEP]"
    tok.pad_token = tok.eos_token
    tok.chat_template = ("{{ bos_token }}{% for message in messages %}"
                         "{{ message['role'] }}: {{ message['content'] }}{{ eos_token }}"
                         "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}")
    return tok


class NERLlamaTests(unittest.TestCase):
    def test_quote_roundtrip_repeated_unicode_and_inside_word(self):
        sample = {"sentence": "α ALT ALTase ALT", "entities": [
            {"name": "ALT", "type": "GENE", "pos": [2, 5]},
            {"name": "ALT", "type": "GENE", "pos": [13, 16]}]}
        answer = ner.target_json(sample)
        self.assertEqual([e["occurrence"] for e in json.loads(answer)], [0, 2])
        pred = ner.parse_entities("```json\n" + answer + "\n```", sample["sentence"], "bc2gm")
        row = evaluation_record(0, sample, pred, "bc2gm", "NERLlama3.1", "instruction_ner")
        self.assertEqual(evaluate_ner([row])["f1"], 1)
        duplicate = json.dumps(json.loads(answer) * 2)
        self.assertEqual(len(ner.parse_entities(duplicate, sample["sentence"], "bc2gm")), 2)
        for invalid in ["{}", '[{"text":"alt","type":"GENE","occurrence":0}]',
                        '[{"text":"ALT","type":"GENE","occurrence":true}]',
                        '[{"text":"ALT","type":"DISEASE","occurrence":0}]']:
            with self.assertRaises(ValueError):
                ner.parse_entities(invalid, sample["sentence"], "bc2gm")

    def test_assistant_only_loss_padding_and_length_guard(self):
        tok = chat_tokenizer()
        sample = {"sentence": "ALT", "entities": [{"name": "ALT", "type": "GENE", "pos": [0, 3]}]}
        config = {"dataset": "bc2gm", "max_length": 1024, "max_new_tokens": 256}
        feature = ner.training_feature(sample, tok, config)
        prefix = ner.chat_ids(tok, ner.instruction("ALT", "bc2gm"), True)
        self.assertEqual(feature["labels"][:len(prefix)], [-100] * len(prefix))
        self.assertEqual(feature["labels"][len(prefix):], feature["input_ids"][len(prefix):])
        self.assertEqual(feature["labels"][-1], tok.eos_token_id)
        short = ner.training_feature({"sentence": "x", "entities": []}, tok, config)
        batch = DataCollatorForTokenClassification(tok)([feature, short])
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        self.assertEqual(batch["labels"][1, len(short["input_ids"]) - 1].item(), tok.eos_token_id)
        with self.assertRaisesRegex(ValueError, "truncated"):
            ner.training_feature(sample, tok, {**config, "max_length": 20, "max_new_tokens": 10})

    def test_invalid_retry_is_gold_blind_and_empty_is_valid(self):
        tok = chat_tokenizer()
        config = {"dataset": "bc2gm", "model_name": "NERLlama3.1", "max_length": 2048,
                  "max_new_tokens": 256, "max_attempts": 3}
        sample = {"sentence": "ALT", "entities": [{"name": "ALT", "type": "GENE", "pos": [0, 3]}]}
        prompts = []
        real_ids = ner.generation_ids
        def track(tokenizer, messages, config):
            prompts.append(messages)
            return real_ids(tokenizer, messages, config)
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(ner, "generation_ids", side_effect=track), patch.object(
                    ner, "generate_answer", side_effect=[("{}", True), ("[]", True)]) as gen:
                rows = ner.predict(Mock(), tok, [sample], config, "test", Path(temp) / "trace.jsonl")
            self.assertEqual(gen.call_count, 2)
            self.assertEqual(rows[0]["attempts"], 2)
            self.assertEqual(evaluate_ner(rows)["false_negative"], 1)
            self.assertNotIn("gold", json.dumps(prompts).lower())
            with patch.object(ner, "generate_answer", return_value=("[]", False)):
                with self.assertRaisesRegex(RuntimeError, "no partial PRF"):
                    ner.predict(Mock(), tok, [sample], config, "test", Path(temp) / "bad.jsonl")

    def test_generate_slices_prompt_and_requires_end_token(self):
        tok = chat_tokenizer()
        mock = Mock()
        mock.get_input_embeddings.return_value.weight = torch.zeros(1)
        prompt = [2, 5]
        mock.generate.return_value = torch.tensor([prompt + [5, tok.eos_token_id]])
        raw, finished = ner.generate_answer(mock, tok, prompt, {"max_new_tokens": 8})
        self.assertEqual(raw, "ALT")
        self.assertTrue(finished)
        self.assertFalse(mock.generate.call_args.kwargs["do_sample"])
        mock.generate.return_value = torch.tensor([prompt + [5]])
        self.assertFalse(ner.generate_answer(mock, tok, prompt, {"max_new_tokens": 8})[1])

    def test_finish_refuses_missing_or_failed_test_rows(self):
        sample = {"sentence": "ALT", "entities": []}
        row = evaluation_record(0, sample, [], "bc2gm", "NERLlama3.1")
        with self.assertRaises(RuntimeError):
            finish_run({}, [row], 2, {}, None)
        row["fully_executed"] = False
        with self.assertRaises(RuntimeError):
            finish_run({}, [row], 1, {}, None)

    def test_quantized_loader_wires_nf4_and_lora_without_network(self):
        base = Mock()
        base.config.model_type = "llama"
        base.config.max_position_embeddings = 4096
        base.is_loaded_in_4bit = True
        peft = SimpleNamespace(LoraConfig=Mock(), PeftModel=Mock(), get_peft_model=Mock(),
                               prepare_model_for_kbit_training=Mock(return_value=base))
        config = {"pretrained_model_path": "baseline/model/pretrained/llama31-8b", "max_length": 4096,
                  "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
        with patch.dict("sys.modules", {"peft": peft}), patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.is_bf16_supported", return_value=True), patch("torch.cuda.current_device", return_value=0), \
             patch("transformers.BitsAndBytesConfig") as quant, \
             patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=base) as load:
            ner.load_qlora(config, chat_tokenizer())
            self.assertTrue(quant.call_args.kwargs["load_in_4bit"])
            self.assertEqual(quant.call_args.kwargs["bnb_4bit_quant_type"], "nf4")
            self.assertTrue(load.call_args.kwargs["local_files_only"])
            self.assertEqual(peft.LoraConfig.call_args.kwargs["target_modules"], "all-linear")
            ner.load_qlora(config, chat_tokenizer(), Path("adapter"))
            self.assertFalse(peft.PeftModel.from_pretrained.call_args.kwargs["is_trainable"])

    def test_cpu_smoke_training_selection_reload_and_final_report(self):
        """Real tiny causal-LM training; quantization/adapter loading is substituted."""
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pre = root / "pretrained"
            tok = chat_tokenizer()
            tok.save_pretrained(pre)
            model_config = LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=24,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                       max_position_embeddings=2048, eos_token_id=tok.eos_token_id,
                                       pad_token_id=tok.pad_token_id)
            sample = {"sentence": "ALT", "entities": [{"name": "ALT", "type": "GENE", "pos": [0, 3]}]}
            for split in ("train", "dev", "test"):
                (root / f"{split}.json").write_text(json.dumps([sample]))
            config = {"dataset": "bc2gm", "model_name": "NERLlama3.1", "pretrained_model_path": str(pre),
                      **{f"{s}_file_path": str(root / f"{s}.json") for s in ("train", "dev", "test")},
                      "checkpoint_dir": str(root / "best"), "prediction_path": str(root / "result/predictions.jsonl"),
                      "summary_csv": str(root / "summary.csv"), "seed": 42, "max_length": 1024,
                      "max_new_tokens": 256, "learning_rate": 2e-4, "epochs": 2, "batch_size": 1,
                      "gradient_accumulation_steps": 4, "lora_r": 16, "lora_alpha": 32,
                      "lora_dropout": 0.05, "max_attempts": 3}
            path = root / "config.json"
            path.write_text(json.dumps(config))
            loaded = []
            def local_loader(config, tokenizer, adapter_path=None):
                loaded.append(adapter_path)
                model = (LlamaForCausalLM(model_config) if adapter_path is None else
                         LlamaForCausalLM.from_pretrained(adapter_path, local_files_only=True))
                return model, torch.float32
            with patch.object(ner, "load_qlora", side_effect=local_loader), \
                 patch.object(ner, "version", return_value="offline-test"), \
                 patch.object(ner, "generate_answer", return_value=("[]", True)) as generate:
                metrics = ner.run(path)
            self.assertEqual(generate.call_count, 3)  # dev each epoch, then test once
            self.assertEqual(loaded, [None, (root / "best").resolve()])
            self.assertEqual(metrics["false_negative"], 1)
            self.assertEqual(metrics["evaluated_samples"], 1)
            info = json.loads((root / "best/run_config.json").read_text())
            self.assertEqual(info["best_epoch"], 1)  # earliest checkpoint wins a tie
            with (root / "summary.csv").open() as f:
                self.assertEqual(list(csv.DictReader(f))[0]["variant"], "supervised_baseline")
            rows = [json.loads(x) for x in Path(config["prediction_path"]).read_text().splitlines()]
            self.assertEqual(evaluate_ner(rows), metrics)
            with self.assertRaises(FileExistsError):
                ner.run(path)


if __name__ == "__main__":
    unittest.main()
