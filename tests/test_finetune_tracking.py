"""Tracking is optional and must preserve inference/training error semantics."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from baseline.finetune.utils import tracking_session, track


class TrackingTests(unittest.TestCase):
    def test_disabled_does_not_need_config_or_sdk(self):
        with tracking_session("nonexistent", "disabled") as tracker:
            self.assertIsNone(tracker)
            track(tracker, {"train/loss": 1})

    def test_success_failure_abort_and_metric_forwarding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps({"model_name": "BioBERT", "dataset": "bc2gm", "seed": 42,
                                        "prediction_path": str(Path(temp) / "predictions.jsonl")}))
            sdk = Mock()
            with patch.dict("sys.modules", {"swanlab": sdk}):
                with tracking_session(path, "offline") as tracker:
                    track(tracker, {"dev/f1": 0.5}, step=10)
                self.assertEqual(sdk.init.call_args.kwargs["mode"], "offline")
                sdk.log.assert_called_once_with({"dev/f1": 0.5}, step=10)
                sdk.finish.assert_called_with(state="success", error=None)
                with self.assertRaises(RuntimeError):
                    with tracking_session(path, "offline"):
                        raise RuntimeError("training failed")
                sdk.finish.assert_called_with(state="crashed", error="RuntimeError")
                with self.assertRaises(KeyboardInterrupt):
                    with tracking_session(path, "offline"):
                        raise KeyboardInterrupt()
                sdk.finish.assert_called_with(state="aborted", error=None)
                sdk.log.side_effect = OSError("logging unavailable")
                track(sdk, {"train/loss": 0.2})  # Must not abort a valid training step.


if __name__ == "__main__":
    unittest.main()
