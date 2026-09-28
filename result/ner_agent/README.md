# Independent NER results

New runs are saved as `dataset/model/variant/run_timestamp/`:

- `run.log`: sample progress, stage status, errors and every raw model response.
- `predictions.jsonl`: one immediately flushed record per sample, containing all
  stage inputs/outputs, attempts, final entities and evaluation labels.
- `metrics.json`: successful-sample exact-span micro PRF, TP/FP/FN, evaluated
  sample count, success rate, failure counts and per-module effects.
- `run_config.json`: secret-free configuration, exact prompts, input file hash
  and source-code hash, system version and training-pool hash.

V2 uses the same extraction, independent coverage, boundary and verification
modules for all datasets. Training examples actually supplied to each stage are
saved in its trace. Candidate `sources` record provenance, not correctness.

`summary_prf1.csv` receives one row per completed configuration. On the next save,
missing evaluation metadata columns are appended while preserving existing column
order and historical scores. Historical rows are marked `evaluation_scope=all_samples`;
new rows use `successful_samples_only`. Do not directly compare different scopes.
An interrupted run keeps its already flushed logs
and predictions but has no completed metrics/summary entry. Every new execution
has a new run directory; earlier results are not overwritten.
HTTP 400/401/403/404 stops the run immediately after saving the failed response;
fix the model/account/request configuration before running it again.

Only samples with successful final responses and valid parsing for every required
stage enter TP/FP/FN. A retry that eventually succeeds is included. A valid empty
entity list is included and can produce FN; optional stages skipped because their
candidate list is empty are valid. Exhausted network, HTTP or parsing failures
exclude the whole sample, including when a later stage preserves earlier output.
The raw responses and fallback predictions remain in the logs and JSONL.

`num_samples` counts all processed samples; `evaluated_samples` counts scored
samples, and `failed_samples` counts excluded samples. `success_rate` is their
completion rate. If no samples qualify, P/R/F1 are JSON null / empty CSV cells,
not zero. Stage failure counts cover all processed samples at that stage, but entity-change
effects use only fully successful samples. `status=complete` means every sample
was processed, not that every stage succeeded.

Compare variants on the intersection of their successful sample IDs after checking
that sentences and gold labels match; independently filtered summaries may cover
different subsets. Processes already running keep their loaded evaluation code.
Existing predictions can be rescored with `evaluate_ner` without API requests;
historical metrics are not automatically replaced.

With `--variant compare` (extraction/full) or `all` (six variants), identical stage inputs share the same output, including
failed attempts, within that sample. `api_calls` counts actual calls charged to
that variant; `logical_attempts` also counts reused attempts. Runtime under cache
sharing is not a standalone latency comparison. Separate runs do not share cache.

`paired_samples` and `paired_precision/recall/f1` score the same successful sample
intersection for every requested variant. When the paired extraction F1 is positive,
`paired_f1_relative_gain` compares against that shared extraction control, and
`target_20pct_met` requires F1 relative gain >= 20% with neither P nor R decreasing.
These fields are absent when the comparison is undefined; absence is not success.
This is a goal check, not a performance guarantee or a comparison with historical
baseline runs. Inspect completion rates and sample coverage alongside gains.
