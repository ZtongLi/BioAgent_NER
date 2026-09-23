# Independent NER configurations

Each dataset has one DeepSeek config and one GPT config. The same config is used for
all ablations; select the variant on the command line. API keys are local plaintext
fields, and these JSON files are ignored by Git. Run snapshots exclude API keys.

Fields: `agent_system` (`ner_agent`), `dataset`, `model_name`, `base_url`,
`api_keys` (or `api_key_env`), `test_file_path`, `max_loop`, `timeout`,
`max_attempts`, `retry_delay`, `sleep_seconds`, `temperature`.

`max_attempts` includes the first request. Request failures and invalid model output
share this bounded attempt budget. `sleep_seconds` applies before each API attempt.
The six initial configurations evaluate the existing fixed 500-sentence dev subsets.
No baseline predictions or rules are loaded. Training examples are stored with
their source path and row index in `prompts/annotation_rules/`.
