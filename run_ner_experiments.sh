#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

variant="${1:-full}"
if [[ $# -gt 0 ]]; then shift; fi

for dataset in bc2gm bc5cdr ncbi; do
  for model in deepseek gpt; do
    python3 main.py --config "config/ner_agent/$dataset/${dataset}_${model}.json" \
      --variant "$variant" "$@"
  done
done
