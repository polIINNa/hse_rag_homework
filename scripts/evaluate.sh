#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m rag.prepare
python -m eval.run --split test --output results/test_repeat.jsonl "$@"
python -m eval.export_excel results/test_repeat.jsonl --output test_repeat.xlsx
