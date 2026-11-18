#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then
    rag_python=.venv/bin/python
else
    rag_python=python
fi
"$rag_python" -m rag.prepare
"$rag_python" -u -m eval.run --split test --output results/test_clean.jsonl --resume
"$rag_python" -m eval.export_excel results/test_clean.jsonl --output test_clean.xlsx
"$rag_python" -m eval.compare_runs results/test.jsonl results/test_clean.jsonl --allow-fingerprint-change --output results/reproducibility.json
