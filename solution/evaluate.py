"""Оценка RAG в Excel: eval/evaluator_excel.py из репозитория, но судья повторяет запрос при 429 и сбоях сети.

Из корня репозитория: python -m solution.evaluate data/test.jsonl --output test_results.xlsx

Колонки, листы Excel и замер time — те же, что в evaluator_excel. Модель и промпт судьи не меняются:
вызывается та же score_answer, только с повторами (solution/judge.py).
"""

import argparse
from pathlib import Path

import eval.evaluator_excel as evaluator_excel
from eval.rag_example import SimpleRAG
from solution.generate import llm_tokens
from solution.judge import judge


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_path")
    parser.add_argument("--output", type=Path, help="Путь для сохранения Excel-отчёта")
    parser.add_argument("--simple", action="store_true", help="оценить SimpleRAG (без агента) вместо RAG")
    args = parser.parse_args()
    evaluator_excel.score_answer = judge  # evaluate() вызывает score_answer из своего модуля
    if args.simple:
        evaluator_excel.RAG = SimpleRAG
    print(evaluator_excel.evaluate(args.dataset_path, args.output))

    with open(args.dataset_path, encoding="utf-8") as source:
        questions = sum(1 for line in source if line.strip())
    total = llm_tokens["input"] + llm_tokens["output"]
    print(f"Токены LLM без судьи: вход {llm_tokens['input']}, выход {llm_tokens['output']}, "
          f"всего {total}, на вопрос {total / questions:.0f}. Судья — примерно ещё 1 тыс. на вопрос.")


if __name__ == "__main__":
    main()
