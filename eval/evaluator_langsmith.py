"""Из корня репозитория: python -m eval.evaluator_langsmith ИМЯ_ДАТАСЕТА"""

import argparse
from time import perf_counter

from langsmith import Client

from eval.eval_functions import score_answer, score_context
from eval.rag_example import RAG


def metrics(outputs, reference_outputs):
    context = score_context(reference_outputs["gold_context"], outputs["context"])
    answer = score_answer(reference_outputs["gold_answer"], outputs["answer"])
    return [
        {"key": "context_recall", "score": context["context_recall"]},
        {"key": "context_precision", "score": context["context_precision"]},
        {"key": "answer_correctness", "score": answer["answer_correctness"], "comment": answer["reason"]},
        {"key": "time", "score": outputs["time"]},
    ]


def evaluate(dataset_name):
    system = RAG()

    def answer_question(inputs):
        start = perf_counter()
        result = system.rag(inputs["question"])
        return {**result, "time": perf_counter() - start}

    return Client().evaluate(
        answer_question, data=dataset_name, evaluators=[metrics], experiment_prefix="rag",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name")
    evaluate(parser.parse_args().dataset_name)
