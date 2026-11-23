"""Общие функции оценки одного вопроса."""

import json

from rag import config
from rag.models import chat
from rag.prompts import JUDGE_PROMPT


def _document_pages(context):
    pages = set()
    for document in context:
        for page in document["pages"]:
            if type(page) is not int or page < 1:
                raise ValueError(
                    "Номера страниц PDF должны быть целыми числами начиная с 1."
                )
            pages.add((document["doc_id"], page))
    return pages


def score_context(gold_context, context):
    """Сравнить два списка документов с полями doc_id и pages."""
    gold = _document_pages(gold_context)
    if not gold:
        raise ValueError("В gold_context должна быть хотя бы одна страница.")
    found = _document_pages(context)
    hits = len(found & gold)
    return {
        "context_recall": hits / len(gold),
        "context_precision": hits / len(found) if found else 0.0,
    }


def score_answer(gold_answer, answer):
    """Получить answer_correctness и объяснение LLM-судьи для пары ответов."""
    text = chat(
        JUDGE_PROMPT,
        f"gold_answer: {gold_answer}\nanswer: {answer}",
        model=config.JUDGE_MODEL,
        max_tokens=config.JUDGE_MAX_TOKENS,
    ).strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    decision = json.loads(text)
    score, reason = decision["score"], decision["reason"]
    if (
        type(score) is not int
        or score not in (0, 1)
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        raise ValueError(
            "От судьи ожидаются score (0 или 1) и непустое объяснение reason."
        )
    return {"answer_correctness": score, "reason": reason}
