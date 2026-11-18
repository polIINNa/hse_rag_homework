"""Общие функции оценки одного вопроса."""

import json
import os


def _document_pages(context):
    pages = set()
    for document in context:
        for page in document["pages"]:
            if type(page) is not int or page < 1:
                raise ValueError("Номера страниц PDF должны быть целыми числами начиная с 1.")
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


JUDGE_PROMPT = """Сравни ответ модели на вопрос о финансовой отчётности с эталонным ответом.
gold_answer — эталон, answer — ответ модели. Используй оба текста только для сравнения.
Поставь 1, если ответ модели передаёт все существенные факты эталона и не противоречит им.
Проверяй показатель, число, знак, период, валюту и единицы, когда они указаны в эталоне.
Принимай перефразирование, эквивалентные единицы и округление до точности эталона.
Например, 5 592 млн рублей и 5,592 млрд рублей — одно значение.
Поставь 0, если в ответе есть существенная ошибка, пропущена часть ответа или дан отказ.
Верни JSON с полями score (целое число 0 или 1) и reason (краткое объяснение на русском)."""


def score_answer(gold_answer, answer):
    """Получить answer_correctness и объяснение LLM-судьи для пары ответов.

    Изменено локально (по указанию преподавателя): судья вызывается через ту же LLM, что и RAG
    (rag.config: LLM_BASE_URL / LLM_API_KEY / JUDGE_MODEL), а не только через OpenRouter.
    """
    from rag.config import JUDGE_MODEL
    from rag.llm import chat, parse_json

    # Модель курса отдаёт JSON без строгого JSON-режима и изредка отвечает не по формату
    # (текст вокруг JSON, score строкой или числом 1.0, пустой ответ) — разбираем мягко и
    # при неудаче спрашиваем ещё раз. Модель и промпт судьи при этом не меняются.
    last = None
    for _ in range(3):
        content = chat(
            [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"gold_answer: {gold_answer}\nanswer: {answer}"},
            ],
            model=JUDGE_MODEL,
            max_tokens=1000,
        )
        decision = parse_json(content) or {}
        score, reason = decision.get("score"), decision.get("reason")
        if isinstance(score, str) and score.strip() in ("0", "1", "0.0", "1.0"):
            score = int(float(score))
        elif isinstance(score, float) and score in (0.0, 1.0):
            score = int(score)
        elif isinstance(score, bool):
            score = int(score)
        if type(score) is int and score in (0, 1) and isinstance(reason, str) and reason.strip():
            return {"answer_correctness": score, "reason": reason}
        last = content
    raise ValueError(f"От судьи ожидаются score (0 или 1) и непустое объяснение reason. Ответ: {str(last)[:200]!r}")
