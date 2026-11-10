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
    """Получить answer_correctness и объяснение LLM-судьи для пары ответов."""
    import requests

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
        json={
            "model": "google/gemini-3.8-flash",
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"gold_answer: {gold_answer}\nanswer: {answer}"},
            ],
        },
        timeout=90,
    )
    response.raise_for_status()
    decision = json.loads(response.json()["choices"][0]["message"]["content"])
    score, reason = decision["score"], decision["reason"]
    if type(score) is not int or score not in (0, 1) or not isinstance(reason, str) or not reason.strip():
        raise ValueError("От судьи ожидаются score (0 или 1) и непустое объяснение reason.")
    return {"answer_correctness": score, "reason": reason}
