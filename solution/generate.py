"""Генерация ответа по найденным страницам. LLM курса — тот же сервис, что у судьи, ключ в LLM_API_KEY."""

import json
import os
import re
import time

import requests

LLM_URL = "https://claude.gigaparsers.ru/v1/messages"
LLM_MODEL = "deepseek-flash"
ATTEMPTS = 8  # попыток на запрос, если сервис недоступен или перегружен
RETRY_STATUSES = (429, 500, 502, 503, 504, 529)

SYSTEM_PROMPT = """Ты отвечаешь на вопросы по финансовой отчётности банков. Отвечай только по страницам отчётов, которые даны в сообщении.

Правила:
1. Дай точный и полный ответ: все запрошенные показатели с единицами измерения, валютой и периодом, как в отчёте. Если вопрос просит посчитать (разницу, долю, отношение), посчитай и приведи результат вместе с исходными числами.
2. Пиши числа в русской записи: пробел разделяет разряды, запятая отделяет дробную часть (1 151; 26 926,8). В отчётах ЗАО «Альфа-Банк» (Беларусь) разряды разделены запятой на английский манер: там 1,151 означает 1 151, переводи такие числа в русскую запись.
3. Не отказывайся отвечать. Если данных не хватает, дай лучший ответ, который можно получить из страниц.
4. В pages перечисли только страницы, на которых основан ответ, их метками, например "vtb_2024_annual:45".

Верни только JSON без Markdown: {"answer": "ответ на русском", "pages": ["doc_id:page", ...]}"""


def retry_wait(response, attempt):
    """Сколько ждать перед повтором: сколько попросил сервис (Retry-After), иначе 5, 10, 20, 40, 60 с."""
    retry_after = response.headers.get("retry-after", "") if response is not None else ""
    return float(retry_after) if retry_after.isdigit() else min(5 * 2 ** attempt, 60)


def call_llm(system, user):
    for attempt in range(ATTEMPTS):
        response = None
        try:
            response = requests.post(
                LLM_URL,
                headers={"x-api-key": os.environ["LLM_API_KEY"], "anthropic-version": "2023-06-01"},
                json={
                    "model": LLM_MODEL,
                    "max_tokens": 1024,
                    "temperature": 0,  # одинаковые ответы при повторном прогоне: проверка допускает отклонение 5%
                    "thinking": {"type": "disabled"},
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                timeout=(10, 120),
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt == ATTEMPTS - 1:
                raise
        else:
            if response.status_code not in RETRY_STATUSES:
                break
            if attempt == ATTEMPTS - 1:
                break
        time.sleep(retry_wait(response, attempt))  # сервис недоступен или перегружен — ждём и пробуем ещё раз
    if response.status_code >= 400:  # текст ответа объясняет причину, например какой лимит превышен
        raise requests.HTTPError(f"{response.status_code}: {response.text[:300]}", response=response)
    return "".join(block["text"] for block in response.json()["content"] if block["type"] == "text").strip()


def page_label(document):
    return f"{document.metadata['doc_id']}:{document.metadata['page']}"


def generate_answer(question, documents):
    """Ответ на вопрос и те страницы из documents, на которых он основан."""
    pages = "\n\n".join(f"=== Страница {page_label(d)} ===\n{d.page_content}" for d in documents)
    reply = call_llm(SYSTEM_PROMPT, f"Страницы отчётов:\n\n{pages}\n\nВопрос: {question}")
    try:
        decision = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))  # модель иногда оборачивает JSON в ```
        answer = str(decision["answer"]).strip()
        cited = set(decision.get("pages", []))
    except (AttributeError, ValueError, KeyError, TypeError):
        return reply, documents  # модель вернула не JSON: отдаём её текст и все найденные страницы
    used = [d for d in documents if page_label(d) in cited]
    return answer, used or documents
