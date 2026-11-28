"""Генерация ответа по найденным страницам. LLM курса — тот же сервис, что у судьи, ключ в LLM_API_KEY."""

import json
import os
import re
import sys
import time
from collections import deque

import requests
from langsmith import traceable

LLM_URL = "https://claude.gigaparsers.ru/v1/messages"
LLM_MODEL = "deepseek-flash"
ATTEMPTS = 8  # попыток на запрос, если сервис недоступен или перегружен
RETRY_STATUSES = (429, 500, 502, 503, 504, 529)
# Лимит ключа LLM_API_KEY: сервис отвечает 429 «rate limit 10/min exceeded for this key».
LLM_REQUESTS_PER_MINUTE = int(os.environ.get("LLM_REQUESTS_PER_MINUTE", "10"))
LLM_WINDOW = 61.0  # секунд: минута с запасом, чтобы окно сервиса точно успело освободиться
llm_requests = deque()  # время последних запросов к LLM из этого процесса
# Ещё у ключа дневной лимит токенов (2 млн): тогда сервис просит ждать часы — ждать нет смысла.
MAX_RETRY_WAIT = 300  # секунд
llm_tokens = {"input": 0, "output": 0}  # токены генерации и шагов агента за процесс (судья не учитывается)


class LLMQuotaExceeded(RuntimeError):
    """Сервис LLM просит ждать дольше MAX_RETRY_WAIT: обычно исчерпан дневной лимит токенов ключа."""


def count_tokens(usage):
    """Добавляет токены из ответа сервиса: usage в JSON-ответе или usage_metadata сообщения LangChain."""
    llm_tokens["input"] += (usage or {}).get("input_tokens", 0)
    llm_tokens["output"] += (usage or {}).get("output_tokens", 0)

SYSTEM_PROMPT = """Ты отвечаешь на вопросы по финансовой отчётности банков. Отвечай только по страницам отчётов, которые даны в сообщении.

Правила:
1. Дай точный и полный ответ: все запрошенные показатели с единицами измерения, валютой и периодом, как в отчёте. Если вопрос просит посчитать или сравнить (разницу, долю, отношение, что больше), сначала приведи числа к одним единицам (тыс., млн, млрд), потом посчитай и приведи результат вместе с исходными числами.
2. Пиши числа в русской записи: пробел разделяет разряды, запятая отделяет дробную часть (1 151; 26 926,8). В отчётах ЗАО «Альфа-Банк» (Беларусь) разряды разделены запятой на английский манер: там 1,151 означает 1 151, переводи такие числа в русскую запись.
3. Не отказывайся отвечать. Если данных не хватает, дай лучший ответ, который можно получить из страниц.
4. В pages перечисли метками (например "vtb_2024_annual:45") все страницы, где есть числа и факты из твоего ответа. Если одно и то же число приведено и в основной форме отчётности, и в примечании к ней, укажи обе страницы. Если нужная таблица начинается на одной странице и продолжается на следующей, укажи и страницу, где она начинается. Не указывай страницы, на которых нет данных из ответа.
5. Отвечай только на то, что спрошено. Не добавляй неспрошенные показатели, разбивки по этапам или статьям и комментарии о расхождениях в отчёте. Исходные числа приводи, только если вопрос требует расчёта или сравнения.

Верни только JSON без Markdown: {"answer": "ответ на русском", "pages": ["doc_id:page", ...]}"""


def wait_for_llm_slot():
    """Ждёт, пока за последние LLM_WINDOW секунд будет меньше LLM_REQUESTS_PER_MINUTE запросов.
    Вызывается перед каждым запросом к LLM: шаги агента, ответ, судья — у них общий лимит ключа."""
    while True:
        now = time.monotonic()
        while llm_requests and now - llm_requests[0] >= LLM_WINDOW:
            llm_requests.popleft()
        if len(llm_requests) < LLM_REQUESTS_PER_MINUTE:
            break
        time.sleep(LLM_WINDOW - (now - llm_requests[0]))
    llm_requests.append(time.monotonic())


def retry_wait(response, attempt):
    """Сколько ждать перед повтором: сколько попросил сервис (Retry-After), иначе 5, 10, 20, 40, 60 с.
    Если сервис просит ждать дольше MAX_RETRY_WAIT, останавливаемся с LLMQuotaExceeded."""
    retry_after = response.headers.get("retry-after", "") if response is not None else ""
    wait = float(retry_after) if retry_after.isdigit() else min(5 * 2 ** attempt, 60)
    if wait > MAX_RETRY_WAIT:
        raise LLMQuotaExceeded(f"Сервис LLM просит подождать {wait / 3600:.1f} ч: {response.text[:300]}")
    reason = f"ответ {response.status_code}" if response is not None else "сбой соединения"
    print(f"LLM: {reason}, повтор через {wait:.0f} с", file=sys.stderr, flush=True)  # видно в логе прогона
    return wait


@traceable(run_type="llm", name="deepseek-flash")
def call_llm(system, user):
    for attempt in range(ATTEMPTS):
        response = None
        wait_for_llm_slot()
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
    body = response.json()
    count_tokens(body.get("usage"))
    return "".join(block["text"] for block in body["content"] if block["type"] == "text").strip()


def page_label(document):
    return f"{document.metadata['doc_id']}:{document.metadata['page']}"


def page_heading(document):
    """Строка перед текстом страницы для модели. Единицы обычно указаны один раз в шапке формы отчётности,
    а на страницах примечаний их нет, поэтому показываем основные единицы отчёта (metadata["units"])."""
    units = document.metadata.get("units")
    note = f" (суммы в отчёте, если на странице не указано иное: {units})" if units else ""
    return f"=== Страница {page_label(document)}{note} ==="


@traceable(name="generate_answer")
def generate_answer(question, documents):
    """Ответ на вопрос и те страницы из documents, на которых он основан."""
    pages = "\n\n".join(f"{page_heading(d)}\n{d.page_content}" for d in documents)
    reply = call_llm(SYSTEM_PROMPT, f"Страницы отчётов:\n\n{pages}\n\nВопрос: {question}")
    try:
        decision = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))  # модель иногда оборачивает JSON в ```
        answer = str(decision["answer"]).strip()
        cited = set(decision.get("pages", []))
    except (AttributeError, ValueError, KeyError, TypeError):
        return reply, documents  # модель вернула не JSON: отдаём её текст и все найденные страницы
    used = [d for d in documents if page_label(d) in cited]
    return answer, used or documents
