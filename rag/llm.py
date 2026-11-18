"""Вызов LLM через любой OpenAI-совместимый API (Ollama, OpenRouter и др.) и разбор JSON-ответа модели."""

import json
import re
import threading
import time

import requests

from rag.config import (IS_OPENROUTER, LLM_API_FORMAT, LLM_MODEL, LLM_ROOT_URL, LLM_RPM, LLM_TIMEOUT,
                        REASONING_EFFORT, llm_api_key)

try:
    from langsmith import traceable
except ImportError:  # LangSmith не установлен — трассировка просто выключена
    def traceable(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap(args[0]) if args and callable(args[0]) else wrap


RETRY_STATUSES = (429, 500, 502, 503, 504, 529)
_lock = threading.Lock()
_next_slot = 0.0


def throttle():
    """Не чаще LLM_RPM запросов в минуту на весь процесс (с запасом 10%), чтобы не получать 429."""
    global _next_slot
    if not LLM_RPM:
        return
    interval = 60.0 / LLM_RPM * 1.1
    with _lock:
        now = time.monotonic()
        slot = max(now, _next_slot)
        _next_slot = slot + interval
    time.sleep(max(0.0, slot - now))


def _status(error):
    status = getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)
    if status is None and "429" in str(error):
        status = 429
    return status


def call_with_retries(fn, attempts=6):
    """Вызов LLM через throttle; на 429/5xx ждём (на 429 — минуту, лимит поминутный) и повторяем."""
    for attempt in range(attempts):
        throttle()
        try:
            return fn()
        except Exception as error:  # ошибки SDK разных провайдеров
            status = _status(error)
            transient = isinstance(error, (requests.ConnectionError, requests.Timeout)) or any(
                word in error.__class__.__name__ for word in ("Connection", "Timeout"))
            if status not in RETRY_STATUSES and not transient:
                raise
            if attempt == attempts - 1:
                raise
            wait = 61 if status == 429 else min(5 * 2 ** attempt, 60)
            print(f"   LLM: {status or error.__class__.__name__}, жду {wait} с и повторяю")
            time.sleep(wait)


def strip_thinking(text):
    """Убрать рассуждения <think>…</think>, которые некоторые модели пишут прямо в ответ."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _openai_request(messages, model, json_mode, max_tokens):
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if IS_OPENROUTER:
        payload["reasoning"] = {"effort": REASONING_EFFORT}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": "Bearer " + llm_api_key()}
    return f"{LLM_ROOT_URL}/v1/chat/completions", headers, payload


def _anthropic_request(messages, model, max_tokens):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    payload = {"model": model, "messages": turns, "temperature": 0, "max_tokens": max_tokens}
    if system:
        payload["system"] = system
    key = llm_api_key()
    headers = {"x-api-key": key, "Authorization": "Bearer " + key, "anthropic-version": "2023-06-01"}
    return f"{LLM_ROOT_URL}/v1/messages", headers, payload


def _content(data):
    if LLM_API_FORMAT == "anthropic":
        return "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
    return data["choices"][0]["message"]["content"]


@traceable(name="llm_chat", run_type="llm")
def chat(messages, model=LLM_MODEL, json_mode=True, max_tokens=4000):
    if LLM_API_FORMAT == "anthropic":
        url, headers, payload = _anthropic_request(messages, model, max_tokens)
    else:
        url, headers, payload = _openai_request(messages, model, json_mode, max_tokens)

    def request():
        response = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        if response.status_code == 400 and "response_format" in payload:
            payload.pop("response_format")           # сервис не умеет JSON-режим — повторим без него, JSON просим промптом
            raise requests.HTTPError("сервис не поддерживает response_format", response=_Fake503())
        response.raise_for_status()
        try:
            content = strip_thinking(_content(response.json()))
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise requests.HTTPError(f"неожиданный ответ: {error!r}", response=_Fake503())
        if not content:
            raise requests.HTTPError("пустой ответ модели", response=_Fake503())
        return content

    return call_with_retries(request)


class _Fake503:
    status_code = 503


def parse_json(text):
    """Первый JSON-объект из ответа модели (в т.ч. внутри ```json … ``` или после пояснений), иначе None."""
    text = strip_thinking(text)
    decoder = json.JSONDecoder()
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        for candidate in (text[start:], re.sub(r",\s*([}\]])", r"\1", text[start:])):
            try:
                data, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return None


def parse_answer(text):
    """{"answer": str, "sources": [(doc_id, page)]} из ответа модели (JSON или текст со ссылками)."""
    text = strip_thinking(text)
    data = parse_json(text)
    if not isinstance(data, dict):
        sources = [(d, int(p)) for d, p in re.findall(r"([a-z]+_20\d\d_(?:annual|6m))\D{1,20}?(\d{1,3})", text)]
        return {"answer": text, "sources": sources}
    answer = data.get("answer", "")
    if not isinstance(answer, str):
        answer = json.dumps(answer, ensure_ascii=False)
    sources = []
    for item in data.get("sources") or []:
        if not isinstance(item, dict):
            continue
        pages = item.get("pages", item.get("page"))
        pages = pages if isinstance(pages, list) else [pages]
        for page in pages:
            try:
                sources.append((str(item.get("doc_id", "")).strip(), int(page)))
            except (TypeError, ValueError):
                continue
    return {"answer": answer.strip(), "sources": sources}
