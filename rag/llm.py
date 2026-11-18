import json
import os
import time

import requests
from langsmith import traceable
from rag.rate_limit import wait_for_slot


class LLM:
    def __init__(self):
        self.key = os.getenv("LLM_API_KEY")
        if not self.key:
            raise RuntimeError("Заполните LLM_API_KEY в .env ключом преподавателя")
        self.model = os.getenv("RAG_MODEL", "deepseek-flash")
        self.session = requests.Session()

    @traceable(run_type="llm", name="rag_decision")
    def complete(self, system, user):
        for attempt in range(4):
            try:
                wait_for_slot()
                response = self.session.post(
                    "https://claude.gigaparsers.ru/v1/messages",
                    headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                    json={"model": self.model, "max_tokens": 2400, "temperature": 0,
                          "thinking": {"type": "disabled"}, "system": system,
                          "messages": [{"role": "user", "content": user}]}, timeout=120,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                result = response.json()
                if result.get("stop_reason") == "max_tokens":
                    raise ValueError("Ответ модели обрезан по max_tokens")
                text = "".join(b["text"] for b in result["content"] if b["type"] == "text").strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                # Модель иногда вставляет буквальные переводы строк внутри answer.
                # Допускаем их, сохраняя содержимое ответа и проверку структуры JSON.
                return json.loads(text, strict=False)
            except requests.HTTPError as exc:
                if exc.response.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                delay = exc.response.headers.get("Retry-After", "")
                time.sleep(min(float(delay), 60) if delay.isdigit() else 6 * (attempt + 1))
            except (requests.Timeout, requests.ConnectionError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
