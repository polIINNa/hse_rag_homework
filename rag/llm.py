import json
import os
import time

import requests

from rag.common import load_env, traced
from rag.budget import budget


def parse_json(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Модель вернула JSON, который не является объектом")
    return result


class CourseLLM:
    def __init__(self):
        load_env()
        if not os.environ.get("LLM_API_KEY"):
            raise RuntimeError("Нужен LLM_API_KEY в .env или переменных среды; EMB_KEY его не заменяет")
        self.model = os.environ.get("RAG_MODEL", "deepseek-flash")
        self.session = requests.Session()

    @traced("generate_or_search", "llm")
    def complete(self, system, payload, *, json_output=True):
        for attempt in range(4):
            response = budget().request(self.session.post,
                "https://claude.gigaparsers.ru/v1/messages",
                headers={"x-api-key": os.environ["LLM_API_KEY"], "anthropic-version": "2023-06-01"},
                json={
                    "model": self.model, "max_tokens": 2400, "temperature": 0,
                    "thinking": {"type": "disabled"}, "system": system,
                    "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                }, timeout=90,
            )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 3:
                    delay = min(float(response.headers.get("Retry-After", 2 ** (attempt + 1))), 60)
                    time.sleep(max(delay, 0))
                    continue
            if not response.ok:
                raise RuntimeError(f"Шлюз LLM вернул HTTP {response.status_code}; ключ и лимиты проверьте отдельно")
            result = response.json()
            if result.get("stop_reason") == "max_tokens":
                raise RuntimeError("Ответ LLM обрезан по max_tokens; итоговый прогон остановлен")
            text = "".join(b["text"] for b in result["content"] if b.get("type") == "text")
            return parse_json(text) if json_output else text.strip()
        raise RuntimeError("Исчерпаны попытки обращения к LLM")
