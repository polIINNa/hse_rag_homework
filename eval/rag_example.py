"""RAG по финансовой отчётности. Формат ответа: {"answer": str, "context": [{"doc_id": str, "pages": [int]}]}.

Режим задаётся переменной окружения RAG_MODE: agentic (по умолчанию, LangGraph-агент) или simple (базовый RAG).
"""

import logging

from rag.config import RAG_MODE
from rag.pipeline import simple_rag
from rag.retriever import get_index

log = logging.getLogger(__name__)


class RAG:
    """Индекс загружается один раз при создании объекта, rag() отвечает на один вопрос."""

    def __init__(self, mode: str | None = None):
        self.mode = mode or RAG_MODE
        get_index()
        self.agent = None
        if self.mode == "agentic":
            from rag.agent import AgenticRAG

            self.agent = AgenticRAG()

    def rag(self, question: str) -> dict:
        if self.agent is not None:
            try:
                result = self.agent.run(question)
            except Exception as error:  # сбой агента не должен ронять весь прогон — отвечаем базовым RAG
                log.warning("Агент упал (%s), отвечаю базовым RAG", error)
                result = simple_rag(question)
        else:
            result = simple_rag(question)
        return {"answer": result["answer"], "context": result["context"]}
