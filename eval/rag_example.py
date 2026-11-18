"""RAG по финансовой отчётности: векторный поиск страниц с фильтром по банку и ответ LLM по найденным страницам."""

import re

from solution.build_index import open_index
from solution.generate import generate_answer

# Как банки называют в вопросах. Ключ — код банка, как в doc_id и в метаданных индекса.
BANK_PATTERNS = {
    "sber": r"\bсбербанк|\bсбер\w{0,2}\b",
    "vtb": r"\bвтб\b",
    "tbank": r"\bт-?банк|тинькофф",
    "domrf": r"дом\.?\s?рф",
    "mkb": r"\bмкб\b|московск\w* кредитн",
    "rshb": r"россельхоз|\bрсхб\b|сельскохозяйственн",
    "sovkom": r"совкомбанк",
    "alfa": r"альфа",
    "gpb": r"газпромбанк",
}


def find_banks(question):
    """Банки, названные в вопросе. Пустой список — банк не распознан."""
    return [bank for bank, pattern in BANK_PATTERNS.items() if re.search(pattern, question, re.IGNORECASE)]


def to_context(documents):
    """Найденные страницы в формате score_context: [{"doc_id": ..., "pages": [...]}]."""
    pages = {}
    for document in documents:
        pages.setdefault(document.metadata["doc_id"], []).append(document.metadata["page"])
    return [{"doc_id": doc_id, "pages": doc_pages} for doc_id, doc_pages in pages.items()]


class RAG:
    """Индекс загружается один раз при создании."""

    def __init__(self, k=3, use_bank_filter=True):
        self.store = open_index()
        self.k = k
        self.use_bank_filter = use_bank_filter

    def search(self, question, k=None):
        """Страницы, ближайшие к вопросу. Если в вопросе назван банк, ищем только среди его страниц."""
        banks = find_banks(question) if self.use_bank_filter else []
        if len(banks) == 1:
            bank_filter = {"bank": banks[0]}
        elif banks:
            bank_filter = {"bank": {"$in": banks}}
        else:
            bank_filter = None
            
        return self.store.similarity_search(question, k=k or self.k, filter=bank_filter)

    def rag(self, question: str) -> dict:
        documents = self.search(question)
        answer, used = generate_answer(question, documents)
        return {"answer": answer, "context": to_context(used)}
