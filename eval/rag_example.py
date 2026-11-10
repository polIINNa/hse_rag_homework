class RAG:
    """Шаблон. Здесь можно подготовить или загрузить свой индекс один раз."""

    def rag(self, question: str) -> dict:
        """Верните, например, {"answer": "...", "context": [{"doc_id": "domrf_2025_6m", "pages": [14]}]}."""
        raise NotImplementedError("Подключите здесь своё решение RAG.")
