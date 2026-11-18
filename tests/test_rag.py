import pytest

from eval.eval_functions import score_context
from rag.build_index import chunks
from rag.retrieval import context_of
from rag.system import RAG
from rag.ocr import remove_grid
import numpy as np


class Retriever:
    def __init__(self):
        self.pages = {("bank_2025_annual", n): {"doc_id": "bank_2025_annual", "page": n, "text": f"page {n}"} for n in (1, 2)}
        self.searches = []

    def search(self, query, top_k=4, doc_ids=None):
        self.searches.append(query)
        return [self.pages[("bank_2025_annual", min(len(self.searches), 2))]]

    def get_page(self, doc_id, page):
        return [self.pages[(doc_id, page)]] if (doc_id, page) in self.pages else []


class LLM:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return next(self.decisions)


def test_agent_decides_to_search_again():
    retriever = Retriever()
    llm = LLM([{"action": "search", "query": "вторая часть", "doc_ids": []}, {"action": "answer", "answer": "оба факта"}])
    result = RAG(retriever, llm).rag("составной вопрос")
    assert retriever.searches == ["составной вопрос", "вторая часть"]
    assert result == {"answer": "оба факта", "context": [{"doc_id": "bank_2025_annual", "pages": [1, 2]}]}
    assert "page 2" in llm.prompts[-1]


def test_agent_can_finish_without_extra_search():
    retriever = Retriever()
    result = RAG(retriever, LLM([{"action": "answer", "answer": "факт"}])).rag("вопрос")
    assert len(retriever.searches) == 1
    assert result["context"][0]["pages"] == [1]


def test_agent_reads_next_page_and_deduplicates():
    llm = LLM([{"action": "read_page", "doc_id": "bank_2025_annual", "page": 2}, {"action": "answer", "answer": "факт"}])
    assert RAG(Retriever(), llm).rag("вопрос")["context"][0]["pages"] == [1, 2]
    assert context_of([{"doc_id": "a", "page": 1}] * 2) == [{"doc_id": "a", "pages": [1]}]


def test_search_budget_is_bounded():
    with pytest.raises(ValueError, match="исчерпания"):
        RAG(Retriever(), LLM([{"action": "search"}]), max_searches=0).rag("вопрос")


def test_no_phantom_document():
    with pytest.raises(ValueError, match="отсутствующий"):
        RAG(Retriever(), LLM([{"action": "search", "query": "q", "doc_ids": ["fake"]}])).rag("вопрос")


def test_chunk_coverage():
    text = "\n".join(f"line {i:04}" for i in range(200))
    parts = list(chunks(text, size=100, overlap=20))
    assert parts[-1].endswith("line 0199")
    assert all(len(p) <= 100 for p in parts)
    assert all(f"line {i:04}" in "\n".join(parts) for i in range(200))
    with pytest.raises(ValueError):
        list(chunks(text, size=10, overlap=10))


def test_official_context_scoring():
    score = score_context([{"doc_id": "d", "pages": [1, 2]}], [{"doc_id": "d", "pages": [2, 3]}])
    assert score == {"context_recall": .5, "context_precision": .5}
    with pytest.raises(ValueError):
        score_context([{"doc_id": "d", "pages": [0]}], [])


def test_scan_grid_removal_preserves_short_character_strokes():
    gray = np.full((200, 300), 255, dtype=np.uint8)
    gray[20, 10:290] = 0
    gray[20:180, 10] = 0
    gray[80:90, 80:82] = 0
    result = remove_grid(gray)
    assert np.all(result[20, 10:290] == 255)
    assert np.all(result[20:180, 10] == 255)
    assert np.array_equal(result[80:90, 80:82], gray[80:90, 80:82])


def test_llm_preserves_literal_line_breaks_in_answer(monkeypatch):
    from rag import llm

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(llm, "wait_for_slot", lambda: None)

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"content": [{"type": "text", "text": '{"action":"answer","answer":"Первая строка\nВторая строка"}'}]}

    model = llm.LLM()
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return Response()

    monkeypatch.setattr(model.session, "post", post)
    assert model.complete("instruction", "question") == {"action": "answer", "answer": "Первая строка\nВторая строка"}
    assert len(calls) == 1
