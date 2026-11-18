"""Базовый RAG: вопрос → фильтр по метаданным → гибридный поиск → LLM (ответ + страницы-источники)."""

import re

from rag.config import TOP_K
from rag.llm import chat, parse_answer, traceable
from rag.prompts import SIMPLE_SYSTEM, SIMPLE_USER, format_pages
from rag.query_parser import candidate_docs, parse_question
from rag.retriever import get_index

MAX_PAGES_PER_DOC = 3


@traceable(name="retrieve_pages", run_type="retriever")
def initial_retrieval(question, k=TOP_K):
    """Первичный поиск. Если в вопросе несколько банков — ищем по каждому отдельно, чтобы ни один не потерялся."""
    index = get_index()
    priors = candidate_docs(question)
    banks = parse_question(question).banks
    if len(banks) <= 1:
        return index.search(question, priors, k=k)
    per_bank = max(3, k // len(banks) + 1)
    hits = []
    for bank in banks:
        bank_priors = {d: p for d, p in priors.items() if d.startswith(bank + "_")}
        hits += index.search(question, bank_priors, k=per_bank)
    return hits


def answer_numbers(answer):
    """Числа из ответа, по которым можно узнать страницу: от 4 цифр или с дробной частью (годы не считаем)."""
    numbers = set()
    for match in re.finditer(r"\d[\d  ]*(?:[,.]\d+)?", answer):
        token = re.sub(r"\s", "", match.group()).strip(",.")
        digits = re.sub(r"\D", "", token)
        if re.fullmatch(r"20[12]\d", token):
            continue
        if len(digits) >= 4 or (len(digits) >= 3 and re.search(r"[,.]", token)):
            numbers.add(token)
    return numbers


def select_context(answer, cited, shown):
    """Контекст для оценки: страницы, на которые сослалась модель, + показанные ей страницы тех же
    документов, где встречаются числа из ответа (одно значение часто есть и в отчёте, и в примечании)."""
    shown_keys = [(p["doc_id"], p["page"]) for p in shown]
    selected = [key for key in dict.fromkeys(cited) if key in shown_keys]
    numbers = answer_numbers(answer)
    if numbers:
        docs = {d for d, _ in selected} or {d for d, _ in shown_keys}
        for page in shown:
            key = (page["doc_id"], page["page"])
            if key in selected or page["doc_id"] not in docs:
                continue
            if sum(k[0] == page["doc_id"] for k in selected) >= MAX_PAGES_PER_DOC:
                continue
            text = re.sub(r"\s", "", page["text"])
            if any(n in text for n in numbers):
                selected.append(key)
    if not selected and shown_keys:
        selected = [shown_keys[0]]
    context = {}
    for doc_id, page in selected:
        context.setdefault(doc_id, []).append(page)
    return [{"doc_id": d, "pages": sorted(set(p))} for d, p in context.items()]


@traceable(name="simple_rag")
def simple_rag(question):
    pages = initial_retrieval(question, k=TOP_K)
    messages = [
        {"role": "system", "content": SIMPLE_SYSTEM},
        {"role": "user", "content": SIMPLE_USER.format(question=question, context=format_pages(pages))},
    ]
    parsed = parse_answer(chat(messages))
    return {"answer": parsed["answer"], "context": select_context(parsed["answer"], parsed["sources"], pages)}
