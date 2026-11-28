"""RAG по финансовой отчётности.

RAG — agentic RAG: LangGraph-цикл «модель ⇄ поиск», агент сам решает,
какие запросы делать и нужен ли ещё поиск. Ответ по всем найденным страницам формирует generate_answer.

SimpleRAG — тот же поиск без агента: один поиск по вопросу и ответ. Нужен для сравнения.

Поиск: векторный поиск кандидатов среди страниц банка из вопроса, переранжирование BM25 + RRF,
сначала страницы отчёта за год и период из вопроса.
"""

import os
import re
import time
from typing import Literal

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from rank_bm25 import BM25Okapi

from solution.build_index import open_index, page_id
from solution.generate import (ATTEMPTS, LLM_MODEL, count_tokens, generate_answer, page_heading, page_label,
                               retry_wait, wait_for_llm_slot)

CANDIDATES = 50  # сколько страниц берём из векторного поиска для переранжирования
DENSE_WEIGHT = 0.3  # вес векторного поиска в RRF, остальное — BM25 (подобрано на train)
RRF_C = 60
SIX_MONTHS = re.compile(r"шест\w* месяц|6 месяц|30 июня|полугод|промежуточн", re.IGNORECASE)
UNITS = re.compile(r"в\s+(?:миллиардах|миллионах|тысячах)\s+(?:(?:российских|белорусских)\s+)?рубл\w*", re.IGNORECASE)
MAX_SEARCHES = 4  # больше поисков на вопрос агент не делает: лимиты сервисов и время ответа

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
Bank = Literal["sber", "vtb", "tbank", "domrf", "mkb", "rshb", "sovkom", "alfa", "gpb"]  # коды для инструмента агента

AGENT_PROMPT = """Ты ищешь данные для ответа на вопрос по финансовой отчётности банков. Первый поиск по тексту вопроса уже сделан, его результат в истории. Реши, хватает ли найденных страниц, чтобы ответить на вопрос полностью.

Если чего-то не хватает, ищи через search_reports:
- query — что ищем: показатель и, если нужно, дата словами. Например: «средства клиентов 31 декабря 2024».
- bank, year, period — банк, год и период отчёта, если они понятны из вопроса. Данные на 31 декабря — годовой отчёт (annual) этого года, на 30 июня или за шесть месяцев — отчёт за 6 месяцев (6m).
- Если в вопросе несколько дат, периодов, банков или показателей (изменение, отношение, доля, сравнение), ищи каждое недостающее отдельным запросом.
- После каждого поиска проверь, нашлись ли все нужные числа. Не повторяй прежний запрос, меняй формулировку, дату или отчёт.

Когда данных достаточно или новые поиски ничего не дают, ответь без вызова инструмента: перечисли через запятую метки страниц, где есть данные для ответа, например «vtb_2024_annual:45, vtb_2024_annual:46». Если таблица продолжается на соседней странице, укажи обе. Окончательный ответ по этим страницам сформирует следующий шаг."""


def find_banks(question):
    """Банки, названные в вопросе. Пустой список — банк не распознан."""
    return [bank for bank, pattern in BANK_PATTERNS.items() if re.search(pattern, question, re.IGNORECASE)]


def tokenize(text):
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


def target_report(question):
    """Год и период отчёта из вопроса: последний упомянутый год; полугодие, если о нём есть слова."""
    years = [int(year) for year in re.findall(r"\b20[12]\d\b", question)]
    return (max(years) if years else None), ("6m" if SIX_MONTHS.search(question) else "annual")


def report_priority(document, year, period):
    """0 — отчёт за год и период из вопроса, 1 — за этот или следующий год (там данные для сравнения), 2 — остальные."""
    if year is None:
        return 0
    if document.metadata["year"] == year and document.metadata["period"] == period:
        return 0
    return 1 if document.metadata["year"] in (year, year + 1) else 2


def without_self(inputs):
    """Для трейсов LangSmith: у методов RAG не записываем self (там объект индекса Chroma)."""
    return {name: value for name, value in inputs.items() if name != "self"}


def pages_text(documents):
    """Что видит агент: метка, единицы отчёта и полный текст каждой найденной страницы."""
    return "\n\n".join(f"{page_heading(d)}\n{d.page_content}" for d in documents)


def report_units(ids, texts):
    """Основные единицы каждого отчёта: самая частая фраза вида «в миллионах российских рублей».
    В отчёте их пишут в шапке форм, а примерно на 3/4 страниц единиц нет совсем."""
    phrases = {}
    for page, text in zip(ids, texts):
        doc_id = page.split(":")[0]
        for phrase in UNITS.findall(text):
            counts = phrases.setdefault(doc_id, {})
            phrase = " ".join(phrase.lower().split())
            counts[phrase] = counts.get(phrase, 0) + 1
    return {doc_id: max(counts, key=counts.get) for doc_id, counts in phrases.items()}


def chosen_pages(message, documents):
    """Страницы, которые агент перечислил в последнем ответе. Пусто — агент ничего подходящего не назвал."""
    labels = set(re.findall(r"[a-z]+_20\d\d_(?:annual|6m):\d+", message.text))
    return [document for document in documents if page_label(document) in labels]


def to_context(documents):
    """Найденные страницы в формате score_context: [{"doc_id": ..., "pages": [...]}]."""
    pages = {}
    for document in documents:
        pages.setdefault(document.metadata["doc_id"], []).append(document.metadata["page"])
    return [{"doc_id": doc_id, "pages": doc_pages} for doc_id, doc_pages in pages.items()]


class SimpleRAG:
    """Один поиск по вопросу и ответ. Индекс загружается один раз при создании."""

    def __init__(self, k=5, use_bank_filter=True):
        self.store = open_index()
        stored = self.store.get(include=["documents"])  # те же тексты страниц, что в векторном индексе
        self.bm25_rows = {page: row for row, page in enumerate(stored["ids"])}
        self.bm25 = BM25Okapi([tokenize(text) for text in stored["documents"]])
        self.report_units = report_units(stored["ids"], stored["documents"])
        self.k = k
        self.use_bank_filter = use_bank_filter

    @traceable(run_type="retriever", name="search", process_inputs=without_self)
    def search(self, question, k=None, bank=None, year=None, period=None):
        """Лучшие страницы для вопроса. Если назван банк, ищем только среди его страниц.
        bank, year, period передаёт агент; если их нет, они определяются по тексту вопроса."""
        if bank:
            banks = [bank]
        else:
            banks = find_banks(question) if self.use_bank_filter else []
        if len(banks) == 1:
            bank_filter = {"bank": banks[0]}
        elif banks:
            bank_filter = {"bank": {"$in": banks}}
        else:
            bank_filter = None

        candidates = self.store.similarity_search(question, k=CANDIDATES, filter=bank_filter)

        # RRF
        bm25_scores = self.bm25.get_scores(tokenize(question))
        by_bm25 = sorted(candidates, key=lambda d: -bm25_scores[self.bm25_rows[page_id(d)]])
        fused = {}
        for weight, ranking in ((DENSE_WEIGHT, candidates), (1 - DENSE_WEIGHT, by_bm25)):
            for rank, document in enumerate(ranking, start=1):
                fused[page_id(document)] = fused.get(page_id(document), 0.0) + weight / (RRF_C + rank)

        text_year, text_period = target_report(question)
        year, period = year or text_year, period or text_period
        ranked = sorted(candidates, key=lambda d: (report_priority(d, year, period), -fused[page_id(d)]))
        for document in ranked:  # единицы отчёта для модели: на странице их часто нет
            document.metadata["units"] = self.report_units.get(document.metadata["doc_id"])
        return ranked[:k or self.k]

    @traceable(name="rag", process_inputs=without_self)
    def rag(self, question: str) -> dict:
        documents = self.search(question)
        answer, used = generate_answer(question, documents)
        return {"answer": answer, "context": to_context(used)}


class RAG(SimpleRAG):
    """Agentic RAG: поиск ведёт агент. Индекс и гибридный поиск — те же, что в SimpleRAG."""

    def __init__(self, k=5):
        super().__init__(k=k)

        @tool(response_format="content_and_artifact")
        def search_reports(query: str, bank: Bank | None = None, year: int | None = None,
                           period: Literal["annual", "6m"] | None = None):
            """Найди страницы финансовых отчётов банков. Возвращает несколько страниц с метками вида vtb_2024_annual:45.
            query — что ищем: показатель и, если нужно, дата словами.
            bank — код банка: sber (Сбербанк), vtb (ВТБ), tbank (Т-Банк), domrf (Банк ДОМ.РФ), mkb (Московский кредитный банк),
            rshb (Россельхозбанк), sovkom (Совкомбанк), alfa (ЗАО «Альфа-Банк», Беларусь), gpb (АО «НПФ Газпромбанк-фонд»).
            year — год отчёта. period — annual (годовой отчёт, данные на 31 декабря) или 6m (отчёт за 6 месяцев, на 30 июня)."""
            documents = self.search(query, bank=bank, year=year, period=period)
            return pages_text(documents), documents  # текст видит модель, сами страницы — в ToolMessage.artifact

        model = ChatAnthropic(
            model=LLM_MODEL,
            api_key=os.environ["LLM_API_KEY"],
            base_url="https://claude.gigaparsers.ru",
            temperature=0,
            max_tokens=1024,
            thinking={"type": "disabled"},
            max_retries=0,  # повторяем сами в model_node, чтобы каждый запрос проходил через общий лимит
            timeout=120,
        ).bind_tools([search_reports])

        def model_node(state: MessagesState):
            messages = [SystemMessage(content=AGENT_PROMPT)] + state["messages"]
            for attempt in range(ATTEMPTS):
                wait_for_llm_slot()
                try:
                    reply = model.invoke(messages)
                except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as error:
                    if attempt == ATTEMPTS - 1:  # 429, сбой сети или таймаут, 5xx
                        raise
                    time.sleep(retry_wait(getattr(error, "response", None), attempt))
                else:
                    count_tokens(reply.usage_metadata)
                    return {"messages": [reply]}

        def route_after_model(state: MessagesState):
            searches = sum(isinstance(message, ToolMessage) for message in state["messages"])
            if state["messages"][-1].tool_calls and searches < MAX_SEARCHES:
                return "tools"
            return END

        builder = StateGraph(MessagesState)
        builder.add_node("model", model_node)
        builder.add_node("tools", ToolNode([search_reports]))
        builder.add_edge(START, "model")
        builder.add_conditional_edges("model", route_after_model, {"tools": "tools", END: END})
        builder.add_edge("tools", "model")
        self.agent = builder.compile()

    @traceable(name="agentic_rag", process_inputs=without_self)
    def rag(self, question: str) -> dict:
        # Первый поиск — по тексту вопроса, без LLM. В историю он записан так, будто его сделал агент,
        # поэтому модель сразу решает, хватает ли найденного или нужен дополнительный поиск.
        first = self.search(question)
        call = {"name": "search_reports", "args": {"query": question}, "id": "first_search", "type": "tool_call"}
        state = self.agent.invoke({"messages": [
            HumanMessage(content=question),
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(content=pages_text(first), artifact=first, tool_call_id="first_search", name="search_reports"),
        ]})
        documents = {}  # страницы из всех поисков, без повторов, в порядке находки
        for message in state["messages"]:
            if isinstance(message, ToolMessage) and message.artifact:
                for document in message.artifact:
                    documents.setdefault(page_id(document), document)
        documents = list(documents.values())
        last = state["messages"][-1]
        # Агент перечислил нужные страницы — отвечаем по ним. Если не перечислил (или упёрся в лимит поисков) — по всем.
        chosen = [] if last.tool_calls else chosen_pages(last, documents)
        answer, used = generate_answer(question, chosen or documents)
        return {"answer": answer, "context": to_context(used)}
