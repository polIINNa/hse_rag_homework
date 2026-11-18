"""Agentic RAG на LangGraph (как в семинаре 3): модель сама решает, хватает ли найденного или нужен ещё поиск.

START → model ── ответ (JSON без tool call) ──→ END
          │                                     ↑
          └─ tool call → tools → model ─────────┘

- Первичный поиск делаем до агента и кладём результаты в первое сообщение: на простых вопросах
  модель отвечает сразу одним вызовом, без лишних шагов.
- Инструменты: search_reports(query, bank, year, period) — гибридный поиск с фильтрами по метаданным;
  read_page(doc_id, page) — прочитать конкретную страницу (продолжение таблицы).
- Жёсткий лимит: после MAX_AGENT_SEARCHES раундов инструментов модель обязана ответить по найденному.
"""

from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from rag.config import (IS_OPENROUTER, LLM_API_FORMAT, LLM_MODEL, LLM_ROOT_URL, LLM_TIMEOUT, MAX_AGENT_SEARCHES,
                        REASONING_EFFORT, TOP_K, llm_api_key)
from rag.llm import call_with_retries, parse_answer, strip_thinking, traceable
from rag.pipeline import initial_retrieval, select_context
from rag.prompts import AGENT_SYSTEM, AGENT_USER, SIMPLE_SYSTEM, SIMPLE_USER, format_pages
from rag.query_parser import candidate_docs
from rag.retriever import get_index

Bank = Literal["sber", "vtb", "tbank", "domrf", "mkb", "rshb", "sovkom", "alfa", "gpb"]
Period = Literal["annual", "6m"]


def make_llm():
    if LLM_API_FORMAT == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=LLM_MODEL, api_key=llm_api_key(), base_url=LLM_ROOT_URL, temperature=0,
                             max_tokens=4000, timeout=LLM_TIMEOUT, max_retries=0,
                             default_headers={"Authorization": "Bearer " + llm_api_key()})
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=llm_api_key(),
        base_url=LLM_ROOT_URL + "/v1",
        use_responses_api=False,
        temperature=0,
        max_tokens=4000,
        timeout=LLM_TIMEOUT,
        max_retries=0,                    # повторы делает call_with_retries с учётом лимита
        extra_body={"reasoning": {"effort": REASONING_EFFORT}} if IS_OPENROUTER else None,
    )


def make_tools(seen: dict):
    """Инструменты одного запуска: seen — все страницы, которые видела модель (из них берём источники)."""

    @tool
    def search_reports(query: str, bank: Optional[Bank] = None, year: Optional[int] = None,
                       period: Optional[Period] = None) -> str:
        """Ищет страницы в МСФО-отчётах. query — самодостаточный запрос на русском: показатель, дата или
        период, банк. Фильтры: bank — код банка; year — год отчёта (2022–2025); period — annual (годовой)
        или 6m (за 6 месяцев, на 30 июня). Данные на 31 декабря прошлого года есть и в отчёте следующего года.
        Возвращает до 4 ещё не показанных страниц с doc_id и номером страницы."""
        priors = candidate_docs(query, bank=bank, year=year, period=period)
        hits = get_index().search(query, priors, k=4, exclude=set(seen))
        for hit in hits:
            seen[(hit["doc_id"], hit["page"])] = hit
        return format_pages(hits) if hits else "Новых страниц по этому запросу и фильтрам нет."

    @tool
    def read_page(doc_id: str, page: int) -> str:
        """Возвращает текст страницы page документа doc_id, например продолжение таблицы на следующей странице."""
        found = get_index().get_page(doc_id, page)
        if found is None:
            return f"Страницы {page} в документе {doc_id} нет."
        seen[(found["doc_id"], found["page"])] = found
        return format_pages([found])

    return [search_reports, read_page]


def route_after_model(state: MessagesState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_agent(llm, tools, question, seen, max_rounds=MAX_AGENT_SEARCHES):
    model_with_tools = llm.bind_tools(tools)

    def model_node(state: MessagesState):
        rounds = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        if rounds >= max_rounds:
            # лимит поиска исчерпан: один вызов без инструментов по всем найденным страницам
            response = call_with_retries(lambda: llm.invoke([
                SystemMessage(content=SIMPLE_SYSTEM),
                HumanMessage(content=SIMPLE_USER.format(question=question, context=format_pages(list(seen.values())))),
            ]))
        else:
            response = call_with_retries(
                lambda: model_with_tools.invoke([SystemMessage(content=AGENT_SYSTEM)] + state["messages"]))
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("model", model_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route_after_model, {"tools": "tools", END: END})
    builder.add_edge("tools", "model")
    return builder.compile()


def message_text(message):
    content = message.content
    if not isinstance(content, str):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return strip_thinking(content)


class AgenticRAG:
    def __init__(self):
        self.llm = make_llm()

    @traceable(name="agentic_rag")
    def run(self, question):
        pages = initial_retrieval(question, k=TOP_K)
        seen = {(p["doc_id"], p["page"]): p for p in pages}
        agent = build_agent(self.llm, make_tools(seen), question, seen)
        result = agent.invoke(
            {"messages": [HumanMessage(content=AGENT_USER.format(question=question, context=format_pages(pages)))]},
            config={"recursion_limit": 4 * MAX_AGENT_SEARCHES + 4, "run_name": "agentic_rag_graph"},
        )
        final = message_text(result["messages"][-1])
        parsed = parse_answer(final)
        searches = sum(len(m.tool_calls) for m in result["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        shown = list(seen.values())
        return {
            "answer": parsed["answer"],
            "context": select_context(parsed["answer"], parsed["sources"], shown),
            "tool_calls": searches,
        }
