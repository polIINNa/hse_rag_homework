import json
import os
from rag.config import env_bool
from rag.llm import LLM, parse_json
from rag.prompts import SELECT_PROMPT, ANSWER_PROMPT
from rag.retrieval import Retriever, context_from_pages

class RAG:
    def __init__(self, retriever=None, llm=None, agentic=None):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLM()
        self.agentic = env_bool('RAG_AGENTIC') if agentic is None else agentic
        self.last_trace = {}

    def rag(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Question must be nonempty')
        candidates = self.retriever.search(question, limit=14)
        trace = {'question': question, 'searches': [question], 'decisions': []}
        selected = candidates[:4]
        if self.agentic:
            for step in range(2):
                excerpts = '\n\n'.join(f"[{p['doc_id']}, page={p['page']}]\n{p['snippet']}" for p in candidates)
                instruction = '\nЭто последний шаг: search_queries и read_pages должны быть пустыми.' if step else ''
                raw = self.llm.complete(SELECT_PROMPT + instruction, f'ВОПРОС: {question}\n\nКАНДИДАТЫ:\n{excerpts}', max_tokens=1000)
                decision = parse_json(raw)
                trace['decisions'].append(decision)
                known = {(p['doc_id'], p['page']): p for p in candidates}
                selected = []
                for item in decision.get('selected', []):
                    key = (item.get('doc_id'), item.get('page'))
                    if key in known and known[key] not in selected:
                        selected.append(known[key])
                selected = selected[:6]
                if step == 1:
                    break
                additions = []
                for query in decision.get('search_queries', [])[:2]:
                    if isinstance(query, str) and query.strip():
                        trace['searches'].append(query)
                        additions.extend(self.retriever.search(query, limit=6))
                for item in decision.get('read_pages', [])[:3]:
                    if not isinstance(item, dict) or type(item.get('page')) is not int:
                        continue
                    page = self.retriever.read_page(item.get('doc_id'), item['page'])
                    if page:
                        additions.append({**page, 'snippet': page['text'][:1300]})
                if not additions:
                    break
                # Keep selected evidence and fresh discoveries before earlier candidates.
                merged = {}
                for page in selected + additions + candidates:
                    merged.setdefault((page['doc_id'], page['page']), page)
                candidates = list(merged.values())[:18]
        if not selected:
            selected = candidates[:3]
            trace['selection_fallback'] = True
        if hasattr(self.retriever, 'add_unit_sources'):
            selected = self.retriever.add_unit_sources(question, selected)
        # Return exactly the pages actually supplied to the final generator.
        # The selector never sees gold answers or gold_context.
        context_text = '\n\n'.join(f"[{p['doc_id']}, стр. {p['page']}]\n{p['text']}" for p in selected)
        answer = self.llm.complete(ANSWER_PROMPT, f'ВОПРОС: {question}\n\nИСТОЧНИКИ:\n{context_text}', max_tokens=1800).strip()
        if not answer:
            raise ValueError('Empty answer from generator')
        context = context_from_pages(selected)
        self.last_trace = {**trace, 'context': context, 'answer': answer, 'context_characters': len(context_text)}
        return {'context': context, 'answer': answer}
