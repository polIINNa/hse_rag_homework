import json
import pytest
from rag.build_index import chunks_for_page
from rag.llm import parse_json
from rag.pipeline import RAG
from rag.retrieval import organizations, context_from_pages, unit_excerpt, Retriever
from eval.eval_functions import score_context

def page(doc='domrf_2025_6m', number=5):
    return {'doc_id': doc, 'page': number, 'text': 'Чистые комиссионные доходы 5 592 млн руб.',
            'snippet': 'Чистые комиссионные доходы 5 592 млн руб.', 'organization': 'Банк ДОМ.РФ'}

class FakeRetriever:
    def __init__(self):
        self.queries = []
    def search(self, question, limit):
        self.queries.append(question)
        return [page(number=5), page(number=14)]
    def read_page(self, doc, number):
        return page(doc, number) if doc == 'domrf_2025_6m' and number in (5, 14) else None

class FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
    def complete(self, system, user, max_tokens):
        self.calls.append((system, user))
        return next(self.replies)

def decision(pages=(5,), queries=()):
    return json.dumps({'selected': [{'doc_id': 'domrf_2025_6m', 'page': p} for p in pages],
                       'search_queries': list(queries), 'read_pages': []})

def test_context_metrics_use_physical_pages_and_deduplicate():
    context = context_from_pages([page(number=5), page(number=5), page(number=14)])
    assert context == [{'doc_id': 'domrf_2025_6m', 'pages': [5, 14]}]
    assert score_context([{'doc_id': 'domrf_2025_6m', 'pages': [14]}], context) == {'context_recall': 1, 'context_precision': 0.5}

def test_chunks_preserve_metadata_and_tail():
    original = page()
    original['text'] = '\n'.join(f'row {i}  12 345  67 890' for i in range(200))
    chunks = list(chunks_for_page(original, 300, 50))
    assert len(chunks) > 10
    assert all(c['page'] == 5 and c['doc_id'] == original['doc_id'] for c in chunks)
    assert chunks[-1]['text'].endswith('row 199  12 345  67 890')
    assert all(len(c['text']) <= 300 for c in chunks)

def test_organization_aliases_and_multi_bank():
    assert set(organizations('Сбербанк, ВТБ, Т-Банк и Банк ДОМ.РФ')) == {'sber', 'vtb', 'tbank', 'domrf'}
    assert organizations('ЗАО Альфа-Банк (Беларусь)') == ['alfa']
    assert organizations('НПФ Газпромбанк-фонд') == ['gpb']

def test_agent_stops_when_sufficient_and_generator_sees_only_selected():
    retriever = FakeRetriever()
    llm = FakeLLM([decision(), '5 592 млн руб.'])
    rag = RAG(retriever, llm, agentic=True)
    result = rag.rag('Комиссионные доходы ДОМ.РФ?')
    assert len(retriever.queries) == 1
    assert len(llm.calls) == 2
    assert result['context'] == [{'doc_id': 'domrf_2025_6m', 'pages': [5]}]
    assert 'стр. 14' not in llm.calls[-1][1]
    assert 'gold_answer' not in llm.calls[-1][1]

def test_agent_searches_again_only_when_requested():
    retriever = FakeRetriever()
    llm = FakeLLM([decision(queries=['ДОМ.РФ комиссии 2025']), decision((5, 14)), '5 592 млн руб.'])
    result = RAG(retriever, llm, agentic=True).rag('Комиссионные доходы?')
    assert len(retriever.queries) == 2
    assert len(llm.calls) == 3
    assert result['context'][0]['pages'] == [5, 14]

def test_invalid_citations_are_not_returned():
    llm = FakeLLM([decision((999,)), 'Данных недостаточно.'])
    result = RAG(FakeRetriever(), llm, agentic=True).rag('Вопрос')
    assert 999 not in result['context'][0]['pages']

def test_json_validation():
    assert parse_json('```json\n{"selected": []}\n```') == {'selected': []}
    with pytest.raises(ValueError):
        parse_json('[]')

def test_teacher_rejects_zero_based_page_numbers():
    with pytest.raises(ValueError):
        score_context([{'doc_id': 'x', 'pages': [1]}], [{'doc_id': 'x', 'pages': [0]}])

def test_units_are_retrieved_from_source_and_counted_in_context():
    data = page('vtb_2024_annual', 45)
    data['text'] = 'Средства клиентов 26 926,8'
    units = page('vtb_2024_annual', 6)
    units['text'] = 'Отчет о прибылях и убытках (В МИЛЛИАРДАХ РОССИЙСКИХ РУБЛЕЙ)'
    assert unit_excerpt(units)
    retriever = object.__new__(Retriever)
    retriever.unit_sources = {'vtb_2024_annual': units}
    selected = retriever.add_unit_sources('Каков общий объём средств клиентов?', [data])
    assert context_from_pages(selected) == [{'doc_id': 'vtb_2024_annual', 'pages': [6, 45]}]
    assert retriever.add_unit_sources('Какие факторы риска?', [data]) == [data]

def test_unit_labels_are_not_confused_with_specific_amounts():
    assert unit_excerpt({'text': 'Отчет\nПрим.      млн руб.       млн руб.\n'})
    assert unit_excerpt({'text': 'Бухгалтерский баланс\n(тыс. руб.)'})
    assert unit_excerpt({'text': 'Стоимость отдельного объекта 100 млн руб.'}) is None
