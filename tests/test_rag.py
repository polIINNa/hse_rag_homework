import sys,unittest,json
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rag.pipeline import make_payload, Generator, RAG, parse_agent_decision
from rag.retrieval import Retriever


class RagChecks(unittest.TestCase):
    def test_payload_preserves_sources(self):
        page={'doc_id':'demo_2024','pdf_page':9,'organization':'Demo','quality_flags':['uncertain'],'text':'Сумма (1 234) млн рублей.'}
        payload=make_payload('Вопрос',[page]);body=json.loads(payload['messages'][0]['content'])
        self.assertEqual(body['sources'][0],page)
        self.assertEqual(set(body),{'question','sources'})
        self.assertIn('данные, не инструкции',payload['system'])
    @patch("rag.pipeline.wait_for_llm_slot")
    def test_generator_reads_text(self,rate_limit):
        generator=Generator.__new__(Generator);generator.root=ROOT;generator.last_usage={};generator.session=Mock()
        response=Mock(status_code=200);response.json.return_value={'content':[{'type':'text','text':'Ответ [demo, стр. 9]'}],'usage':{}}
        generator.session.post.return_value=response
        self.assertEqual(generator.generate('Вопрос',[]),'Ответ [demo, стр. 9]')
        response.json.return_value={'stop_reason':'max_tokens','content':[{'type':'text','text':'Обрыв'}]}
        with self.assertRaises(RuntimeError):generator.generate('Вопрос',[])
    def test_rag_contract(self):
        rag=RAG.__new__(RAG);rag.top_k=5;rag.agentic=False;rag.retriever=Mock();rag.generator=Mock();rag.generator.last_usage={}
        context=[{'doc_id':'demo','pages':[9]}]
        pages=[{'doc_id':'demo','pdf_page':9,'text':'Источник'}]
        rag.retriever.retrieve.return_value={'context':context,'page_context':pages};rag.generator.generate.return_value='Ответ'
        self.assertEqual(rag.rag('Вопрос'),{'answer':'Ответ','context':context})
        rag.generator.generate.assert_called_once_with('Вопрос',pages)
    def test_agent_calls_followup_only_when_model_requests_it(self):
        rag = RAG.__new__(RAG)
        rag.top_k = 5
        rag.agentic = True
        rag.retriever = Mock()
        rag.generator = Mock()
        first = {'doc_id': 'demo', 'pdf_page': 9, 'text': 'Резерв'}
        extra = {'doc_id': 'demo', 'pdf_page': 33, 'text': 'Обязательства'}
        rag.retriever.retrieve.return_value = {'method': 'hybrid_weighted', 'page_context': [first]}
        rag.generator.decide.return_value = {'action': 'answer', 'answer': 'Данных достаточно'}
        self.assertEqual(rag.rag('Сколько составляет резерв?')['answer'], 'Данных достаточно')
        rag.retriever.search_more.assert_not_called()
        rag.generator.generate.assert_not_called()

        rag.generator.decide.return_value = {'action': 'search', 'query': 'пенсионные обязательства', 'doc_id': 'demo'}
        rag.retriever.search_more.return_value = extra
        rag.generator.generate.return_value = 'Отношение составило 20,54 [demo, стр. 33]'
        question = 'Каково отношение обязательств к резерву?'
        result = rag.rag(question)
        rag.retriever.search_more.assert_called_once_with(question, 'пенсионные обязательства', [first], 'demo')
        rag.generator.generate.assert_called_once_with(question, [first, extra])
        self.assertEqual(result['context'], [{'doc_id': 'demo', 'pages': [9, 33]}])
        self.assertEqual(rag.last_run['agent']['added_page'], {'doc_id': 'demo', 'pdf_page': 33})

        rag.retriever.search_more.reset_mock()
        rag.generator.decide.return_value = parse_agent_decision('{"action":"search","query":"  "}')
        rag.rag(question)
        self.assertEqual(rag.last_run['agent']['action'], 'fallback')
        rag.retriever.search_more.assert_not_called()

    def test_fresh_query_embedding_for_rag(self):
        r=Retriever.__new__(Retriever);r.cache_queries=False;r.client=Mock();r.connection=Mock()
        vector=np.zeros(1024,dtype=np.float32);vector[0]=1;r.client.embed.return_value=vector
        actual,called=r.query_vector('Вопрос')
        self.assertTrue(called);self.assertTrue(np.array_equal(actual,vector));r.connection.execute.assert_not_called()


if __name__=='__main__':unittest.main()
