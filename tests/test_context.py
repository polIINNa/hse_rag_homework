import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.context import ContextBuilder, declarations
from rag.pipeline import make_payload


class ContextChecks(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.index = Path(self.folder.name) / 'pages.sqlite'
        with sqlite3.connect(self.index) as connection:
            connection.execute('CREATE TABLE pages(doc_id TEXT, pdf_page INTEGER, organization TEXT, text TEXT, quality_flags TEXT)')

    def tearDown(self):
        self.folder.cleanup()

    def put(self, doc, number, text):
        with sqlite3.connect(self.index) as connection:
            connection.execute('INSERT INTO pages VALUES (?,?,?,?,?)', (doc, number, 'Bank', text, '[]'))
        return {'doc_id': doc, 'pdf_page': number, 'organization': 'Bank', 'text': text, 'quality_flags': []}

    def build(self, seeds, budget=3):
        builder = ContextBuilder(self.index, max_extra_pages=budget)
        try:
            return builder.build(seeds)
        finally:
            builder.close()

    def test_report_default_from_two_columns_is_cited_and_sent(self):
        seed = self.put('bank_2025', 66, 'Кредиты клиентам 23 197,5')
        evidence = 'Все суммы приведены в миллиардах российских    Новые стандарты\nрублей, если не указано иное.                  Обзор изменений'
        self.put('bank_2025', 86, evidence)
        pages, diagnostics = self.build([seed])
        self.assertEqual(pages[0]['unit_context']['unit'], 'млрд российских рублей')
        self.assertEqual(pages[0]['unit_context']['source_pdf_page'], 86)
        self.assertEqual(pages[1]['text'], evidence)
        self.assertTrue(pages[1]['excerpt'])
        self.assertEqual(diagnostics['total_pages'], 2)
        body = json.loads(make_payload('Сколько кредитов?', pages)['messages'][0]['content'])
        self.assertEqual(body['sources'][0]['unit_context'], pages[0]['unit_context'])
        self.assertEqual(body['sources'][1]['pdf_page'], 86)

    def test_single_amount_is_not_document_unit(self):
        self.assertEqual(declarations('Вознаграждение составило 10 миллионов российских рублей.'), [])
        seed = self.put('a_2025', 1, 'Активы 123,4')
        self.put('a_2025', 2, 'Вознаграждение составило 10 миллионов российских рублей.')
        # Для страницы применяются единицы измерения только её отчёта.
        self.put('b_2025', 3, 'Все суммы приведены в миллиардах российских рублей.')
        pages, _ = self.build([seed])
        self.assertEqual(len(pages), 1)
        self.assertNotIn('unit_context', pages[0])

    def test_local_header_takes_precedence(self):
        seed = self.put('a_2025', 1, '(В ТЫСЯЧАХ РОССИЙСКИХ РУБЛЕЙ)\nАктивы 123,4')
        self.put('a_2025', 2, 'Все суммы приведены в миллиардах российских рублей.')
        pages, _ = self.build([seed])
        self.assertEqual(len(pages), 1)
        self.assertNotIn('unit_context', pages[0])

    def test_conflicting_defaults_are_not_guessed(self):
        seed = self.put('a_2025', 1, 'Активы 123,4')
        self.put('a_2025', 2, 'Все суммы приведены в миллиардах российских рублей.')
        self.put('a_2025', 3, 'Данная финансовая отчетность представлена в тысячах российских рублей.')
        pages, diagnostics = self.build([seed])
        self.assertNotIn('unit_context', pages[0])
        self.assertEqual(diagnostics['documents_with_conflicting_default_units'], ['a_2025'])

    def test_table_continuation_requires_shared_section_and_columns(self):
        header = '2. СЕГМЕНТНЫЙ АНАЛИЗ (ПРОДОЛЖЕНИЕ)\nКорпоративный центр    Прочий бизнес    Казначейство\n2023 г. 2022 г. 2023 г. 2022 г.\n'
        seed = self.put('a_2023', 11, header + 'Прибыль 32,6 12,3 22,1')
        self.put('a_2023', 12, header + 'Активы 510,7 111,1 22,1')
        self.put('a_2023', 10, header.replace('2.', '3.') + 'Активы 123,4 15,2')
        pages, diagnostics = self.build([seed])
        self.assertEqual([p['pdf_page'] for p in pages], [11, 12])
        self.assertEqual(pages[1]['context_role'], 'table_continuation')
        self.assertEqual(pages[1]['related_to_page'], 11)
        self.assertEqual(diagnostics['roles']['table_continuation'], 1)

    def test_budget_does_not_leave_unsupported_unit_metadata(self):
        seed = self.put('a_2025', 1, 'Активы 123,4')
        self.put('a_2025', 2, 'Все суммы приведены в миллиардах российских рублей.')
        pages, diagnostics = self.build([seed], budget=0)
        self.assertEqual(len(pages), 1)
        self.assertNotIn('unit_context', pages[0])
        self.assertEqual(diagnostics['unit_references_skipped_by_budget'], ['a_2025'])


if __name__ == '__main__':
    unittest.main()
