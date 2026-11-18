"""Проверки разбиения текста и поискового запроса по готовому локальному индексу."""
import sqlite3
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from scripts.build_index import windows, table_blocks, chunks_for_page
from rag.retrieval import stem_search


def cell(text, row, column, header=False, columns=1):
    return {'text': text, 'start_row_offset_idx': row, 'end_row_offset_idx': row+1,
            'start_col_offset_idx': column, 'end_col_offset_idx': column+columns,
            'column_header': header, 'row_span': 1, 'col_span': columns}


class ChunkChecks(unittest.TestCase):
    def test_source_coverage(self):
        for source in ['', 'Короткий текст', 'я'*9000, 'Сумма: (1 234,56) млн руб.\n'*1000]:
            spans = list(windows(source,size=2200,overlap=350))
            if not source:
                self.assertEqual(spans, [])
                continue
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], len(source))
            for start, end, body in spans:
                self.assertEqual(body, source[start:end])
                self.assertLessEqual(end-start, 2200)
            for left, right in zip(spans, spans[1:]):
                self.assertLessEqual(right[0], left[1])
                self.assertGreater(right[0], left[0])

    def test_period_associations_and_merged_cells(self):
        cells = [cell('Показатель',0,0,True),cell('За шесть месяцев',0,1,True,2),cell('За три месяца',0,3,True,2)]
        cells += [cell(year,1,col,True) for col,year in enumerate(['2025','2024','2025','2024'],1)]
        cells += [cell('Чистые комиссионные доходы',2,0)]
        cells += [cell(value,2,col) for col,value in enumerate(['5 592','1 742','3 401','1 176'],1)]
        cells += [cell('Объединённая строка',3,0),cell('100',3,1,columns=2)]
        blocks = list(table_blocks({'rows':4,'cols':5,'cells':cells}))
        self.assertIn('За шесть месяцев / 2025: 5 592', blocks[0])
        self.assertIn('За шесть месяцев / 2024: 1 742', blocks[0])
        self.assertIn('За три месяца / 2025: 3 401', blocks[0])
        self.assertIn('За три месяца / 2024: 1 176', blocks[0])
        self.assertEqual(blocks[1].count(': 100'), 1)
        self.assertIn('[объединённая ячейка]', blocks[1])

    def test_source_metadata_and_units(self):
        page = {'doc_id':'demo','pdf_page':14,'organization':'Банк','text':'(в миллионах рублей)\n'+('Сумма (1 234)\n'*500),
                'text_available':True,'quality_flags':['table_structure_pending'],'tables':{'status':'pending'}}
        chunks = list(chunks_for_page(page))
        self.assertGreater(len(chunks),1)
        for chunk in chunks:
            self.assertIn('Документ: demo\nPDF-страница: 14', chunk['text'])
            self.assertIn('(в миллионах рублей)', chunk['text'])
            self.assertIn('table_structure_pending',chunk['flags'])

    @unittest.skipUnless((ROOT / 'index/lexical_stems.sqlite').exists(), 'Build index first')
    def test_known_question_returns_original_page(self):
        results = stem_search(
            'чистые комиссионные доходы', ROOT / 'index/lexical_stems.sqlite', 3,
            doc_ids={'domrf_2025_6m'},
        )
        self.assertIn(('domrf_2025_6m', 5), results)
        self.assertEqual(len(set(results)), len(results))
        corpus = (ROOT / 'index/corpus.sqlite').as_uri() + '?mode=ro'
        with sqlite3.connect(corpus, uri=True) as db:
            page = db.execute(
                'SELECT text FROM pages WHERE doc_id = ? AND pdf_page = ?', ('domrf_2025_6m', 5)
            ).fetchone()
        self.assertIn('5 592', page[0])


if __name__=='__main__':
    unittest.main()
