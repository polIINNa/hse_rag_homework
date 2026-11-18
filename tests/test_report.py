from pathlib import Path
from openpyxl import load_workbook
from eval.run import COLUMNS, write_excel

def test_report_preserves_required_columns_numeric_types_and_literal_answers(tmp_path):
    row = dict(zip(COLUMNS, ['q1', '=dangerous()', [{'doc_id': 'd', 'pages': [1]}],
                            '100 млн руб.', [{'doc_id': 'd', 'pages': [1]}],
                            '=1+1', 1.0, 1.0, 0, 12.5]))
    target = tmp_path / 'report.xlsx'
    write_excel([row], target, {'complete': True, 'dataset': 'fixture.jsonl'})
    workbook = load_workbook(target)
    detail = workbook['По вопросам']
    assert list(next(detail.values)) == COLUMNS
    assert detail['B2'].data_type == 's'
    assert detail['F2'].data_type == 's'
    assert detail['J2'].value == 12.5
    assert workbook['Итоги']['B5'].value == "=AVERAGE('По вопросам'!G2:G2)"
    assert detail.freeze_panes == 'B2'
    workbook.close()
