"""Verify a completed run and insert measured test means into README."""
import argparse
import json
from pathlib import Path
from statistics import mean
from rag.config import ROOT
from eval.run import METRICS

START = '<!-- TEST_METRICS_START -->'
END = '<!-- TEST_METRICS_END -->'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    run = json.loads(args.report.with_suffix('.run.json').read_text(encoding='utf-8'))
    events = [json.loads(line) for line in args.report.with_suffix('.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [event['row'] for event in events if event['event'] == 'result']
    dataset = [json.loads(line) for line in (ROOT / 'data/test.jsonl').read_text(encoding='utf-8').splitlines()]
    if not run['complete'] or len(rows) != 141 or {r['question_id'] for r in rows} != {r['question_id'] for r in dataset}:
        raise ValueError('Not a complete 141-question test run')
    averages = {metric: mean(row[metric] for row in rows) for metric in METRICS}
    for metric in METRICS:
        if abs(averages[metric] - run['averages'][metric]) > 1e-10:
            raise ValueError(f'Mean mismatch: {metric}')
    text = '## Итоговые метрики на test\n\nВсе 141 вопрос; полный независимый прогон.\n\n| Метрика | Среднее |\n| --- | ---: |\n'
    for metric, value in averages.items():
        text += f'| {metric} | {value:.6f}' + (' с' if metric == 'time' else '') + ' |\n'
    text += '\nExcel: [results/test_results.xlsx](results/test_results.xlsx). '
    text += 'Подробный журнал и параметры находятся рядом с отчетом.\n'
    readme = ROOT / 'README.md'
    source = readme.read_text(encoding='utf-8')
    block = START + '\n' + text + END
    if START in source:
        source = source[:source.index(START)] + block + source[source.index(END) + len(END):]
    else:
        split = source.index('## Решение')
        source = source[:split] + block + '\n\n' + source[split:]
    readme.write_text(source, encoding='utf-8')
    print(json.dumps(averages, ensure_ascii=False))

if __name__ == '__main__':
    main()
