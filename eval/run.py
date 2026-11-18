"""Reproducible, resumable evaluation with the unmodified teacher's judge."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from statistics import mean
from time import perf_counter

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from rag.config import ROOT, INDEX
from rag.pipeline import RAG
from rag.llm import retry_call
from eval.eval_functions import score_context, score_answer
from eval.evaluator_excel import read_items

COLUMNS = ['question_id', 'question', 'gold_context', 'gold_answer', 'context', 'answer',
           'context_recall', 'context_precision', 'answer_correctness', 'time']
METRICS = COLUMNS[-4:]

def write_metadata(path, metadata):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def fingerprint(dataset):
    files = sorted((ROOT / 'rag').glob('*.py')) + sorted((ROOT / 'eval').glob('*.py'))
    code = hashlib.sha256()
    for path in files:
        code.update(path.relative_to(ROOT).as_posix().encode())
        code.update(path.read_bytes())
    return {'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'code_sha256': code.hexdigest(), 'model': os.getenv('RAG_MODEL', 'deepseek-flash'),
            'agentic': os.getenv('RAG_AGENTIC', 'true'),
            'api_interval': os.getenv('RAG_API_INTERVAL', '6.2'),
            'index_manifest': json.loads((INDEX / 'manifest.json').read_text())}

def write_excel(results, path, metadata):
    """Keep the provided Python/openpyxl report format and exact required columns."""
    workbook = Workbook()
    summary = workbook.active
    summary.title = 'Итоги'
    summary.append(['RAG по финансовой отчетности', 'Екатерина Шумакова'])
    summary.append(['Вопросов оценено', len(results)])
    summary.append(['Статус', 'Завершён' if metadata['complete'] else 'Частичный прогон'])
    summary.append(['Метрика', 'Среднее'])
    details = workbook.create_sheet('По вопросам')
    details.append(COLUMNS)
    for result in results:
        details.append([json.dumps(result[key], ensure_ascii=False) if key in ('context', 'gold_context') else result[key] for key in COLUMNS])
    for index, metric in enumerate(METRICS, start=5):
        column = chr(ord('G') + METRICS.index(metric))
        summary.cell(index, 1, metric)
        summary.cell(index, 2, f"=AVERAGE('По вопросам'!{column}2:{column}{len(results) + 1})")
        summary.cell(index, 2).number_format = '0.0000' if metric != 'time' else '0.00" с"'
    summary.append(['Данные', str(metadata['dataset'])])
    summary.append(['Судья', 'deepseek-flash; исходные модель и промпт сохранены'])
    summary.append(['Время', 'От готового RAG до полного ответа; судья не включен'])
    summary.append(['Источник', 'https://github.com/polIINNa/hse_rag_homework'])
    summary.column_dimensions['A'].width = 34
    summary.column_dimensions['B'].width = 75
    widths = [24, 65, 45, 65, 45, 80, 21, 21, 22, 16]
    for index, width in enumerate(widths, start=1):
        details.column_dimensions[chr(64 + index)].width = width
    for sheet in (summary, details):
        sheet.freeze_panes = 'B2' if sheet == details else 'A5'
        for row in sheet:
            for cell in row:
                cell.font = Font(name='Calibri', size=11)
                cell.alignment = Alignment(vertical='top', wrap_text=True)
                if isinstance(cell.value, str) and (sheet == details or cell.data_type != 'f'):
                    cell.data_type = 's'
        for cell in sheet[1]:
            cell.fill = PatternFill('solid', fgColor='17365D')
            cell.font = Font(name='Calibri', size=11, color='FFFFFF', bold=True)
        sheet.row_dimensions[1].height = 32
    details.auto_filter.ref = details.dimensions
    for row in range(2, len(results) + 2):
        line_count = max(sum(max(1, math.ceil(len(line) / (widths[col - 1] * 0.9)))
                             for line in str(details.cell(row, col).value or '').split('\n'))
                         for col in range(1, 7))
        details.row_dimensions[row].height = min(409, max(45, line_count * 15 + 12))
        for column in range(7, 11):
            details.cell(row, column).number_format = '0.0000' if column < 10 else '0.00'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.xlsx')
    workbook.save(temporary)
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/test_results.xlsx')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    items = list(read_items(args.dataset))
    if args.limit:
        items = items[:args.limit]
    if not items or len({item['question_id'] for item in items}) != len(items):
        raise ValueError('Dataset is empty or has duplicate question ids')
    if args.output.resolve() == args.dataset.resolve():
        raise ValueError('Output must differ from input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal = args.output.with_suffix('.jsonl')
    manifest_path = args.output.with_suffix('.run.json')
    expected = fingerprint(args.dataset)
    results = []
    predictions = {}
    previous = {}
    cumulative_usage = {'input_tokens': 0, 'output_tokens': 0, 'calls': 0}
    if args.resume and journal.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['fingerprint'] != expected:
            raise ValueError('Configuration, code, dataset or index changed: use a fresh output path')
        for line in journal.read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            if event['event'] == 'prediction':
                predictions[event['question_id']] = event
                for key in cumulative_usage:
                    cumulative_usage[key] = max(cumulative_usage[key], event.get('generator_usage', {}).get(key, 0))
            elif event['event'] == 'result':
                results.append(event['row'])
    elif journal.exists() or args.output.exists():
        raise ValueError('Report exists: use --resume or a different --output for a fresh run')
    completed = {row['question_id'] for row in results}
    if not completed.issubset({item['question_id'] for item in items}):
        raise ValueError('Resume selection does not include completed questions')
    if len(results) == len(items):
        print('Run is already complete. Choose a new --output for an independent rerun.')
        return
    metadata = {'fingerprint': expected, 'dataset': args.dataset.name, 'started_utc': previous.get('started_utc', datetime.now(timezone.utc).isoformat()),
                'expected_questions': len(items), 'complete': False,
                'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'logical_cpus': os.cpu_count()}}
    write_metadata(manifest_path, metadata)
    system = RAG()
    system.llm.usage.update(cumulative_usage)
    with journal.open('a', encoding='utf-8') as stream:
        def append(event):
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        for item in items:
            qid = item['question_id']
            if qid in completed:
                continue
            if qid in predictions:
                event = predictions[qid]
                prediction, elapsed = event['prediction'], event['time']
            else:
                start = perf_counter()
                prediction = system.rag(item['question'])
                elapsed = perf_counter() - start
                append({'event': 'prediction', 'question_id': qid, 'prediction': prediction, 'time': elapsed,
                        'trace': system.last_trace, 'generator_usage': system.llm.usage})
            # Labels become visible only AFTER rag(question) has finished.
            scores = score_context(item['gold_context'], prediction['context'])
            judged = retry_call(lambda: score_answer(item['gold_answer'], prediction['answer']))
            row = {key: item[key] for key in COLUMNS[:4]}
            row.update(**prediction, **scores, answer_correctness=judged['answer_correctness'], time=elapsed)
            append({'event': 'result', 'row': row, 'judge_reason': judged['reason']})
            results.append(row)
            completed.add(qid)
            averages = {metric: mean(result[metric] for result in results) for metric in METRICS}
            metadata.update(averages=averages, completed_questions=len(results), generator_usage=system.llm.usage,
                            complete=len(results) == len(items))
            write_metadata(manifest_path, metadata)
            write_excel(results, args.output, metadata)
            print(json.dumps({'done': len(results), 'total': len(items), 'question_id': qid,
                              'averages': averages, 'usage': system.llm.usage}, ensure_ascii=False), flush=True)
    print(f'Saved {args.output}', flush=True)

if __name__ == '__main__':
    main()
