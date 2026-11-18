"""Подготовить страницы PDF для индекса. Запуск из корня репозитория.

Стадия native читает текстовый слой PDF, ocr распознаёт страницы с непригодным текстом,
tables извлекает структуру таблиц через Docling. В результате сохраняются номера страниц PDF.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import re
import tempfile
import time

REVISION = 1
# Страницы, для которых в итоговый индекс вошли результаты Docling.
TABLE_PAGES = {
    'alfa_2022_annual': (27,),
    'alfa_2024_annual': (14, 20, 40),
    'domrf_2025_6m': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
    'gpb_2023_annual': (10,),
    'gpb_2025_annual': (20,),
    'mkb_2024_annual': (41,),
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def alnum(text):
    return sum(c.isalnum() for c in text)


def table_candidate(text):
    return sum(len(re.findall(r'\d+', line)) >= 3 for line in text.splitlines()) >= 3


def ocr_layout(observations, width, height):
    rows = []
    for item in sorted(observations, key=lambda x: sum(p[1] for p in x['box']) / 4):
        center = sum(p[1] for p in item['box']) / 4 / height
        if not rows or abs(rows[-1][0] - center) > 0.004:
            rows.append((center, []))
        rows[-1][1].append((min(p[0] for p in item['box']) / width, item['text']))
    lines = []
    for _, items in rows:
        line = ''
        for x, text in sorted(items):
            line += ' ' * max(1 if line else 0, round(x * 180) - len(line)) + text
        lines.append(line.rstrip())
    return '\n'.join(lines)


class LocalOCR:
    def __init__(self):
        from rapidocr import RapidOCR, ModelType, OCRVersion
        from rapidocr.utils.typings import EngineType
        # Используем те же мобильные веса PP-OCRv5, что и Docling для русского языка.
        # Вызываем RapidOCR напрямую, чтобы не загружать модели анализа разметки страниц.
        self.reader = RapidOCR(params={
            'Global.text_score': 0.0, 'Global.use_cls': False, 'Global.log_level': 'warning',
            'EngineConfig.onnxruntime.intra_op_num_threads': 4,
            'Det.engine_type': EngineType.ONNXRUNTIME, 'Rec.engine_type': EngineType.ONNXRUNTIME,
            'Det.ocr_version': OCRVersion.PPOCRV5, 'Det.lang_type': 'ch', 'Det.model_type': ModelType.MOBILE,
            'Rec.ocr_version': OCRVersion.PPOCRV5, 'Rec.lang_type': 'eslav', 'Rec.model_type': ModelType.MOBILE})

    def read(self, image):
        import numpy as np
        result = self.reader(np.asarray(image), use_cls=False)
        if result is None or result.boxes is None:
            return []
        return [{'text': str(t), 'score': float(s), 'box': b.tolist()}
                for b, t, s in zip(result.boxes, result.txts, result.scores)]

    @staticmethod
    def quality(items):
        return sum(alnum(x['text']) * x['score'] for x in items if x['score'] >= 0.5)

    def extract(self, image):
        original = self.read(image)
        best, clockwise, frame = original, 0, image
        score = self.quality(best)
        text = ' '.join(x['text'] for x in best if x['score'] >= 0.5)
        cyrillic = len(re.findall('[А-Яа-яЁё]', text))
        # Ориентацию страницы выбираем по результатам пробного распознавания.
        if score < 150 or (cyrillic < 10 and sum(c.isdigit() for c in text) > 20):
            for angle in (90, 270, 180):
                rotated = image.rotate(-angle, expand=True)
                candidate = self.read(rotated)
                candidate_score = self.quality(candidate)
                if candidate_score > score * 1.5 + 20:
                    best, clockwise, frame, score = candidate, angle, rotated, candidate_score
        good = [x for x in best if x['score'] >= 0.5]
        return {'status': 'complete', 'engine': 'RapidOCR/eslav/onnxruntime',
                'rotation_clockwise': clockwise, 'rotation_selection': 'heuristic',
                'frame_width': frame.width, 'frame_height': frame.height,
                'coordinate_origin': 'top_left_of_rotated_render', 'render_dpi': 216,
                'text': ocr_layout(good, frame.width, frame.height),
                'observations': best,
                'mean_confidence': sum(x['score'] for x in good) / len(good) if good else 0.0,
                'discarded_low_confidence_lines': len(best) - len(good)}


class TableExtractor:
    def __init__(self):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, TableFormerMode
        from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
        from docling.document_converter import DocumentConverter, PdfFormatOption
        options = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
        options.ocr_options = RapidOcrOptions(lang=['iso:ru'], backend='onnxruntime', use_cls=False)
        options.table_structure_options.mode = TableFormerMode.ACCURATE
        options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=4)
        self.converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})

    def extract(self, source, number, rotation, image=None):
        with tempfile.TemporaryDirectory(prefix='hse-table-') as temporary:
            input_page = number
            if rotation:
                if image is None:
                    raise ValueError('Rotation requires a rendered original page')
                normalized = image.rotate(-rotation, expand=True)
                source = Path(temporary) / 'normalized.pdf'
                normalized.save(source, resolution=216)
                input_page = 1
            result = self.converter.convert(source, page_range=(input_page, input_page))
            if str(result.status) != 'ConversionStatus.SUCCESS':
                raise RuntimeError(f'Docling conversion not complete: {result.status}')
            document = result.document
            return {'status': 'complete', 'engine': 'Docling/TableFormer/accurate',
                    'rotation_clockwise': rotation, 'docling_input_page': input_page,
                    'document_markdown': document.export_to_markdown(),
                    'items': [{'rows': t.data.num_rows, 'cols': t.data.num_cols,
                               'cells': [c.model_dump(mode='json') for c in t.data.table_cells],
                               'html': t.export_to_html(doc=document)} for t in document.tables],
                    'errors': [str(e) for e in result.errors], 'verified': False}


def finalise(record):
    native, ocr, tables = record['native'], record.get('ocr', {}), record.get('tables', {})
    text = native.get('layout', '') or native.get('plain', '')
    record['text_source'] = 'pypdf'
    if ocr.get('status') == 'complete' and alnum(ocr['text']) > alnum(text):
        text, record['text_source'] = ocr['text'], 'rapidocr'
    record['text'] = text
    flags = []
    if native.get('status') == 'error':
        flags.append('native_extraction_failed')
    if record['requires_ocr'] and ocr.get('status') not in ('complete',):
        flags.append('ocr_' + ocr.get('status', 'pending'))
    if not alnum(text):
        flags.append('blank_or_unreadable')
    elif alnum(text) < 40:
        flags.append('very_short_text')
    if '\ufffd' in text or re.search(r'\(cid:\d+\)', text):
        flags.append('broken_character_mapping')
    if ocr.get('rotation_clockwise', 0):
        flags.append('page_rotation_inferred')
    if ocr.get('status') == 'complete' and ocr.get('discarded_low_confidence_lines', 0):
        flags.append('low_confidence_ocr_lines')
    record['table_candidate'] = table_candidate(text)
    if tables.get('status') == 'complete':
        if tables['items']:
            flags.append('table_structure_unverified')
    elif record['table_candidate']:
        flags.append('table_structure_' + tables.get('status', 'pending'))
    if tables.get('status') == 'error':
        flags.append('table_extraction_failed')
    record['quality_flags'] = flags
    record['text_available'] = alnum(text) >= 40


def export_corpus(output, documents, reports, native_config):
    counts, flags = Counter(), Counter()
    export = output / 'pages.jsonl.tmp'
    issues = output / 'issues.jsonl.tmp'
    with export.open('w', encoding='utf-8') as pages, issues.open('w', encoding='utf-8') as review:
        for document in documents:
            with (reports / document['path']).open('rb') as stream:
                source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            expected_key = digest({'config': native_config, 'source_sha256': source_hash})
            for number in range(1, document['pages'] + 1):
                record = read_json(output / 'pages' / document['doc_id'] / f'{number:04d}.json')
                if record is None or record.get('native', {}).get('fingerprint') != expected_key:
                    counts['not_processed'] += 1
                    continue
                record['organization'] = document['organization']
                record['source'] = document['path']
                finalise(record)
                pages.write(json.dumps(record, ensure_ascii=False) + '\n')
                counts['pages_saved'] += 1
                counts['text_available'] += record['text_available']
                counts['ocr_required'] += record['requires_ocr']
                counts['ocr_complete'] += record.get('ocr', {}).get('status') == 'complete'
                counts['tables_complete'] += record.get('tables', {}).get('status') == 'complete'
                counts['tables_found'] += len(record.get('tables', {}).get('items', []))
                if record['quality_flags']:
                    review.write(json.dumps({'doc_id': record['doc_id'], 'pdf_page': number,
                        'flags': record['quality_flags']}, ensure_ascii=False) + '\n')
                    flags.update(record['quality_flags'])
    export.replace(output / 'pages.jsonl')
    issues.replace(output / 'issues.jsonl')
    summary = {'expected_documents': len(documents), 'expected_pages': sum(d['pages'] for d in documents),
               'counts': dict(counts), 'quality_flags': dict(flags),
               'note': 'Text availability and OCR confidence do not prove correctness. Table structure is unverified.'}
    atomic_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('outputs/corpus'))
    parser.add_argument('--stage', choices=['native', 'ocr', 'tables'], required=True)
    parser.add_argument('--sample', action='append', default=[], help='DOC_ID:PAGE; repeatable')
    parser.add_argument('--doc-id', action='append', default=[])
    parser.add_argument('--limit-pages', type=int)
    args = parser.parse_args()
    if args.stage == 'tables' and not args.sample and not args.doc_id:
        args.sample = [f'{doc}:{page}' for doc, pages in TABLE_PAGES.items() for page in pages]
    if args.limit_pages is not None and args.limit_pages < 1:
        parser.error('--limit-pages must be positive')
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger('pypdf').setLevel(logging.ERROR)
    from pypdf import PdfReader
    import pypdfium2
    documents = [json.loads(line) for line in (args.root / 'data/documents.jsonl').read_text().splitlines() if line.strip()]
    reports = (args.root / 'data/reports').resolve()
    seen = set()
    for d in documents:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', d['doc_id']) or d['doc_id'] in seen:
            raise ValueError('Invalid/duplicate doc_id')
        if type(d['pages']) is not int or d['pages'] < 1:
            raise ValueError('Invalid page count')
        if not (reports / d['path']).resolve().is_relative_to(reports):
            raise ValueError('Source path escapes reports directory')
        seen.add(d['doc_id'])
    valid_samples = {f"{d['doc_id']}:{p}" for d in documents for p in range(1, d['pages']+1)}
    if set(args.sample) - valid_samples or set(args.doc_id) - seen:
        parser.error('Unknown document or out-of-range sample')
    args.output.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in ['pypdf', 'docling', 'rapidocr', 'onnxruntime']}
    native_config = {'revision': REVISION, 'pypdf': versions['pypdf'], 'ocr_threshold_alnum': 40}
    ocr_config = {'revision': REVISION, 'versions': versions, 'lang': 'iso:ru', 'use_cls': False, 'dpi': 216, 'orientation': 'bounded_score_check'}
    tables_config = {'revision': REVISION, 'versions': versions, 'mode': 'accurate', 'use_cls': False}
    from filelock import FileLock
    lock = FileLock(args.output / '.prepare.lock')
    lock.acquire(timeout=0)
    processor = None
    processed, skipped, attempted = 0, 0, 0
    try:
        for d in documents:
            if args.doc_id and d['doc_id'] not in args.doc_id:
                continue
            selected = [p for p in range(1, d['pages']+1) if not args.sample or f"{d['doc_id']}:{p}" in args.sample]
            if not selected:
                continue
            source = reports / d['path']
            with source.open('rb') as stream:
                source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            native_key = digest({'config': native_config, 'source_sha256': source_hash})
            reader = None
            pdf = None
            try:
                for number in selected:
                    if args.limit_pages is not None and attempted >= args.limit_pages:
                        break
                    path = args.output / 'pages' / d['doc_id'] / f'{number:04d}.json'
                    record = read_json(path)
                    current = record and record.get('native', {}).get('fingerprint') == native_key
                    if args.stage != 'native' and not current:
                        raise ValueError(f"Run native stage first: {d['doc_id']}:{number}")
                    if args.stage == 'native':
                        if current and record['native']['status'] == 'complete':
                            skipped += 1
                            continue
                        record = {'schema_version': 1, 'doc_id': d['doc_id'], 'organization': d['organization'],
                                  'source': d['path'], 'source_sha256': source_hash, 'pdf_page': number,
                                  'ocr': {'status': 'pending'}, 'tables': {'status': 'pending'}}
                        key = native_key
                    elif args.stage == 'ocr':
                        if not record['requires_ocr']:
                            skipped += 1
                            continue
                        key = digest({'native': native_key, 'config': ocr_config})
                        if record.get('ocr', {}).get('fingerprint') == key and record['ocr']['status'] == 'complete':
                            skipped += 1
                            continue
                    else:
                        if record['requires_ocr'] and record.get('ocr', {}).get('status') != 'complete':
                            raise ValueError(f"Run OCR stage first: {d['doc_id']}:{number}")
                        key = digest({'native': native_key, 'ocr': record.get('ocr', {}).get('fingerprint'), 'config': tables_config})
                        if record.get('tables', {}).get('fingerprint') == key and record['tables']['status'] == 'complete':
                            skipped += 1
                            continue
                    attempted += 1
                    start = time.perf_counter()
                    try:
                        if args.stage == 'native':
                            if reader is None:
                                reader = PdfReader(source)
                                if len(reader.pages) != d['pages']:
                                    raise ValueError('PDF page count differs from documents manifest')
                            page = reader.pages[number-1]
                            plain = page.extract_text() or ''
                            layout = page.extract_text(extraction_mode='layout') or ''
                            record['native'] = {'status': 'complete', 'fingerprint': key,
                                'engine': 'pypdf', 'plain': plain, 'layout': layout, 'version': versions['pypdf']}
                            record['requires_ocr'] = alnum(plain) < 40 or '\ufffd' in plain or bool(re.search(r'\(cid:\d+\)', plain))
                            if not record['requires_ocr']:
                                record['ocr'] = {'status': 'not_needed'}
                        elif args.stage == 'ocr':
                            if processor is None:
                                processor = LocalOCR()
                            if pdf is None:
                                pdf = pypdfium2.PdfDocument(source)
                            image = pdf[number-1].render(scale=3).to_pil().convert('RGB')
                            record['ocr'] = {**processor.extract(image), 'fingerprint': key}
                            # После повторного распознавания помечаем таблицы для повторной обработки.
                            record['tables'] = {'status': 'pending'}
                        else:
                            if processor is None:
                                processor = TableExtractor()
                            rotation = record.get('ocr', {}).get('rotation_clockwise', 0)
                            image = None
                            if rotation:
                                if pdf is None:
                                    pdf = pypdfium2.PdfDocument(source)
                                image = pdf[number-1].render(scale=3).to_pil().convert('RGB')
                            record['tables'] = {**processor.extract(source, number, rotation, image), 'fingerprint': key}
                    except Exception as error:
                        record[args.stage] = {'status': 'error', 'fingerprint': key,
                                             'error': f'{type(error).__name__}: {error}'}
                        if args.stage == 'native':
                            record['requires_ocr'] = True
                    record[args.stage]['seconds'] = time.perf_counter() - start
                    finalise(record)
                    atomic_json(path, record)
                    processed += 1
                    print(json.dumps({'stage': args.stage, 'doc_id': d['doc_id'], 'page': number,
                        'status': record[args.stage]['status'], 'chars': len(record['text']),
                        'seconds': round(record[args.stage]['seconds'], 2)}, ensure_ascii=False), flush=True)
                    if args.stage != 'native' and record[args.stage]['status'] == 'error':
                        message = record[args.stage]['error']
                        raise SystemExit(
                            f"Стадия {args.stage} остановлена на {d['doc_id']}, стр. {number}. {message}. "
                            'После устранения ошибки повторите ту же команду.'
                        )
            finally:
                if pdf is not None:
                    pdf.close()
            if args.limit_pages is not None and attempted >= args.limit_pages:
                break
    finally:
        try:
            summary = export_corpus(args.output, documents, reports, native_config)
            print(json.dumps({'processed': processed, 'skipped': skipped, 'summary': summary}, ensure_ascii=False), flush=True)
        finally:
            lock.release()


if __name__ == '__main__':
    main()
