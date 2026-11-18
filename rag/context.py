"""Собирает контекст из извлечённых страниц и сохраняет ссылки на источники.

Здесь обрабатывается текст страниц без вызовов эмбеддера и языковой модели.
"""
from collections import Counter
import re
import sqlite3

UNIT = (
    r'(?P<scale>тыс\.?|тысяч[а-я]*|млн\.?|миллион[а-я]*|млрд\.?|миллиард[а-я]*)'
    r'\s+(?P<country>российск[а-я]*\s+|белорусск[а-я]*\s+)?рубл[а-я]*'
)
DECLARATION = re.compile(
    r'(?:все\s+суммы\s+(?:приведены|представлены|выражены)|'
    r'(?:данная|настоящая|эта)\s+'
    r'(?:(?:обобщенная|консолидированная|промежуточная|сокращенная|финансовая)\s+){0,7}'
    r'отчетность\s+(?:представлена|выражена))\s+в\s+' + UNIT,
    re.I,
)
PAGE_UNIT = re.compile(r'(?:^|\()\s*в\s+' + UNIT, re.I | re.M)
SECTION = re.compile(r'^\s*(\d{1,2})(?:\.|\))?\s+([А-ЯЁA-Z][А-ЯЁA-Z\s,«»()\-–]{5,})$')
YEAR = re.compile(r'\b(?:19|20)\d{2}\s*(?:г\.|год)')
DECIMAL = re.compile(r'\d[.,]\d')


def normalize(text):
    return text.lower().replace('ё', 'е').replace('\u00a0', ' ')


def declarations(text):
    """Находит общие единицы измерения отчёта, в том числе в двухколоночном тексте.

    Единицы отдельной суммы или таблицы действуют только для неё.
    Для каждого совпадения сохраняет выдержку из исходных строк.
    """
    lines = text.splitlines()
    results = []
    seen = set()
    for column in [None, 0, 1]:
        fragments, offsets = [], []
        length = 0
        for number, line in enumerate(lines):
            parts = re.split(r' {3,}', line.strip())
            fragment = line if column is None else (parts[column] if len(parts) > column else '')
            fragment = normalize(' '.join(fragment.split()))
            if fragment:
                offsets.append((length, number))
                fragments.append(fragment)
                length += len(fragment) + 1
        flow = ' '.join(fragments)
        for match in DECLARATION.finditer(flow):
            start_line = max(number for offset, number in offsets if offset <= match.start())
            end_line = max(number for offset, number in offsets if offset < match.end()) + 1
            scale = match.group('scale')
            scale = 'тыс.' if scale.startswith('ты') else ('млрд' if scale.startswith(('млрд', 'миллиард')) else 'млн')
            country = match.group('country') or ''
            currency = 'белорусских рублей' if country.startswith('белорус') else ('российских рублей' if country else 'рублей')
            unit = f'{scale} {currency}'
            key = (unit, start_line, end_line)
            if key in seen:
                continue
            # Сохраняем исходные строки вместе с соседней оговоркой «если не указано иное».
            excerpt = '\n'.join(lines[start_line:min(len(lines), end_line + 2)])
            if len(excerpt) > 1800:
                continue
            seen.add(key)
            results.append({'unit': unit, 'text': excerpt, 'source_line_start': start_line + 1,
                            'source_line_end': min(len(lines), end_line + 2)})
    return results


def table_header(text):
    """Признаки продолжения таблицы по номеру раздела и общим колонкам."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    section = None
    position = None
    for i, line in enumerate(lines[:12]):
        match = SECTION.match(line)
        if match:
            title = re.sub(r'\(?ПРОДОЛЖЕНИЕ\)?', '', match.group(2)).strip()
            section = (match.group(1), ' '.join(title.split()))
            position = i
            break
    if section is None:
        return None
    header = []
    has_numeric_rows = False
    for line in lines[position + 1:]:
        if len(DECIMAL.findall(line)) >= 2:
            has_numeric_rows = True
            break
        header.append(line)
        if len(' '.join(header)) > 3000:
            break
    body = ' '.join(header)
    words = set(re.findall(r'[а-яa-z]{3,}', normalize(body)))
    if not has_numeric_rows or len(YEAR.findall(body)) < 2 or len(words) < 4:
        return None
    return {'section': section, 'words': words}


def is_table_neighbor(first, second):
    left, right = table_header(first), table_header(second)
    if left is None or right is None or left['section'] != right['section']:
        return False
    common = left['words'] & right['words']
    union = left['words'] | right['words']
    return len(common) >= 4 and len(common) / len(union) >= 0.6


class ContextBuilder:
    def __init__(self, index, max_extra_pages=3):
        if max_extra_pages < 0:
            raise ValueError('max_extra_pages must be nonnegative')
        self.connection = sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True)
        self.connection.row_factory = sqlite3.Row
        self.max_extra_pages = max_extra_pages
        self._documents = {}
        self._defaults = {}

    def close(self):
        self.connection.close()

    def document(self, doc_id):
        if doc_id not in self._documents:
            rows = self.connection.execute('SELECT * FROM pages WHERE doc_id=? ORDER BY pdf_page', (doc_id,))
            self._documents[doc_id] = {row['pdf_page']: dict(row) for row in rows}
        return self._documents[doc_id]

    def default_units(self, doc_id):
        if doc_id not in self._defaults:
            found = []
            for number, page in self.document(doc_id).items():
                for declaration in declarations(page['text']):
                    found.append({'doc_id': doc_id, 'pdf_page': number, **declaration})
            units = {item['unit'] for item in found}
            self._defaults[doc_id] = (found[0] if len(units) == 1 else None, len(units) > 1)
        return self._defaults[doc_id]

    def source_page(self, doc_id, number, role, text=None):
        import json
        row = self.document(doc_id)[number]
        return {'doc_id': doc_id, 'pdf_page': number, 'organization': row['organization'],
                'text': row['text'] if text is None else text,
                'quality_flags': json.loads(row['quality_flags']), 'context_role': role}

    def build(self, pages):
        result = [dict(page, context_role='retrieved_page') for page in pages]
        seen = {(p['doc_id'], p['pdf_page']) for p in result}
        additions = []
        # Добавляем максимум одну соседнюю страницу таблицы.
        # Проверяем совпадение раздела и сходство заголовков столбцов.
        for page in pages:
            if len(additions) >= min(1, self.max_extra_pages):
                break
            doc, number = page['doc_id'], page['pdf_page']
            document = self.document(doc)
            for neighbor in [number + 1, number - 1]:
                if (doc, neighbor) in seen or neighbor not in document:
                    continue
                if is_table_neighbor(page['text'], document[neighbor]['text']):
                    extra = self.source_page(doc, neighbor, 'table_continuation')
                    extra['related_to_page'] = number
                    result.append(extra)
                    additions.append(extra)
                    seen.add((doc, neighbor))
                    break

        unresolved, conflicts, budget_limited = [], [], []
        for doc in dict.fromkeys(p['doc_id'] for p in result):
            targets = [p for p in result if p['doc_id'] == doc and not PAGE_UNIT.search(p['text'])]
            if not targets:
                continue
            declaration, conflict = self.default_units(doc)
            if declaration is None:
                (conflicts if conflict else unresolved).append(doc)
                continue
            key = (doc, declaration['pdf_page'])
            if key not in seen:
                if len(additions) >= self.max_extra_pages:
                    budget_limited.append(doc)
                    continue
                extra = self.source_page(doc, declaration['pdf_page'], 'unit_reference', declaration['text'])
                extra['excerpt'] = True
                extra['source_line_start'] = declaration['source_line_start']
                extra['source_line_end'] = declaration['source_line_end']
                result.append(extra)
                additions.append(extra)
                seen.add(key)
            for page in targets:
                if page['context_role'] != 'unit_reference':
                    page['unit_context'] = {
                        'unit': declaration['unit'], 'scope': 'document_default',
                        'source_doc_id': doc, 'source_pdf_page': declaration['pdf_page'],
                        'rule': 'Применять, только если для конкретной таблицы или показателя не указаны другие единицы.',
                    }
        diagnostics = {
            'retrieved_pages': len(pages), 'total_pages': len(result),
            'added_pages': [{'doc_id': p['doc_id'], 'pdf_page': p['pdf_page'], 'role': p['context_role']} for p in additions],
            'roles': dict(Counter(p['context_role'] for p in result)),
            'documents_without_explicit_default_units': unresolved,
            'documents_with_conflicting_default_units': conflicts,
            'unit_references_skipped_by_budget': budget_limited,
        }
        return result, diagnostics
