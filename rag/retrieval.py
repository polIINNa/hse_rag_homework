"""Лексический поиск и подготовка выдержек без изменения чисел исходного PDF."""
import re
from bisect import bisect_left
from collections import Counter

_TOKEN = re.compile(r"[a-zа-я]\d+(?:[.,]\d+)?|[a-zа-я]{2,}|\d+(?:[.,]\d+)?")
_SUFFIX = re.compile(r"(иями|ями|ами|ого|ему|ому|ыми|ими|иях|ах|ях|ов|ев|ий|ии|ый|ой|ая|яя|ое|ее|ые|ие|ую|юю|ых|их|ым|им|ом|ем|ам|ям|ия|ию|ью|ей|ся|а|я|ы|и|у|ю|е|о|ь)$")
_TABLE_CELL = re.compile(r"\(?[+−-]?\d[\d ,.]*\)?%?|[–—-]")
_PERIOD = re.compile(r"\b20\d{2}\b|\b\d{1,2}\s+(?:января|июня|декабря)\b|\d{1,2}\.\d{1,2}\.\d{2,4}")
_UNIT = re.compile(r"\b(?:тыс|млн|млрд|трлн)\b|\b(?:руб|byn|rub|usd)\b", re.I)


def stem(word):
    if re.fullmatch("[а-я]+", word):
        root = _SUFFIX.sub("", word)
        return root if len(root) >= 3 else word
    return word


_STOP = {stem(w) for w in (
    "как каков какова каковы какие какой какая какую сколько где когда почему "
    "на по за от до из для при об со во не ни и а но или с к в у о что это "
    "который которая которые этих этот данной данный согласно составляет составили "
    "было были является являются используется используются "
    "банк банки группа группой групп консолидированная консолидированный консолидированно "
    "финансовая финансовой отчетность отчетности отчет отчетов обобщенная промежуточная "
    "пао оао ао зао рф год года году годы месяцев месяц месяца "
    "шесть три первое второе январь января февраль февраля март марта апрель апреля "
    "май мая июнь июня июль июля август августа сентябрь сентября октябрь октября "
    "ноябрь ноября декабрь декабря закончившийся закончившихся закончившиеся "
    "сбербанк сбер втб дом альфа газпромбанк фонд московский кредитный совкомбанк "
    "россельхозбанк тинькофф тбанк tbank vtb domrf mkb gpb rshb sber sovkom alfa"
).split()}
# «Кредитный» сохраняется как содержательный термин вне полного имени МКБ.
_STOP.discard(stem("кредитный"))


def tokens(text):
    text = text.lower().replace("ё", "е")
    result = []
    for word in _TOKEN.findall(text):
        if re.fullmatch(r"(?:19|20)\d\d", word):
            continue  # Годы обрабатываются отдельно как метаданные.
        if word[0].isdigit():
            result.append("#" + word)
        else:
            word = stem(word)
            if word not in _STOP:
                result.append(word)
    return result


def compact_text(text):
    # Пробел внутри 5 592 и десятичная запятая сохраняются.
    lines = []
    for line in text.replace("\u00ad", "").replace("\xa0", " ").splitlines():
        line = re.sub(r"[ \t]{2,}", " | ", line.strip())
        if line or (lines and lines[-1]):
            lines.append(line)
    return "\n".join(lines).strip()


def block_layout(page, fallback):
    """Для двух текстовых колонок сохраняем целые PDF-блоки, для таблиц — строки."""
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    width = page.rect.width
    narrative = [b for b in blocks if len(b[4].strip()) > 90
                 and sum(c.isalpha() for c in b[4]) / max(len(b[4]), 1) > 0.55]
    left = [b for b in narrative if b[0] < width * 0.35 and b[2] < width * 0.58]
    right = [b for b in narrative if b[0] > width * 0.47]
    if len(left) >= 2 and len(right) >= 2:
        return compact_text("\n\n".join(b[4] for b in blocks)), "columns"
    return compact_text(fallback), "rows"


def report_hint(question):
    q = question.lower().replace("ё", "е")
    years = re.findall(r"\b(?:19|20)\d\d\b", q)
    source = re.search(r"отчетност[а-я\s«»(),-]{0,65}?(?:за\s+|от\s+)((?:19|20)\d\d)", q)
    primary = source.group(1) if source else (years[0] if years else None)
    half = bool(re.search(r"полугод|шесть\s+месяц|6\s+месяц|30\s+июня", q))
    period = "6m" if half else ("annual" if "год" in q else None)
    return primary, set(years), period


def safe_prefix(text, limit):
    """Предпочитаем конец строки; не режем слово или число с разрядными пробелами."""
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    line_end = text.rfind("\n", 0, limit + 1)
    if line_end >= limit // 2 and line_end > 0:
        return text[:line_end]
    boundary = 0
    for match in re.finditer(r"\s+", text[:limit + 1]):
        if match.start() and not text[match.start() - 1].isdigit():
            boundary = match.start()
    return text[:boundary]


def excerpt(text, query, limit=2200, idf=None):
    """Выбираем окно по содержательным словам, сохраняя полные строки и продолжения."""
    if len(text) <= limit:
        return text
    if limit < 100:
        return safe_prefix(text, limit)
    query_terms = {t for t in tokens(query) if not t.startswith("#")}
    if not query_terms:
        return safe_prefix(text, limit)
    idf = idf or {}
    offsets, position = [], 0
    lines = text.splitlines(keepends=True)
    for line in lines:
        if query_terms.intersection(tokens(line)):
            offsets.append(position)
        position += len(line)
    line_starts = [0] + [match.end() for match in re.finditer("\n", text)]
    tail = max(0, len(text) - limit)
    tail_index = bisect_left(line_starts, tail)
    starts = {0}
    if tail_index < len(line_starts):
        starts.add(line_starts[tail_index])
    for offset in offsets:
        start = max(0, offset - limit // 3)
        start = line_starts[bisect_left(line_starts, start)]
        starts.add(start)
    best, best_score = "", -1.0
    for start in sorted(starts):
        window = safe_prefix(text[start:], limit)
        counts = Counter(tokens(window))
        score = sum(float(idf.get(t, 1.0)) * (1 + 0.1 * min(counts[t], 3))
                    for t in query_terms if t in counts)
        if score > best_score:
            best, best_score = window, score
    return best


def row_query(query, idf):
    words = [word for word in tokens(query) if not word.startswith("#")]
    weights = {word: float(idf.get(word, 1.0)) ** 1.5 for word in words}
    total = sum(weights.values()) or 1.0
    return {word: weight / total for word, weight in weights.items()}, set(zip(words, words[1:]))


def row_match(label, weights, pairs):
    words = tokens(label)
    coverage = sum(weights.get(word, 0.0) for word in set(words))
    adjacency = len(pairs.intersection(zip(words, words[1:])))
    return coverage + min(adjacency * 0.06, 0.24)


def rerank_by_rows(pages, query, idf):
    weights, pairs = row_query(query, idf)
    if not weights:
        return pages

    def rank(page):
        previous = ""
        best = 0.0
        for line in page["text"].splitlines():
            label, separator, values = line.partition("|")
            if separator and sum(char.isdigit() for char in line) >= 4:
                best = max(best, row_match(previous + " " + label, weights, pairs))
            previous = label
        return page["score"] + 0.5 * best

    return sorted(pages, key=rank, reverse=True)


def table_excerpt(text, query, limit, idf):
    """Вернуть релевантные строки с исходными заголовками и единицами, если они помещаются."""
    if limit < 240:
        return ""
    weights, pairs = row_query(query, idf)
    if not weights:
        return ""
    lines = text.splitlines()
    groups = []
    for position, line in enumerate(lines):
        cells = [cell.strip() for cell in line.strip("| ").split("|")]
        values = [cell for cell in cells if _TABLE_CELL.fullmatch(cell)]
        if len(cells) < 3 or len(values) < 2 or not any(any(c.isdigit() for c in v) for v in values):
            continue
        if all(not any(c.isdigit() for c in v) or re.fullmatch(r"20\d{2}", v) for v in values):
            continue
        if groups and position - groups[-1][-1] <= 6:
            gap = lines[groups[-1][-1] + 1:position]
            if not any(_PERIOD.search(part) and "|" in part for part in gap):
                groups[-1].append(position)
                continue
        groups.append([position])

    best = ""
    best_score = 0.0
    previous_end = -1
    for group in groups:
        start = group[0]
        prelude = lines[max(previous_end + 1, start - 16):start]
        previous_end = group[-1]
        dates = [i for i, line in enumerate(prelude) if _PERIOD.search(line)]
        units = [i for i, line in enumerate(prelude) if _UNIT.search(line)]
        if len(group) < 2 or not dates or not units:
            continue
        header_start = max(0, min(dates[-1], units[-1]) - 3)
        header_end = max(dates[-1], units[-1]) + 1
        header = "\n".join(prelude[header_start:header_end]).strip()
        rows = []
        previous = start - len(prelude) + header_end - 1
        for position in group:
            lead = [line for line in lines[max(previous + 1, position - 5):position] if line and "|" not in line]
            label = lines[position].lstrip("| ").split("|", 1)[0]
            row = "\n".join([*lead, lines[position]])
            score = row_match(" ".join([*lead, label]), weights, pairs)
            if score >= 0.3 and len(header) + len(row) + 1 <= limit:
                rows.append((score, position, row))
            previous = position
        if not rows:
            continue
        strength = max(value for value, _, _ in rows)
        if strength <= best_score:
            continue
        selected = []
        remaining = limit - len(header)
        for _, position, row in sorted(rows, key=lambda item: (-item[0], item[1])):
            if len(row) + 1 <= remaining:
                selected.append((position, row))
                remaining -= len(row) + 1
        best = "\n".join([header, *(row for _, row in sorted(selected))])
        best_score = strength
    return best
