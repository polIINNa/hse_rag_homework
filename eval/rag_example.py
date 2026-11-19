import json
import os
import re
from pathlib import Path

import numpy as np


# Опциональная трассировка в LangSmith (для +1 балла). Если langsmith не установлен
# или трассировка выключена, декоратор становится no-op и код работает как раньше.
def _make_traceable():
    try:
        from langsmith import traceable  # type: ignore

        return traceable
    except Exception:
        def traceable(*dargs, **dkwargs):
            def deco(fn):
                return fn

            if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
                return dargs[0]
            return deco

        return traceable


traceable = _make_traceable()


class RAG:
    """Векторный RAG по финансовым PDF-отчётам (улучшенная, агентная конфигурация).

    Что делает:
    1) Извлекает текст постранично из PDF (PyMuPDF), включая текст таблиц
       (find_tables), с сохранением doc_id и номера страницы.
    2) Разбивает страницы на перекрывающиеся чанки для более точного поиска.
    3) Строит/загружает кэш чанков и эмбеддингов (eval/cache/chunks_v2.*).
    4) Агентный retrieval: выполняет начальный поиск, оценивает покрытие
       сущностей/лет/периодов из вопроса и при необходимости сам запускает
       дополнительные целевые поиски, затем объединяет результаты.
    5) Ранжирует кандидаты по косинусной близости + метаданные (банк/год/период)
       + лексическое совпадение ключевых терминов вопроса.
    6) Возвращает контекст на уровне страниц: [{"doc_id": ..., "pages": [...]}].
    7) Генерирует ответ по полному тексту найденных страниц через LLM
       (GEN_API_KEY/LLM_API_KEY), иначе возвращает извлекающий fallback.
    """

    # Паттерны для эвристического определения банка/организации в вопросе
    ORG_HINTS = [
        (r"альфа[\s-]?банк|зао\s*[«\"]?альфа", "alfa"),
        (r"дом\.?\s*рф|банк\s+дом", "domrf"),
        (r"газпромбанк[\s-]?фонд|нпф\s+газпромбанк", "gpb"),
        (r"\bмкб\b|московск(?:ий|ого)\s+кредитн", "mkb"),
        (r"россельхозбанк|\bрсхб\b", "rshb"),
        (r"сбербанк|\bсбер\b", "sber"),
        (r"совкомбанк", "sovkom"),
        (r"\bт-?банк\b|\bтинькофф\b|тбанк", "tbank"),
        (r"\bвтб\b", "vtb"),
    ]

    _STOPWORDS = {
        "каков", "какова", "каково", "каковы", "какой", "какая", "какие", "какого",
        "сколько", "каким", "какую", "каком", "каких", "какие", "какую", "какая",
        "было", "были", "была", "будет", "является", "являются", "составил", "составила",
        "составили", "составило", "указать", "укажите", "назовите", "можно", "если",
        "этого", "этой", "этом", "этих", "эти", "этот", "году", "года", "году",
        "отчет", "отчете", "отчёте", "отчёт", "отчетности", "отчётности", "отчётный",
        "отчетный", "отчетном", "отчётном", "отчетного", "отчётного", "согласно",
        "данным", "размер", "значение", "величина", "какой", "банк", "банка", "банке",
        "группы", "группа", "группе", "консолидированной", "консолидированном",
        "консолидированной", "итогам", "итогов", "период", "периода", "периоду",
        "год", "месяцев", "месяца", "первое", "первого", "полугодие", "полугодия",
        "ответ", "руб", "рублей", "рублях", "тыс", "млн", "млрд", "трлн",
    }

    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.data_dir = self.root / "data"
        self.reports_dir = self.data_dir / "reports"
        self.docs_path = self.data_dir / "documents.jsonl"
        self.cache_dir = Path(__file__).resolve().parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # v2 cache: chunk-level index (separately from old page-level cache)
        # (извлечение улучшено: добавлены таблицы; для их учёта нужен пересчёт индекса)
        self.chunks_path = self.cache_dir / "chunks_v2.jsonl"
        self.emb_path = self.cache_dir / "embeddings_v2.npy"

        self.chunks = []
        self.embeddings = None
        self.model = None
        self._page_chunks = None  # (doc_id, page) -> [chunk_text, ...]
        self._chunk_tokens = None  # lexical tokens per chunk for hybrid boost
        self._all_doc_ids = None

        self._ensure_index()

        # Прогреваем модель эмбеддингов и лексический индекс до первого вопроса.
        # Иначе инициализация попадёт в измеряемое время первого вопроса (в eval
        # `time` считается вокруг вызова rag()), что делает среднее время нестабильным.
        if self.embeddings is not None and len(self.chunks):
            model = self._get_model()
            model.encode(
                ["query: warmup"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._ensure_chunk_tokens()

    # ---------- Index building ----------
    def _load_documents_meta(self):
        docs = []
        with self.docs_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        return docs

    @staticmethod
    def _split_text(text, chunk_size=1200, overlap=200):
        """Разбивает текст страницы на перекрывающиеся чанки.
        Зачем: одна страница может содержать разные таблицы/разделы; чанки дают
        более точное совпадение эмбеддинга с конкретным фрагментом вопроса."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(0, end - overlap)
        return chunks

    @staticmethod
    def _extract_page_tables(page):
        """Извлекает таблицы со страницы как текст (по возможности с сохранением структуры).
        Зачем: финансовые показатели часто в таблицах; текстовый слой теряет связи
        между ячейками, а табличное представление улучшает retrieval/generation."""
        try:
            finder = page.find_tables()
            tables = getattr(finder, "tables", None) or []
        except Exception:
            return ""

        rendered = []
        for t in tables:
            try:
                rows = t.extract() or []
            except Exception:
                continue
            lines = []
            for row in rows:
                cells = []
                for cell in row:
                    cell = re.sub(r"\s+", " ", str(cell or "")).strip()
                    cells.append(cell)
                line = " | ".join(cells).strip(" |")
                if line and re.search(r"\w", line, flags=re.UNICODE):
                    lines.append(line)
            if lines:
                rendered.append("ТАБЛИЦА:\n" + "\n".join(lines[:80]))
        if not rendered:
            return ""
        return "\n\n".join(rendered[:6])

    def _extract_chunks(self):
        try:
            import pymupdf as fitz  # PyMuPDF
        except Exception:
            import fitz  # type: ignore

        docs = self._load_documents_meta()
        chunks = []
        for meta in docs:
            pdf_path = self.reports_dir / meta["path"]
            if not pdf_path.exists():
                continue
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                text = page.get_text("text") or ""
                table_text = self._extract_page_tables(page)
                if table_text:
                    text = f"{text}\n\n{table_text}"
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                if not text:
                    continue
                page_no = i + 1  # 1-based
                for ci, chunk_text in enumerate(self._split_text(text)):
                    chunks.append(
                        {
                            "doc_id": meta["doc_id"],
                            "page": page_no,
                            "chunk_id": ci,
                            "path": meta["path"],
                            "organization": meta.get("organization", ""),
                            "text": chunk_text,
                        }
                    )
            doc.close()
        return chunks

    def _get_model(self):
        from sentence_transformers import SentenceTransformer

        if self.model is None:
            model_name = os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-small")
            self.model = SentenceTransformer(model_name)
        return self.model

    def _embed_texts(self, texts, prefix="passage: "):
        model = self._get_model()
        inputs = [f"{prefix}{t}" for t in texts]
        emb = model.encode(
            inputs,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def _ensure_index(self):
        if self.chunks_path.exists() and self.emb_path.exists():
            self.chunks = [json.loads(l) for l in self.chunks_path.open(encoding="utf-8") if l.strip()]
            self.embeddings = np.load(self.emb_path)
            if len(self.chunks) == self.embeddings.shape[0]:
                self._all_doc_ids = sorted({c["doc_id"] for c in self.chunks})
                return
            self.chunks = []
            self.embeddings = None

        if not self.reports_dir.exists():
            raise FileNotFoundError(
                f"Не найдена папка с отчётами: {self.reports_dir}. "
                "Распакуйте data/reports.zip в data/reports."
            )

        # Автосборка «в один поток» без индикатора может выглядеть как зависание.
        # Рекомендуемый путь — заранее: python -m eval.build_index
        print(
            "[RAG] Индекс v2 не найден — строю автоматически. "
            "Это может занять долго; для прогресса лучше запустить: "
            "python -m eval.build_index",
            flush=True,
        )
        self.chunks = self._extract_chunks()
        texts = [c["text"][:2000] for c in self.chunks]
        print(f"[RAG] Чанков: {len(self.chunks)}. Считаю эмбеддинги (без прогресс-бара)...", flush=True)
        self.embeddings = self._embed_texts(texts, prefix="passage: ") if texts else np.zeros((0, 384), dtype=np.float32)

        with self.chunks_path.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        np.save(self.emb_path, self.embeddings)
        self._all_doc_ids = sorted({c["doc_id"] for c in self.chunks})
        print(f"[RAG] Индекс v2 сохранён: {self.emb_path} shape={self.embeddings.shape}", flush=True)

    # ---------- Query analysis / boosting ----------
    def _infer_filters(self, question):
        """Извлекает подсказки из вопроса: банк(и), годы, период.
        Зачем: вопросы часто указывают организацию/период; буст по метаданным
        повышает вероятность попадания нужного отчёта в top-k."""
        q = question.lower()
        orgs = set()
        for pat, prefix in self.ORG_HINTS:
            if re.search(pat, q):
                orgs.add(prefix)

        years = set(re.findall(r"\b(20\d{2})\b", q))
        period = None
        if re.search(r"6\s*месяц|полугод|1-?е?\s*полугод|перв\w*\s+полугод|30\s+июн|six\s+months|\b6m\b", q):
            period = "6m"
        elif re.search(r"годов\w*\s+отч[её]т|за\s+20\d{2}\s+год\b|годовая\s+отч[её]тн|annual|31\s+декабр", q):
            period = "annual"

        comparative = bool(re.search(r"сравн|динамик|измен|по\s+сравнени|разниц|вырос|снизил|увелич|уменьш", q))
        return {
            "orgs": orgs,
            "years": years,
            "period": period,
            "comparative": comparative or len(years) > 1 or len(orgs) > 1,
        }

    def _score_boost(self, doc_id, filters):
        """Мягкий бонус/штраф к косинусной близости на основе метаданных doc_id.
        doc_id имеет вид: <org>_<year>_<period>, напр. domrf_2025_6m, sber_2024_annual."""
        boost = 0.0
        parts = doc_id.split("_")
        org = parts[0] if parts else ""
        year = parts[1] if len(parts) > 1 else ""
        period = parts[2] if len(parts) > 2 else ""

        if filters.get("orgs"):
            boost += 0.10 if org in filters["orgs"] else -0.06
        if filters.get("years"):
            boost += 0.06 if year in filters["years"] else -0.03
        if filters.get("period"):
            if period == filters["period"]:
                boost += 0.03
            elif period:
                boost -= 0.02
        return boost

    @staticmethod
    def _tokenize(text):
        return set(re.findall(r"[0-9]+[.,]?[0-9]*|[a-zа-яё]{3,}", text.lower()))

    def _question_terms(self, question):
        toks = self._tokenize(question)
        return {t for t in toks if t not in self._STOPWORDS and not t.isdigit() or len(t) >= 4}

    def _ensure_chunk_tokens(self):
        if self._chunk_tokens is None:
            self._chunk_tokens = [self._tokenize(c.get("text", "")[:1200]) for c in self.chunks]
        return self._chunk_tokens

    def _lexical_boosts(self, question):
        """Небольшой лексический буст (гибридный сигнал): по совпадению значимых
        терминов/чисел вопроса с текстом чанка. Помогает, когда важны точные слова."""
        q_terms = self._question_terms(question)
        if not q_terms:
            return np.zeros(len(self.chunks), dtype=np.float32)
        chunk_tokens = self._ensure_chunk_tokens()
        boosts = np.zeros(len(self.chunks), dtype=np.float32)
        for i, toks in enumerate(chunk_tokens):
            if not toks:
                continue
            overlap = len(q_terms & toks)
            if overlap:
                boosts[i] = min(0.04, 0.008 * overlap)
        return boosts

    # ---------- Retrieval ----------
    @traceable(run_type="retriever", name="retrieve_chunks")
    def _search(self, question, top_k_pages=8, forced_filters=None):
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        model = self._get_model()
        q_emb = model.encode([f"query: {question}"], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)[0]
        base_scores = self.embeddings @ q_emb

        filters = dict(self._infer_filters(question))
        if forced_filters:
            if forced_filters.get("orgs"):
                filters["orgs"] = set(forced_filters["orgs"])
            if forced_filters.get("years"):
                filters["years"] = set(forced_filters["years"])
            if forced_filters.get("period"):
                filters["period"] = forced_filters["period"]

        boosts = np.array([self._score_boost(c["doc_id"], filters) for c in self.chunks], dtype=np.float32)
        lex = self._lexical_boosts(question)
        scores = base_scores + boosts + lex

        # Берём больше чанков, затем агрегируем до уникальных страниц
        cand_k = min(max(top_k_pages * 8, 60), len(scores))
        idx = np.argpartition(-scores, cand_k - 1)[:cand_k]
        idx = idx[np.argsort(-scores[idx])]

        # Агрегация: лучший скор чанка -> скор страницы
        page_best = {}
        for i in idx:
            c = self.chunks[int(i)]
            key = (c["doc_id"], c["page"])
            sc = float(scores[i])
            if key not in page_best or sc > page_best[key][0]:
                page_best[key] = (sc, c)

        ranked_pages = sorted(page_best.values(), key=lambda x: -x[0])[:top_k_pages]
        return ranked_pages

    def _merge_hits(self, hit_lists, top_k_pages):
        """Объединяет результаты нескольких поисков: для каждой страницы оставляет
        максимальный скор, затем возвращает top_k_pages."""
        best = {}
        for hits in hit_lists:
            for sc, c in hits:
                key = (c["doc_id"], c["page"])
                if key not in best or sc > best[key][0]:
                    best[key] = (sc, c)
        return sorted(best.values(), key=lambda x: -x[0])[:top_k_pages]

    @traceable(run_type="chain", name="agentic_retrieve")
    def _agentic_retrieve(self, question, top_k_pages=8):
        """Агентный retrieval (+1 балл): решает, нужен ли дополнительный поиск.

        1) Делает базовый поиск.
        2) Анализирует вопрос и покрытие результатов (упомянутые банки/годы/периоды).
        3) Если важных сущностей не хватает в выдаче — запускает целевые доп. поиски
           (по каждому недостающему банку/году) и объединяет результаты.

        Так агент сам принимает решение о дополнительном поиске для сравнительных
        и многоаспектных вопросов."""
        filters = self._infer_filters(question)
        initial = self._search(question, top_k_pages=top_k_pages)
        hit_doc_ids = {c["doc_id"] for _, c in initial}

        extra_searches = []  # list of (question, forced_filters, reason)

        orgs = sorted(filters.get("orgs") or [])
        years = sorted(filters.get("years") or [])
        period = filters.get("period")

        # Если в вопросе несколько банков — проверяем покрытие каждого
        if len(orgs) > 1:
            for org in orgs:
                covered = any(d.startswith(org + "_") for d in hit_doc_ids)
                if not covered:
                    extra_searches.append((question, {"orgs": {org}, "years": set(years), "period": period}, f"missing org {org}"))

        # Если несколько лет и это сравнение/динамика — проверяем покрытие каждого года
        if len(years) > 1 and (filters.get("comparative") or len(orgs) <= 1):
            target_orgs = orgs if orgs else [None]
            for y in years:
                covered = any((not orgs or any(d.startswith(o + "_") for o in orgs)) and f"_{y}_" in d for d in hit_doc_ids)
                if not covered:
                    ff = {"years": {y}, "orgs": set(orgs), "period": period}
                    extra_searches.append((question, ff, f"missing year {y}"))

        # Ограничиваем число доп. поисков, чтобы не раздувать время
        extra_searches = extra_searches[:3]

        if not extra_searches:
            return initial

        hit_lists = [initial]
        # Для доп. поисков берём чуть меньше страниц, потом объединяем до top_k
        sub_k = max(4, top_k_pages // 2 + 1)
        for q, ff, _reason in extra_searches:
            hit_lists.append(self._search(q, top_k_pages=sub_k, forced_filters=ff))

        merged = self._merge_hits(hit_lists, top_k_pages=top_k_pages)

        # Гарантируем представительство целевых сущностей, если они нашлись
        merged_keys = {(c["doc_id"], c["page"]) for _, c in merged}
        for hits in hit_lists[1:]:
            for sc, c in hits:
                key = (c["doc_id"], c["page"])
                parts = c["doc_id"].split("_")
                org = parts[0] if parts else ""
                year = parts[1] if len(parts) > 1 else ""
                relevant = (orgs and org in orgs) or (years and year in years)
                if relevant and key not in merged_keys and len(merged) < top_k_pages + 2:
                    merged.append((sc, c))
                    merged_keys.add(key)
        merged = sorted(merged, key=lambda x: -x[0])[:top_k_pages]
        return merged

    @staticmethod
    def _group_context(hits):
        by_doc = {}
        for _, p in hits:
            by_doc.setdefault(p["doc_id"], set()).add(p["page"])
        context = [{"doc_id": doc_id, "pages": sorted(pages)} for doc_id, pages in by_doc.items()]
        context.sort(key=lambda x: x["doc_id"])
        return context

    def _page_chunk_map(self):
        """(doc_id, page) -> все чанки страницы (для генерации полного текста страницы).
        Зачем: при retrieval мы храним лучший чанк, но ответу нужен весь текст страницы
        (таблицы/соседние строки), иначе можно потерять существенные числа."""
        if self._page_chunks is None:
            mapping = {}
            for c in self.chunks:
                mapping.setdefault((c["doc_id"], c["page"]), []).append(c)
            for chunks in mapping.values():
                chunks.sort(key=lambda x: x.get("chunk_id", 0))
            self._page_chunks = mapping
        return self._page_chunks

    # ---------- Generation ----------
    @traceable(run_type="llm", name="generate_answer")
    def _generate(self, question, hits):
        if not hits:
            return "В предоставленных документах недостаточно информации для ответа."

        page_map = self._page_chunk_map()
        snippets = []
        for score, p in hits:
            key = (p["doc_id"], p["page"])
            chunks = page_map.get(key, [p])
            full_text = " ".join(c["text"] for c in chunks)
            snippets.append(
                f"[{p['doc_id']} стр. {p['page']}, score={score:.3f}]\n{full_text[:5000]}"
            )
        context_text = "\n\n".join(snippets)

        api_key = os.environ.get("GEN_API_KEY") or os.environ.get("LLM_API_KEY")
        if api_key:
            try:
                # Иногда модель возвращает пустой текст (например, если всё
                # max_tokens ушло на reasoning). Пробуем ещё раз, затем fallback.
                for _ in range(2):
                    answer = self._call_llm(question, context_text, api_key)
                    if answer and answer.strip():
                        return answer
                return self._extractive_answer(question, context_text, error="LLM returned empty answer")
            except Exception as e:
                return self._extractive_answer(question, context_text, error=str(e))

        return self._extractive_answer(question, context_text)

    def _call_llm(self, question, context_text, api_key):
        import requests

        base_url = os.environ.get("GEN_BASE_URL", "https://claude.gigaparsers.ru/v1/messages")
        model = os.environ.get("GEN_MODEL", "deepseek-flash")
        prompt = (
            "Ты помощник по финансовой отчётности. Ответь на вопрос, опираясь ТОЛЬКО на приведённый контекст. "
            "Дай точный и краткий ответ на русском. "
            "Строго сохраняй числа, валюты, единицы измерения (тыс./млн/млрд/трлн, %, п.п., б.п.), знаки и периоды "
            "в точности как в документе — не пересчитывай и не меняй масштаб. "
            "Если вопрос составной или сравнительный — ответь на все части и явно сравни значения, когда это требуется. "
            "Если в контексте недостаточно данных для точного ответа, прямо укажи, чего не хватает."
        )
        user = f"Контекст:\n{context_text}\n\nВопрос: {question}\nОтвет:"
        resp = requests.post(
            base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": model,
                "max_tokens": 700,
                "temperature": 0,
                "thinking": {"type": "disabled"},
                "system": prompt,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()

    @staticmethod
    def _extractive_answer(question, context_text, error=None):
        if error:
            print(f"[RAG] generation fallback: {error}", flush=True)
        head = context_text[:3500]
        return "На основе найденных фрагментов:\n" + head

    @traceable(run_type="chain", name="rag")
    def rag(self, question: str) -> dict:
        top_k = int(os.environ.get("RAG_TOP_K", "8"))
        hits = self._agentic_retrieve(question, top_k_pages=top_k)
        context = self._group_context(hits)
        answer = self._generate(question, hits)
        return {"answer": answer, "context": context}
