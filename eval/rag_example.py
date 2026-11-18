import json
import os
import re
from pathlib import Path

import numpy as np


class RAG:
    """Векторный RAG по финансовым PDF-отчётам (улучшенная конфигурация).

    Что делает:
    1) Извлекает текст постранично из PDF (PyMuPDF) с сохранением doc_id и номера страницы.
    2) Разбивает страницы на перекрывающиеся чанки для более точного поиска.
    3) Строит/загружает кэш чанков и эмбеддингов (eval/cache/chunks_v2.*).
    4) Ищет релевантные чанки по косинусной близости + применяет мягкий буст
       по метаданным (банк/год/период), извлечённым из вопроса.
    5) Возвращает контекст на уровне страниц: [{"doc_id": ..., "pages": [...]}].
    6) Генерирует ответ по найденному контексту через LLM (GEN_API_KEY/LLM_API_KEY),
       иначе возвращает извлекающий fallback.
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

    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.data_dir = self.root / "data"
        self.reports_dir = self.data_dir / "reports"
        self.docs_path = self.data_dir / "documents.jsonl"
        self.cache_dir = Path(__file__).resolve().parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # v2 cache: chunk-level index (separately from old page-level cache)
        self.chunks_path = self.cache_dir / "chunks_v2.jsonl"
        self.emb_path = self.cache_dir / "embeddings_v2.npy"

        self.chunks = []
        self.embeddings = None
        self.model = None
        self._page_chunks = None  # (doc_id, page) -> [chunk_text, ...]

        self._ensure_index()

        # Прогреваем модель эмбеддингов до первого вопроса. Иначе загрузка модели
        # и первый encode попадут в измеряемое время первого вопроса (в eval `time`
        # считается вокруг вызова rag()), что делает среднее время нестабильным.
        if self.embeddings is not None and len(self.chunks):
            model = self._get_model()
            model.encode(
                ["query: warmup"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )

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

    def _extract_chunks(self):
        try:
            import fitz  # PyMuPDF
        except Exception:
            import pymupdf as fitz  # type: ignore

        docs = self._load_documents_meta()
        chunks = []
        for meta in docs:
            pdf_path = self.reports_dir / meta["path"]
            if not pdf_path.exists():
                continue
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                text = page.get_text("text") or ""
                text = re.sub(r"\s+", " ", text).strip()
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
        print(f"[RAG] Индекс v2 сохранён: {self.emb_path} shape={self.embeddings.shape}", flush=True)

    # ---------- Query analysis / boosting ----------
    def _infer_filters(self, question):
        """Извлекает подсказки из вопроса: банк (префикс doc_id), годы, период.
        Зачем: вопросы почти всегда указывают организацию/период; буст по метаданным
        повышает вероятность попадания нужного отчёта в top-k."""
        q = question.lower()
        orgs = set()
        for pat, prefix in self.ORG_HINTS:
            if re.search(pat, q):
                orgs.add(prefix)

        years = set(re.findall(r"\b(20\d{2})\b", q))
        period = None
        if re.search(r"6\s*месяц|полугод|1-?е\s*полугод|перв\w*\s+полугод|six months|6m", q):
            period = "6m"
        elif re.search(r"годов\w*\s+отчётн|за\s+20\d{2}\s+год\b|годовая|annual", q):
            period = "annual"

        return {"orgs": orgs, "years": years, "period": period}

    def _score_boost(self, doc_id, filters):
        """Мягкий бонус/штраф к косинусной близости на основе метаданных doc_id.
        doc_id имеет вид: <org>_<year>_<period>, напр. domrf_2025_6m, sber_2024_annual."""
        boost = 0.0
        parts = doc_id.split("_")
        org = parts[0] if parts else ""
        year = parts[1] if len(parts) > 1 else ""
        period = parts[2] if len(parts) > 2 else ""

        if filters["orgs"]:
            boost += 0.08 if org in filters["orgs"] else -0.08
        if filters["years"]:
            boost += 0.05 if year in filters["years"] else -0.03
        if filters["period"]:
            if period == filters["period"]:
                boost += 0.03
            elif period:
                boost -= 0.02
        return boost

    # ---------- Retrieval ----------
    def _search(self, question, top_k_pages=8):
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        model = self._get_model()
        q_emb = model.encode([f"query: {question}"], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)[0]
        base_scores = self.embeddings @ q_emb

        filters = self._infer_filters(question)
        # Буст считаем по doc_id чанка
        boosts = np.array([self._score_boost(c["doc_id"], filters) for c in self.chunks], dtype=np.float32)
        scores = base_scores + boosts

        # Берём больше чанков, затем агрегируем до уникальных страниц
        cand_k = min(max(top_k_pages * 6, 40), len(scores))
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
            "Дай точный и краткий ответ на русском. Сохраняй числа, валюты, единицы измерения, знаки и периоды "
            "в точности как в документе. Если вопрос составной — ответь на все части. "
            "Если в контексте недостаточно данных для точного ответа, прямо укажи это."
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

    def rag(self, question: str) -> dict:
        top_k = int(os.environ.get("RAG_TOP_K", "8"))
        hits = self._search(question, top_k_pages=top_k)
        context = self._group_context(hits)
        answer = self._generate(question, hits)
        return {"answer": answer, "context": context}
