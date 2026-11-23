import hashlib
import json
import re
from collections import defaultdict

import numpy as np

from rag import config
from rag.models import chat, embed
from rag.prompts import RAG_PROMPT


class RAG:
    def __init__(self, index_dir=config.INDEX, top_k=config.TOP_K):
        vector_file = index_dir / config.VECTORS_FILE
        if not vector_file.is_file():
            raise FileNotFoundError("Соберите индекс командой python -m rag.index.")
        chunks = (index_dir / "chunks.json").read_bytes()
        self.chunks = json.loads(chunks)
        with np.load(vector_file) as index:
            if (
                index["model"].item() != config.EMBEDDING_MODEL
                or index["chunks_hash"].item() != hashlib.sha256(chunks).hexdigest()
            ):
                raise ValueError(
                    "Индекс устарел. Пересоберите его командой python -m rag.index."
                )
            self.vectors = index["vectors"]
        if self.vectors.ndim != 2 or len(self.vectors) != len(self.chunks):
            raise ValueError("Размер индекса не совпадает с числом фрагментов.")
        self.top_k = top_k

    def rag(self, question):
        scores = self.vectors @ embed([question])[0]
        query = question.casefold()
        banks = [
            bank
            for bank, names in config.BANKS.items()
            if any(name in query for name in names)
        ]
        years = set(re.findall(r"\b20\d{2}\b", query))
        half_year = bool(
            re.search(r"шест[ьи]\s+месяц|перв[а-я]*\s+полугод|30\s+июня", query)
        )
        for i, chunk in enumerate(self.chunks):
            doc = chunk["doc_id"]
            scores[i] += config.PERIOD_BONUS * (doc.split("_")[1] in years)
            scores[i] += config.PERIOD_BONUS * (half_year and doc.endswith("_6m"))
        selected = []
        order = np.argsort(-scores, kind="stable")
        groups = banks or [None]
        count, remainder = divmod(self.top_k, len(groups))
        for number, bank in enumerate(groups):
            matches = [
                i
                for i in order
                if (bank is None or self.chunks[i]["doc_id"].startswith(bank + "_"))
            ]
            selected.extend(matches[: count + (number < remainder)])
        sources = defaultdict(set)
        passages = []
        for i in selected:
            chunk = self.chunks[i]
            sources[chunk["doc_id"]].add(chunk["page"])
            passages.append(
                f"[{chunk['doc_id']}, стр. {chunk['page']}]\n{chunk['text']}"
            )
        answer = chat(
            RAG_PROMPT,
            f"Вопрос: {question}\n\nОтчёты:\n" + "\n\n".join(passages),
        )
        return {
            "answer": answer,
            "context": [
                {"doc_id": doc, "pages": sorted(pages)}
                for doc, pages in sources.items()
            ],
        }
