"""Векторный индекс страниц отчётов в index/chroma.

Из корня репозитория: python -m solution.build_index
Перед этим: python -m solution.extract (создаёт outputs/pages.jsonl).

Одна страница = один чанк. Эмбеддинги — сервис курса (deepvk/USER-bge-m3), ключ в EMB_KEY.
"""

import json
import os
import time

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from solution.clean import clean_text
from solution.documents import ROOT, load_documents
from solution.extract import PAGES_PATH

INDEX_DIR = ROOT / "index" / "chroma"
COLLECTION = "pages"
EMB_URL = "https://embeddings.parsers360.ru:10443"
MIN_INTERVAL = 1.0  # секунд между запросами к сервису эмбеддингов
ATTEMPTS = 8  # попыток на один текст, если сервис недоступен или перегружен
RETRY_STATUSES = (429, 500, 502, 503, 504)
BATCH_SIZE = 25  # страниц за раз: пачка сохраняется в индекс целиком, при сбое теряется только она
PERIODS = {"annual": "годовой отчёт за {year} год", "6m": "отчёт за 6 месяцев {year} года"}


class CourseEmbeddings(Embeddings):
    """Эмбеддер курса для LangChain: один текст — один запрос к сервису, в ответ вектор из 1024 чисел."""

    def __init__(self, api_key, url=EMB_URL):
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.url = url
        self.last_request = 0.0

    def embed_query(self, text):
        for attempt in range(ATTEMPTS):
            # лимит ключа — 60 запросов в минуту (GET /me), поэтому не чаще одного запроса в MIN_INTERVAL
            time.sleep(max(0.0, MIN_INTERVAL - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                response = requests.post(f"{self.url}/embed", headers=self.headers,
                                         json={"text": text[:20000]},  # сервис принимает до 20 000 символов
                                         timeout=(10, 60))  # 10 с на подключение, 60 с на ответ
            except (requests.ConnectionError, requests.Timeout):
                if attempt == ATTEMPTS - 1:
                    raise
            else:
                if response.status_code not in RETRY_STATUSES:
                    break
            time.sleep(min(2 ** attempt, 30))  # сервис недоступен или перегружен — ждём и пробуем ещё раз
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def page_header(document, page_no):
    """Строка перед текстом страницы: по ней поиск различает почти одинаковые страницы разных банков и лет."""
    period = PERIODS[document["period"]].format(year=document["year"])
    return f"{document['organization']}, {period}, стр. {page_no}"


def load_pages():
    """Страницы из outputs/pages.jsonl в виде Document: текст для поиска + метаданные."""
    documents = {document["doc_id"]: document for document in load_documents()}
    pages = []
    with open(PAGES_PATH, encoding="utf-8") as source:
        for line in source:
            page = json.loads(line)
            text = clean_text(page["text"])
            if not text:  # пустые страницы (сканы без OCR) в индекс не кладём
                continue
            
            document = documents[page["doc_id"]]
            pages.append(Document(
                page_content=f"{page_header(document, page['page'])}\n{text}",
                metadata={
                    "doc_id": page["doc_id"],
                    "page": page["page"],
                    "bank": document["bank"],
                    "year": document["year"],
                    "period": document["period"],
                    "source": page["source"],
                },
            ))
            
    return pages


def open_index():
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=str(INDEX_DIR),
        embedding_function=CourseEmbeddings(os.environ["EMB_KEY"]),
        collection_metadata={"hnsw:space": "cosine"},
    )


def page_id(page):
    return f"{page.metadata['doc_id']}:{page.metadata['page']}"


def main():
    """Эмбеддинги считаем только для новых и изменённых страниц: лимит сервиса — 60 страниц в минуту.
    Для полной пересборки (например, при смене модели эмбеддингов) удалите папку index/chroma."""
    pages = load_pages()
    store = open_index()
    stored = store.get(include=["documents", "metadatas"])
    stored_pages = {i: (text, meta) for i, text, meta in zip(stored["ids"], stored["documents"], stored["metadatas"])}

    current_ids = {page_id(page) for page in pages}
    new_pages = [page for page in pages if stored_pages.get(page_id(page)) != (page.page_content, page.metadata)]
    outdated_ids = [i for i in stored_pages if i not in current_ids]  # страниц больше нет в pages.jsonl
    outdated_ids += [page_id(page) for page in new_pages if page_id(page) in stored_pages]  # текст изменился
    if outdated_ids:
        store.delete(ids=outdated_ids)
    print(f"Страниц: {len(pages)}. Без изменений: {len(pages) - len(new_pages)}, "
          f"посчитать эмбеддинги: {len(new_pages)}, удалено из индекса: {len(outdated_ids)}", flush=True)

    for start in range(0, len(new_pages), BATCH_SIZE):
        batch = new_pages[start:start + BATCH_SIZE]
        store.add_documents(batch, ids=[page_id(page) for page in batch])
        print(f"добавлено {start + len(batch)} из {len(new_pages)} стр.", flush=True)
    print(f"Готово: индекс в {INDEX_DIR}, в нём {len(store.get(include=[])['ids'])} стр.")


if __name__ == "__main__":
    main()
