# RAG по финансовой отчётности

Сначала извлекаем текст и сохраняем его вместе с айдишником, номером страницы и эмбеддингом, для поиска используем BM25 и cosine similarity, также модель может воспользоваться поиском еще раз, если ей не хватает контекста.

## Библиотеки

- `PyMuPDF`
- `NumPy`
- `requests`
- `openpyxl`

## Воспроизведение

Положите `EMB_KEY` и `LLM_API_KEY` в `.env` и распакуйте [архив отчётов](https://media.githubusercontent.com/media/polIINNa/hse_rag_homework/main/data/reports.zip) в `data/reports`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python build_index.py build &&
.venv/bin/python -m eval.evaluator_excel data/test.jsonl --output test_results.xlsx
```

## Результаты на test

| Метрика | Среднее |
| --- | ---: |
| `context_recall` | 0.49775414 |
| `context_precision` | 0.20070922 |
| `answer_correctness` | 0.65957447 |
| `time`, с | 12.00142672 |
