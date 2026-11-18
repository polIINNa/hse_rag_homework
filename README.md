# RAG по финансовой отчётности

Сначала извлекаем текст и сохраняем его вместе с айдишником, номером страницы и эмбеддингом, для поиска используем BM25 и cosine similarity, также модель может воспользоваться поиском еще раз, если ей не хватает контекста.

## Библиотеки

- `PyMuPDF`
- `NumPy`
- `requests`
- `openpyxl`

## Воспроизведение

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python build_index.py build &&
.venv/bin/python -m eval.evaluator_excel data/test.jsonl --output test_results.xlsx
```

## Результаты на test

| Метрика | Среднее |
| --- | ---: |
| `context_recall` | TBD |
| `context_precision` | TBD |
| `answer_correctness` | TBD |
| `time`, с | TBD |
