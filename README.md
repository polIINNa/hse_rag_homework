# RAG по финансовой отчётности банков

Артур Керопьян

## Результаты на test

Прогнал агентный RAG на тесте из 141 вопроса. Официальный судья засчитал 102 ответа. По каждому вопросу сохранил ответ, страницы и оценки в [test_results.xlsx](test_results.xlsx).

| Метрика | Среднее |
| --- | ---: |
| `context_recall` | 0.739125 |
| `context_precision` | 0.159431 |
| `answer_correctness` | 0.723404 |
| `time` | 8.322371 с |

Время включает поиск и генерацию, в том числе дополнительный запрос агента и ожидание лимита API. Судья вызывается после замера.

## Как устроено решение

Из 55 PDF извлёк текст с помощью PyPDF. Для сканов использовал RapidOCR, для части таблиц — Docling. При разбиении на чанки сохранил `doc_id` и номер страницы PDF. В итоговом индексе 4718 чанков.

На запросе объединяю поиск по эмбеддингам и полнотекстовый поиск SQLite через RRF. Учитываю банк и год отчёта из вопроса. Модели передаю пять найденных страниц, а при необходимости добавляю продолжение таблицы и выдержки с единицами измерения. Если данных не хватает, агент может сформулировать один дополнительный запрос и получить ещё одну страницу.

Генерация идёт через `deepseek-flash`. В колонке `context` сохраняю номера всех страниц, которые увидела модель. Туда же попадают страницы с единицами измерения, даже если их нет в эталоне.

- `rag/` — поиск, подготовка контекста, генерация и решение агента.
- `scripts/` — подготовка отчётов и построение индексов.
- `eval/` — оценка, экспорт Excel и трассировка LangSmith.

## Запуск

Я запускал проект на Python 3.12. Команды ниже выполняйте из корня репозитория. Если архива ещё нет или распаковка завершается ошибкой, скачайте [reports.zip](https://media.githubusercontent.com/media/polIINNa/hse_rag_homework/main/data/reports.zip) и сохраните его в папку `data/`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
test -f .env || cp .env.example .env
python -m zipfile -e data/reports.zip data/reports
```

В `.env` укажите ключи от преподавателя. `EMB_KEY` нужен для построения индекса и поиска по вопросам, `LLM_API_KEY` — для генерации ответов и судьи.

```bash
python -m scripts.prepare_corpus --stage native &&
python -m scripts.prepare_corpus --stage ocr &&
python -m scripts.prepare_corpus --stage tables &&
python -m scripts.build_index &&
python -m scripts.build_embeddings
```

На первом запуске RapidOCR и Docling скачивают веса моделей. Для этапа `tables` выбраны 20 страниц, которые вошли в итоговый индекс. Построение эмбеддингов с нуля занимает от 80 минут при лимите 60 запросов в минуту, обработка PDF идёт отдельно. После сбоя повторите прерванную команду. Готовые страницы и векторы сохраняются. При готовых файлах `corpus.sqlite`, `lexical_stems.sqlite` и `embedding_cache.sqlite` в `index/` можно сразу запускать оценку.

```bash
python -m eval.evaluator_excel data/test.jsonl --agent \
  --checkpoint outputs/test_checkpoint.json --output outputs/test_results.xlsx
```

При остановке повторите ту же команду. Она продолжит с первого вопроса без готовой оценки. Функции оценки взяты из задания.

## LangSmith

Полный прогон по всем 141 вопросам `test` я по ошибке запустил с отключённой трассировкой. Повторить его с трассировкой до дедлайна не удалось, поскольку суточная квота API закончилась.

В [эксперименте LangSmith](https://smith.langchain.com/public/b2e31080-e262-4784-a351-3165d9c09cb2/d/compare?selectedSessions=7e2eb190-6695-4a5e-b9a2-46cd8c429f6b) проверил два train-вопроса. Видно, какие страницы получил агент, запрашивал ли он дополнительные данные и какой балл поставил судья.

Чтобы запустить свой эксперимент, заполните `LANGSMITH_API_KEY` в `.env`.

```bash
python -m eval.evaluator_langsmith hse-rag-agent-pilot --pilot
```

## Проверки

```bash
python -m unittest discover -s tests
```
