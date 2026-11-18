# RAG по финансовой отчётности

PyMuPDF извлекает текст с сохранением расположения строк и столбцов. Для страниц без полноценного текстового слоя используется OCR Tesseract на русском и английском. Каждый фрагмент связан с идентификатором отчёта и номером страницы PDF.

Поиск объединяет косинусную близость векторов MiniLM (FastEmbed и ONNX Runtime) с TF-IDF по словам и биграммам (scikit-learn). При ранжировании учитываются банк, год и период отчётности. DeepSeek Flash получает найденные страницы и либо отвечает со ссылками на них, либо формулирует дополнительные запросы и читает второй набор страниц. В `context` записываются страницы, выбранные моделью как источники ответа.

Средние по 141 вопросу из test:

- `context_recall` — 0,8870
- `context_precision` — 0,7430
- `answer_correctness` — 0,9007 (127 из 141)
- `time` — 6,4319 с

### Зависимости и запуск

Понадобятся Python 3.12, архив `data/reports.zip` и `LLM_API_KEY` в `.env` или в переменных окружения.

```bash
python -m pip install -r requirements.txt
python -m zipfile -e data/reports.zip data/reports
python -m rag_retrieval
python -m eval.evaluator_excel data/test.jsonl --output test_results.xlsx --workers 1
```

При первом построении индекса загружаются языковые данные Tesseract и модель MiniLM. Индекс и модели кешируются в `index/`. Индекс строится по PDF и метаданным документов. Скрипт оценки передаёт в RAG только вопрос, записывает метрики в Excel через openpyxl и измеряет время поиска и генерации до обращения к судье. Между запросами к LLM выдерживается не менее 6,1 секунды, чтобы уложиться в лимит API в 10 запросов в минуту. Для возобновления прерванного прогона можно указать `--checkpoint outputs/test-progress.jsonl` при первом и повторном запуске.

### LangSmith

С `LANGSMITH_API_KEY` в `.env` следующая команда создаёт эксперимент с трассами поиска и генерации и Excel по тому же прогону. Скрипт загружает датасет в LangSmith при первом запуске и сверяет его с локальным файлом при повторном. Для рабочего пространства в регионе EU в `.env` нужна строка `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`.

```bash
python -m eval.evaluator_langsmith financial-rag-test --dataset-path data/test.jsonl --workers 1 --output test_results.xlsx
```



Ссылка на эксперимент: https://eu.smith.langchain.com/public/a3008c43-4e6d-4e84-bca2-4b7e416f4025/d
