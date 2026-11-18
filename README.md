# rag по финансовой отчётности

решение александра болдырева
ветка aleksandr_boldyrev_main

результаты на всех 141 вопросах test

context_recall - 0.7812
context_precision - 0.2628
answer_correctness - не оценена без ключа преподавателя
time - 6.8369

для оценки answer_correctness нужен ключ преподавателя
без него эта колонка остаётся пустой
метрики поиска считаются функцией score_context
модель и промпт судьи в eval/eval_functions.py не менялись

из 55 pdf извлечены 3925 страниц с исходными номерами
pdftotext сохраняет расположение текста и таблиц
страницы без текстового слоя распознаются tesseract на русском и английском при 300 dpi

для векторного индекса соседние страницы собраны в 511 групп до 8 страниц
от каждой страницы берутся первые 1800 символов
эмбеддинги размерности 1024 получены сервисом преподавателя
поиск объединяет bm25 по отдельным страницам и косинусную близость групп
учитываются организация год и тип отчёта
четыре найденные страницы передаются локальной qwen2.5-3b-instruct в 4 битах через mlx
модель может запросить дополнительный поиск если не хватает данных
тогда объединяются до 8 страниц из двух поисков
это решение сохраняется в results/agent_trace.jsonl
эталонные ответы и страницы используются только при оценке
langsmith не подключён

прогон выполнялся локально на apple m3 pro с 36 gib памяти
время измеряется от передачи вопроса готовому rag до получения полного ответа и контекста
загрузка модели и подготовка индекса в него не входят

приведённый прогон проверен на mac с apple silicon и python 3.12

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export EMB_KEY='ключ сервиса эмбеддингов'
python run_eval.py --no-judge
```

на linux зависимости выбирают cpu backend mlx по [официальной инструкции](https://ml-explore.github.io/mlx/build/html/install.html#cpu-only-linux)
этот вариант в данном прогоне не проверялся

готовый текст и индекс находятся в assets поэтому pdf заново обрабатывать не нужно
локальная модель автоматически скачивается с hugging face при первом запуске
результаты сохраняются в results/test_results.xlsx и results/test_predictions.jsonl
после прерывания можно продолжить командой python run_eval.py --no-judge --resume

когда появится ключ преподавателя ответы повторно генерировать не нужно

```bash
export LLM_API_KEY='ключ преподавателя'
python run_eval.py --judge-only
```

эта команда добавит официальную оценку в excel и обновит средние в этом файле
полный прогон с оценкой ответов запускается командой python run_eval.py

чтобы заново подготовить корпус нужны poppler tesseract и языки rus и eng

```bash
brew install poppler tesseract tesseract-lang
curl -L 'https://media.githubusercontent.com/media/polIINNa/hse_rag_homework/main/data/reports.zip' -o /tmp/rag-reports.zip
OMP_THREAD_LIMIT=1 python prepare.py --archive /tmp/rag-reports.zip
python embeddings.py
```

используются numpy scipy scikit-learn pymorphy3 requests pymupdf openpyxl и mlx-lm
