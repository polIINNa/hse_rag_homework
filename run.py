"""Запуск всего одной командой:  python3 run.py

1. Ключ эмбеддера курса EMB_KEY в .env (если нет — спросит и сохранит; .env в git не попадает).
2. LLM курса (claude.gigaparsers.ru): если ключа LLM_API_KEY нет в .env — спросит; формат API
   (OpenAI или Anthropic) и модель определит сам. Другие варианты (OpenRouter, локальная Ollama) — см. README.
3. Индекс эмбеддингов: готовый лежит в index/ — пропускается; если его нет — считается (около часа).
4. Eval на test → results/test_results.xlsx и 4 средних в README.md (при обрыве продолжает;
   python3 run.py --fresh — посчитать заново, не продолжая прошлый прогон).

Перед первым запуском:  python3 -m pip install -r requirements.txt
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from getpass import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COURSE_LLM_URL = "https://claude.gigaparsers.ru"


def ensure_emb_key():
    env_path = ROOT / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    present = {line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
               for line in lines if "=" in line and not line.lstrip().startswith("#")}
    if present.get("EMB_KEY"):
        return
    value = getpass("Вставьте EMB_KEY — ключ эмбеддера курса (ввод не отображается) и нажмите Enter: ").strip()
    lines = [line for line in lines if not line.strip().startswith("EMB_KEY=")] + [f"EMB_KEY={value}"]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("EMB_KEY сохранён в .env")


def read_env():
    env_path = ROOT / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    return {line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
            for line in lines if "=" in line and not line.lstrip().startswith("#")}


def save_env(values):
    env_path = ROOT / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    lines = [line for line in lines if line.split("=", 1)[0].strip() not in values]
    lines += [f"{key}={value}" for key, value in values.items()]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick_model(models):
    for family in ("sonnet", "haiku", "opus"):
        found = sorted((m for m in models if family in m.lower()), reverse=True)
        if found:
            return found[0]
    return models[0] if models else None


def detect_llm():
    """Для LLM_BASE_URL из .env понять формат API (OpenAI или Anthropic) и модель; сохранить в .env."""
    import requests

    env = read_env()
    root = env["LLM_BASE_URL"].rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    key = env.get("LLM_API_KEY", "")
    if not key:
        sys.exit("В .env задан LLM_BASE_URL, но нет LLM_API_KEY — впишите ключ и запустите снова.")
    bearer = {"Authorization": f"Bearer {key}"}
    anthropic = {"x-api-key": key, "Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"}
    report = []

    models = []
    for headers in (bearer, anthropic):
        try:
            response = requests.get(f"{root}/v1/models", headers=headers, timeout=30)
            report.append(f"GET /v1/models → {response.status_code}")
            if response.ok:
                models = [m.get("id") for m in response.json().get("data", []) if m.get("id")]
                break
        except (requests.RequestException, ValueError) as error:
            report.append(f"GET /v1/models → {error.__class__.__name__}")
    if models:
        print("Модели на сервере:", ", ".join(models))
    model = env.get("LLM_MODEL") or pick_model(models)
    if not model:
        sys.exit("Не удалось узнать список моделей. Впишите в .env строку LLM_MODEL=<название модели> "
                 "(его даёт курс) и запустите снова.\n" + "\n".join(report))

    probe = [{"role": "user", "content": "Ответь одним словом: ок"}]
    tries = [
        ("openai", f"{root}/v1/chat/completions", bearer, {"model": model, "messages": probe, "max_tokens": 20}),
        ("anthropic", f"{root}/v1/messages", anthropic, {"model": model, "messages": probe, "max_tokens": 20}),
    ]
    for api_format, url, headers, payload in tries:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            report.append(f"{api_format}: {response.status_code} {response.text[:150]}")
            data = response.json() if response.ok else {}
        except (requests.RequestException, ValueError) as error:
            report.append(f"{api_format}: {error.__class__.__name__}")
            continue
        if data.get("choices") or data.get("content"):
            save_env({"LLM_BASE_URL": root, "LLM_API_FORMAT": api_format, "LLM_MODEL": model})
            print(f"LLM работает: формат {api_format}, модель {model} — записано в .env")
            return
    sys.exit("Сервер LLM не ответил ни в формате OpenAI, ни в формате Anthropic. "
             "Пришлите этот вывод (ключа в нём нет):\n" + "\n".join(report))


def ollama_models():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as response:
            return [m["name"] for m in json.load(response).get("models", [])]
    except OSError:
        return None


def ensure_ollama(config):
    models = ollama_models()
    if models is None:
        sys.exit("\nНе вижу Ollama. Установите её с https://ollama.com/download, откройте приложение Ollama "
                 "(иконка ламы появится в строке меню) и запустите python3 run.py ещё раз.")
    if any(name.split(":")[0] == config.LLM_MODEL for name in models):
        print(f"Локальная модель {config.LLM_MODEL} готова")
        return
    ollama = shutil.which("ollama")
    if not ollama:
        sys.exit("Ollama запущена, но команды ollama нет в терминале — переустановите приложение Ollama.")
    print(f"Скачиваю модель {config.OLLAMA_BASE_MODEL} (один раз, ~4,7 ГБ)…")
    subprocess.run([ollama, "pull", config.OLLAMA_BASE_MODEL], check=True)
    modelfile = (f"FROM {config.OLLAMA_BASE_MODEL}\n"
                 f"PARAMETER num_ctx {config.OLLAMA_CONTEXT}\n"
                 "PARAMETER temperature 0\n")
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as file:
        file.write(modelfile)
    subprocess.run([ollama, "create", config.LLM_MODEL, "-f", file.name], check=True)
    os.unlink(file.name)
    print(f"Модель {config.LLM_MODEL} создана (окно контекста {config.OLLAMA_CONTEXT} токенов)")


def main():
    os.chdir(ROOT)
    ensure_emb_key()

    env = read_env()
    if not env.get("LLM_BASE_URL") and not env.get("OPENROUTER_API_KEY"):
        key = getpass("Вставьте LLM_API_KEY — ключ к модели курса claude.gigaparsers.ru "
                      "(ввод не отображается) и нажмите Enter: ").strip()
        save_env({"LLM_BASE_URL": COURSE_LLM_URL, "LLM_API_KEY": key})
        env = read_env()
    if env.get("LLM_BASE_URL") and not (env.get("LLM_API_FORMAT") and env.get("LLM_MODEL")):
        detect_llm()

    # импорты после настройки .env: rag.config читает .env при импорте
    from rag import config

    print(f"LLM: {config.LLM_MODEL} через {config.LLM_BASE_URL}")
    if config.IS_LOCAL:
        ensure_ollama(config)

    from eval.run_eval import evaluate
    from rag import index

    print("\n=== 1/2 Индекс эмбеддингов ===")
    index.main()
    print("\n=== 2/2 Eval на test ===")
    evaluate(Path("data/test.jsonl"), Path("results/test_results.xlsx"), fresh="--fresh" in sys.argv)


if __name__ == "__main__":
    main()
