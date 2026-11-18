"""Запуск только моделей в уже подготовленном окружении, без установки и парсинга."""
import argparse
import getpass
import os
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Первые 3 train-вопроса, чтобы проверить API до полного test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repeat", action="store_true", help="Переоценить test и сверить допуск в пять процентов")
    parser.add_argument("--langsmith", action="store_true", help="При --repeat записать также эксперимент в LangSmith")
    parser.add_argument("--tag", default="", help="Метка отдельного smoke-прогона, например v2")
    args = parser.parse_args()
    if args.tag and (not args.smoke or not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag)):
        parser.error("--tag используется только со --smoke и содержит буквы, цифры, _ или -")
    if args.repeat and args.smoke:
        parser.error("Выберите --repeat или --smoke")
    if args.langsmith and (not args.repeat or args.resume):
        parser.error("--langsmith требует --repeat и отдельного запуска без --resume")
    root = Path(__file__).resolve().parent
    runtime = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not runtime.exists() or not (root / "artifacts/index.joblib").exists():
        raise RuntimeError("Сначала требуется подготовленное окружение и индекс")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if args.repeat and not (root / "test_results.xlsx").is_file():
        raise FileNotFoundError("Сначала нужен test_results.xlsx для сравнения")
    if args.repeat:
        repeat_output = "artifacts/test_repeat.xlsx"
        if args.langsmith:
            from rag.common import load_env

            if (root / repeat_output).exists():
                raise FileExistsError(f"Файл уже существует: {repeat_output}")
            load_env()
            key = os.environ.get("LANGSMITH_API_KEY") or getpass.getpass("LangSmith API key: ").strip()
            if not key:
                parser.error("Для LangSmith нужен ключ API")
            subprocess.run([str(runtime), "-m", "pip", "install", "-r", "requirements-langsmith.txt"],
                           cwd=root, env={k: v for k, v in env.items() if not k.endswith(("_KEY", "_TOKEN"))}, check=True)
            env.update(LANGSMITH_API_KEY=key, LANGSMITH_TRACING="true")
            dataset = "hse-rag-bolgov-test"
            subprocess.run([str(runtime), "langsmith_dataset.py", dataset], cwd=root, env=env, check=True)
            command = [str(runtime), "-m", "eval.evaluator_langsmith", dataset, "--output", repeat_output]
        else:
            command = [str(runtime), "-m", "evaluate", "--split", "test", "--output", repeat_output, "--no-readme"]
            if args.resume:
                command.append("--resume")
        subprocess.run(command, cwd=root, env=env, check=True)
        subprocess.run([str(runtime), "check_repeat.py", "test_results.xlsx", repeat_output],
                       cwd=root, env=env, check=True)
        return
    command = [str(runtime), "-m", "evaluate"]
    if args.smoke:
        name = f"smoke_{args.tag}_results.xlsx" if args.tag else "smoke_results.xlsx"
        command += ["--split", "train", "--limit", "3", "--output", "artifacts/" + name]
    else:
        command += ["--split", "test", "--output", "test_results.xlsx"]
    if args.resume:
        command.append("--resume")
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
