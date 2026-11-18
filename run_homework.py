"""Одна команда подготовки: python run_homework.py. Полный прогон: добавить --run."""
import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="После подготовки запустить модель и официальный eval")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-install", action="store_true", help="Использовать подготовленное окружение без pip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # .env не попадает в архив сдачи; при запуске из архива ключ вводится без эха.
    from rag.common import load_env
    load_env()
    env.update({k: v for k, v in os.environ.items() if k.startswith(("LLM_", "RAG_", "LANGSMITH_"))})
    if args.run and not env.get("LLM_API_KEY"):
        env["LLM_API_KEY"] = getpass.getpass("Курсовой LLM_API_KEY (ввод скрыт): ").strip()
        if not env["LLM_API_KEY"]:
            raise RuntimeError("Ключ не введён")
    python = Path(sys.executable)
    if not args.skip_install:
        venv_python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not venv_python.exists():
            subprocess.run([str(python), "-m", "venv", str(root / ".venv")], check=True, env=env)
        python = venv_python
        subprocess.run([str(python), "-m", "pip", "install", "-r", "requirements.txt"], check=True, env=env)

    def run(*parts):
        subprocess.run([str(python), *parts], check=True, env=env)

    run("-m", "unittest", "discover", "-s", "tests", "-v")
    run("-m", "preflight")
    run("-m", "rag.prepare")
    if not (root / "artifacts/index.joblib").exists():
        run("-m", "rag.index")
    run("-m", "preflight", "--index")
    print("Подготовка завершена. До этой точки запросов к модели нет.", flush=True)
    if not args.run:
        print("Для полного eval: python run_homework.py --run")
        return
    command = ["-m", "evaluate", "--split", "test", "--output", "test_results.xlsx"]
    if args.resume:
        command.append("--resume")
    run(*command)
    print("Excel готов: test_results.xlsx; средние добавлены в README.md", flush=True)


if __name__ == "__main__":
    main()
