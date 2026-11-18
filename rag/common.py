import functools
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path=None):
    path = Path(path or ROOT / ".env")
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def traced(name, run_type="chain"):
    # Без установленного SDK базовая часть работает точно так же.
    def decorate(func):
        try:
            from langsmith import traceable
        except ImportError:
            if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
                raise RuntimeError("Для LANGSMITH_TRACING установите requirements-langsmith.txt")
            return func
        return traceable(name=name, run_type=run_type)(func)
    return decorate


load_env()
