import argparse
import json
from .pipeline import RAG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Задать вопрос финансовым отчётам")
    parser.add_argument("question")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    result = RAG(mode="baseline" if args.baseline else "agent").rag(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))

