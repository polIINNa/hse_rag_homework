"""Чат с агентом по отчётам: уточняющие вопросы учитывают историю диалога.

Из корня репозитория: python -m rag.chat
"""

from rag.agent import Agent


def main():
    agent = Agent()
    history = []
    print("Вопрос по отчётности банков (пустая строка — выход).")
    while True:
        question = input("\n> ").strip()
        if not question:
            break
        result = agent.run(question, history=history)
        history = result["messages"]
        print(result["answer"])
        for search in result["searches"]:
            print(f"  [поиск] {search['query']} {search['doc_ids'] or ''}")
        print("  [источники]", ", ".join(f"{c['doc_id']} с. {', '.join(map(str, c['pages']))}"
                                         for c in result["context"]))


if __name__ == "__main__":
    main()
