import argparse
import json
import os
from pathlib import Path
from statistics import mean

from rag.corpus import DEFAULT_INDEX, ROOT, extract_chunks, read_documents
from rag.index import ReportIndex, save_index
from rag.models import EMBEDDING_MODEL, EMBEDDING_REVISION, Embedder
from rag.pipeline import RAG, source_context


def build(args):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    documents = read_documents(args.manifest)
    print(f"Extracting {sum(doc.pages for doc in documents)} pages from {len(documents)} PDFs", flush=True)
    chunks, skipped, ocr_pages = extract_chunks(documents, args.reports_dir, args.ocr_dir)
    if skipped:
        print(f"{len(skipped)} image pages had sparse native text. Supply --ocr-dir to include OCR.")
    print(f"Embedding {len(chunks)} passages ({ocr_pages} pages OCRed)", flush=True)
    revision = args.embedding_revision or (EMBEDDING_REVISION if args.embedding_model == EMBEDDING_MODEL else None)
    embedder = Embedder(args.embedding_model, revision)
    save_index(args.index, documents, chunks, args.embedding_model, embedder, args.batch_size)
    print(f"Index saved to {args.index}")


def ask(args):
    system = RAG(mode=args.mode)
    print(json.dumps(system.rag(args.question), ensure_ascii=False, indent=2))


def measure_retrieval(args):
    from eval.eval_functions import score_context

    index = ReportIndex.load(os.environ.get("RAG_INDEX", DEFAULT_INDEX))
    with args.dataset.open(encoding="utf-8") as source:
        items = [json.loads(line) for line in source if line.strip()]
    if not items:
        raise ValueError("The dataset is empty")
    scores = []
    for item in items:
        evidence = index.search(item["question"], args.top_pages)
        scores.append(score_context(item["gold_context"], source_context(evidence)))
    print(json.dumps({
        "questions": len(items),
        "context_recall": mean(score["context_recall"] for score in scores),
        "context_precision": mean(score["context_precision"] for score in scores),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Financial report RAG")
    commands = parser.add_subparsers(required=True)
    build_parser = commands.add_parser("build", help="Index the PDF corpus")
    build_parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "documents.jsonl")
    build_parser.add_argument("--reports-dir", type=Path, default=ROOT / "data" / "reports")
    build_parser.add_argument("--index", type=Path, default=Path(os.environ.get("RAG_INDEX", DEFAULT_INDEX)))
    build_parser.add_argument("--ocr-dir", type=Path, default=os.environ.get("TESSDATA_PREFIX"))
    build_parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    build_parser.add_argument("--embedding-revision")
    build_parser.add_argument("--batch-size", type=int, default=32)
    build_parser.set_defaults(run=build)

    ask_parser = commands.add_parser("ask", help="Answer one question and show the supporting PDF pages")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--mode", choices=["baseline", "agentic"])
    ask_parser.set_defaults(run=ask)

    retrieval_parser = commands.add_parser("retrieval", help="Measure page retrieval without an LLM judge")
    retrieval_parser.add_argument("dataset", type=Path)
    retrieval_parser.add_argument("--top-pages", type=int, choices=range(1, 13), default=4)
    retrieval_parser.set_defaults(run=measure_retrieval)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
