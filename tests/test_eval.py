import json

import pytest
from openpyxl import load_workbook

from eval import evaluator_excel as evaluation


def test_excel_contains_all_questions_and_exact_means(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-only")
    calls = []

    class RAG:
        def rag(self, question):
            calls.append(question)
            return {"answer": "=1+1", "context": [{"doc_id": "doc", "pages": [2]}]}

    monkeypatch.setattr(evaluation, "RAG", RAG)
    monkeypatch.setattr(evaluation, "judge", lambda gold, answer: {"answer_correctness": 1, "reason": "test"})
    clock = iter([10., 12., 20., 24.])
    monkeypatch.setattr(evaluation, "perf_counter", lambda: next(clock))
    dataset = tmp_path / "test.jsonl"
    items = [{"question_id": str(i), "question": f"q{i}", "gold_answer": "secret-reference",
              "gold_context": [{"doc_id": "doc", "pages": [1, 2]}]} for i in (1, 2)]
    dataset.write_text("\n".join(json.dumps(i) for i in items))
    target = tmp_path / "report.xlsx"
    evaluation.evaluate(dataset, target)
    assert calls == ["q1", "q2"]  # No reference answers or pages reach the RAG.
    report = load_workbook(target, data_only=False)
    assert list(report["Итоги"].values)[1] == (.5, 1., 1., 3.)
    details = list(report["По вопросам"].values)
    assert len(details) == 3
    assert details[0] == ("question_id", "question", "gold_context", "gold_answer", "context", "answer", "context_recall", "context_precision", "answer_correctness", "time")
    assert report["По вопросам"]["F2"].data_type == "s"
    report.close()


def test_judge_failure_does_not_create_fake_report(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-only")

    class RAG:
        def rag(self, question):
            return {"answer": "a", "context": []}

    monkeypatch.setattr(evaluation, "RAG", RAG)
    def fail(*args):
        raise RuntimeError("API unavailable")
    monkeypatch.setattr(evaluation, "judge", fail)
    source = tmp_path / "test.jsonl"
    source.write_text(json.dumps({"question_id": "1", "question": "q", "gold_answer": "a", "gold_context": [{"doc_id": "d", "pages": [1]}]}))
    target = tmp_path / "report.xlsx"
    with pytest.raises(RuntimeError, match="API unavailable"):
        evaluation.evaluate(source, target)
    assert not target.exists()


def test_resume_only_retries_missing_work(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import langsmith

    monkeypatch.setenv("LLM_API_KEY", "test-only")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-only")
    items = [{"question_id": str(i), "question": f"q{i}", "gold_answer": f"gold{i}",
              "gold_context": [{"doc_id": "d", "pages": [1]}]} for i in (1, 2, 3)]
    source = tmp_path / "test.jsonl"
    source.write_text("\n".join(json.dumps(i) for i in items))
    target = tmp_path / "report.xlsx"
    context = items[0]["gold_context"]
    saved = {**items[0], "gold_context": json.dumps(context), "context": json.dumps(context),
             "answer": "a1", "context_recall": 1., "context_precision": 1.,
             "answer_correctness": 1, "time": 2., "judge_reason": "already scored"}
    target.with_suffix('.jsonl').write_text(json.dumps(saved) + '\n')
    examples = [SimpleNamespace(id=str(i), inputs={k: item[k] for k in ('question_id', 'question')},
                                outputs={k: item[k] for k in ('gold_context', 'gold_answer')})
                for i, item in enumerate(items, 1)]
    project = SimpleNamespace(id="project", reference_dataset_id="dataset", url="https://example.test/experiment")
    rag_calls, judge_calls, feedback = [], [], []

    class RAG:
        llm = SimpleNamespace(model="test")

        def rag(self, question):
            rag_calls.append(question)
            return {"answer": "a3", "context": context}

    class Client:
        def read_project(self, **kwargs):
            return project

        def read_dataset(self, **kwargs):
            return SimpleNamespace(id="dataset")

        def list_examples(self, **kwargs):
            return examples

        def list_runs(self, **kwargs):
            return [SimpleNamespace(id="run1", reference_example_id="1", error=None,
                                    inputs=examples[0].inputs,
                                    feedback_stats={"context_recall": {}, "context_precision": {}},
                                    outputs={"answer": "a1", "context": context, "time": 2.}),
                    SimpleNamespace(id="run2", reference_example_id="2", error=None,
                                    inputs=examples[1].inputs,
                                    outputs={"answer": "a2", "context": context, "time": 4.}),
                    SimpleNamespace(id="run3", reference_example_id="3", inputs=examples[2].inputs,
                                    error="quota", outputs=None)]

        def create_feedback(self, run_id, **kwargs):
            feedback.append((run_id, kwargs))

        def evaluate(self, predict, *, data, evaluators, experiment, **kwargs):
            assert experiment is project
            assert [e.inputs['question_id'] for e in data] == ['3']
            for example in data:
                evaluators[0](SimpleNamespace(error=None, outputs=predict(example.inputs)), example)

            class Results(list):
                experiment_name = "existing-experiment"
            return Results()

    def judge(gold, answer):
        judge_calls.append((gold, answer))
        return {"answer_correctness": 1, "reason": "test"}

    monkeypatch.setattr(evaluation, "RAG", RAG)
    monkeypatch.setattr(evaluation, "judge", judge)
    monkeypatch.setattr(langsmith, "Client", Client)
    clock = iter([10., 16.])
    monkeypatch.setattr(evaluation, "perf_counter", lambda: next(clock))
    evaluation.evaluate(source, target, langsmith=True, resume_experiment="existing-experiment")
    assert rag_calls == ["q3"]
    assert judge_calls == [("gold2", "a2"), ("gold3", "a3")]
    assert len(feedback) == 6
    assert {f['key'] for run_id, f in feedback if run_id == 'run1'} == {'answer_correctness', 'time'}
    report = load_workbook(target, data_only=True)
    assert list(report["Итоги"].values)[1] == (1., 1., 1., 4.)
    assert report["По вопросам"].max_row == 4
    report.close()
