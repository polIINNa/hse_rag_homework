import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import fitz
from openpyxl import load_workbook

from eval.eval_functions import JUDGE_PROMPT, score_answer, score_context
from evaluate import COLUMNS, evaluate, save_excel
from rag.budget import ApiBudget, call_official_judge
from rag.common import ROOT
from rag.index import PageIndex, organizations, parse_corpus, ocr_page
from rag.pipeline import RAG, context_of, format_sources
from rag.retrieval import block_layout, compact_text, excerpt, safe_prefix, table_excerpt, tokens
from rag.scans import read_scan_table


class NoNetwork(unittest.TestCase):
    def setUp(self):
        self.network = patch("requests.sessions.Session.send", side_effect=AssertionError("Network disabled in tests"))
        self.network.start()
        self.addCleanup(self.network.stop)


class ParsingTests(NoNetwork):
    def test_budget_cut_does_not_turn_5592_into_5(self):
        text = "Доходы составили 5 592 млн руб. за период"
        result = safe_prefix(text, text.index("592") + 2)
        self.assertNotIn("5", result)
        result = safe_prefix(text, len(text) - 1)
        self.assertIn("5 592", result)

    @unittest.skipUnless((ROOT / "data/reports/vtb/2025/annual.pdf").exists(), "Real column fixture not downloaded")
    def test_two_columns_keep_full_macro_factor_in_prompt(self):
        with fitz.open(ROOT / "data/reports/vtb/2025/annual.pdf") as pdf:
            page = pdf[57]
            text, layout = block_layout(page, page.get_text("text", sort=True))
        self.assertEqual(layout, "columns")
        question = "Какие макроэкономические факторы используются ВТБ при расчете PD в течение 12 месяцев?"
        first = {"doc_id": "vtb_2025_annual", "page": 58, "text": text, "queries": [question]}
        others = [{"doc_id": "fixture", "page": n, "text": "other page"} for n in [1, 2, 3]]
        shown = format_sources([first, *others], 10000)[0]["text"]
        self.assertIn("RUONIA", shown)
        self.assertIn("усредненный объем экспорта", shown)

    def test_context_compaction_keeps_numbers_and_signs(self):
        raw = "  В миллионах рублей\nДоходы       5 592       (782)     1,25%"
        compact = compact_text(raw)
        for value in ["5 592", "(782)", "1,25%", "миллионах рублей"]:
            self.assertIn(value, compact)

    def test_excerpt_keeps_continuation_after_query_terms(self):
        text = ("Посторонний текст.\n" * 100 +
                "Макроэкономические факторы:\nусредненный объем экспорта, темп роста\n"
                "среднегодового значения ставки RUONIA.\n" + "Посторонний текст.\n" * 100)
        result = excerpt(text, "макроэкономические факторы", limit=350)
        self.assertIn("RUONIA", result)
        self.assertLessEqual(len(result), 350)

    def test_table_excerpt_keeps_column_dates_after_wrapped_row(self):
        source = (
            "На 1 января | На 31 декабря\n"
            "2024 г. | 2024 г.\n"
            "млн руб. | млн руб.\n"
            "Инвестиционные бумаги | 32 085 | 26 318\n"
            "Финансовые инструменты,\n"
            "оцениваемые по\n"
            "справедливой стоимости\n"
            "через прочий\n"
            "совокупный доход | 37 501 | 22 756\n"
            "Итого активы уровня 3 | 109 931 | 70 610\n"
        )
        shown = table_excerpt(source, "Каковы итого активы уровня 3 на конец 2024 года?", 410, {})
        self.assertIn("На 1 января | На 31 декабря\n2024 г. | 2024 г.\nмлн руб. | млн руб.", shown)
        self.assertIn("Итого активы уровня 3 | 109 931 | 70 610", shown)
        self.assertLessEqual(len(shown), 410)

    def test_financial_inflections_share_retrieval_tokens(self):
        self.assertEqual(tokens("чистые комиссионные доходы"), tokens("чистых комиссионных доходов"))

    @unittest.skipUnless((ROOT / "data/reports/gpb/2025/annual.pdf").exists()
                         and (ROOT / "data/tessdata/rus.traineddata").exists(), "Real OCR fixture not downloaded")
    def test_rotated_scan_keeps_original_pdf_unchanged(self):
        path = ROOT / "data/reports/gpb/2025/annual.pdf"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with fitz.open(path) as pdf:
            text, rotation = ocr_page(pdf[92], ROOT / "data/tessdata")
            self.assertEqual(rotation, 90)
            self.assertIn("Прошнуровано", text)
            self.assertEqual(len(pdf), 93)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    @unittest.skipUnless((ROOT / "data/reports/gpb/2025/annual.pdf").exists()
                         and (ROOT / "data/tessdata/rus.traineddata").exists(), "Real OCR fixture not downloaded")
    def test_scan_table_recovers_two_cash_balance_columns(self):
        with fitz.open(ROOT / "data/reports/gpb/2025/annual.pdf") as report:
            text = read_scan_table(report[14], ROOT / "data/tessdata")
        self.assertIn("344 064 | 1 063 355", text)

    def test_parsed_cache_reused_only_for_identical_pdf(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def create(text):
                with fitz.open() as pdf:
                    pdf.new_page().insert_text((50, 70), text)
                    pdf.save(root / "a.pdf")
            docs = [{"doc_id": "fixture", "path": "a.pdf", "organization": "Bank", "pages": 1}]
            create("Revenue first version 123")
            first, _ = parse_corpus(root, docs, cache_dir=root / "cache")
            with patch("rag.index.fitz.open", side_effect=AssertionError("Cache should be used")):
                second, _ = parse_corpus(root, docs, cache_dir=root / "cache")
            self.assertEqual(first, second)
            create("Revenue changed version 456")
            third, _ = parse_corpus(root, docs, cache_dir=root / "cache")
            self.assertIn("456", third[0]["text"])

    def test_physical_pdf_pages_and_table_values(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "report.pdf"
            with fitz.open() as pdf:
                for text in ["Statement 2024\nAmounts in millions\nRevenue 1 250", "Note 2\nLoss (350)\nPrevious year 2023"]:
                    page = pdf.new_page()
                    page.insert_text((50, 70), text)
                pdf.save(path)
            pages, warnings = parse_corpus(d, [{"doc_id": "bank_2024", "path": "report.pdf",
                                               "organization": "Bank", "pages": 2}])
            self.assertEqual([p["page"] for p in pages], [1, 2])
            self.assertIn("1 250", pages[0]["text"])
            self.assertIn("(350)", pages[1]["text"])
            self.assertFalse(warnings)

    def test_wrong_page_count_fails(self):
        with tempfile.TemporaryDirectory() as d:
            with fitz.open() as pdf:
                pdf.new_page().insert_text((50, 70), "Report")
                pdf.save(Path(d) / "a.pdf")
            with self.assertRaises(ValueError):
                parse_corpus(d, [{"doc_id": "x", "path": "a.pdf", "organization": "X", "pages": 2}])

    def test_official_judge_is_byte_identical(self):
        b = (ROOT / "eval/eval_functions.py").read_bytes()
        sha = hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()
        self.assertEqual(sha, "841185ed54cd1db2a6ebe72fd7e624edcb9586d9")


class RetrievalTests(NoNetwork):
    @classmethod
    def setUpClass(cls):
        cls.pages = [
            {"doc_id": "sber_2024_annual", "page": 10, "organization": "Сбербанк",
             "text": "Чистые комиссионные доходы Сбербанка за 2024 год составили 850 млн руб."},
            {"doc_id": "vtb_2024_annual", "page": 8, "organization": "ВТБ",
             "text": "Чистые комиссионные доходы ВТБ за 2024 год составили 250 млн руб."},
            {"doc_id": "sber_2025_annual", "page": 14, "organization": "Сбербанк",
             "text": "Сравнительные показатели за 2023 год: резерв кредитного риска 710 млн руб. Стадия 2."},
            {"doc_id": "sber_2024_annual", "page": 15, "organization": "Сбербанк",
             "text": "Операционная аренда помещений. Стоимость автомобилей и зданий. Обязательства."},
        ]
        cls.index = PageIndex(cls.pages)

    def test_financial_query_selects_relevant_bank_and_page(self):
        hits = self.index.search("Чистые комиссионные доходы Сбербанка за 2024 год", top_k=1)
        self.assertEqual((hits[0]["doc_id"], hits[0]["page"]), ("sber_2024_annual", 10))

    def test_multiple_banks_are_covered(self):
        hits = self.index.search("Сравни комиссионные доходы Сбербанка и ВТБ за 2024 год", top_k=2)
        self.assertEqual({p["doc_id"].split("_")[0] for p in hits}, {"sber", "vtb"})

    def test_comparative_data_in_later_report_is_not_filtered_out(self):
        hits = self.index.search("Резерв кредитного риска Сбербанка за 2023 год", top_k=1)
        self.assertEqual(hits[0]["doc_id"], "sber_2025_annual")

    def test_joblib_roundtrip(self):
        import joblib
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "index.joblib"
            joblib.dump(self.index, p)
            restored = PageIndex.load(p)
            self.assertEqual(restored.search("комиссионные доходы Сбербанка", top_k=1)[0]["page"], 10)

    def test_aliases(self):
        self.assertEqual(set(organizations("Т-Банк и банк ДОМ.РФ")), {"tbank", "domrf"})


class AgentTests(NoNetwork):
    def test_context_character_budget_is_hard_for_many_long_pages(self):
        pages = [{"doc_id": f"d{i}", "page": 1, "text": "заголовок " * 1000,
                  "excerpt": "значение 123 " * 1000} for i in range(12)]
        result = format_sources(pages, 2000)
        self.assertLessEqual(sum(len(p["text"]) for p in result), 2000)
        self.assertTrue(all(p["text"] for p in result))

    def test_new_excerpt_from_same_page_is_not_lost(self):
        first = {"doc_id": "a", "page": 1, "text": "full " * 3000, "excerpt": "OLD_VALUE"}
        second = {**first, "excerpt": "NEW_VALUE"}
        index, llm = Mock(), Mock()
        index.search.side_effect = [[first], [second]]
        llm.complete.side_effect = [{"action": "search", "queries": ["more"]},
                                    {"action": "answer", "answer": "NEW_VALUE"}]
        result = RAG(index=index, llm=llm).rag("question")
        source = llm.complete.call_args.args[1]["sources"][0]["text"]
        self.assertIn("OLD_VALUE", source)
        self.assertIn("NEW_VALUE", source)
        self.assertEqual(result["context"], [{"doc_id": "a", "pages": [1]}])

    def test_model_decides_extra_search_and_all_shown_pages_are_counted(self):
        first = {"doc_id": "sber_2024_annual", "page": 1, "text": "Заголовок"}
        extra = {"doc_id": "sber_2024_annual", "page": 5, "text": "Показатель 123 млн руб."}
        index = Mock()
        index.search.side_effect = [[first], [extra]]
        llm = Mock()
        llm.complete.side_effect = [
            {"action": "search", "queries": ["Сбербанк показатель за 2024 год"], "reason": "нет суммы"},
            {"action": "answer", "answer": "123 млн руб. [sber_2024_annual, с. 5]"},
        ]
        result = RAG(index=index, llm=llm, agentic=True).rag("Показатель Сбербанка?")
        self.assertEqual(result["context"], [{"doc_id": "sber_2024_annual", "pages": [1, 5]}])
        self.assertEqual(index.search.call_count, 2)
        self.assertTrue(llm.complete.call_args_list[0].args[1]["allow_search"])
        self.assertFalse(llm.complete.call_args_list[1].args[1]["allow_search"])
        for call in llm.complete.call_args_list:
            self.assertNotIn("gold_answer", call.args[1])
            self.assertNotIn("gold_context", call.args[1])

    def test_no_extra_search_when_model_can_answer(self):
        index, llm = Mock(), Mock()
        index.search.return_value = [{"doc_id": "x", "page": 2, "text": "123"}]
        llm.complete.return_value = {"action": "answer", "answer": "123"}
        result = RAG(index=index, llm=llm).rag("Вопрос")
        self.assertEqual(index.search.call_count, 1)
        self.assertEqual(result["answer"], "123")

    def test_repeated_search_after_final_round_gets_text_answer(self):
        index, llm = Mock(), Mock()
        first = {"doc_id": "report", "page": 1, "text": "Заголовок"}
        evidence = {"doc_id": "report", "page": 8, "text": "Показатель 123 млн руб."}
        index.search.side_effect = [[first], [evidence]]
        llm.complete.side_effect = [
            {"action": "search", "queries": ["показатель"]},
            {"action": "search", "queries": ["тот же показатель"]},
            "123 млн руб. [report, с. 8]",
        ]
        result = RAG(index=index, llm=llm, agentic=True).rag("Каков показатель?")
        self.assertEqual(index.search.call_count, 2)
        self.assertEqual(result["answer"], "123 млн руб. [report, с. 8]")
        self.assertEqual(result["context"], [{"doc_id": "report", "pages": [1, 8]}])
        system, payload = llm.complete.call_args.args
        self.assertIn("Поиск завершён", system)
        self.assertNotIn("allow_search", payload)
        self.assertEqual(payload["sources"][1]["text"], evidence["text"])
        self.assertFalse(llm.complete.call_args.kwargs["json_output"])
        self.assertEqual(result["steps"][-1]["action"], "answer_only_retry")

    def test_invalid_json_decision_fails(self):
        index, llm = Mock(), Mock()
        index.search.return_value = []
        llm.complete.return_value = {"action": "search", "queries": []}
        with self.assertRaises(ValueError):
            RAG(index=index, llm=llm).rag("Вопрос")


class EvaluationTests(NoNetwork):
    def test_resume_reuses_real_pending_prediction_after_judge_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data").mkdir(); (root / "eval").mkdir()
            item = {"question_id": "q1", "question": "Q", "gold_answer": "A",
                    "gold_context": [{"doc_id": "a", "pages": [1]}]}
            (root / "data/train.jsonl").write_text(json.dumps(item), encoding="utf-8")
            (root / "eval/eval_functions.py").write_text("synthetic fixture", encoding="utf-8")
            system = SimpleNamespace(index=SimpleNamespace(fingerprint="synthetic"), llm=SimpleNamespace(model="mock"),
                                     top_k=1, agentic=False, rag=Mock(return_value={"answer": "A",
                                     "context": [{"doc_id": "a", "pages": [1]}]}))
            judge = Mock(side_effect=[RuntimeError("temporary failure"), {"answer_correctness": 1, "reason": "fixture"}])
            with patch("evaluate.ROOT", root):
                with self.assertRaises(RuntimeError):
                    evaluate("train", root / "output.xlsx", system=system, judge=judge)
                self.assertFalse((root / "output.xlsx").exists())
                self.assertTrue((root / "output.pending.json").exists())
                evaluate("train", root / "output.xlsx", system=system, judge=judge, resume=True)
                self.assertEqual(system.rag.call_count, 1)
                self.assertFalse((root / "output.pending.json").exists())
                completed = json.loads((root / "output.jsonl").read_text(encoding="utf-8"))
                stale = {"question_id": completed["question_id"], "time": completed["time"],
                         "prediction": {k: completed.get(k, []) for k in ("answer", "context", "steps")}}
                (root / "output.pending.json").write_text(json.dumps(stale), encoding="utf-8")
                (root / "output.xlsx").unlink()
                evaluate("train", root / "output.xlsx", system=system, judge=judge, resume=True)
                self.assertEqual(system.rag.call_count, 1)

    def test_final_test_rejects_injected_models(self):
        with self.assertRaises(ValueError):
            evaluate("test", system=Mock())

    def test_context_scores_unique_pdf_pages(self):
        gold = [{"doc_id": "a", "pages": [1, 2]}]
        found = [{"doc_id": "a", "pages": [2, 2]}, {"doc_id": "b", "pages": [1]}]
        self.assertEqual(score_context(gold, found), {"context_recall": 0.5, "context_precision": 0.5})

    def test_excel_has_exact_columns_and_safe_text(self):
        row = {"question_id": "q", "question": "=1+2", "gold_context": [{"doc_id": "a", "pages": [1]}],
               "gold_answer": "123", "context": [{"doc_id": "a", "pages": [1]}], "answer": "=SUM(1)",
               "context_recall": 1., "context_precision": 1., "answer_correctness": 1,
               "time": 2.5, "judge_reason": "=not a formula"}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "result.xlsx"
            save_excel([row], p)
            book = load_workbook(p)
            self.assertEqual(tuple(c.value for c in book["По вопросам"][1]), COLUMNS)
            self.assertEqual(book["По вопросам"]["F2"].data_type, "s")
            self.assertEqual(book["По вопросам"]["B2"].data_type, "s")
            self.assertEqual(book["Итоги"]["D2"].value, 2.5)
            book.close()

    def test_eval_times_only_rag_and_passes_only_question(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data").mkdir(); (root / "eval").mkdir()
            item = {"question_id": "q1", "question": "Тестовый вопрос", "gold_answer": "SECRET_GOLD",
                    "gold_context": [{"doc_id": "a", "pages": [1]}]}
            (root / "data/train.jsonl").write_text(json.dumps(item), encoding="utf-8")
            (root / "eval/eval_functions.py").write_text("synthetic fixture", encoding="utf-8")
            system = SimpleNamespace(index=SimpleNamespace(fingerprint="synthetic"), llm=SimpleNamespace(model="mock"),
                                     top_k=1, agentic=False, rag=Mock(return_value={"answer": "answer",
                                     "context": [{"doc_id": "a", "pages": [1]}]}))
            judge = Mock(return_value={"answer_correctness": 0, "reason": "synthetic test"})
            with patch("evaluate.ROOT", root), patch("evaluate.perf_counter", side_effect=[10., 12.5]):
                summary = evaluate("train", root / "output.xlsx", system=system, judge=judge)
            self.assertEqual(summary["time"], 2.5)
            system.rag.assert_called_once_with("Тестовый вопрос")
            judge.assert_called_once_with("SECRET_GOLD", "answer")

    def test_test_cannot_be_silently_shortened(self):
        with self.assertRaises(ValueError):
            evaluate("test", limit=1)


class BudgetTests(NoNetwork):
    def test_empty_usage_keeps_conservative_reservation(self):
        response = Mock()
        response.json.return_value = {"usage": {}}
        with tempfile.TemporaryDirectory() as d:
            b = ApiBudget(Path(d), limit=100000)
            b.request(Mock(return_value=response), "https://claude.gigaparsers.ru/v1/messages",
                      json={"model": "x", "max_tokens": 10})
            state = json.loads(b._path().read_text())
            self.assertGreater(state["charged_tokens"], 10)
            self.assertEqual(state["unmeasured_reservations"], 1)

    def test_spending_window_does_not_reset_at_midnight(self):
        clock = [100.]
        response = Mock()
        response.json.return_value = {"usage": {"input_tokens": 25, "output_tokens": 5}}
        with tempfile.TemporaryDirectory() as d, patch("rag.budget.time.time", side_effect=lambda: clock[0]), patch("rag.budget.time.sleep", side_effect=lambda s: clock.__setitem__(0, clock[0] + s)):
            b = ApiBudget(Path(d), limit=100000)
            b.request(Mock(return_value=response), "https://claude.gigaparsers.ru/v1/messages", json={"model": "x"})
            b.request(Mock(return_value=response), "https://claude.gigaparsers.ru/v1/messages", json={"model": "x"})
            state = json.loads(b._path().read_text())
            self.assertEqual(state["charged_tokens"], 60)
            self.assertEqual(len(state["requests"]), 2)
            self.assertGreaterEqual(clock[0], 106.2)
            clock[0] += 86401
            b.request(Mock(return_value=response), "https://claude.gigaparsers.ru/v1/messages", json={"model": "x"})
            state = json.loads(b._path().read_text())
            self.assertEqual(state["charged_tokens"], 30)

    def test_rate_and_actual_tokens_for_eleven_calls(self):
        clock = [100.]
        response = Mock()
        response.json.return_value = {"usage": {"input_tokens": 25, "output_tokens": 5}}
        starts = []
        def send(url, **kwargs):
            starts.append(clock[0])
            return response
        def sleep(seconds):
            clock[0] += seconds
        with tempfile.TemporaryDirectory() as d, patch("rag.budget.time.time", side_effect=lambda: clock[0]), patch("rag.budget.time.sleep", side_effect=sleep):
            b = ApiBudget(Path(d), limit=100000)
            for _ in range(11):
                b.request(send, "https://claude.gigaparsers.ru/v1/messages", json={"model": "x", "max_tokens": 10})
            state = json.loads(b._path().read_text())
            self.assertEqual(state["charged_tokens"], 11 * 30)
            self.assertEqual(state["unmeasured_reservations"], 0)
            self.assertGreaterEqual(starts[10] - starts[0], 60)

    def test_daily_guard_prevents_http_call(self):
        with tempfile.TemporaryDirectory() as d:
            send = Mock()
            with self.assertRaises(RuntimeError):
                ApiBudget(Path(d), limit=10).request(send, "https://claude.gigaparsers.ru/v1/messages",
                                                    json={"model": "x", "max_tokens": 100})
            send.assert_not_called()

    def test_judge_transport_preserves_official_payload(self):
        response = Mock()
        response.ok = True
        response.json.return_value = {"content": [{"type": "text", "text": '{"reason":"проверка","score":1}'}],
                                      "usage": {"input_tokens": 10, "output_tokens": 5}}
        with tempfile.TemporaryDirectory() as d:
            tracker = ApiBudget(Path(d), limit=100000)
            with patch("rag.budget.budget", return_value=tracker), patch("requests.post", return_value=response) as transport:
                result = call_official_judge(score_answer, "123 млн руб.", "123 млн руб.")
            payload = transport.call_args.kwargs["json"]
            self.assertEqual(payload["system"], JUDGE_PROMPT)
            self.assertEqual(payload["model"], "deepseek-flash")
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(result["answer_correctness"], 1)


if __name__ == "__main__":
    unittest.main()
