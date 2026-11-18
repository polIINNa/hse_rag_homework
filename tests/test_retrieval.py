import unittest

from rag.retrieval import banks_in_question, report_years, stem, fuse


class RetrievalChecks(unittest.TestCase):
    def test_banks(self):
        self.assertEqual(banks_in_question('Сбербанк и ВТБ'),{'sber','vtb'})
        self.assertEqual(banks_in_question('Т‑Банк и Банк ДОМ.РФ'),{'tbank','domrf'})
        self.assertEqual(banks_in_question('НПФ Газпромбанк-фонд'),{'gpb'})
        self.assertEqual(banks_in_question('Неизвестная организация'),set())
    def test_snapshot_year_is_not_report_year(self):
        self.assertEqual(report_years('Сколько средств на 31 декабря 2023 года согласно отчетности?'),set())
        self.assertEqual(report_years('Согласно отчетности за 2024 год, сколько было в 2023 году?'),{'2024'})
        self.assertEqual(report_years('По отчетности за 2023 и 2024 годы'),{'2023','2024'})
        self.assertEqual(report_years('Сравни ВТБ и Сбербанк по отчетности за 2024 год'),set())
    def test_word_forms(self):
        self.assertEqual(stem('средств'),stem('средства'))
        self.assertEqual(stem('кредитные'),stem('кредитных'))
    def test_fusion_and_bank_scope(self):
        a=('vtb_2024_annual',1);b=('sber_2024_annual',2);c=('vtb_2023_annual',3)
        result=fuse([[b,a,c],[a,b,c]],{'vtb'})
        self.assertEqual(result,[a,c])
        self.assertEqual(fuse([[a,c]],{'vtb'},{'2023'})[0],c)

if __name__=='__main__':unittest.main()
