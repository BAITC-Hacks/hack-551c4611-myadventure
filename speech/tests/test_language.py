import unicodedata
import unittest

from speech import detectSegmentLanguage


class LanguageTests(unittest.TestCase):
    def test_russian(self) -> None:
        for text in ["До пятницы подготовьте итоговый отчёт.", "Добрый день, коллеги!", "Мы обсудим сроки завтра."]:
            with self.subTest(text=text):
                self.assertEqual(detectSegmentLanguage(text), "ru")

    def test_kazakh(self) -> None:
        for text in ["Әріптестер, бүгін жиналысты бастаймыз.", "Мен осы аптада есеп жасап беремін.", "Осы аптада жаңа график жасап беріңіз."]:
            with self.subTest(text=text):
                self.assertEqual(detectSegmentLanguage(text), "kk")

    def test_required_mixed_example(self) -> None:
        text = "Асқар, осы аптада подрядчикпен сөйлесіп, новый график жасап беріңіз."
        self.assertEqual(detectSegmentLanguage(text), "mixed")
        self.assertEqual(detectSegmentLanguage(text.upper()), "mixed")

    def test_other_mixed_text(self) -> None:
        self.assertEqual(detectSegmentLanguage("Коллеги, бүгін жиналысты бастаймыз."), "mixed")

    def test_short_uncertain_and_unsupported(self) -> None:
        for text in ["", "   ", "...", "123", "ОК", "Иә", "Асқар", "проект график", "Hello world", "да да да"]:
            with self.subTest(text=text):
                self.assertIsNone(detectSegmentLanguage(text))

    def test_kazakh_name_does_not_make_russian_mixed(self) -> None:
        self.assertEqual(detectSegmentLanguage("Асқар, подготовьте новый отчёт."), "ru")

    def test_unicode_normalization(self) -> None:
        text = "Бүгін жиналысты бастаймыз."
        self.assertEqual(detectSegmentLanguage(unicodedata.normalize("NFD", text)), "kk")
