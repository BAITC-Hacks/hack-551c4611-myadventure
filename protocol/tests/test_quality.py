import unittest
from protocol.quality import clean_deadline, clean_person, meaningful_task


class QualityTests(unittest.TestCase):
    def test_deadline_is_not_urgency_or_full_instruction(self):
        self.assertIsNone(clean_deadline('Не надо откладывать'))
        self.assertIsNone(clean_deadline('Согласуем бюджет быстро'))
        self.assertEqual(clean_deadline('К 15 октября жду полный отчет.'), 'К 15 октября')
        self.assertEqual(clean_deadline('до 15 октября 2026 года'), 'до 15 октября 2026 года')
        self.assertEqual(clean_deadline('осы аптада'), 'осы аптада')

    def test_placeholders_never_become_people(self):
        transcript = [{'speakerId': 'VOICE_A'}]
        for value in ['null', ' NULL ', 'UNKNOWN', 'None', 'SPEAKER_00', 'VOICE_A', 'Спикер 2', None]:
            with self.subTest(value=value):
                self.assertIsNone(clean_person(value, transcript))
        self.assertEqual(clean_person(' Ерлан ', transcript), 'Ерлан')
        self.assertEqual(clean_person('Юридический департамент', transcript), 'Юридический департамент')

    def test_acknowledgements_and_objectless_tasks(self):
        for text in ['сделать', 'проверьте!', 'проверить', 'Хорошо, сделаю.', 'Поняла, сделаем']:
            self.assertFalse(meaningful_task(text))
        for text in ['Подготовить отчёт', 'Есепті дайындаңыз', 'Check report']:
            self.assertTrue(meaningful_task(text))
