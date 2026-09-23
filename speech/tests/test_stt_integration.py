"""Real inference; opt in with JINALYS_STT_MODEL_DIR pointing to local weights."""

import os
import socket
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from speech import TranscriptionError, TranscriptionSegment, prepareAudio, transcribeAudio
from speech.stt import _load_model

FIXTURES = Path(__file__).parent / "fixtures"


@unittest.skipUnless(os.environ.get("JINALYS_STT_MODEL_DIR"), "Real local STT weights not configured")
class RealSTTTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("STT must work offline"))
        network.start()
        self.addCleanup(network.stop)

    def assert_valid(self, result: list[TranscriptionSegment], duration: float) -> str:
        self.assertTrue(result)
        starts = [segment["start"] for segment in result]
        self.assertEqual(starts, sorted(starts))
        for segment in result:
            self.assertGreaterEqual(segment["start"], 0)
            self.assertGreaterEqual(segment["end"], segment["start"])
            self.assertLessEqual(segment["end"], duration)
            self.assertTrue(segment["text"].strip())
        return " ".join(segment["text"] for segment in result).lower()

    def test_ru(self) -> None:
        audio = prepareAudio(FIXTURES / "ru.wav")
        text = self.assert_valid(transcribeAudio(audio, language="ru"), audio.duration_seconds)
        self.assertIn("ночного", text)
        self.assertIn("добычу", text)

    def test_automatic_language_detection(self) -> None:
        for name, keyword in [("ru.wav", "ночного"), ("kk.wav", "ғалымдар")]:
            with self.subTest(name=name):
                audio = prepareAudio(FIXTURES / name)
                text = self.assert_valid(transcribeAudio(audio), audio.duration_seconds)
                self.assertIn(keyword, text)

    def test_kk(self) -> None:
        audio = prepareAudio(FIXTURES / "kk.wav")
        text = self.assert_valid(transcribeAudio(audio, language="kk"), audio.duration_seconds)
        self.assertIn("ғалымдар", text)
        self.assertIn("арқылы", text)
        self.assertIn("ойлайды", text)

    def test_mixed(self) -> None:
        path = self.root / "mixed.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            for name in ["ru.wav", "kk.wav"]:
                with wave.open(str(FIXTURES / name), "rb") as source:
                    output.writeframes(source.readframes(source.getnframes()))
                if name == "ru.wav":
                    output.writeframes(b"\0\0" * 16000)
        audio = prepareAudio(path)
        text = self.assert_valid(transcribeAudio(audio, language="mixed"), audio.duration_seconds)
        self.assertIn("ночного", text)
        self.assertIn("ғалымдар", text)
        self.assertIn("арқылы", text)
        # Regression for the hallucination found in the initial real run.
        self.assertNotIn("thank you", text)

    def test_silence(self) -> None:
        path = self.root / "silence.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * 48000)
        self.assertEqual(transcribeAudio(prepareAudio(path)), [])

    def test_missing_model(self) -> None:
        with self.assertRaises(TranscriptionError) as caught:
            transcribeAudio(prepareAudio(FIXTURES / "ru.wav"), model_path=self.root / "missing")
        self.assertEqual(caught.exception.code, "MODEL_MISSING")

    def test_corrupt_model(self) -> None:
        folder = self.root / "broken-model"
        folder.mkdir()
        for name in ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.json"]:
            (folder / name).write_text("broken", encoding="utf-8")
        with self.assertRaises(TranscriptionError) as caught:
            transcribeAudio(prepareAudio(FIXTURES / "ru.wav"), model_path=folder)
        self.assertEqual(caught.exception.code, "MODEL_LOADING_FAILED")

    @classmethod
    def tearDownClass(cls) -> None:
        _load_model.cache_clear()
