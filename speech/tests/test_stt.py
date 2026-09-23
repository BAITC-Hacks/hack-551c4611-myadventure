import socket
import tempfile
import unittest
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speech import TranscriptionError, prepareAudio, transcribeAudio
from speech.stt import Language, _load_model


@dataclass
class Segment:
    start: float
    end: float
    text: str


class STTTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "audio.wav"
        with wave.open(str(self.path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * 16000)
        self.audio = prepareAudio(self.path)
        self.model = self.root / "model"
        self.model.mkdir()
        for name in ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.json"]:
            (self.model / name).write_text("fixture", encoding="utf-8")
        _load_model.cache_clear()
        self.addCleanup(_load_model.cache_clear)
        self.network = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def test_source_languages_and_transcribe_options(self) -> None:
        # Unit-level preservation; real recognition is tested separately.
        for language, text in [("ru", "Добрый день"), ("kk", "Қайырлы күн"), ("mixed", "Коллеги, жиналысты бастаймыз")]:
            with self.subTest(language=language), patch("speech.stt._load_model") as loader:
                loader.return_value.transcribe.return_value = ([Segment(0, 1, text)], object())
                # Literal arguments keep the public type boundary explicit.
                hint: Language = "ru" if language == "ru" else "kk" if language == "kk" else "mixed"
                output = transcribeAudio(self.audio, model_path=self.model, language=hint)
                self.assertEqual(output, [{"start": 0, "end": 1, "text": text}])
                kwargs = loader.return_value.transcribe.call_args.kwargs
                self.assertEqual(kwargs["task"], "transcribe")
                self.assertEqual(kwargs["language"], None if language == "mixed" else language)
                self.assertEqual(kwargs["multilingual"], language == "mixed")
                self.assertFalse(kwargs["without_timestamps"])
                self.assertTrue(kwargs["vad_filter"])

    def test_silence(self) -> None:
        with patch("speech.stt._load_model") as loader:
            loader.return_value.transcribe.return_value = ([], object())
            self.assertEqual(transcribeAudio(self.audio, model_path=self.model), [])

    def test_model_missing_and_incomplete(self) -> None:
        for path in [self.root / "missing", self.root]:
            with self.assertRaises(TranscriptionError) as caught:
                transcribeAudio(self.audio, model_path=path)
            self.assertEqual(caught.exception.code, "MODEL_MISSING")

    def test_model_not_configured(self) -> None:
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(TranscriptionError) as caught:
            transcribeAudio(self.audio)
        self.assertEqual(caught.exception.code, "MODEL_MISSING")

    def test_missing_tokenizer_cannot_trigger_download(self) -> None:
        (self.model / "tokenizer.json").unlink()
        with patch("speech.stt.importlib.import_module") as importer, self.assertRaises(TranscriptionError) as caught:
            transcribeAudio(self.audio, model_path=self.model)
        self.assertEqual(caught.exception.code, "MODEL_MISSING")
        importer.assert_not_called()

    def test_model_loading_failure(self) -> None:
        with patch("speech.stt.importlib.import_module", side_effect=ImportError("missing dependency")):
            with self.assertRaises(TranscriptionError) as caught:
                transcribeAudio(self.audio, model_path=self.model)
        self.assertEqual(caught.exception.code, "MODEL_LOADING_FAILED")

    def test_local_loading_flags_and_cache(self) -> None:
        with patch("speech.stt.importlib.import_module") as importer:
            factory = importer.return_value.WhisperModel
            factory.return_value.model.is_multilingual = True
            _load_model(str(self.model), "cpu", "int8")
            _load_model(str(self.model), "cpu", "int8")
            factory.assert_called_once_with(str(self.model), device="cpu", compute_type="int8", local_files_only=True, use_auth_token=False)

    def test_english_only_model_is_rejected(self) -> None:
        with patch("speech.stt.importlib.import_module") as importer:
            importer.return_value.WhisperModel.return_value.model.is_multilingual = False
            with self.assertRaises(TranscriptionError) as caught:
                _load_model(str(self.model), "cpu", "int8")
        self.assertEqual(caught.exception.code, "MODEL_LOADING_FAILED")

    def test_invalid_or_changed_audio(self) -> None:
        with self.assertRaises(TranscriptionError) as caught:
            transcribeAudio(replace(self.audio, duration_seconds=2), model_path=self.model)
        self.assertEqual(caught.exception.code, "INVALID_AUDIO")
        self.path.write_bytes(b"bad")
        with self.assertRaises(TranscriptionError) as caught:
            transcribeAudio(self.audio, model_path=self.model)
        self.assertEqual(caught.exception.code, "INVALID_AUDIO")

    def test_inference_failure_in_lazy_generator(self) -> None:
        def broken() -> object:
            raise RuntimeError("inference failed during iteration")

        lazy_segments = (broken() for _ in range(1))
        with patch("speech.stt._load_model") as loader:
            loader.return_value.transcribe.return_value = (lazy_segments, object())
            with self.assertRaises(TranscriptionError) as caught:
                transcribeAudio(self.audio, model_path=self.model)
        self.assertEqual(caught.exception.code, "INFERENCE_FAILED")

    def test_direct_inference_failure(self) -> None:
        with patch("speech.stt._load_model") as loader:
            loader.return_value.transcribe.side_effect = RuntimeError("out of memory")
            with self.assertRaises(TranscriptionError) as caught:
                transcribeAudio(self.audio, model_path=self.model)
        self.assertEqual(caught.exception.code, "INFERENCE_FAILED")

    def test_timestamps_are_sorted_and_blank_text_omitted(self) -> None:
        with patch("speech.stt._load_model") as loader:
            loader.return_value.transcribe.return_value = ([Segment(0.5, 1, "два"), Segment(0, 0.4, "бір"), Segment(0, 0, " ")], object())
            result = transcribeAudio(self.audio, model_path=self.model)
        self.assertEqual([item["start"] for item in result], [0, 0.5])

    def test_invalid_model_output(self) -> None:
        for segment in [Segment(-1, 1, "x"), Segment(2, 1, "x"), Segment(0, float("nan"), "x"), SimpleNamespace(start=0, end=1, text=None)]:
            with self.subTest(segment=segment), patch("speech.stt._load_model") as loader:
                loader.return_value.transcribe.return_value = ([segment], object())
                with self.assertRaises(TranscriptionError) as caught:
                    transcribeAudio(self.audio, model_path=self.model)
                self.assertEqual(caught.exception.code, "INFERENCE_FAILED")

    def test_model_timestamps_cannot_extend_past_audio(self) -> None:
        with patch("speech.stt._load_model") as loader:
            loader.return_value.transcribe.return_value = ([Segment(0.5, 2, "сөз"), Segment(2, 3, "extra")], object())
            result = transcribeAudio(self.audio, model_path=self.model)
        self.assertEqual(result, [{"start": 0.5, "end": 1, "text": "сөз"}])
