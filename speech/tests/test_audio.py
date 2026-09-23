import importlib
import importlib.util
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from speech import AudioPreparationError, prepareAudio
from speech.audio import DEFAULT_MAX_BYTES, DEFAULT_MAX_DURATION_SECONDS


class AudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def wav(
        self, name: str = "meeting.wav", channels: int = 1, rate: int = 16000,
        duration_seconds: int = 1,
    ) -> Path:
        path = self.directory / name
        with wave.open(str(path), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(b"\x00\x00" * channels * rate * duration_seconds)
        return path

    def mp3(self, name: str = "meeting.mp3") -> Path:
        module = importlib.import_module("av")
        path = self.directory / name
        output = module.open(str(path), "w")
        stream = output.add_stream("mp3", rate=16000)
        stream.layout = "mono"
        for _ in range(10):
            frame = module.AudioFrame(format="s16", layout="mono", samples=1600)
            frame.sample_rate = 16000
            frame.planes[0].update(bytes(3200))
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        output.close()
        return path

    def assert_error(self, path: Path, code: str) -> None:
        with self.assertRaises(AudioPreparationError) as caught:
            prepareAudio(path)
        self.assertEqual(caught.exception.code, code)
        self.assertTrue(str(caught.exception))

    def test_valid_wav(self) -> None:
        for channels, rate in [(1, 16000), (2, 44100)]:
            with self.subTest(channels=channels, rate=rate):
                path = self.wav(channels=channels, rate=rate)
                original = path.read_bytes()
                result = prepareAudio(path)
                self.assertEqual(result.duration_seconds, 1.0)
                self.assertGreaterEqual(result.duration_seconds, 0)
                self.assertEqual((result.channels, result.sample_rate), (channels, rate))
                self.assertEqual(result.path, path)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(list(self.directory.iterdir()), [path])

    @unittest.skipUnless(importlib.util.find_spec("av"), "PyAV runtime dependency not installed")
    def test_valid_mp3_and_actual_format_checks(self) -> None:
        path = self.mp3()
        original = path.read_bytes()
        result = prepareAudio(path)
        self.assertEqual(result.format, "mp3")
        self.assertAlmostEqual(result.duration_seconds, 1.0, places=2)
        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(path.read_bytes(), original)
        spoofed_wav = self.directory / "spoofed.wav"
        spoofed_wav.write_bytes(original)
        self.assert_error(spoofed_wav, "UNSUPPORTED_FORMAT")
        spoofed_mp3 = self.directory / "spoofed.mp3"
        spoofed_mp3.write_bytes(self.wav("source.wav").read_bytes())
        self.assert_error(spoofed_mp3, "UNSUPPORTED_FORMAT")

    @unittest.skipUnless(importlib.util.find_spec("av"), "PyAV runtime dependency not installed")
    def test_corrupt_mp3_and_missing_decoder_are_controlled(self) -> None:
        path = self.directory / "meeting.mp3"
        path.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00")
        self.assert_error(path, "CORRUPT_AUDIO")
        valid = self.mp3("valid.mp3")
        with patch("speech.audio.importlib.import_module", side_effect=ImportError("missing")):
            self.assert_error(valid, "DECODER_UNAVAILABLE")

    def test_empty_file(self) -> None:
        path = self.directory / "empty.wav"
        path.touch()
        self.assert_error(path, "EMPTY_FILE")

    def test_text_and_spoofed_extension(self) -> None:
        for name in ["notes.txt", "notes.wav"]:
            path = self.directory / name
            path.write_text("This is not an audio recording.", encoding="utf-8")
            self.assert_error(path, "UNSUPPORTED_FORMAT")

    def test_corrupt_audio_and_no_temporary_files(self) -> None:
        path = self.wav()
        original = path.read_bytes()
        for payload in [b"RIFF", original[:-20], original[:36] + b"xxxx" + original[40:]]:
            path.write_bytes(payload)
            self.assert_error(path, "CORRUPT_AUDIO")
            self.assertEqual(list(self.directory.iterdir()), [path])
            self.assertEqual(path.read_bytes(), payload)

    def test_truncated_data_with_repaired_container_length(self) -> None:
        path = self.wav()
        data = bytearray(path.read_bytes()[:-10])
        struct.pack_into("<I", data, 4, len(data) - 8)
        path.write_bytes(data)
        self.assert_error(path, "CORRUPT_AUDIO")

    def test_size_boundary(self) -> None:
        path = self.wav()
        size = path.stat().st_size
        self.assertEqual(prepareAudio(path, max_bytes=size).size_bytes, size)
        with self.assertRaises(AudioPreparationError) as caught:
            prepareAudio(path, max_bytes=size - 1)
        self.assertEqual(caught.exception.code, "FILE_TOO_LARGE")

    def test_meeting_limits_accept_1800_seconds_and_reject_longer(self) -> None:
        self.assertEqual(DEFAULT_MAX_BYTES, 400 * 1024 * 1024)
        self.assertEqual(DEFAULT_MAX_DURATION_SECONDS, 1800)
        accepted = prepareAudio(self.wav("accepted.wav", rate=1, duration_seconds=1800))
        self.assertEqual(accepted.duration_seconds, 1800)
        with self.assertRaises(AudioPreparationError) as caught:
            prepareAudio(self.wav("too-long.wav", rate=1, duration_seconds=1801))
        self.assertEqual(caught.exception.code, "DURATION_EXCEEDED")
        self.assertIn("not truncated", str(caught.exception))

    def test_missing_file_and_directory(self) -> None:
        self.assert_error(self.directory / "missing.wav", "INVALID_FILE")
        self.assert_error(self.directory, "INVALID_FILE")

    def test_invalid_limit(self) -> None:
        with self.assertRaises(ValueError):
            prepareAudio(self.wav(), max_bytes=0)
        for duration in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                prepareAudio(self.wav(), max_duration_seconds=duration)

    def test_io_error_is_controlled(self) -> None:
        path = self.wav()
        with patch.object(Path, "open", side_effect=PermissionError("denied")):
            self.assert_error(path, "UNREADABLE_FILE")

    def test_zero_sample_rate(self) -> None:
        path = self.wav()
        data = bytearray(path.read_bytes())
        struct.pack_into("<I", data, 24, 0)
        path.write_bytes(data)
        self.assert_error(path, "CORRUPT_AUDIO")

    def test_invalid_chunk_length_is_controlled(self) -> None:
        path = self.wav()
        data = bytearray(path.read_bytes())
        data[12:16] = b"JUNK"
        struct.pack_into("<I", data, 16, 0xFFFFFFFF)
        path.write_bytes(data)
        self.assert_error(path, "CORRUPT_AUDIO")

    def test_incomplete_pcm_frame(self) -> None:
        path = self.wav()
        data = bytearray(path.read_bytes()[:-1])
        struct.pack_into("<I", data, 4, len(data) - 8)
        struct.pack_into("<I", data, 40, len(data) - 44)
        path.write_bytes(data)
        self.assert_error(path, "CORRUPT_AUDIO")

    def test_wav_without_samples(self) -> None:
        path = self.wav()
        data = bytearray(path.read_bytes()[:44])
        struct.pack_into("<I", data, 4, 36)
        struct.pack_into("<I", data, 40, 0)
        path.write_bytes(data)
        self.assert_error(path, "CORRUPT_AUDIO")


if __name__ == "__main__":
    unittest.main()
