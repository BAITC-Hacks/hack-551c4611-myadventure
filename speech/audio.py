"""Bounded local validation of PCM WAV and MP3 meeting audio."""

import importlib
import math
import os
import stat
import struct
import wave
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

AudioErrorCode = Literal[
    "INVALID_FILE", "EMPTY_FILE", "FILE_TOO_LARGE", "DURATION_EXCEEDED",
    "UNSUPPORTED_FORMAT", "CORRUPT_AUDIO", "UNREADABLE_FILE", "DECODER_UNAVAILABLE",
]
DEFAULT_MAX_BYTES = 400 * 1024 * 1024
DEFAULT_MAX_DURATION_SECONDS = 1800.0


class AudioPreparationError(Exception):
    """Controlled input error; callers can use code independently of the message."""

    def __init__(self, code: AudioErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class PreparedAudio:
    """Validated source file; ownership stays with the caller. No temporary files."""

    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    size_bytes: int
    format: Literal["wav", "mp3"]


class _AvCodec(Protocol):
    @property
    def name(self) -> str: ...


class _AvStream(Protocol):
    @property
    def type(self) -> str: ...

    @property
    def codec_context(self) -> _AvCodec: ...


class _AvContainerFormat(Protocol):
    @property
    def name(self) -> str: ...


class _AvAudioFormat(Protocol):
    @property
    def bytes(self) -> int: ...


class _AvLayout(Protocol):
    @property
    def channels(self) -> Sequence[object]: ...


class _AvFrame(Protocol):
    @property
    def samples(self) -> int: ...

    @property
    def sample_rate(self) -> int: ...

    @property
    def format(self) -> _AvAudioFormat: ...

    @property
    def layout(self) -> _AvLayout: ...


class _AvContainer(Protocol):
    @property
    def format(self) -> _AvContainerFormat: ...

    @property
    def streams(self) -> Iterable[_AvStream]: ...

    def decode(self, *streams: _AvStream) -> Iterable[_AvFrame]: ...

    def close(self) -> None: ...


class _AvOpen(Protocol):
    def __call__(self, file: str, mode: str) -> _AvContainer: ...


def _duration_error(duration: float, maximum: float) -> AudioPreparationError:
    return AudioPreparationError(
        "DURATION_EXCEEDED",
        f"Audio duration {duration:.3f} seconds exceeds the {maximum:g}-second limit; the recording was not truncated.",
    )


def _prepare_wav(path: Path, size: int, max_duration_seconds: float) -> PreparedAudio:
    try:
        with path.open("rb") as source:
            header = source.read(12)
            if len(header) < 12:
                raise AudioPreparationError("CORRUPT_AUDIO", "WAV header is truncated.")
            if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise AudioPreparationError("UNSUPPORTED_FORMAT", "File content is not RIFF/WAVE audio.")
            if struct.unpack_from("<I", header, 4)[0] + 8 != size:
                raise AudioPreparationError("CORRUPT_AUDIO", "WAV container length does not match file size.")
            source.seek(0)
            with wave.open(source, "rb") as audio:
                rate = audio.getframerate()
                channels = audio.getnchannels()
                width = audio.getsampwidth()
                frames = audio.getnframes()
                if rate <= 0 or channels <= 0 or width not in (1, 2, 3, 4) or frames <= 0:
                    raise AudioPreparationError("CORRUPT_AUDIO", "WAV must contain PCM frames with valid audio parameters.")
                duration = frames / rate
                if duration > max_duration_seconds:
                    raise _duration_error(duration, max_duration_seconds)
                expected_bytes = frames * channels * width
                if expected_bytes > size:
                    raise AudioPreparationError("CORRUPT_AUDIO", "WAV declares more audio data than the file contains.")
                decoded_bytes = 0
                chunk_frames = max(1, 65536 // (channels * width))
                while chunk := audio.readframes(chunk_frames):
                    decoded_bytes += len(chunk)
                    if decoded_bytes > expected_bytes:
                        raise AudioPreparationError("CORRUPT_AUDIO", "WAV contains incomplete frames or changed while reading.")
                if decoded_bytes != expected_bytes:
                    raise AudioPreparationError("CORRUPT_AUDIO", "WAV sample data is truncated or has incomplete frames.")
        return PreparedAudio(
            path=path, duration_seconds=duration, sample_rate=rate, channels=channels,
            sample_width_bytes=width, frame_count=frames, size_bytes=size, format="wav",
        )
    except AudioPreparationError:
        raise
    except (wave.Error, EOFError, struct.error, RuntimeError) as error:
        raise AudioPreparationError(
            "CORRUPT_AUDIO", "WAV is damaged or uses an unsupported encoding; PCM WAV is required.",
        ) from error
    except OSError as error:
        raise AudioPreparationError("UNREADABLE_FILE", "Audio file could not be read.") from error


def _prepare_mp3(path: Path, size: int, max_duration_seconds: float) -> PreparedAudio:
    try:
        module = importlib.import_module("av")
        open_media = cast(_AvOpen, module.open)
    except (ImportError, AttributeError) as error:
        raise AudioPreparationError(
            "DECODER_UNAVAILABLE", "MP3 validation requires PyAV; install speech/requirements-stt.txt.",
        ) from error

    container: _AvContainer | None = None
    try:
        container = open_media(str(path), "r")
        formats = {name.strip().lower() for name in container.format.name.split(",")}
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if "mp3" not in formats or not audio_streams or not audio_streams[0].codec_context.name.lower().startswith("mp3"):
            raise AudioPreparationError("UNSUPPORTED_FORMAT", "File content is not MPEG Layer III audio.")

        sample_rate = 0
        channels = 0
        sample_width = 0
        frame_count = 0
        duration = 0.0
        for frame in container.decode(audio_streams[0]):
            rate = int(frame.sample_rate)
            frame_channels = len(frame.layout.channels)
            samples = int(frame.samples)
            width = int(frame.format.bytes)
            if rate <= 0 or frame_channels <= 0 or samples <= 0 or width <= 0:
                raise AudioPreparationError("CORRUPT_AUDIO", "MP3 decoder returned invalid audio parameters.")
            if sample_rate and (rate != sample_rate or frame_channels != channels):
                raise AudioPreparationError("CORRUPT_AUDIO", "MP3 audio parameters change unexpectedly inside the file.")
            sample_rate, channels, sample_width = rate, frame_channels, width
            frame_count += samples
            duration += samples / rate
            if duration > max_duration_seconds:
                raise _duration_error(duration, max_duration_seconds)
        if frame_count == 0 or not math.isfinite(duration) or duration <= 0:
            raise AudioPreparationError("CORRUPT_AUDIO", "MP3 contains no decodable audio frames.")
        return PreparedAudio(
            path=path, duration_seconds=duration, sample_rate=sample_rate, channels=channels,
            sample_width_bytes=sample_width, frame_count=frame_count, size_bytes=size, format="mp3",
        )
    except AudioPreparationError:
        raise
    except OSError as error:
        raise AudioPreparationError("UNREADABLE_FILE", "Audio file could not be read.") from error
    except Exception as error:
        raise AudioPreparationError("CORRUPT_AUDIO", "MP3 is damaged or cannot be decoded completely.") from error
    finally:
        if container is not None:
            container.close()


def prepareAudio(
    file: str | os.PathLike[str], *, max_bytes: int = DEFAULT_MAX_BYTES,
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
) -> PreparedAudio:
    """Validate format, size, frames and duration without modifying the source."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if (
        isinstance(max_duration_seconds, bool) or not isinstance(max_duration_seconds, (int, float))
        or not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0
    ):
        raise ValueError("max_duration_seconds must be finite and positive")
    path = Path(file)
    try:
        if not path.is_file():
            raise AudioPreparationError("INVALID_FILE", "Audio input must be an existing regular file.")
        with path.open("rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise AudioPreparationError("INVALID_FILE", "Audio input must be a regular file.")
            if info.st_size == 0:
                raise AudioPreparationError("EMPTY_FILE", "Audio file is empty.")
            if info.st_size > max_bytes:
                raise AudioPreparationError("FILE_TOO_LARGE", f"Audio file exceeds the {max_bytes}-byte limit.")
        extension = path.suffix.lower()
        if extension == ".wav":
            return _prepare_wav(path, info.st_size, float(max_duration_seconds))
        if extension == ".mp3":
            return _prepare_mp3(path, info.st_size, float(max_duration_seconds))
        raise AudioPreparationError("UNSUPPORTED_FORMAT", "Only PCM WAV and MP3 audio are supported.")
    except AudioPreparationError:
        raise
    except OSError as error:
        raise AudioPreparationError("UNREADABLE_FILE", "Audio file could not be read.") from error
