"""Bounded-memory validation of local PCM WAV input. No external services."""

import os
import stat
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AudioErrorCode = Literal[
    "INVALID_FILE", "EMPTY_FILE", "FILE_TOO_LARGE", "UNSUPPORTED_FORMAT",
    "CORRUPT_AUDIO", "UNREADABLE_FILE",
]
DEFAULT_MAX_BYTES = 500 * 1024 * 1024


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
    format: Literal["wav"] = "wav"


def prepareAudio(
    file: str | os.PathLike[str], *, max_bytes: int = DEFAULT_MAX_BYTES,
) -> PreparedAudio:
    """Validate PCM WAV content, size and all frames before returning metadata.

    Limits are applied before decoding. Extension is checked against content.
    MP3 is deliberately unsupported until a local decoder is configured.
    The caller must retain the source unchanged while consuming the result.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
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
            if path.suffix.lower() != ".wav":
                raise AudioPreparationError("UNSUPPORTED_FORMAT", "Only PCM WAV is supported; MP3 requires a local decoder.")
            header = source.read(12)
            if len(header) < 12:
                raise AudioPreparationError("CORRUPT_AUDIO", "WAV header is truncated.")
            if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise AudioPreparationError("UNSUPPORTED_FORMAT", "File content is not a RIFF/WAVE audio file.")
            if struct.unpack_from("<I", header, 4)[0] + 8 != info.st_size:
                raise AudioPreparationError("CORRUPT_AUDIO", "WAV container length does not match file size.")
            source.seek(0)
            with wave.open(source, "rb") as audio:
                rate = audio.getframerate()
                channels = audio.getnchannels()
                width = audio.getsampwidth()
                frames = audio.getnframes()
                if rate <= 0 or channels <= 0 or width not in (1, 2, 3, 4) or frames <= 0:
                    raise AudioPreparationError("CORRUPT_AUDIO", "WAV must contain PCM frames with valid audio parameters.")
                expected_bytes = frames * channels * width
                if expected_bytes > info.st_size:
                    raise AudioPreparationError("CORRUPT_AUDIO", "WAV declares more audio data than the file contains.")
                decoded_bytes = 0
                # Keep each read bounded even for unusual channel counts.
                chunk_frames = max(1, 65536 // (channels * width))
                while chunk := audio.readframes(chunk_frames):
                    decoded_bytes += len(chunk)
                    if decoded_bytes > expected_bytes:
                        raise AudioPreparationError("CORRUPT_AUDIO", "WAV contains incomplete frames or changed while reading.")
                if decoded_bytes != expected_bytes:
                    raise AudioPreparationError("CORRUPT_AUDIO", "WAV sample data is truncated or has incomplete frames.")
            return PreparedAudio(
                path=path, duration_seconds=frames / rate, sample_rate=rate,
                channels=channels, sample_width_bytes=width, frame_count=frames,
                size_bytes=info.st_size,
            )
    except (wave.Error, EOFError, struct.error, RuntimeError) as error:
        raise AudioPreparationError("CORRUPT_AUDIO", "WAV is damaged or uses an unsupported encoding; PCM WAV is required.") from error
    except OSError as error:
        raise AudioPreparationError("UNREADABLE_FILE", "Audio file could not be read.") from error
