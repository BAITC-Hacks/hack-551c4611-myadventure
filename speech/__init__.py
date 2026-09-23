"""Local speech pipeline internals, separate from the public transcript contract."""

from .alignment import AlignmentError, AlignmentResult, SpeechPipelineResult, TranscriptSegment, alignTranscript
from .audio import AudioPreparationError, PreparedAudio, prepareAudio
from .diarization import DiarizationError, DiarizationResult, DiarizationSegment, PyannoteLocalBackend, diarizeAudio
from .language import detectSegmentLanguage
from .stt import TranscriptionError, TranscriptionSegment, transcribeAudio

__all__ = [
    "AudioPreparationError", "PreparedAudio", "prepareAudio",
    "TranscriptionError", "TranscriptionSegment", "transcribeAudio",
    "DiarizationError", "DiarizationResult", "DiarizationSegment", "PyannoteLocalBackend", "diarizeAudio",
    "AlignmentError", "AlignmentResult", "SpeechPipelineResult", "TranscriptSegment", "alignTranscript",
    "detectSegmentLanguage",
]
