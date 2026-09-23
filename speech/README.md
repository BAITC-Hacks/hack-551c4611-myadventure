# Local audio preparation

For application integration, use `processMeetingAudio(file)` from `speech`.
It hides preprocessing, model inference, speaker alignment and language marking
behind the existing `SpeechPipelineResult`. See `PIPELINE.md` for the public API,
error handling, deployment configuration and explicitly marked DEMO support.

Requires Python 3.11+. Runtime uses only the standard library.

```python
from speech import AudioPreparationError, prepareAudio

try:
    audio = prepareAudio("meeting.wav", max_bytes=500 * 1024 * 1024)
    print(audio.duration_seconds, audio.sample_rate, audio.channels)
except AudioPreparationError as error:
    print(error.code, str(error))
```

Supports uncompressed PCM RIFF WAV (8/16/24/32-bit samples). Checks the file,
byte limit, extension, signature, container length, audio parameters and complete
sample data. Duration is computed from validated frames. Memory consumption is
bounded while reading the payload. Python 3.12+ also accepts PCM WAVE_EXTENSIBLE
through the standard library.

The input validation interface currently accepts WAV only. MP3 produces
`UNSUPPORTED_FORMAT`; the optional STT decoder does not expand the accepted
input formats. No system FFmpeg dependency or cloud fallback is added.
Resampling and mono conversion are deferred until an STT backend specifies its
requirements in `prepareAudio`. The STT adapter below performs this conversion
in memory. The source is never modified; no temporary files are created.
The returned path belongs to the caller, who must retain the file unchanged until
downstream processing completes. This internal interface does not alter the
external transcript contract in `contracts/speech`.

From the repository root:

```sh
python -m pip install -r speech/requirements-dev.txt
python -m unittest discover -s speech/tests -v
python -m ruff check --config speech/pyproject.toml speech
python -m mypy --config-file speech/pyproject.toml speech
python -c "from speech import prepareAudio, PreparedAudio, AudioPreparationError"
```

Errors use stable codes: `INVALID_FILE`, `EMPTY_FILE`, `FILE_TOO_LARGE`,
`UNSUPPORTED_FORMAT`, `CORRUPT_AUDIO`, `UNREADABLE_FILE`. Invalid limit configuration
raises `ValueError`. This module needs no build step.

## Local multilingual STT

Install `python -m pip install -r speech/requirements-stt.txt` in the speech worker
environment. `faster-whisper` uses CTranslate2 and bundled PyAV libraries; no
system FFmpeg command is needed. CPU int8 is the default. CUDA is optional and
requires a compatible local CUDA/cuDNN setup; it was not used in validation.

Provision weights separately from inference. Tested model:
[large-v3-turbo CTranslate2 conversion](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo),
MIT license, revision `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` (about 1.6 GB).
On a machine with download access, provision the five required files:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dropbox-dash/faster-whisper-large-v3-turbo",
    revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    local_dir="speech/models/large-v3-turbo",
    allow_patterns=["model.bin", "config.json", "tokenizer.json",
                    "preprocessor_config.json", "vocabulary.json"],
    token=False,
)
```

Alternatively copy these files from an approved local model store. Set
`JINALYS_STT_MODEL_DIR` to that directory in the worker environment, or supply
`model_path` explicitly. `speech/.env.example` documents configuration; it is not
automatically loaded. Model weights are ignored by Git. No API key is needed.

```python
from speech import prepareAudio, transcribeAudio, TranscriptionError

try:
    segments = transcribeAudio(prepareAudio("meeting.wav"))
    # [{"start": 0.2, "end": 2.4, "text": "..."}, ...]
except TranscriptionError as error:
    print(error.code, str(error))
```

Optional `language="ru"` or `"kk"` gives a monolingual hint. The default and
`language="mixed"` use per-segment language detection. The task is always
`transcribe`, never translate. English-only models are rejected. The adapter keeps
source-language text, applies local VAD and word alignment to reduce silence
hallucinations, validates/sorts timestamps and bounds them by the audio duration.
Whisper can still misrecognize words; mixed-language accuracy needs evaluation on
real meetings. No translation/post-editing step is applied.

The loader requires local model and tokenizer files before importing the backend,
sets `local_files_only=True` and `use_auth_token=False`, and caches one model.
Missing assets produce `MODEL_MISSING`; corrupt/incompatible assets or missing
dependencies produce `MODEL_LOADING_FAILED`; bad input produces `INVALID_AUDIO`;
inference errors, including generator iteration, produce `INFERENCE_FAILED`.
PyAV resamples validated WAV to mono 16 kHz in memory. Temporary audio is not
created. The external transcript contract remains unchanged; speaker assignment
is a later pipeline stage.

Tests without weights exercise the adapter and controlled failures using doubles.
To also run actual RU/KZ/mixed/silence inference, set `JINALYS_STT_MODEL_DIR` and
run the same unittest command above. Seven real-model tests are skipped only when
the variable is absent. They use attributed fixtures and block Python socket
connections during inference. For deployment, network isolation provides an
additional boundary. See `STT_VALIDATION.md` for the measured validation scope.

## Speaker diarization

`diarizeAudio(preparedAudio)` provides anonymous speaker turns through an optional
local pyannote backend. Without its separately provisioned model it returns a
controlled error. `mode="demo"` explicitly enables deterministic simulation,
marked `DEMO` with a warning in the result. See `DIARIZATION.md` for examples,
offline setup and the distinction between adapter tests and real-model evaluation.

## Speaker/transcript alignment

`alignTranscript(transcript, diarization, duration_seconds=...)` assigns speaker
IDs by maximum timestamp overlap and returns an internal `AlignmentResult`.
Its `.result` is the exact public `SpeechPipelineResult` JSON; DEMO provenance and
warnings stay outside that JSON. See `ALIGNMENT.md` for boundary/empty-input rules
and schema validation. The alignment schema test needs `npm ci --prefix
contracts/speech` in addition to the Python test tools listed above.

## Optional segment language

`detectSegmentLanguage(text)` returns `ru`, `kk`, `mixed`, or `None` using a small
local word/character heuristic. Alignment adds the already-supported `language`
field only when evidence is sufficient. It never changes text or STT settings.
Distinct Kazakh markers plus a Russian marker recognize mixed speech, including
`Асқар, осы аптада подрядчикпен сөйлесіп, новый график жасап беріңіз.`
Short/ambiguous text stays unclassified. A lone Kazakh name does not trigger mixed;
shared loanwords alone are not Russian evidence. This is an approximate helper,
not a linguistic model: transliteration, unknown vocabulary and subtle switches
can remain unclassified or be misclassified. No dependency or network call is added.
