# Local audio preparation

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

MP3 is not supported in this environment because no local decoder is installed.
It produces `UNSUPPORTED_FORMAT`; no FFmpeg dependency or cloud fallback is added.
Resampling and mono conversion are deferred until an STT backend specifies its
requirements. The source is never modified; no temporary files are created.
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
