# Speech Pipeline

## Назначение

Speech pipeline принимает локальный файл совещания, проверяет аудио, выполняет
локальное распознавание речи и diarization, сопоставляет текст со спикерами и
возвращает model-independent `SpeechPipelineResult`. Аудио и транскрипт не
передаются во внешние AI-сервисы. Имена реальных людей по голосу не определяются:
спикеры обозначаются как `SPEAKER_00`, `SPEAKER_01` и далее.

## Architecture

```text
Audio
→ preprocessing
→ local STT
→ diarization
→ alignment
→ TranscriptSegment[]
```

`speech.audio.prepareAudio` валидирует вход и вычисляет длительность.
`speech.stt.transcribeAudio` выполняет локальный Whisper inference и возвращает
текст с временными метками. `speech.diarization.diarizeAudio` получает интервалы
анонимных спикеров. `speech.alignment.alignTranscript` детерминированно выбирает
спикера по максимальному пересечению интервалов, добавляет стабильные segment ID
и необязательную языковую метку. Публичный orchestration находится в
`speech.processMeetingAudio`.

## Technologies

- Python 3.11+ и стандартная библиотека для preprocessing, orchestration,
  alignment и определения языка.
- `faster-whisper==1.2.1` с CTranslate2 и `av==18.1.0` для локального
  multilingual STT и проверки MP3.
- Silero VAD, который поставляется и вызывается через `faster-whisper`.
- `pyannote.audio==4.0.7` для необязательного реального локального diarization
  backend в отдельном Python-процессе.
- JSON Schema draft-07, TypeScript 5.9.3 и Ajv 8.17.1 для общего внешнего
  контракта в `contracts/speech`.
- `unittest`, Ruff 0.12.12 и mypy 1.18.2 для проверок Python-кода.

## Models

- STT: `dropbox-dash/faster-whisper-large-v3-turbo`, CTranslate2 conversion модели
  `openai/whisper-large-v3-turbo`, проверенная revision
  `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`. Это фактически проверенная модель.
- Diarization: `pyannote/speaker-diarization-community-1`. Код поддерживает полный
  локальный bundle этой модели, но в текущем окружении веса отсутствуют и качество
  реальной diarization ещё не проверялось.
- В DEMO-режиме модель diarization не используется: спикеры назначаются по
  фиксированным временным окнам. Это симуляция, а не результат speaker AI.
- Для `ru` / `kk` / `mixed` применяется небольшая локальная текстовая эвристика;
  отдельной модели language identification нет.

## Dependencies

Runtime STT:

```powershell
python -m pip install -r speech/requirements-stt.txt
```

Необязательный real diarization worker устанавливается отдельно, чтобы тяжёлые
зависимости pyannote не конфликтовали с основным процессом:

```powershell
py -3 -m venv speech/.venv-diarization
& speech/.venv-diarization/Scripts/python.exe -m pip install --upgrade pip
& speech/.venv-diarization/Scripts/python.exe -m pip install -r speech/requirements-diarization.txt
```

Инструменты разработки и зависимости проверки контракта:

```powershell
python -m pip install -r speech/requirements-dev.txt
npm ci --prefix contracts/speech
```

Точные прямые зависимости хранятся в `speech/requirements-stt.txt`,
`speech/requirements-diarization.txt`, `speech/requirements-dev.txt` и
`contracts/speech/package-lock.json`. Не добавляйте cloud STT SDK или API keys.

## System Dependencies

- Python 3.11 или новее.
- Node.js и npm нужны только для schema/type tests и интеграции TypeScript-контракта.
- Для PCM WAV, MP3 и `faster-whisper` отдельный FFmpeg executable не нужен:
  pinned PyAV декодирует и преобразует звук в mono 16 kHz внутри процесса.
- Git LFS нужен только для официального offline clone gated модели
  `pyannote/speaker-diarization-community-1`.
- Реальный pyannote worker может потребовать совместимые TorchCodec/FFmpeg shared
  libraries. В текущем окружении этот путь не проверен. DEMO и STT от них не зависят.
- Проверенная STT-модель занимает примерно 1.6 GB на диске; измеренный working
  set CPU worker после inference — около 0.96 GB. Публичный pipeline использует
  CPU int8 и не занимает общую RTX 4060 Laptop 8 GB, оставляя её Ollama/Qwen3 4B.
  Внутренний CUDA-режим не проверен совместно с Qwen3 и не используется интеграцией.

## Environment Variables

| Variable | Required | Использование |
| --- | --- | --- |
| `JINALYS_STT_MODEL_DIR` | Да | Полный путь к локальному CTranslate2 STT bundle. |
| `JINALYS_DIARIZATION_MODEL_DIR` | Для real diarization | Полный путь к локальному `community-1` bundle с `config.yaml`. |
| `JINALYS_DIARIZATION_PYTHON` | Если pyannote в отдельном venv | Python executable diarization worker. Без него используется текущий Python. |
| `HF_HUB_OFFLINE=1` | Рекомендуется | Запрещает Hugging Face Hub искать файлы в сети во время runtime. |
| `HF_HUB_DISABLE_TELEMETRY=1` | Рекомендуется | Отключает telemetry Hub-библиотек. |

`.env` автоматически не загружается. Переменные должны быть заданы процессу
worker. Diarization adapter дополнительно сам принудительно задаёт offline,
no-token и no-telemetry настройки своему дочернему процессу.

## Model Setup

### STT

Сначала установите `speech/requirements-stt.txt`. На машине, которой разрешён
доступ к Hugging Face, выполните одноразовое provision точной проверенной revision:

```powershell
New-Item -ItemType Directory -Force speech/models/large-v3-turbo | Out-Null
@'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dropbox-dash/faster-whisper-large-v3-turbo",
    revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    local_dir="speech/models/large-v3-turbo",
    allow_patterns=[
        "model.bin",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "vocabulary.json",
    ],
    token=False,
)
'@ | python -
```

Runtime проверяет наличие всех пяти файлов до загрузки backend. Затем модель
открывается с `local_files_only=True` и кэшируется один раз на процесс. Каталог
`speech/models` игнорируется Git; веса нельзя коммитить.

### Diarization

`community-1` является gated-моделью. На provisioning-машине сначала примите
условия доступа к `pyannote/speaker-diarization-community-1`. Затем выполните
официальный offline clone, используя Hugging Face access token только как пароль
при запросе Git. Не сохраняйте токен в repository или runtime environment.

```powershell
git lfs install
New-Item -ItemType Directory -Force speech/models | Out-Null
git clone https://hf.co/pyannote/speaker-diarization-community-1 speech/models/community-1
```

Передайте полный bundle в закрытое runtime-окружение. Во время inference worker
загружает только локальный путь с `token=False`; сетевой diarization API не
используется. До реальной интеграции проверьте bundle и native audio dependencies
на целевой машине.

## Run

Все команды ниже выполняются в PowerShell из корня repository. Этот запуск
использует реальный локальный STT и явно отмеченный DEMO diarization:

```powershell
$env:JINALYS_STT_MODEL_DIR = (Resolve-Path speech/models/large-v3-turbo).Path
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
@'
import json
from speech import PipelineDiagnostics, PipelineError, processMeetingAudio

diagnostics = PipelineDiagnostics()
try:
    result = processMeetingAudio(
        "speech/tests/fixtures/ru.wav",
        mode="demo",
        diagnostics=diagnostics,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(diagnostics)
except PipelineError as error:
    print(error.stage, error.code, str(error))
    raise
'@ | python -
```

Для real diarization дополнительно настройте локальный worker, уберите
`mode="demo"` и `diagnostics` можно оставить для внутренних предупреждений:

```powershell
$env:JINALYS_DIARIZATION_MODEL_DIR = (Resolve-Path speech/models/community-1).Path
$env:JINALYS_DIARIZATION_PYTHON = (Resolve-Path speech/.venv-diarization/Scripts/python.exe).Path
@'
import json
from speech import PipelineError, processMeetingAudio

try:
    print(json.dumps(
        processMeetingAudio("speech/tests/fixtures/ru.wav"),
        ensure_ascii=False,
        indent=2,
    ))
except PipelineError as error:
    print(error.stage, error.code, str(error))
    raise
'@ | python -
```

## Public Entry Point

```python
from speech import processMeetingAudio

result = processMeetingAudio("meeting.wav")
```

Signature:

```python
processMeetingAudio(
    file: str | os.PathLike[str],
    *,
    mode: Literal["local", "demo"] = "local",
    diagnostics: PipelineDiagnostics | None = None,
    demo_speakers: int = 1,
    demo_turn_seconds: float = 5.0,
) -> SpeechPipelineResult
```

Другой код должен зависеть только от этого entry point и публичного контракта,
а не от `prepareAudio`, `transcribeAudio`, конкретной Whisper-модели или pyannote.
Контролируемые ошибки выходят как `PipelineError` с полями `stage` и `code`.

## Input

- `str` или `os.PathLike[str]` существующего локального regular file.
- Принимаются RIFF PCM WAV (8/16/24/32-bit samples) и MPEG Layer III MP3.
- Максимальный размер — 400 MiB; максимальная длительность — 1800 секунд.
- Проверяются extension, фактический container/codec, параметры и полная
  декодируемость. Файл длиннее 1800 секунд возвращает `DURATION_EXCEEDED` и не
  обрезается. Повреждённый или подменённый файл даёт контролируемую ошибку.
- Pipeline не изменяет и не удаляет source file. Caller должен сохранять файл до
  завершения вызова.

## Output

Возвращается только `SpeechPipelineResult`; internal diagnostics в него не
добавляются. `speakerName` и `language` являются необязательными полями. Ниже —
фактический output локального STT для публичной RU-записи FLEURS; speaker label
получен в явно отмеченном DEMO-режиме и не является AI diarization.

```json
{
  "durationSeconds": 13.68,
  "detectedSpeakers": 1,
  "segments": [
    {
      "id": "seg-1",
      "speakerId": "SPEAKER_00",
      "start": 0.24,
      "end": 6.26,
      "text": "Они умеют отлично видеть в темноте при помощи ночного видения и почти незаметно передвигаться."
    },
    {
      "id": "seg-2",
      "speakerId": "SPEAKER_00",
      "start": 6.78,
      "end": 12.66,
      "text": "Оцелоты выслеживают добычу, сливаясь с окружающей обстановкой, а затем набрасываются на добычу."
    }
  ]
}
```

Segment ID уникален внутри результата, timestamps выражены в секундах и
отсортированы. `detectedSpeakers` не учитывает `SPEAKER_UNKNOWN`. При тишине
`segments` может быть пустым, а `detectedSpeakers` — равным нулю.
Каждый compact UTF-8 JSON segment короче либо равен 6000 символам. Если исходный
model segment длиннее 800 символов, текст делится у предложения/пробела, части
сохраняют speaker и исходные model start/end, а ID остаются уникальными во всём
транскрипте. Точные промежуточные timestamps не выдумываются.
Перед возвратом также проверяются общие лимиты AI-модуля: не более 120000
символов исходного текста и 5000 segments. Аномальный вывод отклоняется как
`PipelineError(stage="alignment", code="TRANSCRIPT_TOO_LARGE")`.

## Testing

Минимальные проверки без скачивания моделей; real-model тесты автоматически
skip, если соответствующий local model variable не задан:

```powershell
python -m unittest discover -s speech/tests -v
python -m ruff check --config speech/pyproject.toml speech
python -m mypy --config-file speech/pyproject.toml speech
python -m compileall -q speech
python -c "from speech import PipelineError, processMeetingAudio; print('IMPORT PASS')"
npm test --prefix contracts/speech
npm run lint --prefix contracts/speech
npm run typecheck --prefix contracts/speech
```

Полная проверка с уже provisioned STT-моделью:

```powershell
$env:JINALYS_STT_MODEL_DIR = (Resolve-Path speech/models/large-v3-turbo).Path
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
python -m unittest discover -s speech/tests -v
python -m unittest speech.tests.test_pipeline.PipelineEndToEndTests.test_real_mp3_real_stt_explicit_demo_diarization -v
python -m speech.fixture_suite --mode demo --report "$env:TEMP/jinalys-speech-fixtures.json"
```

Полностью model-based local E2E также требует
`JINALYS_DIARIZATION_MODEL_DIR` и совместимый `JINALYS_DIARIZATION_PYTHON`.

Измерение 2026-09-23: cold-start обработка публичного RU WAV длительностью 13.68
секунды заняла 19.152 секунды на CPU int8, включая загрузку локальной STT-модели,
полный pipeline и explicit DEMO diarization. Это одно измерение на текущем
ноутбуке, не throughput-гарантия. Реальный MP3 того же публичного аудио также
прошёл end-to-end; временный transcoded MP3 не коммитится.

## RU Test

```powershell
$env:JINALYS_STT_MODEL_DIR = (Resolve-Path speech/models/large-v3-turbo).Path
$env:HF_HUB_OFFLINE = "1"
python -m unittest speech.tests.test_stt_integration.RealSTTTests.test_ru -v
```

Fixture: `speech/tests/fixtures/ru.wav`, public FLEURS RU recording.

## KZ Test

```powershell
$env:JINALYS_STT_MODEL_DIR = (Resolve-Path speech/models/large-v3-turbo).Path
$env:HF_HUB_OFFLINE = "1"
python -m unittest speech.tests.test_stt_integration.RealSTTTests.test_kk -v
```

Fixture: `speech/tests/fixtures/kk.wav`, public FLEURS KZ recording.

## Mixed RU/KZ Test

```powershell
$env:JINALYS_STT_MODEL_DIR = (Resolve-Path speech/models/large-v3-turbo).Path
$env:HF_HUB_OFFLINE = "1"
python -m unittest speech.tests.test_stt_integration.RealSTTTests.test_mixed -v
```

Fixture: `speech/tests/fixtures/mixed.wav`, synthetic montage из публичных RU и KZ
записей с секундой тишины. Это последовательное переключение языков, а не
естественная фраза с code-switching.

## Multi Speaker Test

Полный fixture runner обрабатывает `multi-speaker.wav` и требует минимум два
speaker ID. В DEMO спикеры симулируются на известной границе montage:

```powershell
$env:JINALYS_STT_MODEL_DIR = (Resolve-Path speech/models/large-v3-turbo).Path
$env:HF_HUB_OFFLINE = "1"
python -m speech.fixture_suite --mode demo --report "$env:TEMP/jinalys-speech-fixtures.json"
```

`multi-speaker.wav` — synthetic montage двух разных публичных голосов FLEURS.
PASS в DEMO подтверждает pipeline и контракт, но не качество AI diarization.

## Demo/Fallback Mode

DEMO включается только явно и симулирует исключительно diarization. STT остаётся
реальным локальным inference. Автоматического fallback при ошибке модели нет.

```python
from speech import PipelineDiagnostics, processMeetingAudio

diagnostics = PipelineDiagnostics()
result = processMeetingAudio(
    "meeting.wav",
    mode="demo",
    diagnostics=diagnostics,
    demo_speakers=2,
    demo_turn_seconds=5.0,
)
assert diagnostics.mode == "DEMO"
```

`PipelineDiagnostics` обязателен для DEMO, должен храниться рядом с job state и
показываться пользователю. Публичный `SpeechPipelineResult` намеренно не содержит
provenance. DEMO нельзя представлять как определение реальных спикеров.

## Known Limitations

- Проверенная STT-модель требует около 1.6 GB веса и заметную RAM/CPU; accuracy на
  шумных совещаниях, дальнем микрофоне и перекрывающейся речи не измерялась.
- Казахское распознавание проверено на одном публичном fixture и содержит ошибки;
  WER target не установлен.
- Mixed fixture является монтажом, поэтому natural RU/KZ code-switching ещё не
  оценивался. Whisper может галлюцинировать текст на тишине.
- Language marker — эвристика. Короткий или неоднозначный segment может не получить
  `language` либо быть классифицирован ошибочно.
- Real `community-1` bundle и native dependencies отсутствовали в проверенном
  окружении; real diarization accuracy и fully local E2E пока не подтверждены.
- Alignment назначает один speaker на целый STT segment. Одновременная речь не
  разделяется на два transcript segment.
- `speakerId` стабилен только внутри одного результата и не идентифицирует человека.
- Два повторных запуска не показали накопления памяти, но длительный production
  soak test ещё не выполнялся.
- Граница 1800 секунд и поручение на отметке 1792 секунды проверены synthetic
  PCM-container test с injected STT/diarization output. Это проверка orchestration,
  глобальных timestamps и отсутствия truncation, а не реальное 30-минутное STT
  benchmark. Реальная 30-минутная встреча пока не измерялась.

## Integration Notes

Developer 3 должен:

1. Передать сохранённый локальный WAV или MP3 path Python worker и вызвать только
   `speech.processMeetingAudio`.
2. Catch `PipelineError`; отображать/логировать его `stage` и `code`, не разбирать
   текст сообщения как API.
3. Сериализовать возвращённый dict без добавления internal полей. TypeScript UI
   должен использовать `TranscriptSegment` и `SpeechPipelineResult` из
   `contracts/speech/index.ts`; runtime validation доступна в
   `contracts/speech/schema.json`.
4. Передать `result["segments"]` Developer 2 как `TranscriptSegment[]`.
5. Не ожидать `speakerName`: текущий pipeline выдаёт anonymous `speakerId`.
   Отображаемое имя можно сопоставить отдельно, не меняя speech implementation.
6. Хранить `PipelineDiagnostics` отдельно для каждого job. В DEMO обязательно
   показать предупреждение и не называть speaker count результатом AI.
7. Не удалять input file до возврата функции. Pipeline сам не создаёт temporary
   audio и не владеет исходным файлом.
8. Запускать долгоживущий worker, чтобы кэш `faster-whisper` переиспользовал одну
   модель между запросами. Настройки модели не передаются через frontend.
9. Для production local mode provision обе модели заранее, закрыть worker от
   outbound network и не передавать аудио, транскрипты или API keys наружу.
10. Запускать speech worker в CPU int8 profile. RTX 4060 Laptop 8 GB остаётся
    выделенной Ollama/Qwen3 4B; совместное размещение CUDA speech не сертифицировано.
