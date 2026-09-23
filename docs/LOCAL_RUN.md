# Локальная интеграция: пошаговый запуск

Код объединяет существующие speech и protocol без второго frontend. Backend — локальный Python HTTP server с одной очередью обработки. Next.js проксирует поток файла и короткие запросы статуса. Реального облачного fallback нет.

## Этап 1 — зависимости
PowerShell, корень репозитория. Node 22.18+; Python 3.11/3.12 рекомендуется для совместимости ML-пакетов.
```powershell
npm.cmd ci
python -m venv .venv
.venv/Scripts/python -m pip install -r integration/requirements.txt
python -m venv speech/.venv-diarization
speech/.venv-diarization/Scripts/python -m pip install -r speech/requirements-diarization.txt
```
Веса моделей этими командами не скачиваются. Системные требования и подробности speech: speech/README.md и speech/DIARIZATION.md. GPU не требуется UI; speech использует CPU int8, qwen3:4b желательно запускать на GPU. Скорость 30-минутного аудио на этом компьютере ещё не измерена.

## Этап 2 — модели
1. Установить Ollama с https://ollama.com/download.
2. В отдельном терминале: `powershell -File protocol/start-local.ps1`.
3. В другом терминале:
```powershell
$env:OLLAMA_HOST='127.0.0.1:11435'
ollama pull qwen3:4b
```
4. Whisper: выполнить Python snapshot_download из speech/README.md в speech/models/large-v3-turbo. Требуются все пять перечисленных файлов, закреплённая там revision.
5. Pyannote: получить разрешённый полный offline bundle community-1 по speech/DIARIZATION.md и положить в speech/models/community-1. Владелец аккаунта самостоятельно принимает условия доступа модели на Hugging Face. Нужны реальные веса, а не Git LFS pointer-файлы. Для декодера pyannote могут потребоваться совместимые TorchCodec/FFmpeg shared libraries.

Модель pyannote нельзя заменять demo-диаризацией при проверке реальной цепочки. Не выдавать результаты без модели за распознавание.

## Этап 3 — запустить job server
Из корня репозитория, в отдельном PowerShell:
```powershell
$env:LOCAL_LLM_BASE_URL='http://127.0.0.1:11435'
$env:LOCAL_LLM_MODEL='qwen3:4b'
$env:LOCAL_LLM_TIMEOUT='600'
$env:JINALYS_STT_MODEL_DIR=(Resolve-Path speech/models/large-v3-turbo).Path
$env:JINALYS_DIARIZATION_MODEL_DIR=(Resolve-Path speech/models/community-1).Path
$env:JINALYS_DIARIZATION_PYTHON=(Resolve-Path speech/.venv-diarization/Scripts/python.exe).Path
$env:HF_HUB_OFFLINE='1'
$env:HF_HUB_DISABLE_TELEMETRY='1'
.venv/Scripts/python -m integration.server
```
Сервер слушает только 127.0.0.1:8765. /health показывает доступность процесса, НЕ готовность моделей. Переменные задаются в окружении процесса, .env автоматически не читается. До загрузки весов Resolve-Path завершится ошибкой — это ожидаемо.

## Этап 4 — запустить интерфейс
В другом терминале:
```powershell
npm.cmd run dev
```
Открыть http://127.0.0.1:3000. Если порт занят прежним demo, остановить его или использовать `npm.cmd run dev -- --port 3001` и открыть 3001. Оба сервера и Ollama должны оставаться запущенными.

## Этап 5 — ручная проверка
1. Ввести название, реальную дату и IANA timezone (например Asia/Almaty). Если дата неизвестна — оставить пустой.
2. Загрузить speech/tests/fixtures/ru.wav, kk.wav или mixed.wav. Записи в репозитории относятся к проверке распознавания; для проверки конкретных поручений дополнительно записать смоделированное совещание с известными задачами.
3. Нажать «Хаттама жасау». POST возвращает jobId; UI опрашивает GET статуса.
4. Проверить preprocessing → stt → diarization → alignment → chunk/extracting/verifying/summarizing → done. Количество частей не означает точный процент.
5. Проверить весь транскрипт, источник поручений, null-значения, исходные формулировки сроков. Confidence — оценка модели, не точность.
6. Скачать DOCX. Саммари — ключевые цитаты; все строки сохраняются, в том числе более пяти реплик. Дата добавляется адаптером из metadata.
7. Проверить повторный запуск, повреждённый WAV, отсутствие модели, файл более 400 МиБ и запись более 30 минут. Нельзя обходить серверную проверку изменением расширения.
8. Проверить около 30 минут с поручением в конце, сопоставить весь текст и источники вручную. Эта реальная проверка пока НЕ выполнена.

## Протокол HTTP
- POST /api/meetings: raw binary body (не multipart), Content-Type application/octet-stream, X-Audio-Format wav|mp3, X-Meeting-Metadata — UTF-8 JSON в base64. Content-Length выставляет браузер для File.
- Ответ 202: {jobId}.
- GET /api/meetings?jobId=...: status, stage, result при completed, error при failed.
- Next.js → Python: POST /jobs и GET /jobs/{jobId} на фиксированном loopback:8765.
- Сначала сервер потоково сохраняет файл с ограничением размера; полная проверка формата/длительности идёт в preprocessing до моделей. Поэтому неверное содержимое может дать 202, затем failed.
- Максимум два принятых задания: одно обрабатывается, второе ждёт/загружается. Дополнительный запрос — 429.

## Жизненный цикл и ограничения прототипа
«Күтуді тоқтату» отменяет запросы браузера, НЕ backend. После получения jobId можно нажать «Күтуді жалғастыру». До получения jobId при обрыве загрузки результат может оказаться недоступен пользователю. Перезагрузка страницы не сохраняет jobId в UI; HTTP-результат можно получить по сохранённому ID.

Результаты хранятся в памяти процесса до 6 часов, максимум 100 записей. После перезапуска исчезают. Аудио удаляется после завершения/ошибки; при аварийном завершении могут остаться jinalys-* в системном temp — владелец машины должен удалить только такие файлы после остановки обработки. Постоянной очереди, server-side отмены и автоматического перезапуска worker нет. Зависшая модель занимает worker до её внутреннего timeout; общий watchdog пока не реализован.

Решение предназначено для доверенного локального использования, не для публикации Python API в сеть. Нет авторизации, многопользовательского разграничения, промышленного retention. Производственный reverse proxy должен разрешать 400 МиБ; загрузка ограничена 120 сек на стороне Python и 150 сек на стороне Next. Модель обрабатывается отдельно и не держит POST открытым.

Контекст между AI-частями — до двух реплик. Поздние отмены/далёкие уточнения глобально не сверяются, семантические дубликаты могут оставаться.

## Автоматические проверки без весов
```powershell
.venv/Scripts/python -m unittest discover -s integration/tests -v
.venv/Scripts/python -m unittest discover -s protocol/tests -v
npm.cmd test
npm.cmd run build
```
Интеграционные success-тесты используют подменённую модель, а валидация WAV — настоящий speech validator. HTTP проверка отсутствующей модели выполнена через Next → Python. Это не доказательство работы реальных моделей.
