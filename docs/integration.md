# Текущий интеграционный контракт

Ранее описанный multipart + единственный финальный ответ заменён job API. Актуальный запуск и контракт: [LOCAL_RUN.md](LOCAL_RUN.md).

POST /api/meetings принимает raw MP3/WAV, X-Audio-Format, X-Meeting-Metadata (base64 UTF-8 JSON), возвращает 202 {jobId}. GET /api/meetings?jobId=... возвращает status/stage и result при завершении.

Python integration/server.py вызывает публичные стадии speech, затем generate_meeting_protocol. Исходные модули не изменены. metadata: title, meetingDate, timeZone; неизвестная дата не подставляется. date для UI/DOCX добавляет адаптер.

Интеграционный код реализован и тестируется без весов. Реальный RU/KK/mixed/30-minute end-to-end требует установленной конфигурации моделей и ещё не подтверждён.
