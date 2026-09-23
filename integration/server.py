"""Local bounded job server. Run from repository root: python -m integration.server."""
import datetime as dt
import json
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MAX_BYTES = 400 * 1024 * 1024
TTL = 6 * 3600
jobs = {}
lock = threading.Lock()
pending = queue.Queue(maxsize=2)


def metadata(value):
    if not isinstance(value, dict) or set(value) - {'title', 'meetingDate', 'timeZone'}:
        raise ValueError('Invalid metadata fields')
    result = {}
    for key, limit in [('title', 200), ('meetingDate', 10), ('timeZone', 100)]:
        v = value.get(key)
        if v in (None, ''):
            continue
        if not isinstance(v, str) or len(v) > limit:
            raise ValueError('Invalid metadata value')
        result[key] = v.strip()
    if 'meetingDate' in result:
        date = dt.date.fromisoformat(result['meetingDate'])
        if date.isoformat() != result['meetingDate']:
            raise ValueError('meetingDate must be YYYY-MM-DD')
    if 'timeZone' in result:
        try:
            ZoneInfo(result['timeZone'])
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError('Unknown IANA time zone') from None
    return result


def run_pipeline(path, meta, progress):
    # Use participant 1's public stages so reported stages reflect actual execution.
    from speech import prepareAudio, transcribeAudio, diarizeAudio, alignTranscript
    from protocol import generate_meeting_protocol
    progress('preprocessing')
    prepared = prepareAudio(path, max_bytes=MAX_BYTES, max_duration_seconds=1800)
    progress('stt')
    transcript = transcribeAudio(prepared)
    progress('diarization')
    speakers = diarizeAudio(prepared, mode='local')
    progress('alignment')
    aligned = alignTranscript(transcript, speakers, duration_seconds=prepared.duration_seconds)
    if aligned.diarization_mode != 'LOCAL':
        raise ValueError('Real jobs require local diarization')
    if not aligned.result['segments']:
        raise ValueError('No speech detected')
    # done is emitted only after the final result is stored, not on chunk completion.
    result = generate_meeting_protocol(aligned.result['segments'], meta,
        on_progress=lambda stage: progress(stage) if stage != 'done' else None)
    if meta.get('meetingDate'):
        result['date'] = meta['meetingDate']
    return result, list(aligned.warnings)


def update(job_id, **fields):
    with lock:
        jobs[job_id].update(fields, updatedAt=time.time())


def work():
    while True:
        job_id, path, meta = pending.get()
        try:
            update(job_id, status='running', stage='preprocessing')
            result, warnings = run_pipeline(path, meta, lambda stage: update(job_id, stage=stage))
            update(job_id, status='completed', stage='done', result=result, warnings=warnings)
        except Exception as error:
            code = getattr(error, 'code', 'PROCESSING_FAILED')
            # Controlled messages only; no filesystem paths, transcripts or traceback.
            update(job_id, status='failed', error={'code':code,
                'message':'Local processing failed. Check models, dependencies and audio format.'})
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)
            pending.task_done()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/health':
            return self.reply(200, {'status':'ok', 'modelsVerified':False})
        job_id = self.path.removeprefix('/jobs/')
        if not self.path.startswith('/jobs/'):
            return self.reply(404, {'error':{'code':'NOT_FOUND'}})
        with lock:
            found = jobs.get(job_id)
            value = dict(found) if found and time.time()-found['updatedAt'] < TTL else None
        return self.reply(200 if value else 404, value or {'error':{'code':'JOB_NOT_FOUND','message':'Job expired or server restarted.'}})

    def do_POST(self):
        if self.path != '/jobs':
            return self.reply(404, {'error':{'code':'NOT_FOUND'}})
        # Direct browser cross-origin writes are not part of this local API.
        if self.headers.get('Origin'):
            return self.reply(403, {'error':{'code':'ORIGIN_NOT_ALLOWED'}})
        directory = None
        job_id = None
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_BYTES:
                return self.reply(413 if size > MAX_BYTES else 400, {'error':{'code':'INVALID_SIZE','message':'Expected nonempty audio up to 400 MiB.'}})
            extension = self.headers.get('X-Audio-Format', '').lower()
            if extension not in ('wav', 'mp3'):
                return self.reply(415, {'error':{'code':'UNSUPPORTED_FORMAT'}})
            import base64
            encoded = self.headers.get('X-Meeting-Metadata', 'e30=')
            if len(encoded) > 2048:
                raise ValueError('Metadata too large')
            meta = metadata(json.loads(base64.b64decode(encoded, validate=True).decode('utf-8')))
            with lock:
                for old in [k for k,v in jobs.items() if v['status'] in ('completed','failed') and time.time()-v['updatedAt'] > TTL]:
                    jobs.pop(old)
                if len(jobs) >= 100 or sum(v['status'] in ('uploading','queued','running') for v in jobs.values()) >= 2:
                    return self.reply(429, {'error':{'code':'BUSY','message':'Two jobs already active. Try again later.'}})
                job_id = str(uuid.uuid4())
                jobs[job_id] = {'jobId':job_id,'status':'uploading','stage':'uploading','updatedAt':time.time()}
            directory = Path(tempfile.mkdtemp(prefix='jinalys-'))
            path = directory / ('audio.'+extension)
            self.connection.settimeout(120)
            with path.open('wb') as out:
                remaining = size
                while remaining:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        raise ValueError('Incomplete upload')
                    out.write(chunk)
                    remaining -= len(chunk)
            update(job_id, status='queued', stage='queued')
            pending.put_nowait((job_id,path,meta))
            directory = None
            accepted_id = job_id
            job_id = None
            self.reply(202, {'jobId':accepted_id})
        except (ValueError, UnicodeError, OSError, queue.Full):
            if directory:
                shutil.rmtree(directory, ignore_errors=True)
            if job_id:
                with lock:
                    jobs.pop(job_id, None)
            self.reply(400, {'error':{'code':'INVALID_UPLOAD','message':'Invalid metadata or incomplete upload.'}})


def main():
    threading.Thread(target=work, daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', 8765), Handler)
    print('JINALYS job server: http://127.0.0.1:8765', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()

