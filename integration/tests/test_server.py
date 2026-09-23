import base64
import http.client
import io
import json
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
from integration import server
from speech import AudioPreparationError, prepareAudio

class IntegrationTests(unittest.TestCase):
    def test_metadata_does_not_invent_date(self):
        self.assertNotIn('meetingDate', server.metadata({'title':'Test','meetingDate':''}))
        with self.assertRaises(ValueError): server.metadata({'meetingDate':'2026-02-30'})
        with self.assertRaises(ValueError): server.metadata({'timeZone':'Invented/Zone'})

    def test_real_audio_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.wav'
            p.write_bytes(b'not audio')
            with self.assertRaises(AudioPreparationError): prepareAudio(p)
            with wave.open(str(p),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(1);wav.setframerate(1);wav.writeframes(b'\x80'*1801)
            with self.assertRaises(AudioPreparationError) as caught: prepareAudio(p)
            self.assertEqual(caught.exception.code,'DURATION_EXCEEDED')

    def test_pipeline_uses_real_stage_boundaries_and_date(self):
        from types import SimpleNamespace
        aligned=SimpleNamespace(result={'segments':[{'id':'s1'}]},diarization_mode='LOCAL',warnings=['review'])
        progress=[]
        def protocol(transcript, meta, on_progress):
            self.assertEqual(transcript,[{'id':'s1'}]);on_progress('chunk:1/2');on_progress('done');return {'summary':'line1\nline2'}
        with patch('speech.prepareAudio',return_value=SimpleNamespace(duration_seconds=1)),patch('speech.transcribeAudio',return_value=[]),patch('speech.diarizeAudio',return_value={}),patch('speech.alignTranscript',return_value=aligned),patch('protocol.generate_meeting_protocol',side_effect=protocol):
            result,warnings=server.run_pipeline('unused',{'meetingDate':'2026-09-23'},progress.append)
        self.assertEqual(result['date'],'2026-09-23');self.assertEqual(result['summary'],'line1\nline2')
        self.assertEqual(progress,['preprocessing','stt','diarization','alignment','chunk:1/2'])
        self.assertEqual(warnings,['review'])

    def test_http_jobs_return_early_poll_result_and_failure(self):
        server.jobs.clear()
        httpd=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        threading.Thread(target=httpd.serve_forever,daemon=True).start()
        threading.Thread(target=server.work,daemon=True).start()
        def request(method,path,data=None,headers=None):
            c=http.client.HTTPConnection('127.0.0.1',httpd.server_port,timeout=3)
            c.request(method,path,body=data,headers=headers or {});r=c.getresponse();value=json.loads(r.read());status=r.status;c.close();return status,value
        def fake(path,meta,progress):
            prepareAudio(path);progress('extracting');return {'title':meta['title']},[]
        wav=io.BytesIO()
        with wave.open(wav,'wb') as f: f.setnchannels(1);f.setsampwidth(1);f.setframerate(8000);f.writeframes(b'\x80'*800)
        try:
            with patch.object(server,'run_pipeline',side_effect=fake):
                for payload,expected in [(wav.getvalue(),'completed'),(b'bad','failed'),(wav.getvalue(),'completed')]:
                    status,value=request('POST','/jobs',payload,{'X-Audio-Format':'wav','X-Meeting-Metadata':base64.b64encode(b'{"title":"Test"}').decode()})
                    self.assertEqual(status,202)
                    for _ in range(100):
                        status,job=request('GET','/jobs/'+value['jobId'])
                        if job['status'] in ('completed','failed'):break
                        time.sleep(.01)
                    self.assertEqual(job['status'],expected)
                    if expected=='completed':self.assertEqual(job['stage'],'done')
            status,_=request('POST','/jobs',b'x',{'Content-Length':str(server.MAX_BYTES+1),'X-Audio-Format':'wav'})
            self.assertEqual(status,413)
        finally:httpd.shutdown();httpd.server_close()

if __name__=='__main__':unittest.main()

