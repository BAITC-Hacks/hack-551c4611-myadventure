import { z } from 'zod';
export const segmentSchema = z.object({id:z.string(),speakerId:z.string(),speakerName:z.string().optional(),start:z.number().nonnegative(),end:z.number().nonnegative(),text:z.string(),language:z.enum(['ru','kk','mixed']).optional()});
export const actionSchema = z.object({id:z.string(),task:z.string(),assignee:z.string().nullable(),deadlineText:z.string().nullable(),deadline:z.string().nullable(),assignedBy:z.string().nullable().optional(),sourceSegmentIds:z.array(z.string()),confidence:z.number().min(0).max(1),needsReview:z.boolean()});
export const protocolSchema = z.object({title:z.string(),date:z.string().optional(),transcript:z.array(segmentSchema),summary:z.string(),topics:z.array(z.object({title:z.string(),summary:z.string()})),actionItems:z.array(actionSchema)});
export type MeetingProtocol = z.infer<typeof protocolSchema>;
export const PROCESSING_TIMEOUT_MS = 60 * 60 * 1000;
export const MAX_AUDIO_BYTES = 400 * 1024 * 1024;
export function validateAudio(file: {name:string;size:number}): string | null {
 if (!/\.(mp3|wav)$/i.test(file.name)) return 'MP3 немесе WAV файлын таңдаңыз.';
 if (!file.size) return 'Файл бос. Басқа аудионы таңдаңыз.';
 if (file.size > MAX_AUDIO_BYTES) return 'Файл көлемі 400 МБ-тан аспауы керек.';
 return null;
}

export type MeetingMetadata = {title?:string;meetingDate?:string;timeZone?:string};
export type JobProgress = {jobId:string;status:string;stage:string};
const jobSchema=z.object({jobId:z.string().uuid(),status:z.enum(['uploading','queued','running','completed','failed']),stage:z.string(),result:protocolSchema.optional(),warnings:z.array(z.string()).optional(),error:z.object({code:z.string(),message:z.string()}).optional()});
async function responseJson(response:Response) {
 const data=await response.json();
 if(!response.ok)throw new Error(data.error?.message??'Сервер қатесі. Кейін қайта көріңіз.');
 return data;
}
export async function waitForJob(jobId:string, signal:AbortSignal, onProgress:(p:JobProgress)=>void):Promise<MeetingProtocol> {
 while(!signal.aborted){
  const response=await fetch('/api/meetings?jobId='+encodeURIComponent(jobId),{signal,cache:'no-store'});
  const job=jobSchema.parse(await responseJson(response));onProgress(job);
  if(job.status==='failed')throw new Error(job.error?.code+': '+job.error?.message);
  if(job.status==='completed'){if(!job.result)throw new Error('Нәтиже жоқ.');return job.result;}
  await new Promise<void>((resolve,reject)=>{const abort=()=>{clearTimeout(timer);reject(new DOMException('Aborted','AbortError'));};const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve();},1500);signal.addEventListener('abort',abort,{once:true});if(signal.aborted)abort();});
 }
 throw new DOMException('Aborted','AbortError');
}
export async function processAudio(file:File,signal:AbortSignal,metadata:MeetingMetadata={},onProgress:(p:JobProgress)=>void=()=>{}):Promise<MeetingProtocol>{
 const bytes=new TextEncoder().encode(JSON.stringify(metadata));const encoded=btoa(Array.from(bytes,b=>String.fromCharCode(b)).join(''));
 const response=await fetch('/api/meetings',{method:'POST',body:file,signal,headers:{'content-type':'application/octet-stream','x-audio-format':file.name.split('.').pop()!.toLowerCase(),'x-meeting-metadata':encoded}});
 const {jobId}=z.object({jobId:z.string().uuid()}).parse(await responseJson(response));onProgress({jobId,status:'queued',stage:'queued'});
 return waitForJob(jobId,signal,onProgress);
}
