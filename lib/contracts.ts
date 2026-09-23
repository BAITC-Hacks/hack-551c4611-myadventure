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
export async function processAudio(file: File, signal: AbortSignal): Promise<MeetingProtocol> {
 const body = new FormData(); body.append('audio',file);
 const response = await fetch('/api/meetings',{method:'POST',body,signal});
 if (!response.ok) { if (response.status===404 || response.status===503) throw new Error('Аудио өңдеу модулі әлі қосылмаған. Демо үлгісін ашуға болады.'); throw new Error('Аудионы өңдеу мүмкін болмады. Қайта көріңіз.'); }
 const parsed = protocolSchema.safeParse(await response.json());
 if (!parsed.success) throw new Error('Сервер жауабы келісілген форматқа сәйкес келмейді.');
 return parsed.data;
}
