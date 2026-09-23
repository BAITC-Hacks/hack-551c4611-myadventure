/** Public, model-independent speech output. Times are seconds. */
export interface TranscriptSegment {
  id: string;
  speakerId: string;
  speakerName?: string;
  start: number;
  end: number;
  text: string;
  language?: "ru" | "kk" | "mixed";
}

export interface SpeechPipelineResult {
  durationSeconds: number;
  detectedSpeakers: number;
  segments: TranscriptSegment[];
}

/** Project internal output onto the public contract, stripping extra fields. */
export function toSpeechPipelineResult(
  result: SpeechPipelineResult,
): SpeechPipelineResult {
  return {
    durationSeconds: result.durationSeconds,
    detectedSpeakers: result.detectedSpeakers,
    segments: result.segments.map((segment) => ({
      id: segment.id,
      speakerId: segment.speakerId,
      ...(segment.speakerName !== undefined
        ? { speakerName: segment.speakerName }
        : {}),
      start: segment.start,
      end: segment.end,
      text: segment.text,
      ...(segment.language !== undefined ? { language: segment.language } : {}),
    })),
  };
}
