import type { SpeechPipelineResult, TranscriptSegment } from "../index.js";

const segment: TranscriptSegment = {
  id: "segment-1", speakerId: "speaker-1", start: 0, end: 1, text: "Сәлем",
};

// Both consumers can import only types, without loading speech implementation.
export function protocolInput(segments: TranscriptSegment[]): string {
  return segments.map((item) => item.text).join("\n");
}
export const frontendResult: SpeechPipelineResult = {
  durationSeconds: 1, detectedSpeakers: 1, segments: [segment],
};

// @ts-expect-error Unsupported language must not enter the public contract.
export const wrongLanguage: TranscriptSegment = { ...segment, language: "en" };
// @ts-expect-error Timestamps must be numbers.
export const wrongTime: TranscriptSegment = { ...segment, start: "0" };
// @ts-expect-error Required speaker identity cannot be omitted.
export const missingSpeaker: TranscriptSegment = { id: "1", start: 0, end: 1, text: "" };
// @ts-expect-error Result must contain segments.
export const missingSegments: SpeechPipelineResult = { durationSeconds: 0, detectedSpeakers: 0 };
