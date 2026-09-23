# Speech pipeline contract

Shared TypeScript types and JSON Schema, independent of STT models and audio
dependencies. Import `TranscriptSegment` and `SpeechPipelineResult` using
`import type` from `index.ts`. Protocol AI can consume `TranscriptSegment[]`.
Non-TypeScript consumers can use `schema.json`; an individual segment is defined
at `#/definitions/TranscriptSegment`.

Times are expressed in seconds. Optional properties are omitted when absent.
Language values are exactly `ru`, `kk`, and `mixed`.

The speech implementation should pass its typed result through
`toSpeechPipelineResult` before returning it to consumers. This projection copies
only public fields, including within segments, without modifying the input.
It does not validate untrusted data: validate such data against the JSON Schema.
The schema rejects additional fields. No model-specific metadata belongs in the
external response. Numeric fields retain the requested `number` contract without
additional range or integer restrictions.

## Checks

From this directory, with Node.js 24 and npm:

```sh
npm ci
npm test
npm run lint
npm run typecheck
```

Tests compile the schema and check accepted/rejected payloads and adapter output.
ESLint checks JavaScript tooling/tests; strict TypeScript checks the contract and
consumer type fixtures, including expected compile errors. No runtime dependencies
or separate build are required. No audio or transcripts are sent to external services.
