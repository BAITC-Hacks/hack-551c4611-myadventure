import test from 'node:test';
import assert from 'node:assert/strict';
import { protocolSchema,validateAudio,MAX_AUDIO_BYTES } from '../lib/contracts.ts';
import { demoProtocol } from '../lib/demo.ts';
test('demo respects shared contract and every action references real evidence',()=>{assert.equal(protocolSchema.safeParse(demoProtocol).success,true);const ids=new Set(demoProtocol.transcript.map(s=>s.id));for(const a of demoProtocol.actionItems)for(const id of a.sourceSegmentIds)assert.ok(ids.has(id));});
test('rejects unsupported, empty and oversized audio',()=>{assert.ok(validateAudio({name:'test.exe',size:200}));assert.ok(validateAudio({name:'test.wav',size:0}));assert.ok(validateAudio({name:'test.mp3',size:MAX_AUDIO_BYTES+1}));assert.equal(validateAudio({name:'test.WAV',size:200}),null);});
test('missing assignee and deadline remain explicitly unknown',()=>{const a=protocolSchema.parse(demoProtocol).actionItems[2];assert.equal(a.assignee,null);assert.equal(a.deadline,null);assert.equal(a.needsReview,true);});
test('invalid confidence and malformed response are rejected',()=>{assert.equal(protocolSchema.safeParse({}).success,false);assert.equal(protocolSchema.safeParse({...demoProtocol,actionItems:[{...demoProtocol.actionItems[0],confidence:2}]}).success,false);});

import { createProtocolBlob } from '../lib/export.ts';
test('DOCX export produces a nonempty ZIP-based Word document',async()=>{const blob=await createProtocolBlob(demoProtocol,true);const bytes=new Uint8Array(await blob.arrayBuffer());assert.ok(bytes.length>1000);assert.equal(bytes[0],0x50);assert.equal(bytes[1],0x4b);});
