import assert from 'node:assert/strict';
import { describe, test } from 'node:test';
import type { LlmConnection } from '@maka/core';
import { generateText, streamText } from 'ai';
import { getAIModel } from '@maka/runtime';

/**
 * Anthropic-compatible relays (one-api/new-api style gateways fronting
 * non-Anthropic models) deviate from the strict Anthropic response schema.
 * The model factory normalizes their responses via a fetch wrapper; these
 * tests pin the two observed deviations:
 *
 * 1. SSE keepalive pings arrive as `event: ping` + `data: {}` without the
 *    `type` discriminator the AI SDK's chunk union validates against —
 *    an unrepaired ping kills the whole stream with AI_TypeValidationError.
 * 2. Non-streaming JSON responses omit `signature` on thinking blocks,
 *    which the SDK's non-streaming response schema requires.
 */

function relayConnection(): LlmConnection {
  return {
    slug: 'relay',
    name: 'relay',
    providerType: 'anthropic-compatible',
    defaultModel: 'relay-model',
    enabled: true,
    createdAt: 0,
    updatedAt: 0,
  };
}

const MODEL_ID = 'relay-model';

function sseResponse(chunks: string): Response {
  return new Response(chunks, {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  });
}

const MESSAGE_START = `event: message_start
data: {"type":"message_start","message":{"id":"msg-1","type":"message","role":"assistant","model":"${MODEL_ID}","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}

`;

const TEXT_START = `event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

`;

const MESSAGE_END = `event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}

`;

function textDelta(text: string): string {
  return `event: content_block_delta\ndata: ${JSON.stringify({
    type: 'content_block_delta',
    index: 0,
    delta: { type: 'text_delta', text },
  })}\n\n`;
}

describe('getAIModel: anthropic-compatible relay normalization', () => {
  test('repairs discriminator-less SSE ping chunks instead of failing the stream', async () => {
    // Raw relay framing: `event: ping` with `data: {}` (real Anthropic sends
    // the type inside the payload).
    const rawPing = 'event: ping\ndata: {}\n\n';
    const stubFetch = (async () =>
      sseResponse(MESSAGE_START + TEXT_START + textDelta('Hello') + rawPing + textDelta(' world') + MESSAGE_END)) as typeof fetch;
    const model = getAIModel({
      connection: relayConnection(),
      apiKey: 'test-key',
      modelId: MODEL_ID,
      fetch: stubFetch,
    });

    const result = streamText({ model, prompt: 'hi' });
    let text = '';
    for await (const delta of result.textStream) text += delta;
    assert.equal(text, 'Hello world');
  });

  test('patches missing thinking signatures on non-streaming JSON responses', async () => {
    const stubFetch = (async () =>
      new Response(
        JSON.stringify({
          id: 'msg-1',
          type: 'message',
          role: 'assistant',
          model: MODEL_ID,
          content: [
            { type: 'thinking', thinking: 'reasoning without a signature' },
            { type: 'text', text: 'done' },
          ],
          stop_reason: 'end_turn',
          stop_sequence: null,
          usage: { input_tokens: 10, output_tokens: 3 },
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      )) as typeof fetch;
    const model = getAIModel({
      connection: relayConnection(),
      apiKey: 'test-key',
      modelId: MODEL_ID,
      fetch: stubFetch,
    });

    const { text } = await generateText({ model, prompt: 'hi' });
    assert.equal(text, 'done');
  });
});
