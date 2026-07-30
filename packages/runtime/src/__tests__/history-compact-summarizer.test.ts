/**
 * Tests for buildLlmHistorySummarizer — the AI-SDK-backed LLM summary that
 * replaces the deterministic excerpt draft when wiring injects it.
 *
 * Run: `npm --workspace @maka/runtime run test`
 */
import { describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { expect } from '../test-helpers.js';
import type { RuntimeEvent, RuntimeEventContent } from '@maka/core/runtime-event';
import type { LlmCallRecord } from '@maka/core/usage-stats/types';
import type { HistoryCompactSummaryInput } from '../ai-sdk-compaction-contract.js';
import {
  buildLlmHistorySummarizer,
  replayPlanItemsToModelMessages,
  type AiSdkGenerateTextLike,
} from '../history-compact-summarizer.js';
import type { RuntimeEventModelReplayItem } from '../model-history.js';
import { buildHistoryCompactCheckpoint } from '../history-compact-checkpoint.js';

const ts = 1_700_000_000_000;
let __seq = 0;
function ev(overrides: Partial<RuntimeEvent> & { content?: RuntimeEventContent }): RuntimeEvent {
  __seq += 1;
  return {
    id: `evt-${__seq}`,
    invocationId: 'inv-1',
    runId: 'run-1',
    sessionId: 'sess-1',
    turnId: 'turn-1',
    ts: ts + __seq,
    partial: false,
    ...overrides,
  } as RuntimeEvent;
}

function inputWith(events: RuntimeEvent[], abortSignal?: AbortSignal): HistoryCompactSummaryInput {
  return {
    sessionId: 'sess-1',
    turnId: 'turn-1',
    source: { foldedRuntimeEvents: events },
    ...(abortSignal ? { abortSignal } : {}),
  };
}

describe('buildLlmHistorySummarizer', () => {
  test('returns the LLM summary and sends the tool-bearing conversation to generateText', async () => {
    const seen: Array<{ instructions: string; messages: unknown[] }> = [];
    const generateText: AiSdkGenerateTextLike = async (opts) => {
      seen.push(opts);
      return { text: '## Goal\n做到 X' };
    };

    const summarize = buildLlmHistorySummarizer({ resolveModel: () => 'fake-model', generateText });

    const events: RuntimeEvent[] = [
      ev({ role: 'user', author: 'user', content: { kind: 'text', text: '读 package.json' } }),
      ev({
        role: 'model',
        author: 'agent',
        content: { kind: 'function_call', id: 'fc1', name: 'read', args: { path: 'package.json' } },
      }),
      ev({
        role: 'tool',
        author: 'tool',
        content: { kind: 'function_response', id: 'fc1', name: 'read', result: { name: 'maka' } },
      }),
      ev({ role: 'model', author: 'agent', content: { kind: 'text', text: '项目名是 maka' } }),
    ];

    const result = await summarize(inputWith(events));

    expect(result).toBe('## Goal\n做到 X');
    expect(seen.length).toBe(1);
    const serialized = JSON.stringify(seen[0]!.messages);
    // summarizer 收到的是模型可见的含 tool 对话，而不是纯文本摘要
    expect(serialized).toContain('package.json');
    expect(serialized).toContain('maka');
  });

  test('inherits the session provider options without imposing a compaction-only output cap', async () => {
    let seen: Parameters<AiSdkGenerateTextLike>[0] | undefined;
    const providerOptions = { openaiCompatible: { reasoningEffort: 'high' } };
    const summarize = buildLlmHistorySummarizer({
      resolveModel: () => 'fake-model',
      providerOptions,
      generateText: async (options) => {
        seen = options;
        return { text: '## Goal\nX' };
      },
    });

    await summarize(
      inputWith([ev({ role: 'user', author: 'user', content: { kind: 'text', text: 'hi' } })]),
    );

    expect(seen?.providerOptions).toBe(providerOptions);
    expect(seen?.maxOutputTokens).toBe(undefined);
  });

  test('attributes provider-reported usage to one history-compaction call', async () => {
    const records: LlmCallRecord[] = [];
    let now = 100;
    const summarize = buildLlmHistorySummarizer({
      resolveModel: () => 'fake-model',
      generateText: async () => ({
        text: '## Goal\nX',
        finishReason: 'stop',
        usage: {
          inputTokens: 7,
          outputTokens: 3,
          totalTokens: 10,
        },
      }),
      telemetry: {
        connectionSlug: 'connection',
        providerId: 'provider',
        modelId: 'model',
        newId: () => 'call-id',
        now: () => {
          now += 10;
          return now;
        },
        recordLlmCall: (record) => {
          records.push(record);
        },
      },
    });

    await summarize(
      inputWith([ev({ role: 'user', author: 'user', content: { kind: 'text', text: 'hi' } })]),
    );

    assert.deepEqual(records, [
      {
        sessionId: 'sess-1',
        turnId: 'turn-1',
        callKind: 'history_compact',
        callId: 'history_compact_turn-1_call-id',
        connectionSlug: 'connection',
        providerId: 'provider',
        modelId: 'model',
        inputTokens: 7,
        outputTokens: 3,
        cacheHitInputTokens: 0,
        cacheMissInputTokens: 7,
        cacheMissInputSource: 'derived',
        cachedInputTokens: 0,
        cacheWriteInputTokens: 0,
        reasoningTokens: 0,
        totalTokens: 10,
        rawFinishReason: 'stop',
        latencyMs: 10,
        status: 'success',
        startedAt: 110,
      },
    ]);
  });

  test('produces schema-valid tool-result messages (toolName + wrapped output) and does not fall back', async () => {
    const seen: Array<{ messages: unknown[] }> = [];
    const generateText: AiSdkGenerateTextLike = async (opts) => {
      seen.push(opts);
      return { text: '## Goal\nX' };
    };
    const summarize = buildLlmHistorySummarizer({ resolveModel: () => 'fake-model', generateText });

    const events: RuntimeEvent[] = [
      ev({ role: 'user', author: 'user', content: { kind: 'text', text: '读 package.json' } }),
      ev({
        role: 'model',
        author: 'agent',
        content: { kind: 'function_call', id: 'fc1', name: 'read', args: { path: 'package.json' } },
      }),
      ev({
        role: 'tool',
        author: 'tool',
        content: { kind: 'function_response', id: 'fc1', name: 'read', result: { name: 'maka' } },
      }),
      ev({ role: 'model', author: 'agent', content: { kind: 'text', text: 'ok' } }),
    ];

    const result = await summarize(inputWith(events));
    expect(result).toBe('## Goal\nX');

    const messages = seen[0]!.messages as Array<{
      role: string;
      content: Array<{ type: string; toolName?: string; output?: unknown }>;
    }>;
    const toolPart = messages.find((m) => m.role === 'tool')!.content[0]!;
    expect(toolPart.type).toBe('tool-result');
    // toolName must be present in AI SDK tool-result content.
    expect(toolPart.toolName).toBe('read');
    // output must be the {type, value} wrapper, not the raw result object
    expect(toolPart.output).toEqual({ type: 'json', value: { name: 'maka' } });
  });

  test('surfaces provider failures so the runtime can report the real compact reason', async () => {
    const generateText: AiSdkGenerateTextLike = async () => {
      throw new Error('model down');
    };
    const summarize = buildLlmHistorySummarizer({ resolveModel: () => 'fake-model', generateText });

    await assert.rejects(
      summarize(
        inputWith([ev({ role: 'user', author: 'user', content: { kind: 'text', text: 'hi' } })]),
      ),
      /provider_error/,
    );
  });

  test('surfaces an exhausted output budget instead of reporting a generic empty summary', async () => {
    const summarize = buildLlmHistorySummarizer({
      resolveModel: () => 'fake-model',
      generateText: async () => ({ text: '', finishReason: 'length' }),
    });

    await assert.rejects(
      summarize(
        inputWith([ev({ role: 'user', author: 'user', content: { kind: 'text', text: 'hi' } })]),
      ),
      /output_length/,
    );
  });

  test('rejects non-empty partial text when the provider exhausted its output budget', async () => {
    const summarize = buildLlmHistorySummarizer({
      resolveModel: () => 'fake-model',
      generateText: async () => ({ text: '## Goal\npartial summary', finishReason: 'length' }),
    });

    await assert.rejects(
      summarize(
        inputWith([ev({ role: 'user', author: 'user', content: { kind: 'text', text: 'hi' } })]),
      ),
      /output_length/,
    );
  });

  test('returns undefined without calling generateText when there are no events to summarize', async () => {
    let called = false;
    const generateText: AiSdkGenerateTextLike = async () => {
      called = true;
      return { text: 'should not reach' };
    };
    const summarize = buildLlmHistorySummarizer({ resolveModel: () => 'fake-model', generateText });

    const result = await summarize(inputWith([]));

    expect(result).toBe(undefined);
    expect(called).toBe(false);
  });

  test('rolling summary sends the prior summary plus only newly folded events', async () => {
    const seen: unknown[] = [];
    const summarize = buildLlmHistorySummarizer({
      resolveModel: () => 'fake-model',
      generateText: async (options) => {
        seen.push(options.messages);
        return { text: 'rolled' };
      },
    });
    const old = ev({
      role: 'user',
      author: 'user',
      content: { kind: 'text', text: 'ALREADY_SUMMARIZED_RAW' },
    });
    const newer = ev({
      role: 'model',
      author: 'agent',
      content: { kind: 'text', text: 'NEWLY_EVICTED_RAW' },
    });
    const previousCheckpoint = buildHistoryCompactCheckpoint({
      sessionId: 'sess-1',
      coveredRuntimeEvents: [old],
      summary: 'PRIOR_SUMMARY',
    });
    const input = inputWith([old, newer]);

    const result = await summarize({
      ...input,
      previousCheckpoint,
      newlyFoldedRuntimeEvents: [newer],
    });

    expect(result).toBe('rolled');
    const serialized = JSON.stringify(seen[0]);
    expect(serialized).toContain('PRIOR_SUMMARY');
    expect(serialized).toContain('NEWLY_EVICTED_RAW');
    expect(serialized.includes('ALREADY_SUMMARIZED_RAW')).toBe(false);
  });
});

describe('replayPlanItemsToModelMessages tool pairing', () => {
  let itemSeq = 0;
  const call = (toolCallId: string, toolName = 'read'): RuntimeEventModelReplayItem => {
    itemSeq += 1;
    return {
      kind: 'tool_call',
      toolCallId,
      toolName,
      input: {},
      eventId: `evt-c-${itemSeq}`,
      ts: ts + itemSeq,
    };
  };
  const result = (
    toolCallId: string,
    toolName = 'read',
    output: unknown = 'ok',
  ): RuntimeEventModelReplayItem => {
    itemSeq += 1;
    return {
      kind: 'tool_result',
      toolCallId,
      toolName,
      output,
      isError: false,
      eventId: `evt-r-${itemSeq}`,
      ts: ts + itemSeq,
    };
  };

  type ToolCallish = { type: string; toolCallId?: string };
  const parts = (message: { content: unknown }): ToolCallish[] =>
    Array.isArray(message.content) ? (message.content as ToolCallish[]) : [];

  test('coalesces parallel calls into one assistant message answered by one tool message', () => {
    const messages = replayPlanItemsToModelMessages([
      call('a'),
      call('b'),
      result('a'),
      result('b'),
    ]);

    expect(messages.map((m) => m.role)).toEqual(['assistant', 'tool']);
    expect(parts(messages[0]!).map((p) => p.toolCallId)).toEqual(['a', 'b']);
    expect(parts(messages[1]!).map((p) => p.toolCallId)).toEqual(['a', 'b']);
  });

  test('pulls a late-recorded result forward into its own step', () => {
    // The ledger recorded step 2's calls before step 1's slow Read result.
    const messages = replayPlanItemsToModelMessages([
      call('a'),
      call('b'),
      result('a'),
      call('c'),
      call('d'),
      result('b'),
      result('c'),
      result('d'),
    ]);

    expect(messages.map((m) => m.role)).toEqual(['assistant', 'tool', 'assistant', 'tool']);
    expect(parts(messages[1]!).map((p) => p.toolCallId)).toEqual(['a', 'b']);
    expect(parts(messages[3]!).map((p) => p.toolCallId)).toEqual(['c', 'd']);
  });

  test('synthesizes an error result for a call whose result was never recorded', () => {
    const messages = replayPlanItemsToModelMessages([call('a'), call('b'), result('a')]);

    expect(messages.map((m) => m.role)).toEqual(['assistant', 'tool']);
    const toolParts = parts(messages[1]!) as Array<{
      toolCallId: string;
      output?: { type: string; value: unknown };
    }>;
    expect(toolParts.map((p) => p.toolCallId)).toEqual(['a', 'b']);
    expect(toolParts[1]!.output?.type).toBe('error-text');
    expect(String(toolParts[1]!.output?.value)).toContain('unavailable');
  });

  test('drops orphan results whose call never replayed', () => {
    const messages = replayPlanItemsToModelMessages([
      call('a'),
      result('a'),
      result('ghost'),
    ]);

    expect(messages.map((m) => m.role)).toEqual(['assistant', 'tool']);
    expect(parts(messages[1]!).map((p) => p.toolCallId)).toEqual(['a']);
  });
});
