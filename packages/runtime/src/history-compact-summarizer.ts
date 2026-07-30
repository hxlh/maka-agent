import {
  rawFinishReasonString,
  type ModelMessage,
  type ToolCallPart,
  type ToolResultPart,
} from './model-protocol.js';
import { buildRuntimeEventModelReplayPlan } from './model-history.js';
import { toolResultOutput } from './tool-result-output.js';
import type {
  HistoryCompactSummaryInput,
  LlmTelemetryRecorder,
} from './ai-sdk-compaction-contract.js';
import { HistoryCompactSummarizerError } from './history-compact-error.js';
import { normalizeAiSdkUsage, type AiSdkUsageLike } from './model-adapter.js';
import {
  ProviderRequestTracker,
  type ProviderGenerateResult,
  type ProviderRequestTrackerInput,
} from './provider-request-telemetry.js';
import { llmCallUsageFields } from './telemetry/llm-call-usage.js';

export { HistoryCompactSummarizerError } from './history-compact-error.js';

export interface AiSdkGenerateTextOptions {
  model: unknown;
  instructions: string;
  messages: ModelMessage[];
  providerOptions?: Record<string, unknown>;
  maxOutputTokens?: number;
  abortSignal?: AbortSignal;
}

export type AiSdkGenerateTextLike = (
  options: AiSdkGenerateTextOptions,
) => Promise<{ text: string; finishReason?: unknown; usage?: AiSdkUsageLike }>;

interface ProviderMiddlewareGenerateInput {
  doGenerate: () => PromiseLike<ProviderGenerateResult>;
  params: Record<string, unknown> & { abortSignal?: AbortSignal };
  model: { provider: string; modelId: string };
}

export interface BuildLlmHistorySummarizerOptions {
  /** Resolve the AI SDK model used for summarization. Reuses the session model. */
  resolveModel: () => unknown;
  /** Session provider settings, including the selected reasoning level. */
  providerOptions?: Record<string, unknown>;
  /** Injectable `generateText` for tests; defaults to the real AI SDK export. */
  generateText?: AiSdkGenerateTextLike;
  /** Physical provider-call capture and attempt tracking for generated summaries. */
  providerRequestTracking?: Omit<ProviderRequestTrackerInput, 'traceId' | 'turnId'>;
  /** Usage attribution for the auxiliary history-compaction call. */
  telemetry?: {
    connectionSlug: string;
    providerId: string;
    modelId: string;
    newId: () => string;
    now: () => number;
    recordLlmCall: LlmTelemetryRecorder;
  };
}

// Conversation-summarization prompt (sectioned, modelled on pi/opencode):
// asks for a checkpoint another LLM can continue from. Tool calls and their
// results are part of the conversation sent to the summarizer, because the
// folded events are projected with the same policy the model would see them.
const SUMMARIZATION_SYSTEM_PROMPT = [
  'You are a context summarization assistant.',
  'Read the conversation between a user and an AI assistant, then produce a structured summary another LLM will use to continue the same task.',
  'Do NOT continue the conversation. Do NOT answer questions in it. ONLY output the structured summary.',
  '',
  'Use this exact format:',
  '',
  '## Goal',
  '[What the user is trying to accomplish]',
  '',
  '## Progress',
  '### Done',
  '- [Completed work and changes]',
  '### In Progress',
  '- [Current work]',
  '',
  '## Key Decisions',
  '- **[Decision]**: [Brief rationale]',
  '',
  '## Next Steps',
  '1. [Ordered list of what should happen next]',
  '',
  '## Critical Context',
  '- [Files, commands/results, errors, anything needed to continue; or "(none)"]',
  '',
  'Keep each section concise. Preserve exact file paths, function names, commands, and error messages.',
].join('\n');

// Closing instruction appended as the final user message, so the summarization
// request never ends on an assistant turn (see buildLlmHistorySummarizer).
const SUMMARY_REQUEST_TEXT =
  'Write the structured summary of the conversation above now, using the exact format requested.';

export function buildLlmHistorySummarizer(options: BuildLlmHistorySummarizerOptions) {
  return async (input: HistoryCompactSummaryInput): Promise<string | undefined> => {
    const newlyFoldedRuntimeEvents =
      input.newlyFoldedRuntimeEvents ?? input.source.foldedRuntimeEvents;
    if (newlyFoldedRuntimeEvents.length === 0) return input.previousCheckpoint?.summary;
    try {
      const plan = buildRuntimeEventModelReplayPlan(newlyFoldedRuntimeEvents);
      const messages = replayPlanItemsToModelMessages(plan.items);
      if (input.previousCheckpoint) {
        messages.unshift({
          role: 'user',
          content: [
            {
              type: 'text',
              text: `Previous continuation summary:\n${input.previousCheckpoint.summary}\n\nUpdate it using the newer conversation events that follow.`,
            },
          ],
        });
      }
      // End on a user instruction. A replay typically ends with the assistant's
      // last answer, which asks the provider to CONTINUE an assistant turn —
      // Anthropic treats that as prefill, but stricter relay endpoints (Kimi
      // k3) return an empty response and the gateway surfaces it as a 502,
      // failing every compaction. Merging into a trailing user message avoids
      // consecutive user messages for relays that validate strict alternation.
      const lastMessage = messages[messages.length - 1];
      if (lastMessage?.role === 'user' && Array.isArray(lastMessage.content)) {
        lastMessage.content.push({ type: 'text', text: SUMMARY_REQUEST_TEXT });
      } else {
        messages.push({ role: 'user', content: [{ type: 'text', text: SUMMARY_REQUEST_TEXT }] });
      }
      const providerRequestTracker = options.providerRequestTracking
        ? new ProviderRequestTracker({
            ...options.providerRequestTracking,
            traceId: options.providerRequestTracking.newId(),
            turnId: input.turnId,
          })
        : undefined;
      const ai =
        options.generateText && !providerRequestTracker ? undefined : await loadAiSdkTextModule();
      const generateText = options.generateText ?? ai!.generateText;
      const model = providerRequestTracker
        ? ai!.wrapLanguageModel({
            model: options.resolveModel(),
            middleware: {
              wrapGenerate: async ({
                doGenerate,
                params,
                model: providerModel,
              }: ProviderMiddlewareGenerateInput) =>
                await providerRequestTracker.trackGenerate({
                  providerId: providerModel.provider,
                  modelId: providerModel.modelId,
                  params,
                  abortSignal: input.abortSignal,
                  doGenerate,
                }),
            },
          })
        : options.resolveModel();
      const startedAt = options.telemetry?.now();
      const result = await generateText({
        model,
        instructions: SUMMARIZATION_SYSTEM_PROMPT,
        messages,
        ...(options.providerOptions !== undefined
          ? { providerOptions: options.providerOptions }
          : {}),
        ...(input.abortSignal ? { abortSignal: input.abortSignal } : {}),
      });
      if (options.telemetry && startedAt !== undefined) {
        recordHistoryCompactCall(options.telemetry, input, startedAt, result);
      }
      if (rawFinishReasonString(result.finishReason) === 'length') {
        throw new HistoryCompactSummarizerError('output_length');
      }
      return result.text;
    } catch (error) {
      if (error instanceof HistoryCompactSummarizerError) throw error;
      throw new HistoryCompactSummarizerError('provider_error', { cause: error });
    }
  };
}

function recordHistoryCompactCall(
  telemetry: NonNullable<BuildLlmHistorySummarizerOptions['telemetry']>,
  input: HistoryCompactSummaryInput,
  startedAt: number,
  result: Awaited<ReturnType<AiSdkGenerateTextLike>>,
): void {
  const usage = normalizeAiSdkUsage(result.usage, { rawFinishReason: result.finishReason });
  if (!usage) return;
  const completedAt = telemetry.now();
  try {
    telemetry.recordLlmCall({
      sessionId: input.sessionId,
      turnId: input.turnId,
      callKind: 'history_compact',
      callId: `history_compact_${input.turnId}_${telemetry.newId()}`,
      connectionSlug: telemetry.connectionSlug,
      providerId: telemetry.providerId,
      modelId: telemetry.modelId,
      ...llmCallUsageFields(usage),
      latencyMs: Math.max(0, completedAt - startedAt),
      status: 'success',
      startedAt,
    });
  } catch {
    // Usage telemetry is diagnostic. The summary remains authoritative.
  }
}

interface AiSdkTextModule {
  generateText: AiSdkGenerateTextLike;
  wrapLanguageModel(input: Record<string, unknown>): unknown;
}

async function loadAiSdkTextModule(): Promise<AiSdkTextModule> {
  const ai = await import('ai').catch((err) => {
    throw new Error(
      `Failed to load 'ai' package for history summarization. Run \`npm install ai\`. Inner: ${(err as Error).message}`,
    );
  });
  return ai as unknown as AiSdkTextModule;
}

type ReplayPlanItems = ReturnType<typeof buildRuntimeEventModelReplayPlan>['items'];

export function replayPlanItemsToModelMessages(items: ReplayPlanItems): ModelMessage[] {
  const out: ModelMessage[] = [];
  // Parallel tool calls in one model step arrive as consecutive tool_call
  // items. They must stay in ONE assistant message (and their results in ONE
  // tool message): strict providers (Kimi, OpenAI) reject an assistant
  // message whose tool_calls are not immediately followed by a tool message
  // answering every id, so one-message-per-call replays fail validation.
  let pendingToolCalls: ToolCallPart[] | undefined;
  let pendingToolResults: ToolResultPart[] | undefined;
  const flushToolCalls = (): void => {
    if (pendingToolCalls?.length) out.push({ role: 'assistant', content: pendingToolCalls });
    pendingToolCalls = undefined;
  };
  const flushToolResults = (): void => {
    if (pendingToolResults?.length) out.push({ role: 'tool', content: pendingToolResults });
    pendingToolResults = undefined;
  };
  for (const item of items) {
    if (item.kind === 'text') {
      flushToolCalls();
      flushToolResults();
      // Merge into a directly preceding same-role text-only message: strict
      // relays reject consecutive same-role messages, and a text run has no
      // protocol reason to split.
      const textPart = { type: 'text' as const, text: item.content };
      const previous = out[out.length - 1];
      if (item.role === 'user') {
        if (previous?.role === 'user' && Array.isArray(previous.content)) {
          previous.content.push(textPart);
        } else {
          out.push({ role: 'user', content: [textPart] });
        }
      } else {
        if (
          previous?.role === 'assistant' &&
          Array.isArray(previous.content) &&
          previous.content.every((part) => part.type === 'text')
        ) {
          previous.content.push(textPart);
        } else {
          out.push({ role: 'assistant', content: [textPart] });
        }
      }
    } else if (item.kind === 'tool_call') {
      flushToolResults();
      pendingToolCalls ??= [];
      pendingToolCalls.push({
        type: 'tool-call',
        toolCallId: item.toolCallId,
        toolName: item.toolName,
        input: item.input,
      });
    } else if (item.kind === 'tool_result') {
      flushToolCalls();
      pendingToolResults ??= [];
      pendingToolResults.push({
        type: 'tool-result',
        toolCallId: item.toolCallId,
        toolName: item.toolName,
        output: toolResultOutput(item.output, item.isError),
      });
    }
    // thinking entries are intentionally skipped for summarization
  }
  flushToolCalls();
  flushToolResults();
  repairToolMessagePairing(out);
  return out;
}

/**
 * The event ledger can record a slow tool's result AFTER the next model step's
 * calls (results append on completion; store write order is not conversation
 * order). Strict providers require every assistant tool_call to be answered by
 * the IMMEDIATELY following tool message, so pull late results forward into
 * their own step's tool message. A call whose result was never recorded
 * (interrupted session) gets a synthesized error result, and orphan results
 * whose call never replayed are dropped — either deviation fails provider
 * validation and would block compaction (and session recap) entirely.
 */
function repairToolMessagePairing(messages: ModelMessage[]): void {
  const toolCallsOf = (message: ModelMessage | undefined): ToolCallPart[] =>
    message?.role === 'assistant' && Array.isArray(message.content)
      ? message.content.filter((part): part is ToolCallPart => part.type === 'tool-call')
      : [];

  for (let i = 0; i < messages.length; i++) {
    const calls = toolCallsOf(messages[i]);
    if (calls.length === 0) continue;
    let answeringMessage = messages[i + 1];
    if (answeringMessage?.role !== 'tool') {
      answeringMessage = { role: 'tool', content: [] };
      messages.splice(i + 1, 0, answeringMessage);
    }
    const answered = new Set(
      answeringMessage.content
        .filter((part): part is ToolResultPart => part.type === 'tool-result')
        .map((part) => part.toolCallId),
    );
    for (const call of calls) {
      if (answered.has(call.toolCallId)) continue;
      let moved = false;
      for (let j = i + 2; j < messages.length && !moved; j++) {
        const candidate = messages[j];
        if (candidate.role !== 'tool') continue;
        const index = candidate.content.findIndex(
          (part) => part.type === 'tool-result' && part.toolCallId === call.toolCallId,
        );
        if (index < 0) continue;
        const [part] = candidate.content.splice(index, 1);
        answeringMessage.content.push(part as ToolResultPart);
        moved = true;
      }
      if (!moved) {
        answeringMessage.content.push({
          type: 'tool-result',
          toolCallId: call.toolCallId,
          toolName: call.toolName,
          output: {
            type: 'error-text',
            value:
              '[tool result unavailable — the session ended before this result was recorded]',
          },
        });
      }
    }
  }

  // Drop orphan results (their call never replayed) and now-empty tool messages.
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role !== 'tool') continue;
    let k = i - 1;
    while (k >= 0 && messages[k].role === 'tool') k--;
    const callIds = new Set(toolCallsOf(messages[k]).map((call) => call.toolCallId));
    message.content = message.content.filter(
      (part) => part.type !== 'tool-result' || callIds.has(part.toolCallId),
    );
    if (message.content.length === 0) messages.splice(i, 1);
  }
}
