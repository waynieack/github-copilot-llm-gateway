import * as vscode from 'vscode';
import { randomBytes } from 'node:crypto';
import {
  OpenAIChatCompletionRequest,
  OpenAIModelsResponse,
  GatewayConfig
} from './types';

/**
 * Trim trailing slashes and a trailing `/v1` (or `/openai/v1`) segment so the
 * client can safely append `/v1/models` / `/v1/chat/completions` regardless of
 * how the user typed their Server URL in settings.
 *
 * Users often paste the exact URL they tested with curl (which already ends in
 * `/v1`) or leave a trailing slash from copy-paste; both produced 404s before
 * this normalisation existed.
 */
export function normalizeBaseUrl(rawUrl: string): string {
  let url = rawUrl.trim();
  // Strip trailing slashes.
  url = url.replace(/\/+$/, '');
  // If the URL already ends with /v1 or /openai/v1, drop it.
  url = url.replace(/\/(openai\/)?v1$/i, '');
  return url;
}

/**
 * Strip a leading `Bearer ` (case-insensitive) from the configured API key
 * and trim whitespace. The client always prepends `Bearer`, so users who
 * paste their full `Authorization: Bearer …` header would otherwise send
 * `Bearer Bearer …` and get 401s.
 */
export function normalizeApiKey(rawKey: string | undefined): string {
  if (!rawKey) { return ''; }
  return rawKey.trim().replace(/^Bearer\s+/i, '');
}

/**
 * Accumulated tool call during streaming
 */
interface StreamingToolCall {
  id: string;
  name: string;
  arguments: string;
}

/**
 * State for tracking tool calls during streaming
 */
interface ToolCallState {
  toolCallsByIndex: Map<number, StreamingToolCall>;
  finalizedIndices: Set<number>;
  requestId: string;
  toolCallCounter: number;
}

/**
 * Parsed SSE chunk data
 */
interface ParsedChunk {
  delta?: {
    content?: string;
    /** LM Studio / DeepSeek-R1 separate reasoning field */
    reasoning_content?: string;
    tool_calls?: Array<{
      index?: number;
      id?: string;
      function?: { name?: string; arguments?: string };
    }>;
    function_call?: { name?: string; arguments?: string };
  };
  message?: {
    content?: string;
    /** LM Studio / DeepSeek-R1 separate reasoning field */
    reasoning_content?: string;
    text?: string;
    tool_calls?: Array<{
      index?: number;
      id?: string;
      function?: { name?: string; arguments?: string };
    }>;
    function_call?: { name?: string; arguments?: string };
  };
  finishReason?: string;
  id?: string;
}

/**
 * HTTP client for OpenAI-compatible inference servers
 */
export type GatewayLogger = (message: string) => void;

export class GatewayClient {
  private config: GatewayConfig;
  private readonly log: GatewayLogger;

  constructor(config: GatewayConfig, logger?: GatewayLogger) {
    this.config = config;
    this.log = logger ?? (() => { /* no-op logger */ });
  }

  /**
   * Update client configuration
   */
  public updateConfig(config: GatewayConfig): void {
    this.config = config;
  }

  /**
   * Fetch available models from the server's models endpoint.
   *
   * Tries `/v1/models` first and falls back to `/models` so the client works
   * against servers that mount the OpenAI API at the root (e.g. LM Studio in
   * some configurations).
   */
  public async fetchModels(cancellationToken?: vscode.CancellationToken): Promise<OpenAIModelsResponse> {
    const base = normalizeBaseUrl(this.config.serverUrl);
    const candidates = [`${base}/v1/models`, `${base}/models`];
    let lastError: Error | undefined;

    for (const url of candidates) {
      try {
        const response = await this.fetch(url, {
          method: 'GET',
          headers: this.getHeaders(),
        }, cancellationToken);

        if (response.ok) {
          return await response.json();
        }

        // 404 against the first candidate? Try the fallback.
        if (response.status === 404 && url !== candidates[candidates.length - 1]) {
          this.log(`Models endpoint not found at ${url}, trying fallback...`);
          continue;
        }

        const bodyText = await response.text().catch(() => '');
        const truncated = bodyText.length > 200 ? `${bodyText.slice(0, 200)}...` : bodyText;
        throw new Error(
          `Failed to fetch models from ${url}: ${response.status} ${response.statusText}${truncated ? ` — ${truncated}` : ''}`
        );
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        if (url === candidates[candidates.length - 1]) {
          break;
        }
      }
    }

    const message = lastError?.message ?? 'unknown error';
    throw new Error(`Failed to connect to inference server at ${base}: ${message}`);
  }

  /**
   * Create initial tool call tracking state
   */
  private createToolCallState(): ToolCallState {
    return {
      toolCallsByIndex: new Map<number, StreamingToolCall>(),
      finalizedIndices: new Set<number>(),
      requestId: `req_${Date.now()}_${randomBytes(4).toString('hex')}`,
      toolCallCounter: 0,
    };
  }

  /**
   * Process a single streamed tool call delta
   */
  private processToolCallDelta(
    tc: { index?: number; id?: string; function?: { name?: string; arguments?: string } },
    state: ToolCallState
  ): void {
    const index = tc.index ?? state.toolCallCounter++;
    const existing = state.toolCallsByIndex.get(index);

    if (existing) {
      if (tc.id) { existing.id = tc.id; }
      if (tc.function?.name) { existing.name = tc.function.name; }
      if (tc.function?.arguments) { existing.arguments += tc.function.arguments; }
    } else {
      state.toolCallsByIndex.set(index, {
        id: tc.id || '',
        name: tc.function?.name || '',
        arguments: tc.function?.arguments || '',
      });
    }
  }

  /**
   * Process legacy function_call format
   */
  private processLegacyFunctionCall(
    functionCall: { name?: string; arguments?: string },
    parsedId: string,
    state: ToolCallState
  ): void {
    const index = 0;
    const existing = state.toolCallsByIndex.get(index);

    if (existing) {
      if (functionCall.name) { existing.name = functionCall.name; }
      if (functionCall.arguments) { existing.arguments += functionCall.arguments; }
    } else {
      state.toolCallsByIndex.set(index, {
        id: parsedId || '',
        name: functionCall.name || '',
        arguments: functionCall.arguments || '',
      });
    }
  }

  /**
   * Finalize all pending tool calls
   */
  private finalizeToolCalls(state: ToolCallState): StreamingToolCall[] {
    const finishedToolCalls: StreamingToolCall[] = [];

    for (const [index, tc] of state.toolCallsByIndex.entries()) {
      if (!state.finalizedIndices.has(index)) {
        state.finalizedIndices.add(index);
        if (!tc.id) {
          tc.id = `call_${state.requestId}_${index}`;
        }
        finishedToolCalls.push({ ...tc });
      }
    }

    return finishedToolCalls;
  }

  /**
   * Process delta format from streaming response
   */
  private processDeltaFormat(
    parsed: ParsedChunk,
    state: ToolCallState
  ): { content: string; reasoningContent: string; finishedToolCalls: StreamingToolCall[] } {
    const delta = parsed.delta!;
    const finishedToolCalls: StreamingToolCall[] = [];

    // Handle streamed tool_calls
    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls) {
        this.processToolCallDelta(tc, state);
      }
    }

    // Handle legacy function_call format
    if (delta.function_call) {
      this.processLegacyFunctionCall(delta.function_call, parsed.id || '', state);
    }

    // Check if tool calls are complete
    if (parsed.finishReason === 'tool_calls' || parsed.finishReason === 'function_call') {
      finishedToolCalls.push(...this.finalizeToolCalls(state));
    }

    return { content: delta.content || '', reasoningContent: delta.reasoning_content || '', finishedToolCalls };
  }

  /**
   * Process non-delta (final) message format
   */
  private processMessageFormat(
    parsed: ParsedChunk,
    state: ToolCallState
  ): { content: string; reasoningContent: string; finishedToolCalls: StreamingToolCall[] } {
    const message = parsed.message!;
    const finishedToolCalls: StreamingToolCall[] = [];

    // Handle complete tool_calls array
    if (Array.isArray(message.tool_calls)) {
      for (let i = 0; i < message.tool_calls.length; i++) {
        const tc = message.tool_calls[i];
        const index = tc.index ?? i;
        if (!state.finalizedIndices.has(index)) {
          state.finalizedIndices.add(index);
          finishedToolCalls.push({
            id: tc.id || `call_${state.requestId}_${index}`,
            name: tc.function?.name || '',
            arguments: tc.function?.arguments || '',
          });
        }
      }
    }

    // Handle legacy function_call format
    if (message.function_call && !state.finalizedIndices.has(0)) {
      state.finalizedIndices.add(0);
      finishedToolCalls.push({
        id: parsed.id || `call_${state.requestId}_0`,
        name: message.function_call.name || '',
        arguments: message.function_call.arguments || '',
      });
    }

    return { content: message.content || message.text || '', reasoningContent: message.reasoning_content || '', finishedToolCalls };
  }

  /**
   * Parse a raw SSE data string into structured chunk data
   */
  private parseSSEData(data: string): ParsedChunk | null {
    try {
      const parsed = JSON.parse(data);
      return {
        delta: parsed.choices?.[0]?.delta,
        message: parsed.choices?.[0]?.message,
        finishReason: parsed.choices?.[0]?.finish_reason,
        id: parsed.id,
      };
    } catch {
      this.log(`Failed to parse SSE chunk: ${data}`);
      return null;
    }
  }

  /**
   * Process a single SSE line and return yield data if applicable.
   * Throws when the server reports an error via `event: error` / a JSON
   * `error` payload so the caller can surface it rather than seeing an
   * empty stream.
   */
  private processSSELine(
    line: string,
    state: ToolCallState
  ): { content: string; reasoning_content: string; tool_calls: StreamingToolCall[]; finished_tool_calls: StreamingToolCall[] } | null {
    const trimmed = line.trim();

    if (trimmed === '' || trimmed === 'data: [DONE]') {
      return null;
    }

    // Some OpenAI-compatible servers (notably proxies) emit `event: error`
    // on a prior line, but most also pack the payload into the following
    // `data: ...` object with an `error` key. Easier to just look at the
    // data contents.
    if (trimmed.startsWith('event:')) {
      // Caller doesn't see event names; they'll come through on the data line.
      return null;
    }

    if (!trimmed.startsWith('data: ')) {
      return null;
    }

    const data = trimmed.slice(6);

    // Inline error payload: `{ "error": { "message": "..." } }`.
    // Distinguish from a normal chunk (which has `choices`).
    try {
      const maybeErr = JSON.parse(data);
      if (maybeErr && typeof maybeErr === 'object' && 'error' in maybeErr && !('choices' in maybeErr)) {
        const err = (maybeErr as { error: unknown }).error;
        const message =
          (typeof err === 'object' && err !== null && 'message' in err
            ? String((err as { message?: unknown }).message)
            : undefined) ?? JSON.stringify(err);
        throw new Error(`Inference server reported an error mid-stream: ${message}`);
      }
    } catch (e) {
      if (e instanceof Error && e.message.startsWith('Inference server reported')) {
        throw e;
      }
      // Not JSON, or not an error payload — fall through to regular parsing.
    }

    const parsed = this.parseSSEData(data);
    if (!parsed) { return null; }

    if (parsed.delta) {
      const { content, reasoningContent, finishedToolCalls } = this.processDeltaFormat(parsed, state);
      return { content, reasoning_content: reasoningContent, tool_calls: [], finished_tool_calls: finishedToolCalls };
    }

    if (parsed.message) {
      const { content, reasoningContent, finishedToolCalls } = this.processMessageFormat(parsed, state);
      return { content, reasoning_content: reasoningContent, tool_calls: [], finished_tool_calls: finishedToolCalls };
    }

    return null;
  }

  /**
   * Get remaining unfinalised tool calls
   */
  private getRemainingToolCalls(state: ToolCallState): StreamingToolCall[] {
    const remaining: StreamingToolCall[] = [];

    for (const [index, tc] of state.toolCallsByIndex.entries()) {
      if (!state.finalizedIndices.has(index) && (tc.name || tc.arguments)) {
        state.finalizedIndices.add(index);
        if (!tc.id) {
          tc.id = `call_${state.requestId}_${index}`;
        }
        remaining.push({ ...tc });
      }
    }

    return remaining;
  }

  /**
   * Stream chat completions from /v1/chat/completions endpoint
   *
   * IMPORTANT: Tool calls are tracked by INDEX during streaming, not by ID.
   * OpenAI streaming format sends tool calls incrementally with an `index` field
   * to identify which tool call is being updated. The `id` may arrive in a later chunk.
   */
  public async *streamChatCompletion(
    request: OpenAIChatCompletionRequest,
    cancellationToken: vscode.CancellationToken
  ): AsyncGenerator<{ content: string; reasoning_content: string; tool_calls: StreamingToolCall[]; finished_tool_calls: StreamingToolCall[] }, void, unknown> {
    const url = `${normalizeBaseUrl(this.config.serverUrl)}/v1/chat/completions`;
    const state = this.createToolCallState();

    // For streaming we manage the abort + timers directly so that the
    // request timeout only applies until headers arrive; after that, we
    // enforce a separate per-read inactivity timeout that resets on every
    // chunk. Otherwise long generations would be aborted mid-stream even
    // though tokens were arriving continuously.
    const controller = new AbortController();
    const cancelSub = cancellationToken.onCancellationRequested(() => controller.abort());
    const headerTimeoutId = setTimeout(() => controller.abort(), this.config.requestTimeout);
    let inactivityTimeoutId: ReturnType<typeof setTimeout> | undefined;
    const resetInactivityTimer = (): void => {
      if (inactivityTimeoutId) { clearTimeout(inactivityTimeoutId); }
      inactivityTimeoutId = setTimeout(() => controller.abort(), this.config.requestTimeout);
    };

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { ...this.getHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...request, stream: true }),
        signal: controller.signal,
      });

      // Headers received — switch from the fixed request timeout to an
      // inactivity timer that will only fire if the server stalls mid-stream.
      clearTimeout(headerTimeoutId);
      resetInactivityTimer();

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Chat completion failed: ${response.status} ${response.statusText} - ${errorText}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        if (cancellationToken.isCancellationRequested) {
          reader.cancel();
          break;
        }

        const { done, value } = await reader.read();
        if (done) { break; }

        // Any byte from the server counts as activity; reset the inactivity timer.
        resetInactivityTimer();

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const result = this.processSSELine(line, state);
          if (result) { yield result; }
        }
      }

      // Finalize any remaining tool calls
      const remaining = this.getRemainingToolCalls(state);
      if (remaining.length > 0) {
        yield { content: '', reasoning_content: '', tool_calls: [], finished_tool_calls: remaining };
      }
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Chat completion request failed: ${error.message}`);
      }
      throw error;
    } finally {
      clearTimeout(headerTimeoutId);
      if (inactivityTimeoutId) { clearTimeout(inactivityTimeoutId); }
      cancelSub.dispose();
    }
  }

  /**
   * Get headers for API requests
   */
  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {};

    const key = normalizeApiKey(this.config.apiKey);
    if (key) {
      headers['Authorization'] = `Bearer ${key}`;
    }

    return headers;
  }

  /**
   * Fetch wrapper with timeout support. When a cancellationToken is passed,
   * cancellation is wired into the underlying AbortController so the in-flight
   * request is aborted as soon as the user closes the chat / model picker.
   */
  private async fetch(
    url: string,
    options: RequestInit,
    cancellationToken?: vscode.CancellationToken
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.requestTimeout);
    const cancelSub = cancellationToken?.onCancellationRequested(() => controller.abort());

    try {
      return await fetch(url, {
        ...options,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timeoutId);
      cancelSub?.dispose();
    }
  }
}
