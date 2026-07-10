/**
 * Ollama-specific model metadata, discovered from the native
 * `POST /api/show` endpoint. The OpenAI-compatible `/v1/models` list carries
 * none of this, so without it the gateway can't auto-configure per-model
 * context window, sampler params, or capabilities for Ollama backends
 * (issues: agent temperature always applied, context never discovered,
 * top_p defaulted to 1.0, capabilities not read).
 */

export interface OllamaModelInfo {
  /** Runtime context from Modelfile `PARAMETER num_ctx` — what's allocated. */
  readonly numCtx?: number;
  /** Trained context ceiling (`model_info["<arch>.context_length"]`). */
  readonly trainedContext?: number;
  /** Numeric sampler params from the Modelfile (temperature, top_p, ...). */
  readonly params: Readonly<Record<string, number>>;
  /** Capability tags, e.g. ["completion","vision","tools","thinking"]. */
  readonly capabilities: readonly string[];
}

/** Modelfile parameter keys whose values are numeric and worth surfacing. */
const NUMERIC_PARAM_KEYS = new Set<string>([
  'temperature', 'top_p', 'top_k', 'min_p', 'typical_p',
  'presence_penalty', 'frequency_penalty', 'repeat_penalty', 'repeat_last_n',
  'num_ctx', 'num_predict', 'seed', 'tfs_z',
  'mirostat', 'mirostat_tau', 'mirostat_eta',
]);

/**
 * Parse Ollama's `parameters` field — a newline-separated list of
 * `key<whitespace>value` lines (e.g. "temperature 0.7\nnum_ctx 65536") — into
 * a map of the numeric sampler/config params we understand. Non-numeric or
 * unknown keys (e.g. repeated `stop` strings) are ignored.
 */
export function parseOllamaParameters(parameters: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  if (typeof parameters !== 'string') { return out; }
  for (const line of parameters.split('\n')) {
    const match = /^(\S+)\s+(.+)$/.exec(line.trim());
    if (!match) { continue; }
    const key = match[1];
    if (!NUMERIC_PARAM_KEYS.has(key)) { continue; }
    const value = Number(match[2].trim());
    if (Number.isFinite(value)) { out[key] = value; }
  }
  return out;
}

/** Find the `<arch>.context_length` entry in Ollama's `model_info` block. */
function findTrainedContext(modelInfo: unknown): number | undefined {
  if (!modelInfo || typeof modelInfo !== 'object') { return undefined; }
  for (const [key, value] of Object.entries(modelInfo as Record<string, unknown>)) {
    if (
      key.endsWith('.context_length') &&
      typeof value === 'number' && Number.isFinite(value) && value > 0
    ) {
      return value;
    }
  }
  return undefined;
}

/**
 * Parse a raw `POST /api/show` JSON body into {@link OllamaModelInfo}, or
 * `undefined` when the body doesn't look like an Ollama `/api/show` response
 * (so non-Ollama servers that happen to return 200 are ignored).
 */
export function parseOllamaShowResponse(raw: unknown): OllamaModelInfo | undefined {
  if (!raw || typeof raw !== 'object') { return undefined; }
  const obj = raw as Record<string, unknown>;
  const looksLikeOllama =
    'model_info' in obj || 'parameters' in obj || 'capabilities' in obj;
  if (!looksLikeOllama) { return undefined; }

  const params = parseOllamaParameters(obj.parameters);
  const numCtx =
    Number.isFinite(params.num_ctx) && params.num_ctx > 0 ? params.num_ctx : undefined;
  const trainedContext = findTrainedContext(obj.model_info);
  const capabilities = Array.isArray(obj.capabilities)
    ? obj.capabilities.filter((c): c is string => typeof c === 'string')
    : [];

  return { numCtx, trainedContext, params, capabilities };
}
