/**
 * Lenient JSON repair for partial/malformed LLM tool call arguments.
 *
 * Handles the common failure modes seen from streaming tool calls:
 *  - truncated output (unclosed strings, braces, brackets)
 *  - trailing commas
 *  - empty / whitespace-only argument strings
 *
 * The repair logic is implemented as a single string-aware scan so that
 * structural characters inside string literals (e.g. `"{"`) are not mistaken
 * for real braces, and so that nested closures happen in the correct order
 * (`{[{` → close with `}]}`, not `]}}`).
 */

export type RepairLogger = (message: string) => void;

const BRACKET_ESCAPE_PATTERN = /[.*+?^${}()|[\]\\]/g;
const TRAILING_COMMA_PATTERN = /,\s*([}\]])/g;
const NOOP_LOGGER: RepairLogger = () => {
  /* no-op */
};

/**
 * Count literal occurrences of a single character in a string.
 * Does NOT understand string literals — use `balanceStructures` for anything
 * that needs to respect JSON quoting rules. Exported for tests.
 */
export function countChar(str: string, char: string): number {
  const escapedChar = char.replaceAll(BRACKET_ESCAPE_PATTERN, String.raw`\$&`);
  const regex = new RegExp(escapedChar, 'g');
  return (str.match(regex) ?? []).length;
}

/**
 * Scan a candidate JSON string, tracking string state and the stack of
 * open `{`/`[` tokens. Appends whatever closers (and an optional closing
 * quote) are needed so the result is structurally balanced.
 *
 * The function never removes characters — if the input already has more
 * closers than openers it's returned unchanged. Characters inside string
 * literals (including escaped quotes via `\"`) are ignored for bracket
 * bookkeeping.
 */
export function balanceStructures(str: string): string {
  const stack: string[] = [];
  let inString = false;
  let escaped = false;

  for (let i = 0; i < str.length; i++) {
    const c = str[i];

    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (c === '\\') {
        escaped = true;
        continue;
      }
      if (c === '"') {
        inString = false;
      }
      continue;
    }

    if (c === '"') {
      inString = true;
      continue;
    }
    if (c === '{' || c === '[') {
      stack.push(c);
      continue;
    }
    if (c === '}' && stack[stack.length - 1] === '{') {
      stack.pop();
      continue;
    }
    if (c === ']' && stack[stack.length - 1] === '[') {
      stack.pop();
    }
  }

  let result = str;
  if (inString) {
    result += '"';
  }
  while (stack.length > 0) {
    const top = stack.pop();
    result += top === '{' ? '}' : ']';
  }
  return result;
}

/**
 * Back-compat alias — older call sites (and tests) use `balanceBrackets`.
 * The new name is `balanceStructures` because the function also closes
 * unclosed string literals, not just brackets.
 */
export const balanceBrackets = balanceStructures;

/**
 * Try to parse a JSON argument string, applying repair heuristics if the
 * direct parse fails. Returns the parsed value, {} for empty input, or null
 * if repair ultimately fails.
 *
 * The logger (optional) receives diagnostic messages when repair fails so
 * callers can surface them to their own output channel.
 */
export function tryRepairJson(jsonStr: string, log: RepairLogger = NOOP_LOGGER): unknown {
  if (!jsonStr || jsonStr.trim() === '') {
    return {};
  }

  try {
    return JSON.parse(jsonStr);
  } catch {
    // Fall through to repair attempts.
  }

  let repaired = jsonStr.trim();
  repaired = repaired.replaceAll(TRAILING_COMMA_PATTERN, '$1');
  repaired = balanceStructures(repaired);

  try {
    return JSON.parse(repaired);
  } catch {
    log(`JSON repair failed. Original: ${jsonStr}`);
    log(`Repaired attempt: ${repaired}`);
    return null;
  }
}
