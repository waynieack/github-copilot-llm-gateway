import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { parseOllamaParameters, parseOllamaShowResponse } from '../ollamaInfo';

describe('parseOllamaParameters', () => {
  test('parses numeric sampler params, ignores unknown/non-numeric', () => {
    const params = parseOllamaParameters(
      'repeat_penalty                 1.05\n' +
      'temperature                    0.7\n' +
      'top_k                          20\n' +
      'top_p                          0.8\n' +
      'num_ctx                        65536\n' +
      'presence_penalty               0\n' +
      'stop                           "<|im_end|>"'
    );
    assert.equal(params.temperature, 0.7);
    assert.equal(params.top_p, 0.8);
    assert.equal(params.top_k, 20);
    assert.equal(params.num_ctx, 65536);
    assert.equal(params.presence_penalty, 0);
    assert.equal(params.repeat_penalty, 1.05);
    assert.equal(params.stop, undefined);
  });

  test('returns empty object for non-string input', () => {
    assert.deepEqual(parseOllamaParameters(undefined), {});
  });
});

describe('parseOllamaShowResponse', () => {
  test('extracts num_ctx, trained context, params, capabilities', () => {
    const info = parseOllamaShowResponse({
      model_info: { 'qwen3.context_length': 262144, 'qwen3.block_count': 40 },
      parameters: 'temperature 0.7\ntop_p 0.8\nnum_ctx 65536',
      capabilities: ['completion', 'vision', 'tools', 'thinking'],
    });
    assert.ok(info);
    assert.equal(info.numCtx, 65536);
    assert.equal(info.trainedContext, 262144);
    assert.equal(info.params.top_p, 0.8);
    assert.deepEqual([...info.capabilities], ['completion', 'vision', 'tools', 'thinking']);
  });

  test('num_ctx omitted -> undefined, trained context still found', () => {
    const info = parseOllamaShowResponse({
      model_info: { 'gemma3.context_length': 8192 },
      parameters: 'temperature 1',
      capabilities: ['completion'],
    });
    assert.ok(info);
    assert.equal(info.numCtx, undefined);
    assert.equal(info.trainedContext, 8192);
  });

  test('returns undefined for a non-Ollama body (no recognised keys)', () => {
    assert.equal(parseOllamaShowResponse({ id: 'gpt-4', object: 'model' }), undefined);
    assert.equal(parseOllamaShowResponse(null), undefined);
  });
});
