/** API helpers for backend communication. */

const API_BASE = '/api';

export interface HealthStatus {
  status: string;
  llama_server: string;
  llama_server_connected: boolean;
}

export interface SampleSchema {
  name: string;
  type: string;
  path: string;
}

export interface CompletionResult {
  content: string;
  tokens_generated: number;
  timings: Record<string, number>;
  grammar_used: string;
  json_schema_used: string;
  completion_probabilities?: any[];
}

export interface TraceResult {
  trace: any;
  total_tokens: number;
  tokens_masked: number;
  mask_rate: number;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  rules: { name: string; summary: string; is_root: boolean }[];
}

export async function healthCheck(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function convertSchema(schemaText: string): Promise<{ grammar: string; errors: string[] }> {
  const res = await fetch(`${API_BASE}/convert-schema`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ schema_text: schemaText }),
  });
  return res.json();
}

export async function validateGrammar(grammarText: string): Promise<ValidationResult> {
  const res = await fetch(`${API_BASE}/validate-grammar`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ grammar_text: grammarText }),
  });
  return res.json();
}

export async function grammarInfo(grammarText: string): Promise<{
  rules: { name: string; definition: string; is_root: boolean }[];
  num_rules: number;
  has_root: boolean;
}> {
  const res = await fetch(`${API_BASE}/grammar-info`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ grammar_text: grammarText }),
  });
  return res.json();
}

export async function listSampleSchemas(): Promise<SampleSchema[]> {
  const res = await fetch(`${API_BASE}/sample-schemas`);
  const data = await res.json();
  return data.schemas;
}

export async function getSampleSchema(name: string): Promise<{ name: string; type: string; content: string }> {
  const res = await fetch(`${API_BASE}/sample-schema/${encodeURIComponent(name)}`);
  return res.json();
}

export async function sendCompletion(prompt: string, grammar: string, jsonSchema: string, params: {
  n_predict?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  n_probs?: number;
  seed?: number;
}): Promise<CompletionResult> {
  const body: any = {
    prompt,
    grammar,
    json_schema: jsonSchema,
    n_predict: params.n_predict ?? 256,
    temperature: params.temperature ?? 0.7,
    top_k: params.top_k ?? 40,
    top_p: params.top_p ?? 0.95,
    seed: params.seed ?? 42,
  };
  if (params.n_probs) {
    body.n_probs = params.n_probs;
  }
  const res = await fetch(`${API_BASE}/completion`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}

export async function traceCompletion(completionData: any, grammarText: string, jsonSchema: string): Promise<TraceResult> {
  const res = await fetch(`${API_BASE}/trace`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      completion_data: completionData,
      grammar_text: grammarText,
      json_schema: jsonSchema,
    }),
  });
  return res.json();
}
