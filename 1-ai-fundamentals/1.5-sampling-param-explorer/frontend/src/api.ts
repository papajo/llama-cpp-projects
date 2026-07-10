/** API helpers. */
const BASE = '/api';

export interface Preset {
  name: string;
  description: string;
}

export interface Sample {
  name: string;
  title: string;
  description: string;
}

export interface SimulateParams {
  distribution: string;
  temperature: number;
  top_k: number;
  top_p: number;
  min_p: number;
  repeat_penalty: number;
  frequency_penalty: number;
  presence_penalty: number;
  mirostat: number;
  mirostat_tau: number;
  mirostat_eta: number;
  dynatemp_range: number;
  dynatemp_exponent: number;
  sampler_sequence: string;
}

export interface SimulateResult {
  snapshots: any[];
  final_probs: number[];
  final_logits_sample: number[];
  summary: { vocab_size: number; final_entropy: number; final_peak_prob: number; final_zero_count: number; num_snapshots: number };
  distribution_name: string;
  distribution_description: string;
}

export interface ParamPreset {
  name: string;
  label: string;
  description: string;
  params: Record<string, any>;
}

export async function fetchPresets(): Promise<Preset[]> {
  const r = await fetch(`${BASE}/presets`);
  const d = await r.json();
  return d.presets;
}

export async function fetchSamples(): Promise<Sample[]> {
  const r = await fetch(`${BASE}/samples`);
  const d = await r.json();
  return d.samples;
}

export async function fetchSampleLogits(name: string): Promise<{ name: string; description: string; logits: number[] }> {
  const r = await fetch(`${BASE}/samples/${name}`);
  return r.json();
}

export async function simulate(params: SimulateParams): Promise<SimulateResult> {
  const r = await fetch(`${BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return r.json();
}

export async function fetchParamPresets(): Promise<ParamPreset[]> {
  const r = await fetch(`${BASE}/presets/parameters`);
  const d = await r.json();
  return d.presets;
}
