import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  Chart as ChartJS, BarElement, CategoryScale, LinearScale,
  Tooltip, Legend,
} from 'chart.js'
import { Bar } from 'react-chartjs-2'
import {
  fetchPresets, fetchSamples, fetchSampleLogits,
  simulate, fetchParamPresets,
  type Preset, type Sample, type SimulateResult,
  type SimulateParams, type ParamPreset,
} from '../api'

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend)

const DEFAULT_SAMPLER_SEQ = 'repeat,freq,pres,top_k,top_p,min_p,temp,typ'

export function Explorer() {
  const [presets, setPresets] = useState<Preset[]>([])
  const [samples, setSamples] = useState<Sample[]>([])
  const [paramPresets, setParamPresets] = useState<ParamPreset[]>([])
  const [result, setResult] = useState<SimulateResult | null>(null)
  const [loading, setLoading] = useState(false)

  // Distribution
  const [distType, setDistType] = useState<'preset' | 'sample'>('preset')
  const [selectedPreset, setSelectedPreset] = useState('unimodal')
  const [selectedSample, setSelectedSample] = useState('')
  const [customLogits, setCustomLogits] = useState<number[] | null>(null)

  // Sampling params
  const [params, setParams] = useState({
    temperature: 0.8,
    top_k: 40,
    top_p: 0.95,
    min_p: 0.05,
    repeat_penalty: 1.0,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,
    mirostat: 0,
    mirostat_tau: 5.0,
    mirostat_eta: 0.1,
    dynatemp_range: 0.0,
    dynatemp_exponent: 1.0,
    sampler_sequence: DEFAULT_SAMPLER_SEQ,
  })

  useEffect(() => {
    fetchPresets().then(setPresets)
    fetchSamples().then(setSamples)
    fetchParamPresets().then(setParamPresets)
  }, [])

  const handleSimulate = useCallback(async () => {
    setLoading(true)
    try {
      let logits: number[] = []
      if (customLogits) {
        logits = customLogits
      } else if (distType === 'sample' && selectedSample) {
        const s = await fetchSampleLogits(selectedSample)
        logits = s.logits
      }

      const r = await simulate({
        distribution: distType === 'preset' ? selectedPreset : (selectedSample || 'unimodal'),
        ...params,
      })
      setResult(r)
    } catch (e: any) {
      console.error('Simulation failed:', e)
    } finally {
      setLoading(false)
    }
  }, [params, distType, selectedPreset, selectedSample, customLogits])

  // Auto-run on mount
  useEffect(() => {
    handleSimulate()
  }, [])

  const applyParamPreset = useCallback((p: ParamPreset) => {
    setParams(prev => ({ ...prev, ...p.params }))
  }, [])

  const updateParam = useCallback((key: string, value: number | string) => {
    setParams(prev => ({ ...prev, [key]: value }))
  }, [])

  // Chart data from the final probability distribution
  const chartData = result ? {
    labels: result.final_probs.map((_, i) => `${i}`),
    datasets: [{
      label: 'Probability',
      data: result.final_probs,
      backgroundColor: result.final_probs.map((p: number) =>
        p > 0.01 ? '#6366f1' : '#2a2d3a'
      ),
      borderRadius: 2,
    }],
  } : null

  return (
    <div className="layout">
      {/* Left panel: controls */}
      <div className="panel">
        <div className="panel-title">Controls</div>
        <div className="params-scroll">
          {/* Distribution selector */}
          <div className="panel-label">Input Distribution</div>
          <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
            <button
              className={`btn-small ${distType === 'preset' ? 'btn-active' : 'btn-secondary'}`}
              onClick={() => setDistType('preset')}
            >Synthetic</button>
            <button
              className={`btn-small ${distType === 'sample' ? 'btn-active' : 'btn-secondary'}`}
              onClick={() => setDistType('sample')}
            >Real Samples</button>
          </div>

          {distType === 'preset' ? (
            <select value={selectedPreset} onChange={e => setSelectedPreset(e.target.value)}>
              {presets.map(p => (
                <option key={p.name} value={p.name}>{p.name} — {p.description}</option>
              ))}
            </select>
          ) : (
            <select value={selectedSample} onChange={e => setSelectedSample(e.target.value)}>
              <option value="">Select a sample...</option>
              {samples.map(s => (
                <option key={s.name} value={s.name}>{s.title}</option>
              ))}
            </select>
          )}

          {/* Parameter presets */}
          <div className="panel-label" style={{ marginTop: 12 }}>Quick Presets</div>
          <div className="preset-grid">
            {paramPresets.map(p => (
              <button
                key={p.name}
                className="btn-secondary btn-small"
                onClick={() => applyParamPreset(p)}
                title={p.description}
              >{p.label}</button>
            ))}
          </div>

          <hr className="section-divider" />

          {/* Sliders */}
          <SliderParam label="Temperature" value={params.temperature} min={0.01} max={2.5} step={0.01}
            onChange={v => updateParam('temperature', v)} />
          <SliderParam label="Top-K" value={params.top_k} min={0} max={100} step={1}
            onChange={v => updateParam('top_k', v)} />
          <SliderParam label="Top-P" value={params.top_p} min={0} max={1} step={0.01}
            onChange={v => updateParam('top_p', v)} />
          <SliderParam label="Min-P" value={params.min_p} min={0} max={1} step={0.01}
            onChange={v => updateParam('min_p', v)} />
          <SliderParam label="Repeat Penalty" value={params.repeat_penalty} min={0.5} max={2.5} step={0.01}
            onChange={v => updateParam('repeat_penalty', v)} />
          <SliderParam label="Frequency Penalty" value={params.frequency_penalty} min={0} max={2} step={0.01}
            onChange={v => updateParam('frequency_penalty', v)} />
          <SliderParam label="Presence Penalty" value={params.presence_penalty} min={0} max={2} step={0.01}
            onChange={v => updateParam('presence_penalty', v)} />
          <SliderParam label="Mirostat (0=off, 1=v1, 2=v2)" value={params.mirostat} min={0} max={2} step={1}
            onChange={v => updateParam('mirostat', v)} />
          {params.mirostat > 0 && (
            <>
              <SliderParam label="Mirostat Tau" value={params.mirostat_tau} min={0.1} max={10} step={0.1}
                onChange={v => updateParam('mirostat_tau', v)} />
              <SliderParam label="Mirostat Eta" value={params.mirostat_eta} min={0.01} max={1} step={0.01}
                onChange={v => updateParam('mirostat_eta', v)} />
            </>
          )}
          <SliderParam label="Dynatemp Range" value={params.dynatemp_range} min={0} max={2} step={0.1}
            onChange={v => updateParam('dynatemp_range', v)} />

          <div className="panel-label">Sampler Sequence</div>
          <input type="text" value={params.sampler_sequence}
            onChange={e => updateParam('sampler_sequence', e.target.value)}
            placeholder="top_k,top_p,min_p,temp,typ" />

          <div style={{ marginTop: 12 }}>
            <button className="btn-primary" onClick={handleSimulate} disabled={loading}>
              {loading ? '⟳ Simulating...' : '▶ Run Simulation'}
            </button>
          </div>

          {result && (
            <>
              <hr className="section-divider" />
              <div className="panel-label">Distribution: {result.distribution_name}</div>
              <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                {result.distribution_description}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Right panel: charts + steps */}
      <div>
        {/* Distribution chart */}
        <div className="panel">
          <div className="panel-title">
            Final Probability Distribution
            {result && (
              <span style={{ float: 'right', fontWeight: 400, textTransform: 'none' }}>
                {result.summary.final_zero_count} zero · entropy: {result.summary.final_entropy}
              </span>
            )}
          </div>
          {chartData ? (
            <div className="chart-container">
              <Bar
                data={chartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  animation: { duration: 300 },
                  scales: {
                    x: { display: false },
                    y: {
                      beginAtZero: true,
                      grid: { color: 'rgba(255,255,255,0.05)' },
                      ticks: { color: '#8b8fa3', font: { size: 10 } },
                    },
                  },
                  plugins: {
                    legend: { display: false },
                    tooltip: {
                      callbacks: {
                        label: (ctx) => `Token ${ctx.dataIndex}: ${(ctx.raw as number * 100).toFixed(2)}%`,
                      },
                    },
                  },
                }}
              />
            </div>
          ) : (
            <div className="empty-state"><span className="text">Run simulation to see distribution</span></div>
          )}
        </div>

        {/* Summary metrics */}
        {result && (
          <div className="summary-grid">
            <div className="summary-item">
              <div className="summary-value" style={{ color: 'var(--green)' }}>{result.summary.final_peak_prob.toFixed(4)}</div>
              <div className="summary-label">Peak Probability</div>
            </div>
            <div className="summary-item">
              <div className="summary-value" style={{ color: 'var(--blue)' }}>{result.summary.final_entropy.toFixed(2)}</div>
              <div className="summary-label">Entropy (nats)</div>
            </div>
            <div className="summary-item">
              <div className="summary-value" style={{ color: 'var(--amber)' }}>{result.summary.final_zero_count}</div>
              <div className="summary-label">Zeroed Tokens</div>
            </div>
          </div>
        )}

        {/* Pipeline steps */}
        {result && result.snapshots.length > 0 && (
          <div className="panel" style={{ marginTop: 16 }}>
            <div className="panel-title">Sampling Pipeline ({result.snapshots.length} steps)</div>
            <div className="step-list">
              {result.snapshots.map((step: any, i: number) => (
                <div className="step-card" key={i}>
                  <div className="step-header">
                    <span className="step-name">{step.step}</span>
                    <span className="step-stat">
                      peak: <span>{(step.peak_prob * 100).toFixed(1)}%</span>
                      {' · '}entropy: <span>{step.entropy.toFixed(2)}</span>
                      {' · '}zeroed: <span>{step.num_zero}</span>
                    </span>
                  </div>
                  {step.description && <div className="step-desc">{step.description}</div>}
                  {step.top_tokens && (
                    <div style={{ marginTop: 4, fontSize: 10, color: 'var(--text-dim)' }}>
                      Top tokens: {step.top_tokens.slice(0, 3).map((t: any) =>
                        `#${t.id} (${(t.prob * 100).toFixed(1)}%)`
                      ).join(', ')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Slider component ─────────────────────────────────────────────

function SliderParam({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="param-group">
      <div className="panel-label">{label}</div>
      <div className="slider-row">
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(parseFloat(e.target.value))} />
        <span className="slider-value">{value}</span>
      </div>
    </div>
  )
}
