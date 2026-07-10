import React, { useState, useEffect } from 'react';
import { traceCompletion, type TraceResult } from '../api';

interface Props {
  completionData: any;
  grammarText: string;
  jsonSchema: string;
}

export function TokenTrace({ completionData, grammarText, jsonSchema }: Props) {
  const [trace, setTrace] = useState<TraceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState<'cards' | 'inline'>('cards');

  useEffect(() => {
    if (!completionData) return;
    setLoading(true);
    traceCompletion(completionData, grammarText, jsonSchema)
      .then(setTrace)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [completionData, grammarText, jsonSchema]);

  if (loading) {
    return (
      <div className="panel">
        <div className="empty-state">
          <span className="spinner" />
          <span className="text">Analysing token trace...</span>
        </div>
      </div>
    );
  }

  if (!trace) {
    return (
      <div className="panel">
        <div className="empty-state">
          <div className="icon">🔬</div>
          <div className="text">No trace data available</div>
          <div className="hint">Generate with n_probs &gt; 0 to see token-level detail</div>
        </div>
      </div>
    );
  }

  const steps = trace.trace?.constrained ?? [];
  const unconstrainedSteps = trace.trace?.unconstrained ?? [];

  const hasProbs = steps.length > 0;

  return (
    <div>
      {/* Stats bar */}
      <div className="panel" style={{ marginBottom: 20 }}>
        <div className="mask-stats">
          <div className="stat-item">
            <span className="stat-label">Total Tokens</span>
            <span className="stat-value">{trace.total_tokens}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Grammar-Masked</span>
            <span className="stat-value red">{trace.tokens_masked}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Mask Rate</span>
            <span className="stat-value amber">{(trace.mask_rate * 100).toFixed(1)}%</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Grammar Size</span>
            <span className="stat-value" style={{ fontSize: 16 }}>
              {grammarText ? grammarText.split('\n').length : 0} rules
            </span>
          </div>
        </div>

        <div className="controls-row">
          <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>
            <span style={{ color: 'var(--green)', fontWeight: 600 }}>●</span> Accepted token
            {' · '}
            <span style={{ color: 'var(--red)', fontWeight: 600 }}>●</span> Grammar-masked
          </span>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
            <button
              className={`btn-secondary ${viewMode === 'cards' ? '' : ''}`}
              style={viewMode === 'cards' ? { background: 'var(--accent)', color: 'white' } : {}}
              onClick={() => setViewMode('cards')}
            >
              Cards
            </button>
            <button
              className={`btn-secondary ${viewMode === 'inline' ? '' : ''}`}
              style={viewMode === 'inline' ? { background: 'var(--accent)', color: 'white' } : {}}
              onClick={() => setViewMode('inline')}
            >
              Inline
            </button>
          </div>
        </div>
      </div>

      {/* Token traces */}
      {hasProbs && viewMode === 'inline' && (
        <div className="panel">
          <div className="panel-title">Token Stream</div>
          <div className="trace-output">
            {steps.map((step: any, i: number) => (
              <span
                key={i}
                className="token-badge accepted"
                title={`logprob: ${step.logprob.toFixed(3)} | candidates: ${step.num_candidates} | allowed: ${step.num_allowed}`}
              >
                {step.text || '<space>'}
                <span className="token-tooltip">
                  #{step.index} · logprob: {step.logprob.toFixed(3)} · {step.num_allowed}/{step.num_candidates} candidates
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {hasProbs && viewMode === 'cards' && (
        <div className="panel">
          <div className="panel-title">Token Details</div>
          <div style={{ maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-dim)' }}>
                  <th style={{ padding: '6px 8px', textAlign: 'left' }}>#</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left' }}>Token</th>
                  <th style={{ padding: '6px 8px', textAlign: 'right' }}>logprob</th>
                  <th style={{ padding: '6px 8px', textAlign: 'right' }}>Candidates</th>
                  <th style={{ padding: '6px 8px', textAlign: 'right' }}>Allowed</th>
                  <th style={{ padding: '6px 8px', textAlign: 'center' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {steps.map((step: any, i: number) => (
                  <tr
                    key={i}
                    style={{
                      borderBottom: '1px solid var(--border)',
                      background: step.masked_by_grammar
                        ? 'rgba(248, 113, 113, 0.05)'
                        : 'transparent',
                    }}
                  >
                    <td style={{ padding: '4px 8px', color: 'var(--text-dim)' }}>{step.index}</td>
                    <td style={{ padding: '4px 8px', fontFamily: 'var(--font-mono)' }}>
                      {step.text || '␣'}
                    </td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>
                      {step.logprob.toFixed(3)}
                    </td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>
                      {step.num_candidates}
                    </td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>
                      {step.num_allowed}
                    </td>
                    <td style={{ padding: '4px 8px', textAlign: 'center' }}>
                      {step.masked_by_grammar
                        ? <span style={{ color: 'var(--red)' }}>Masked</span>
                        : <span style={{ color: 'var(--green)' }}>Accepted</span>
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Grammar / schema display */}
      {grammarText && (
        <div className="panel" style={{ marginTop: 20 }}>
          <div className="panel-title">Active Grammar</div>
          <div className="code-block" style={{ maxHeight: 200 }}>
            {grammarText}
          </div>
        </div>
      )}
    </div>
  );
}
