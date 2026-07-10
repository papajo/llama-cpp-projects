import React, { useState, useEffect, useCallback } from 'react';
import { listSampleSchemas, getSampleSchema } from '../api';

interface Props {
  onSelectGrammar: (grammar: string) => void;
}

export function SchemaViewer({ onSelectGrammar }: Props) {
  const [schemas, setSchemas] = useState<any[]>([]);
  const [selectedSchema, setSelectedSchema] = useState<any>(null);
  const [content, setContent] = useState('');
  const [convertedGrammar, setConvertedGrammar] = useState('');
  const [copySuccess, setCopySuccess] = useState('');

  useEffect(() => {
    listSampleSchemas().then(setSchemas).catch(() => {});
  }, []);

  const handleSelect = useCallback(async (name: string) => {
    try {
      const data = await getSampleSchema(name);
      setSelectedSchema(data);
      setContent(data.content);
      setConvertedGrammar('');

      // If it's a JSON schema and we're online, convert it
      if (data.type === 'json') {
        try {
          const res = await fetch('/api/convert-schema', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ schema_text: data.content }),
          });
          const result = await res.json();
          if (result.grammar) {
            setConvertedGrammar(result.grammar);
          }
        } catch {}
      }
    } catch {}
  }, []);

  const handleUseGrammar = useCallback(() => {
    if (convertedGrammar) {
      onSelectGrammar(convertedGrammar);
    } else if (selectedSchema?.type === 'gbnf') {
      onSelectGrammar(content);
    }
  }, [convertedGrammar, content, selectedSchema, onSelectGrammar]);

  const handleCopy = useCallback((text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopySuccess('Copied!');
      setTimeout(() => setCopySuccess(''), 2000);
    });
  }, []);

  return (
    <div className="playground-grid">
      {/* Schema list */}
      <div className="panel">
        <div className="panel-title">Sample Schemas & Grammars</div>
        {schemas.length === 0 ? (
          <div className="empty-state">
            <div className="icon">📐</div>
            <div className="text">No sample schemas found</div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {schemas.map((s) => (
              <button
                key={s.name}
                className="btn-secondary"
                style={{
                  justifyContent: 'flex-start',
                  textAlign: 'left',
                  background: selectedSchema?.name === s.name ? 'var(--accent)' : undefined,
                  color: selectedSchema?.name === s.name ? 'white' : undefined,
                  borderColor: selectedSchema?.name === s.name ? 'var(--accent)' : undefined,
                }}
                onClick={() => handleSelect(s.name)}
              >
                <span style={{ opacity: 0.6 }}>
                  {s.type === 'json' ? '📋' : '📝'}
                </span>
                {' '}
                {s.name}
                <span style={{ marginLeft: 'auto', fontSize: 11, opacity: 0.5 }}>
                  .{s.type}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Viewer */}
      <div className="panel">
        {selectedSchema ? (
          <>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div className="panel-title" style={{ margin: 0 }}>
                {selectedSchema.name}.{selectedSchema.type}
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  className="btn-secondary"
                  onClick={() => handleCopy(content)}
                >
                  {copySuccess || 'Copy'}
                </button>
                {(convertedGrammar || selectedSchema.type === 'gbnf') && (
                  <button
                    className="btn-primary"
                    onClick={handleUseGrammar}
                  >
                    Use in Playground
                  </button>
                )}
              </div>
            </div>

            <div className="code-block" style={{ marginTop: 12 }}>
              {content}
            </div>

            {convertedGrammar && (
              <>
                <label style={{ marginTop: 16 }}>Converted GBNF Grammar</label>
                <div className="code-block" style={{ maxHeight: 200, fontSize: 11 }}>
                  {convertedGrammar}
                </div>
                <div className="controls-row" style={{ marginTop: 8 }}>
                  <button
                    className="btn-secondary"
                    onClick={() => handleCopy(convertedGrammar)}
                  >
                    Copy Grammar
                  </button>
                </div>
              </>
            )}

            {selectedSchema.type === 'gbnf' && (
              <div className="controls-row" style={{ marginTop: 12 }}>
                <button
                  className="btn-secondary"
                  onClick={() => handleCopy(content)}
                >
                  Copy Grammar
                </button>
              </div>
            )}
          </>
        ) : (
          <div className="empty-state">
            <div className="icon">📐</div>
            <div className="text">Select a schema to preview</div>
            <div className="hint">
              JSON schemas are auto-converted to GBNF grammar
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
