import React, { useState, useCallback, useRef, useEffect } from 'react';
import { sendCompletion, convertSchema, validateGrammar, grammarInfo } from '../api';

interface Props {
  onCompletion: (data: any, grammar: string, schema: string) => void;
  lastCompletion: any;
  serverConnected: boolean;
}

const SAMPLE_PROMPTS = [
  'List the weather for New York, London, and Tokyo:',
  'Extract all entities from this text: "Apple announced a new iPhone at their Cupertino headquarters in September 2025."',
  'Call the function to check the weather in Paris on 2025-12-25.',
  'Draft a professional email about a project update to the team.',
  'Generate a JSON object describing a book with title, author, year, and genres.',
  'Tell me about the history of computing.',
  'What is 2 + 2? Respond in JSON format.',
];

const DEFAULT_PROMPT = 'Generate a weather report for San Francisco in JSON format.';

const DEFAULT_JSON_SCHEMA = `{
  "type": "object",
  "properties": {
    "city": { "type": "string" },
    "temperature": { "type": "number" },
    "conditions": {
      "type": "string",
      "enum": ["sunny", "cloudy", "rainy"]
    }
  },
  "required": ["city", "temperature", "conditions"]
}`;

export function Playground({ onCompletion, lastCompletion, serverConnected }: Props) {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [jsonSchema, setJsonSchema] = useState(DEFAULT_JSON_SCHEMA);
  const [grammar, setGrammar] = useState('');
  const [inputMode, setInputMode] = useState<'json-schema' | 'gbnf'>('json-schema');

  // Generation params
  const [nPredict, setNPredict] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [nProbs, setNProbs] = useState(5);

  // State
  const [streamingText, setStreamingText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState('');
  const [generatedGrammar, setGeneratedGrammar] = useState('');
  const [grammarValidation, setGrammarValidation] = useState<{
    valid: boolean;
    errors: string[];
    rules: { name: string; summary: string; is_root: boolean }[];
  } | null>(null);
  const outputRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [streamingText]);

  const handleConvert = useCallback(async () => {
    if (!jsonSchema.trim()) return;
    setError('');
    try {
      const result = await convertSchema(jsonSchema);
      if (result.errors.length > 0) {
        setError(`Schema errors: ${result.errors.join('; ')}`);
        return;
      }
      setGeneratedGrammar(result.grammar);
      setGrammar(result.grammar);

      // Validate and show rules
      const val = await validateGrammar(result.grammar);
      setGrammarValidation(val);
    } catch (e: any) {
      setError(`Conversion failed: ${e.message}`);
    }
  }, [jsonSchema]);

  const handleValidateGrammar = useCallback(async () => {
    if (!grammar.trim()) return;
    setError('');
    try {
      const val = await validateGrammar(grammar);
      setGrammarValidation(val);
      if (!val.valid) {
        setError(`Grammar errors: ${val.errors.join('; ')}`);
      }
    } catch (e: any) {
      setError(`Validation failed: ${e.message}`);
    }
  }, [grammar]);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) return;
    setError('');
    setStreamingText('');

    const effectiveGrammar = inputMode === 'json-schema'
      ? grammar
      : grammar;

    if (!effectiveGrammar && inputMode === 'json-schema') {
      setError('Convert a JSON Schema to GBNF first (click "Convert")');
      return;
    }

    setIsGenerating(true);
    try {
      const result = await sendCompletion(prompt, effectiveGrammar, '', {
        n_predict: nPredict,
        temperature,
        n_probs: nProbs,
      });
      setStreamingText(result.content);
      onCompletion(result, effectiveGrammar, jsonSchema);
    } catch (e: any) {
      setError(`Generation failed: ${e.message}`);
    } finally {
      setIsGenerating(false);
    }
  }, [prompt, grammar, jsonSchema, inputMode, nPredict, temperature, nProbs, onCompletion]);

  return (
    <div className="playground-grid">
      {/* Left panel: Controls */}
      <div className="panel">
        <div className="panel-title">Controls</div>

        <label>Prompt</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          placeholder="Enter your prompt..."
        />

        <div className="schema-tabs" style={{ marginTop: 12 }}>
          <button
            className={`schema-tab ${inputMode === 'json-schema' ? 'active' : ''}`}
            onClick={() => setInputMode('json-schema')}
          >
            JSON Schema
          </button>
          <button
            className={`schema-tab ${inputMode === 'gbnf' ? 'active' : ''}`}
            onClick={() => setInputMode('gbnf')}
          >
            GBNF Grammar
          </button>
        </div>

        {inputMode === 'json-schema' ? (
          <>
            <label>JSON Schema</label>
            <textarea
              value={jsonSchema}
              onChange={(e) => setJsonSchema(e.target.value)}
              rows={8}
              placeholder='{"type": "object", ...}'
            />
            <div className="controls-row">
              <button
                className="btn-secondary"
                onClick={handleConvert}
                disabled={!jsonSchema.trim()}
              >
                🔄 Convert to GBNF
              </button>
            </div>
            {generatedGrammar && (
              <>
                <label>Generated GBNF Grammar</label>
                <div className="code-block" style={{ maxHeight: 140, fontSize: 11 }}>
                  {generatedGrammar}
                </div>
              </>
            )}
          </>
        ) : (
          <>
            <label>GBNF Grammar</label>
            <textarea
              value={grammar}
              onChange={(e) => setGrammar(e.target.value)}
              rows={8}
              placeholder={'root ::= "hello" " " "world"'}
            />
            <div className="controls-row">
              <button
                className="btn-secondary"
                onClick={handleValidateGrammar}
                disabled={!grammar.trim()}
              >
                ✅ Validate
              </button>
            </div>
          </>
        )}

        {grammarValidation && grammarValidation.rules.length > 0 && (
          <>
            <label style={{ marginTop: 12 }}>
              Grammar Rules ({grammarValidation.rules.length})
            </label>
            <div style={{ maxHeight: 140, overflowY: 'auto' }}>
              {grammarValidation.rules.map((rule, i) => (
                <div className="rule-card" key={i}>
                  <div className="rule-name">
                    {rule.is_root ? '▶ ' : ''}{rule.name}
                  </div>
                  <div className="rule-def">{rule.summary}</div>
                </div>
              ))}
            </div>
          </>
        )}

        <label style={{ marginTop: 12 }}>Parameters</label>
        <div className="param-row">
          <div className="param-group">
            <label>n_predict</label>
            <input
              type="number"
              value={nPredict}
              onChange={(e) => setNPredict(Number(e.target.value))}
              min={1}
              max={4096}
            />
          </div>
          <div className="param-group">
            <label>temperature</label>
            <input
              type="number"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              min={0}
              max={2}
              step={0.1}
            />
          </div>
          <div className="param-group">
            <label>n_probs</label>
            <input
              type="number"
              value={nProbs}
              onChange={(e) => setNProbs(Number(e.target.value))}
              min={0}
              max={10}
            />
          </div>
        </div>

        <div className="controls-row">
          <button
            className="btn-primary"
            onClick={handleGenerate}
            disabled={isGenerating || !serverConnected}
          >
            {isGenerating ? (
              <><span className="spinner" /> Generating...</>
            ) : (
              '▶ Generate'
            )}
          </button>
          <button
            className="btn-secondary"
            onClick={() => {
              setStreamingText('');
              setGrammar('');
              setGeneratedGrammar('');
              setGrammarValidation(null);
              setError('');
            }}
          >
            Clear
          </button>
        </div>

        {error && <div className="error-box">{error}</div>}
      </div>

      {/* Right panel: Output */}
      <div className="panel">
        <div className="panel-title">
          Output
          {lastCompletion && (
            <span style={{ float: 'right', fontSize: 11, color: 'var(--text-dim)' }}>
              {lastCompletion.tokens_generated} tokens · {lastCompletion.timings?.predicted_per_second
                ? `${lastCompletion.timings.predicted_per_second.toFixed(1)} t/s`
                : ''}
            </span>
          )}
        </div>

        <div className="output-area" ref={outputRef}>
          {streamingText ? (
            <span style={{ color: 'var(--text)' }}>{streamingText}</span>
          ) : isGenerating ? (
            <div className="empty-state">
              <span className="spinner" />
              <span className="text">Generating constrained output...</span>
            </div>
          ) : (
            <div className="empty-state">
              <div className="icon">🎮</div>
              <div className="text">Enter a prompt, configure your schema, and generate</div>
              <div className="hint">
                {serverConnected
                  ? 'Tip: Set n_probs &gt; 0 to enable token-level tracing'
                  : 'Start llama-server to enable generation'}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
