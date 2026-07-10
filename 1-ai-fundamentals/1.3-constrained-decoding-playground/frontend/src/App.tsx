import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Playground } from './components/Playground';
import { SchemaViewer } from './components/SchemaViewer';
import { TokenTrace } from './components/TokenTrace';
import { healthCheck, type HealthStatus } from './api';

type Tab = 'playground' | 'trace' | 'schemas';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('playground');
  const [health, setHealth] = useState<HealthStatus | null>(null);

  // Shared state between tabs
  const [lastCompletion, setLastCompletion] = useState<any>(null);
  const [lastGrammar, setLastGrammar] = useState('');
  const [lastSchema, setLastSchema] = useState('');

  useEffect(() => {
    healthCheck()
      .then(setHealth)
      .catch(() => setHealth({ status: 'error', llama_server: '', llama_server_connected: false }));
  }, []);

  const onCompletion = useCallback((data: any, grammar: string, schema: string) => {
    setLastCompletion(data);
    setLastGrammar(grammar);
    setLastSchema(schema);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>
          <span>🔍</span>
          Constrained Decoding Playground
        </h1>
        <div className="status">
          <span className={`status-dot ${health?.llama_server_connected ? 'connected' : ''}`} />
          {health?.llama_server_connected ? 'llama-server online' : 'llama-server offline'}
        </div>
      </header>

      <nav className="tabs">
        <button
          className={`tab-btn ${activeTab === 'playground' ? 'active' : ''}`}
          onClick={() => setActiveTab('playground')}
        >
          🎮 Playground
        </button>
        <button
          className={`tab-btn ${activeTab === 'trace' ? 'active' : ''}`}
          onClick={() => setActiveTab('trace')}
          disabled={!lastCompletion}
        >
          🔬 Token Trace{lastCompletion ? ' ●' : ''}
        </button>
        <button
          className={`tab-btn ${activeTab === 'schemas' ? 'active' : ''}`}
          onClick={() => setActiveTab('schemas')}
        >
          📐 Schema Browser
        </button>
      </nav>

      {activeTab === 'playground' && (
        <Playground
          onCompletion={onCompletion}
          lastCompletion={lastCompletion}
          serverConnected={health?.llama_server_connected ?? false}
        />
      )}

      {activeTab === 'trace' && lastCompletion && (
        <TokenTrace
          completionData={lastCompletion}
          grammarText={lastGrammar}
          jsonSchema={lastSchema}
        />
      )}

      {activeTab === 'schemas' && (
        <SchemaViewer
          onSelectGrammar={(grammar) => {
            setLastGrammar(grammar);
            setActiveTab('playground');
          }}
        />
      )}
    </div>
  );
}
