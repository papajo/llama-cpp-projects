import React, { useState, useEffect } from 'react'
import { Explorer } from './components/Explorer'

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>🎛️ Sampling Parameter Explorer</h1>
        <span style={{ fontSize: 13, color: 'var(--text-dim)' }}>
          Understand how sampling parameters shape token distributions
        </span>
      </header>
      <Explorer />
    </div>
  )
}
