// =============================================================================
// main.tsx
// -----------------------------------------------------------------------------
// Responsible for: The dashboard's entry point — mounting <App/> into the DOM, and
//                  (additively) initializing Sentry browser monitoring + wrapping the
//                  app in an error boundary so a render-time crash shows a fallback
//                  instead of a blank white screen.
// Role in project: Frontend half of the optional Sentry reliability layer. Sentry is
//                  OFF unless VITE_SENTRY_DSN is set at build time (then init runs). The
//                  ErrorBoundary works either way, so white-screen protection is free
//                  even with monitoring disabled.
// =============================================================================

import React from 'react'
import ReactDOM from 'react-dom/client'
import * as Sentry from '@sentry/react'
import App from './App'
import './index.css'

// Initialize Sentry only when a DSN is configured (mirrors the backend's env-gating):
// no DSN -> no init -> no network, and the dashboard behaves exactly as before.
const SENTRY_DSN = import.meta.env.VITE_SENTRY_DSN
if (SENTRY_DSN) {
  Sentry.init({
    dsn: SENTRY_DSN,
    environment: import.meta.env.VITE_SENTRY_ENVIRONMENT ?? 'demo',
    // Performance tracing, sampled to keep overhead/quota low (20% of transactions).
    integrations: [Sentry.browserTracingIntegration()],
    tracesSampleRate: 0.2,
  })
}

// Minimal, dependency-free fallback shown if a component throws during render. Uses inline
// styles (not Tailwind classes) so it still renders even if styling is part of what broke.
const CrashFallback = (
  <div
    style={{
      padding: '2rem',
      minHeight: '100vh',
      fontFamily: 'system-ui, sans-serif',
      color: '#e5e7eb',
      background: '#0b1020',
    }}
  >
    <h1 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Dashboard hit an error</h1>
    <p style={{ opacity: 0.8 }}>
      The view crashed and was caught. Reload to retry — the error has been reported if
      monitoring is enabled.
    </p>
  </div>
)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {/* ErrorBoundary catches render-time crashes (and reports them when Sentry is on), so the
        operator sees a message instead of a blank screen mid-search. It also protects against a
        bad /state payload that throws somewhere in the render tree below useMapState. */}
    <Sentry.ErrorBoundary fallback={CrashFallback}>
      <App />
    </Sentry.ErrorBoundary>
  </React.StrictMode>,
)
