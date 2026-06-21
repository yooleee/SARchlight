/// <reference types="vite/client" />

interface ImportMetaEnv {
  // WebSocket URL of the voice agent's /transcript feed.
  readonly VITE_AGENT_WS_URL?: string
  // Sentry DSN for browser error/perf monitoring. Unset -> Sentry stays OFF (no init runs).
  readonly VITE_SENTRY_DSN?: string
  // Optional deployment tag attached to Sentry events (defaults to "demo").
  readonly VITE_SENTRY_ENVIRONMENT?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
