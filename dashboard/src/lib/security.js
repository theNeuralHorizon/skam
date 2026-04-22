/*
 * Dashboard security utilities
 * - Sanitize WS/API payloads before rendering
 * - Validate API base & WS origin against an allowlist
 * - Schema-shaped safe getters for untrusted JSON
 */

const MAX_STRING = 2048
const MAX_ARRAY = 500

const ALLOWED_API_PROTOCOLS = new Set(['http:', 'https:', ''])

/**
 * Returns an allowlist of API base origins derived from env + current origin.
 * We deliberately never accept arbitrary user-supplied bases.
 */
export function resolveApiBase() {
  const raw = (import.meta.env.VITE_API_BASE || '').trim()
  if (!raw) return ''
  try {
    const u = new URL(raw, window.location.origin)
    if (!ALLOWED_API_PROTOCOLS.has(u.protocol)) return ''
    if (u.protocol === 'http:' && window.location.protocol === 'https:') return ''
    return u.origin + u.pathname.replace(/\/$/, '')
  } catch {
    return ''
  }
}

/**
 * Builds the WebSocket URL using current page origin only.
 * Rejects any attempt to connect cross-origin by construction.
 */
export function buildWsUrl(path) {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const safePath = String(path || '').replace(/[^A-Za-z0-9/_\-.]/g, '')
  return `${proto}://${window.location.host}${safePath.startsWith('/') ? safePath : `/${safePath}`}`
}

/** Clamp strings, strip control chars, stop XSS via display. */
export function safeStr(v, max = MAX_STRING) {
  if (v === null || v === undefined) return ''
  const s = String(v).slice(0, max)
  return s.replace(/[\u0000-\u001f\u007f]/g, '').trim()
}

export function safeNum(v, fallback = 0) {
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isFinite(n) ? n : fallback
}

export function safeBool(v) {
  return v === true || v === 'true' || v === 1
}

export function safeArr(v, max = MAX_ARRAY) {
  if (!Array.isArray(v)) return []
  return v.slice(0, max)
}

export function safeObj(v) {
  return v && typeof v === 'object' && !Array.isArray(v) ? v : {}
}

/**
 * Parse a WS message safely. Returns null on anything malformed.
 * Preserves the original payload shape (flat recovery events and
 * wrapped prediction events both pass through untouched after size +
 * type checks) so downstream renderers can read fields directly.
 * Caller should still sanitize individual strings before rendering.
 */
export function parseWsMessage(raw) {
  if (typeof raw !== 'string') return null
  if (raw.length > 64 * 1024) return null
  try {
    const msg = JSON.parse(raw)
    if (!msg || typeof msg !== 'object' || Array.isArray(msg)) return null
    return msg
  } catch {
    return null
  }
}

/** True if the message is a typed wrapper envelope (e.g. prediction_raised). */
export function isTypedEnvelope(msg) {
  return !!(msg && typeof msg === 'object' && typeof msg.type === 'string' && msg.data && typeof msg.data === 'object')
}

/** Validates a path used against our proxy — rejects absolute / protocol URLs. */
export function safePath(p) {
  const s = safeStr(p, 256)
  if (!s.startsWith('/')) return '/'
  if (s.includes('..') || s.includes('//')) return '/'
  return s
}
