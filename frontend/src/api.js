/* Thin fetch wrapper over the Flask API. Vite proxies /api in dev (see
   vite.config.js), so there is no base URL to configure per environment. */

class ApiError extends Error {
  constructor(message, { status, fields } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    // Per-field validation messages from the backend, keyed by field name.
    this.fields = fields || null
  }
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(path, options)
  } catch (cause) {
    throw new ApiError('Could not reach the server. Is the backend running?', { status: 0 })
  }

  const isJson = (response.headers.get('content-type') || '').includes('application/json')
  const body = isJson ? await response.json().catch(() => null) : null

  if (!response.ok) {
    throw new ApiError(body?.message || body?.error || `Request failed (${response.status})`, {
      status: response.status,
      fields: body?.fields,
    })
  }
  return body
}

const qs = (params) => {
  const search = new URLSearchParams()
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') search.set(key, value)
  })
  const str = search.toString()
  return str ? `?${str}` : ''
}

export const api = {
  health: () => request('/api/health'),

  // --- hot-spot map ---
  filters: () => request('/api/filters'),
  hotspots: (params) => request(`/api/hotspots${qs(params)}`),

  // --- directory (demo identities, not auth) ---
  members: () => request('/api/members'),
  units: () => request('/api/units'),
  users: (role) => request(`/api/users${qs({ role })}`),
  user: (id) => request(`/api/users/${encodeURIComponent(id)}`),
  updateLocation: (id, payload) =>
    request(`/api/users/${encodeURIComponent(id)}/location`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  suburbs: (q) => request(`/api/suburbs${qs({ q })}`),

  // --- crime prevention units ---
  alerts: (params) => request(`/api/alerts${qs(params)}`),
  patrolPlan: (payload) =>
    request('/api/patrol/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  // --- travel risk + routing ---
  risk: (params) => request(`/api/risk${qs(params)}`),
  riskProfile: () => request('/api/risk/profile'),
  compareRoutes: (payload) =>
    request('/api/routes/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  // --- claims ---
  submitClaim: (formData) =>
    request('/api/claims', { method: 'POST', body: formData }),
  listClaims: (params) => request(`/api/claims${qs(params)}`),
  claim: (id) => request(`/api/claims/${encodeURIComponent(id)}`),
  claimCounts: () => request('/api/claims/counts'),
  approveClaim: (id, payload) =>
    request(`/api/claims/${encodeURIComponent(id)}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  denyClaim: (id, payload) =>
    request(`/api/claims/${encodeURIComponent(id)}/deny`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
}

export { ApiError }

/* ---------- shared formatting ---------- */

export const money = new Intl.NumberFormat('en-ZA', {
  style: 'currency',
  currency: 'ZAR',
  maximumFractionDigits: 0,
})

export const num = new Intl.NumberFormat('en-ZA')

export function formatDateTime(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('en-ZA', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatDate(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString('en-ZA', { year: 'numeric', month: 'short', day: 'numeric' })
}
