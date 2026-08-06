/* Thin fetch wrapper over the Flask API.
   Supports production Vercel -> Railway calls via VITE_API_BASE_URL,
   while falling back to relative paths for Vite proxying in local dev. */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

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
  // Prepend BASE_URL to the relative API endpoint path
  const url = `${BASE_URL}${path}`

  try {
    response = await fetch(url, options)
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

  // --- behavioural analysis ---
  behaviourQueue: (params) => request(`/api/v1/behaviour/review-queue${qs(params)}`),
  behaviourReview: (id) =>
    request(`/api/v1/behaviour/review-queue/${encodeURIComponent(id)}`),
  behaviourHistory: (id) =>
    request(`/api/v1/behaviour/review-queue/${encodeURIComponent(id)}/history`),
  // reviewer_id is supplied by session.jsx and trusted by the API. On an
  // identification decision that field is the audit trail's only signature —
  // see the note in BEHAVIOUR_REVIEW_API.md §6.
  behaviourDecide: (id, decision, payload) =>
    request(`/api/v1/behaviour/review-queue/${encodeURIComponent(id)}/${decision}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  // Is a pipeline pushing frames for this camera right now?
  behaviourLiveStatus: (cameraId) =>
    request(`/api/v1/behaviour/live/status${qs({ camera_id: cameraId })}`),
  // Not fetched — this is an MJPEG stream and the consumer is an <img src>.
  // The cache-buster matters: without it a browser reuses the closed stream
  // from a previous mount and the feed never restarts.
  behaviourLiveUrl: (cameraId, nonce) =>
    `${BASE_URL}/api/v1/behaviour/live${qs({ camera_id: cameraId, t: nonce })}`,

  // --- behavioural analysis ---
  behaviourQueue: (params) => request(`/api/v1/behaviour/review-queue${qs(params)}`),
  behaviourReview: (id) =>
    request(`/api/v1/behaviour/review-queue/${encodeURIComponent(id)}`),
  behaviourHistory: (id) =>
    request(`/api/v1/behaviour/review-queue/${encodeURIComponent(id)}/history`),
  // reviewer_id is supplied by session.jsx and trusted by the API. On an
  // identification decision that field is the audit trail's only signature —
  // see the note in BEHAVIOUR_REVIEW_API.md §6.
  behaviourDecide: (id, decision, payload) =>
    request(`/api/v1/behaviour/review-queue/${encodeURIComponent(id)}/${decision}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  // Is a pipeline pushing frames for this camera right now?
  behaviourLiveStatus: (cameraId) =>
    request(`/api/v1/behaviour/live/status${qs({ camera_id: cameraId })}`),
  // Not fetched — this is an MJPEG stream and the consumer is an <img src>.
  // The cache-buster matters: without it a browser reuses the closed stream
  // from a previous mount and the feed never restarts.
  behaviourLiveUrl: (cameraId, nonce) =>
    `${BASE_URL}/api/v1/behaviour/live${qs({ camera_id: cameraId, t: nonce })}`,

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
  // Victoria's claim-risk scorer (assessor-facing, no UI on it right now).
  safetyScore: (claimId) => request(`/api/safety-score/${encodeURIComponent(claimId)}`),
  // The member-facing reward score — a different thing entirely, see
  // services/member_score_service.py.
  memberSafetyScore: (memberId) =>
    request(`/api/members/${encodeURIComponent(memberId)}/safety-score`),
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