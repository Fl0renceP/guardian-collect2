import { useCallback, useEffect, useState } from 'react'
import { api, ApiError, formatDate, formatDateTime, money, num } from '../api'
import LoadingGraphic from '../components/LoadingGraphic'
import { useSession } from '../session'
import StatusPill from '../components/StatusPill'
import { MediaStrip } from './MyClaims'

const TABS = [
  { key: 'pending', label: 'Awaiting review' },
  { key: 'approved', label: 'Approved' },
  { key: 'denied', label: 'Declined' },
]

export default function ReviewQueue() {
  const { employee } = useSession()
  const [tab, setTab] = useState('pending')
  const [claims, setClaims] = useState([])
  const [counts, setCounts] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [openId, setOpenId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [busyId, setBusyId] = useState(null)
  const [denyFor, setDenyFor] = useState(null)
  const [denyReason, setDenyReason] = useState('')
  const [denyError, setDenyError] = useState(null)
  const [note, setNote] = useState('')
  const [banner, setBanner] = useState(null)

  const load = useCallback(() => {
    setLoading(true)
    Promise.all([api.listClaims({ status: tab }), api.claimCounts()])
      .then(([list, c]) => {
        setClaims(list.claims)
        setCounts(c)
        setError(null)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [tab])

  useEffect(load, [load])

  // New submissions should surface without the assessor reloading.
  useEffect(() => {
    const timer = setInterval(load, 15000)
    return () => clearInterval(timer)
  }, [load])

  function openDetail(incident) {
    if (openId === incident) {
      setOpenId(null)
      setDetail(null)
      return
    }
    setOpenId(incident)
    setDetail(null)
    setNote('')
    api
      .claim(incident)
      .then((data) => setDetail(data.claim))
      .catch(() => setDetail(null))
  }

  async function approve(incident) {
    setBusyId(incident)
    setBanner(null)
    try {
      await api.approveClaim(incident, { employee_id: employee.employee_id, note: note || undefined })
      setBanner({
        kind: 'good',
        text: `${incident} approved — it now counts toward the hot-spot map.`,
      })
      setOpenId(null)
      setDetail(null)
      load()
    } catch (err) {
      setBanner({ kind: 'bad', text: err.message })
    } finally {
      setBusyId(null)
    }
  }

  async function deny(incident) {
    setBusyId(incident)
    setDenyError(null)
    try {
      await api.denyClaim(incident, {
        employee_id: employee.employee_id,
        denial_reason: denyReason,
      })
      setBanner({ kind: 'good', text: `${incident} declined — the member will see your reason.` })
      setDenyFor(null)
      setDenyReason('')
      setOpenId(null)
      load()
    } catch (err) {
      if (err instanceof ApiError && err.fields?.denial_reason) {
        setDenyError(err.fields.denial_reason)
      } else {
        setDenyError(err.message)
      }
    } finally {
      setBusyId(null)
    }
  }

  if (!employee) return <LoadingGraphic label="Loading assessor profile…" />

  return (
    <>
      <h1>Claims review</h1>
      <p className="muted" style={{ margin: '2px 0 18px' }}>
        Reviewing as <strong>{employee.name}</strong> · {employee.role}. Approving a claim adds it
        to the claims dataset and the crime hot-spot map.
      </p>

      {banner ? (
        <div className={`banner banner-${banner.kind}`} style={{ marginBottom: 16 }} role="status">
          {banner.text}
        </div>
      ) : null}

      <div className="nav" style={{ marginBottom: 16 }}>
        {TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            className={`btn${tab === t.key ? ' btn-primary' : ''}`}
            onClick={() => setTab(t.key)}
          >
            {t.label}
            {counts[t.key] != null ? (
              <span style={{ marginLeft: 6, fontVariantNumeric: 'tabular-nums', opacity: 0.85 }}>
                {num.format(counts[t.key])}
              </span>
            ) : null}
          </button>
        ))}
      </div>

      {error ? (
        <div className="banner banner-bad" role="alert">
          {error}
        </div>
      ) : null}

      {loading && !claims.length ? <LoadingGraphic label="Loading review queue…" /> : null}

      {!loading && !claims.length ? (
        <div className="card" style={{ padding: 28, textAlign: 'center' }}>
          <p style={{ margin: 0 }}>
            {tab === 'pending' ? 'No claims are waiting for review.' : `No ${tab} claims.`}
          </p>
        </div>
      ) : null}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {claims.map((claim) => {
          const open = openId === claim.Incident
          const busy = busyId === claim.Incident
          return (
            <article key={claim.Incident} className="card" style={{ padding: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                <StatusPill status={claim.status} />
                <strong style={{ fontSize: 14.5 }}>
                  {claim.PERIL} · {claim.SUBURB}
                </strong>
                <span className="tiny">{claim.Incident}</span>
                <span className="spacer" />
                <span className="muted">{money.format(claim.CLAIM_AMOUNT || 0)}</span>
                <button type="button" className="btn" onClick={() => openDetail(claim.Incident)}>
                  {open ? 'Hide' : 'Review'}
                </button>
              </div>

              <p className="tiny" style={{ margin: '8px 0 0' }}>
                <strong>{claim.member_name}</strong> · policy {claim.policy_number} ·{' '}
                {claim.member_email} · {claim.member_phone}
              </p>
              <p className="tiny" style={{ margin: '3px 0 0' }}>
                Incident {formatDateTime(claim.INCIDENT_DATE_TIME)} · submitted{' '}
                {formatDate(claim.submitted_at)}
                {claim.reviewed_at
                  ? ` · reviewed ${formatDate(claim.reviewed_at)} by ${claim.reviewed_by_name || claim.reviewed_by}`
                  : ''}
              </p>

              {claim.status === 'denied' && claim.denial_reason ? (
                <p className="muted" style={{ marginTop: 8 }}>
                  <strong>Reason given:</strong> {claim.denial_reason}
                </p>
              ) : null}

              {open ? (
                <div style={{ marginTop: 14, borderTop: '1px solid var(--hairline)', paddingTop: 14 }}>
                  <h2 style={{ fontSize: 14 }}>Member's account</h2>
                  <p style={{ margin: '6px 0 0', fontSize: 14 }}>{claim.description}</p>

                  <dl
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'auto 1fr',
                      gap: '4px 14px',
                      margin: '14px 0 0',
                      fontSize: 13.5,
                    }}
                  >
                    <dt className="tiny">Category</dt>
                    <dd style={{ margin: 0 }}>{claim.ITEM_CATEGORY}</dd>
                    {claim.VEHICLE_MAKE ? (
                      <>
                        <dt className="tiny">Vehicle</dt>
                        <dd style={{ margin: 0 }}>
                          {claim.VEHICLE_MAKE} {claim.VEHICLE_MODEL || ''} {claim.VEHICLE_YEAR || ''}
                        </dd>
                      </>
                    ) : null}
                    <dt className="tiny">Door camera</dt>
                    <dd style={{ margin: 0 }}>
                      {claim.camera_consent ? (
                        <>
                          <strong>Permission given</strong>{' '}
                          <span className="tiny">
                            on {formatDate(claim.camera_consent_at)} — footage may be reviewed for
                            this incident only
                          </span>
                        </>
                      ) : (
                        <span className="muted">
                          Not shared — do not pull footage for this claim
                        </span>
                      )}
                    </dd>
                  </dl>

                  <h2 style={{ fontSize: 14, marginTop: 16 }}>Evidence</h2>
                  <MediaStrip media={detail?.media} loading={!detail} />

                  {claim.status === 'pending' ? (
                    <div style={{ marginTop: 18 }}>
                      {denyFor === claim.Incident ? (
                        <div className="field">
                          <label htmlFor={`reason-${claim.Incident}`}>
                            Reason for declining — the member will see this
                          </label>
                          <textarea
                            id={`reason-${claim.Incident}`}
                            value={denyReason}
                            onChange={(e) => {
                              setDenyReason(e.target.value)
                              setDenyError(null)
                            }}
                            placeholder="e.g. The incident date falls outside the policy cover period."
                          />
                          {denyError ? (
                            <span className="err" role="alert">
                              {denyError}
                            </span>
                          ) : null}
                          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                            <button
                              type="button"
                              className="btn btn-danger"
                              disabled={busy}
                              onClick={() => deny(claim.Incident)}
                            >
                              {busy ? 'Declining…' : 'Confirm decline'}
                            </button>
                            <button
                              type="button"
                              className="btn"
                              onClick={() => {
                                setDenyFor(null)
                                setDenyReason('')
                                setDenyError(null)
                              }}
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : (
                        <>
                          <div className="field" style={{ marginBottom: 12 }}>
                            <label htmlFor={`note-${claim.Incident}`}>
                              Assessor note (optional)
                            </label>
                            <input
                              id={`note-${claim.Incident}`}
                              type="text"
                              value={note}
                              onChange={(e) => setNote(e.target.value)}
                              placeholder="e.g. Verified against doorbell footage."
                            />
                          </div>
                          <div style={{ display: 'flex', gap: 8 }}>
                            <button
                              type="button"
                              className="btn btn-good"
                              disabled={busy}
                              onClick={() => approve(claim.Incident)}
                            >
                              {busy ? 'Approving…' : 'Approve claim'}
                            </button>
                            <button
                              type="button"
                              className="btn btn-danger"
                              onClick={() => setDenyFor(claim.Incident)}
                            >
                              Decline…
                            </button>
                          </div>
                          <p className="tiny" style={{ marginTop: 8 }}>
                            Approving adds this incident to the claims dataset and the hot-spot map.
                            Declining requires a reason, which is shown to the member.
                          </p>
                        </>
                      )}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </article>
          )
        })}
      </div>
    </>
  )
}
