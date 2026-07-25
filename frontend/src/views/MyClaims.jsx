import { useCallback, useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { api, formatDate, formatDateTime, money } from '../api'
import { useSession } from '../session'
import StatusPill from '../components/StatusPill'

/* The member's own claims, and — crucially — the reason when one is declined.
   This is where the "member is notified of the rejection with a reason"
   requirement actually lands for the demo; real push delivery is Phase 4. */

export default function MyClaims() {
  const { member } = useSession()
  const location = useLocation()
  const justSubmitted = location.state?.justSubmitted

  const [claims, setClaims] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [openId, setOpenId] = useState(null)
  const [detail, setDetail] = useState(null)

  const load = useCallback(() => {
    if (!member) return
    setLoading(true)
    api
      .listClaims({ member_id: member.member_id })
      .then((data) => setClaims(data.claims))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [member])

  useEffect(load, [load])

  // Poll so a member watching this page sees an assessor's decision arrive.
  useEffect(() => {
    const timer = setInterval(load, 15000)
    return () => clearInterval(timer)
  }, [load])

  function toggle(incident) {
    if (openId === incident) {
      setOpenId(null)
      setDetail(null)
      return
    }
    setOpenId(incident)
    setDetail(null)
    api
      .claim(incident)
      .then((data) => setDetail(data.claim))
      .catch(() => setDetail(null))
  }

  if (!member) return <p className="muted">Loading your details…</p>

  const declined = claims.filter((c) => c.status === 'denied')

  return (
    <>
      <h1>My claims</h1>
      <p className="muted" style={{ margin: '2px 0 18px' }}>
        Reports submitted by {member.name}.{' '}
        <Link to="/report">Report a new incident</Link>
      </p>

      {justSubmitted ? (
        <div className="banner banner-good" style={{ marginBottom: 16 }}>
          Report <strong>{justSubmitted}</strong> submitted. A Discovery assessor will review it —
          the outcome will show here.
        </div>
      ) : null}

      {declined.length ? (
        <div className="banner banner-bad" style={{ marginBottom: 16 }} role="status">
          {declined.length === 1
            ? 'One of your claims was declined. Open it below to see the reason given.'
            : `${declined.length} of your claims were declined. Open them below to see the reasons given.`}
        </div>
      ) : null}

      {error ? (
        <div className="banner banner-bad" role="alert">
          {error}
        </div>
      ) : null}

      {loading && !claims.length ? <p className="muted">Loading…</p> : null}

      {!loading && !claims.length ? (
        <div className="card" style={{ padding: 28, textAlign: 'center' }}>
          <p style={{ margin: 0 }}>You haven't submitted any reports yet.</p>
          <p className="muted" style={{ marginTop: 6 }}>
            <Link to="/report">Report an incident</Link> and it will appear here.
          </p>
        </div>
      ) : null}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {claims.map((claim) => {
          const open = openId === claim.Incident
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
                <button type="button" className="btn" onClick={() => toggle(claim.Incident)}>
                  {open ? 'Hide' : 'Details'}
                </button>
              </div>

              <p className="tiny" style={{ margin: '8px 0 0' }}>
                Incident {formatDateTime(claim.INCIDENT_DATE_TIME)} · submitted{' '}
                {formatDate(claim.submitted_at)}
              </p>

              {claim.status === 'denied' && claim.denial_reason ? (
                <div className="banner banner-bad" style={{ marginTop: 12 }}>
                  <strong>Why this was declined</strong>
                  <p style={{ margin: '4px 0 0' }}>{claim.denial_reason}</p>
                  {claim.reviewed_by_name ? (
                    <p className="tiny" style={{ margin: '6px 0 0' }}>
                      Reviewed by {claim.reviewed_by_name} on {formatDate(claim.reviewed_at)}
                    </p>
                  ) : null}
                </div>
              ) : null}

              {claim.status === 'approved' ? (
                <div className="banner banner-good" style={{ marginTop: 12 }}>
                  Approved{claim.reviewed_by_name ? ` by ${claim.reviewed_by_name}` : ''} on{' '}
                  {formatDate(claim.reviewed_at)}. This incident now contributes to the crime
                  hot-spot map for {claim.SUBURB}.
                  {claim.reviewer_note ? (
                    <p style={{ margin: '4px 0 0' }}>{claim.reviewer_note}</p>
                  ) : null}
                </div>
              ) : null}

              {open ? (
                <div style={{ marginTop: 14, borderTop: '1px solid var(--hairline)', paddingTop: 12 }}>
                  <p style={{ margin: 0, fontSize: 14 }}>{claim.description}</p>
                  <p className="tiny" style={{ marginTop: 10 }}>
                    {claim.ITEM_CATEGORY}
                    {claim.VEHICLE_MAKE
                      ? ` · ${claim.VEHICLE_MAKE} ${claim.VEHICLE_MODEL || ''} ${claim.VEHICLE_YEAR || ''}`
                      : ''}
                    {' · '}
                    Door camera footage:{' '}
                    {claim.camera_consent ? 'permission given' : 'not shared'}
                  </p>
                  <MediaStrip media={detail?.media} loading={!detail} />
                </div>
              ) : null}
            </article>
          )
        })}
      </div>
    </>
  )
}

export function MediaStrip({ media, loading }) {
  if (loading) return <p className="tiny" style={{ marginTop: 10 }}>Loading attachments…</p>
  if (!media || !media.length)
    return <p className="tiny" style={{ marginTop: 10 }}>No photos or video attached.</p>

  return (
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 12 }}>
      {media.map((item) =>
        item.kind === 'video' ? (
          <video
            key={item.blob}
            src={item.url}
            controls
            preload="metadata"
            style={{ width: 220, borderRadius: 8, border: '1px solid var(--border)' }}
          />
        ) : (
          <a key={item.blob} href={item.url} target="_blank" rel="noreferrer">
            <img
              src={item.url}
              alt={item.filename}
              style={{
                width: 120,
                height: 90,
                objectFit: 'cover',
                borderRadius: 8,
                border: '1px solid var(--border)',
                background: 'var(--page)',
              }}
            />
          </a>
        ),
      )}
    </div>
  )
}
