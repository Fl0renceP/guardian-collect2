import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, formatDateTime, num } from '../api'
import LoadingGraphic from '../components/LoadingGraphic'
import { useSession } from '../session'

/* Crime Prevention Unit alert feed.

   Severity uses the reserved status palette and is always paired with a text
   label, so the state never depends on colour alone.

   The "sources" strip is deliberate: two of the four feeds don't exist yet
   (face matches, predicted risk). Showing them as explicitly not-wired is more
   useful than hiding them — an operator seeing no detections should know it's
   because the detector isn't built, not because nobody was detected. */

const SEVERITY_ORDER = { critical: 0, serious: 1, warning: 2, info: 3 }

const SEVERITY_STYLE = {
  critical: { pill: 'pill-denied', label: 'Critical' },
  serious: { pill: 'pill-pending', label: 'Serious' },
  warning: { pill: 'pill-pending', label: 'Warning' },
  info: { pill: 'pill-approved', label: 'Info' },
}

const KIND_LABEL = {
  incident: 'Confirmed incident',
  submission: 'Member report',
  detection: 'Person match',
  predicted: 'Predicted risk',
}

const WINDOWS = [
  { days: 30, label: '30 days' },
  { days: 90, label: '90 days' },
  { days: 180, label: '6 months' },
  { days: 365, label: '12 months' },
]

export default function AlertsFeed() {
  const { role, unit, member } = useSession()
  const isMember = role === 'member'

  const [data, setData] = useState(null)
  const [days, setDays] = useState(90)
  const [severities, setSeverities] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const load = useCallback(() => {
    const params = { since_days: days, limit: 200 }
    if (isMember) {
      if (!member) return
      // The backend decides whether to scope by home — it checks the member's
      // opt-in, so a member who declined simply gets the national feed.
      Object.assign(params, { audience: 'member', member_id: member.member_id })
    } else {
      if (!unit) return
      Object.assign(params, { audience: 'cpu', unit_id: unit.unit_id })
    }
    setLoading(true)
    api
      .alerts(params)
      .then((result) => {
        setData(result)
        setError(null)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [isMember, unit, member, days])

  useEffect(load, [load])

  // A unit watching this screen should see new reports arrive.
  useEffect(() => {
    const timer = setInterval(load, 20000)
    return () => clearInterval(timer)
  }, [load])

  if (isMember ? !member : !unit) return <LoadingGraphic label="Loading audience…" />

  const all = data?.alerts || []
  const shown = severities.length ? all.filter((a) => severities.includes(a.severity)) : all
  const sorted = [...shown].sort(
    (a, b) =>
      (SEVERITY_ORDER[a.severity] ?? 9) - (SEVERITY_ORDER[b.severity] ?? 9) ||
      String(b.at).localeCompare(String(a.at)),
  )

  const counts = data?.summary?.by_severity || {}

  const toggleSeverity = (s) =>
    setSeverities(severities.includes(s) ? severities.filter((x) => x !== s) : [...severities, s])

  return (
    <>
      <h1>Alerts</h1>
      <p className="muted" style={{ margin: '2px 0 16px' }}>
        {isMember
          ? data?.scoped_to_location
            ? `Incidents within ${data.radius_km} km of your home.`
            : 'Incidents across South Africa.'
          : `${unit.name} · ${unit.kind} · operating area ${unit.radius_km} km around ${unit.base_suburb
              .toLowerCase()
              .replace(/\b\w/g, (c) => c.toUpperCase())}`}
      </p>

      {isMember && data && !data.scoped_to_location ? (
        <div className="banner banner-info" style={{ marginBottom: 16 }}>
          You're seeing alerts from across the country.{' '}
          <Link to="/profile">Add a home location</Link> to see only what's near you — it's optional
          and you can remove it at any time.
        </div>
      ) : null}

      {error ? (
        <div className="banner banner-bad" role="alert" style={{ marginBottom: 16 }}>
          {error}
        </div>
      ) : null}

      <div className="card filters">
        <div className="field" style={{ gap: 7 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Window
          </span>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {WINDOWS.map((w) => (
              <button
                key={w.days}
                type="button"
                className={`btn btn-chip${days === w.days ? ' btn-primary' : ''}`}
                onClick={() => setDays(w.days)}
              >
                {w.label}
              </button>
            ))}
          </div>
        </div>

        <div className="field" style={{ gap: 7 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Severity
          </span>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            <button
              type="button"
              className={`btn btn-chip${severities.length === 0 ? ' btn-primary' : ''}`}
              onClick={() => setSeverities([])}
            >
              All
            </button>
            {Object.entries(counts).map(([sev, n]) => (
              <button
                key={sev}
                type="button"
                className={`btn btn-chip${severities.includes(sev) ? ' btn-primary' : ''}`}
                onClick={() => toggleSeverity(sev)}
              >
                {SEVERITY_STYLE[sev]?.label || sev}
                <span style={{ marginLeft: 5, opacity: 0.7, fontSize: 11.5 }}>{num.format(n)}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Honest source status — see the note at the top of this file. */}
      {data?.sources ? (
        <div
          className="card"
          style={{ padding: '10px 14px', marginBottom: 16, display: 'flex', gap: 18, flexWrap: 'wrap' }}
        >
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Feeds
          </span>
          {data.sources.map((s) => (
            <span key={s.source} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span
                aria-hidden="true"
                style={{
                  width: 7,
                  height: 7,
                  borderRadius: '50%',
                  background: s.live ? 'var(--good)' : 'var(--ink-muted)',
                }}
              />
              <span className="muted" style={{ fontSize: 12.5 }}>
                {s.label}
                {s.live ? '' : ` — ${s.note}`}
              </span>
            </span>
          ))}
        </div>
      ) : null}

      {loading && !data ? <LoadingGraphic label="Loading alerts…" /> : null}

      {data && !sorted.length ? (
        <div className="card empty">
          <p style={{ margin: 0 }}>
            No alerts in this window{isMember ? ' near you' : ` for ${unit.name}`}.
          </p>
          <p className="muted" style={{ marginTop: 6 }}>
            Try a longer window — serious incidents are rare enough that 30 days around one area is
            often empty.
          </p>
        </div>
      ) : null}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {sorted.map((alert) => {
          const style = SEVERITY_STYLE[alert.severity] || SEVERITY_STYLE.info
          return (
            <article key={alert.id} className="card list-card roomy-card">
              <div className="card-head">
                <span className={`pill ${style.pill}`}>
                  <span className="dot" aria-hidden="true" />
                  {style.label}
                </span>
                <strong className="alert-title">{alert.title}</strong>
                <span className="tiny alert-kind">{KIND_LABEL[alert.kind] || alert.kind}</span>
                <span className="spacer" />
                {alert.distance_km != null ? (
                  <span className="muted alert-distance" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {alert.distance_km} km away
                  </span>
                ) : null}
              </div>

              <p className="alert-detail">{alert.detail}</p>
              <p className="tiny alert-meta">
                {alert.suburb ? `${alert.suburb} · ` : ''}
                {formatDateTime(alert.at)}
                {alert.meta?.member ? ` · reported by ${alert.meta.member}` : ''}
              </p>
            </article>
          )
        })}
      </div>

      <p className="tiny" style={{ marginTop: 14, maxWidth: 760 }}>
        {isMember
          ? 'You only ever receive alerts about confirmed offenders — never unverified suspects, so a possible match doesn’t become a false alarm.'
          : 'Crime Prevention Units see confirmed incidents, member reports awaiting review, and — once the detector is live — both offender and suspect matches. Discovery members only ever receive offender matches, to avoid raising an alarm over an unverified suspect.'}
      </p>
    </>
  )
}
