import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, num } from '../api'
import { useSession } from '../session'

/* Guardian Safety Score — the member's reward dashboard.

   Colour: each category takes one stop of the brand spectrum, in fixed order.
   That's a categorical use, so the hue is never the only signal — every card
   carries its label, its points and a "x of y" figure, and the meter beneath it
   repeats the same ratio. A colourblind reader loses nothing.

   The score is computed server-side (services/member_score_service.py). Nothing
   here does arithmetic on points, so the dashboard can't drift from the API. */

const CATEGORY_COLOR = {
  app_activity: 'var(--brand-1)',
  route_optimisation: 'var(--brand-2)',
  camera: 'var(--brand-3)',
  claims_response: 'var(--brand-5)',
}

const TIER_PILL = {
  Bronze: 'pill-denied',
  Silver: 'pill-pending',
  Gold: 'pill-pending',
  Platinum: 'pill-approved',
}

const MILES_LOGO_SRC = '/discovery-miles-logo.png?v=4'

export default function MemberSafetyScore() {
  const { member } = useSession()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [open, setOpen] = useState(null)
  const [showMilesLogo, setShowMilesLogo] = useState(true)

  const load = useCallback(() => {
    if (!member) return
    setLoading(true)
    api
      .memberSafetyScore(member.member_id)
      .then((result) => {
        setData(result)
        setError(null)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [member])

  useEffect(load, [load])

  if (!member) return <p className="muted">Loading…</p>
  if (loading && !data) return <p className="muted">Working out your score…</p>
  if (error)
    return (
      <div className="banner banner-bad" role="alert">
        {error}
      </div>
    )
  if (!data) return null

  return (
    <>
      <h1>Your Safety Score</h1>
      <p className="muted" style={{ margin: '2px 0 18px' }}>
        Earned from verified activity — every point below is traceable to something you actually
        did.
      </p>

      {/* Hero */}
      <section className="card score-card" style={{ marginBottom: 16 }}>
        <div className="score-hero">
          <ScoreRing categories={data.categories} score={data.score} />

          <div style={{ flex: 1, minWidth: 260 }}>
            <div
              className="tiny"
              style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}
            >
              Safety score
            </div>
            <p className="muted" style={{ margin: '6px 0 12px', maxWidth: 420 }}>
              Built from app activity, safe route choices, linked cameras and claims response —
              recalculated from your verified activity log.
            </p>
            <span className={`pill ${TIER_PILL[data.tier] || 'pill-pending'}`}>
              <span className="dot" aria-hidden="true" />
              {data.tier} tier
            </span>
          </div>

          <div className="safety-miles" style={{ textAlign: 'right', minWidth: 170 }}>
            <div className="safety-miles-brand">
              {showMilesLogo ? (
                <img
                  src={MILES_LOGO_SRC}
                  alt="Discovery Miles"
                  className="safety-miles-logo"
                  onError={() => setShowMilesLogo(false)}
                />
              ) : (
                <span className="safety-miles-fallback">Discovery Miles</span>
              )}
            </div>
            <div
              style={{
                fontSize: 36,
                fontWeight: 700,
                letterSpacing: '-0.02em',
                color: 'var(--reward)',
                lineHeight: 1.1,
              }}
            >
              {num.format(data.miles)} miles
            </div>
            <div className="muted" style={{ fontSize: 12.5 }}>
              Discovery Miles earned
            </div>
            <div className="tiny" style={{ marginTop: 3 }}>
              {data.miles_per_point} miles per safety point
            </div>
          </div>
        </div>

        {/* Tier progress */}
        <div style={{ marginTop: 22 }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 12.5,
              marginBottom: 6,
            }}
          >
            <span className="muted">Progress toward next tier</span>
            <span className="muted">
              {data.next_tier
                ? `${data.points_to_next_tier} points to ${data.next_tier}`
                : 'Top tier reached'}
            </span>
          </div>
          <div
            style={{
              height: 10,
              borderRadius: 999,
              background: 'var(--hairline)',
              overflow: 'hidden',
            }}
            role="meter"
            aria-valuenow={Math.round(data.tier_progress * 100)}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`Tier progress: ${Math.round(data.tier_progress * 100)} percent`}
          >
            <div
              style={{
                width: `${Math.max(2, data.tier_progress * 100)}%`,
                height: '100%',
                background: 'var(--brand-gradient)',
                borderRadius: 999,
              }}
            />
          </div>
        </div>
      </section>

      {/* Category KPI cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: 12,
          marginBottom: 16,
        }}
        className="grid-4-tiles"
      >
        {data.categories.map((c) => {
          const color = CATEGORY_COLOR[c.key] || 'var(--accent)'
          const expanded = open === c.key
          return (
            <div
              key={c.key}
              className="card"
              style={{ padding: '14px 15px', borderTop: `3px solid ${color}` }}
            >
              <div
                className="tiny"
                style={{
                  fontWeight: 650,
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  minHeight: 28,
                }}
              >
                {c.label}
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 5, marginTop: 4 }}>
                <span className="tile-v" style={{ fontWeight: 700 }}>
                  {c.points}
                </span>
                <span className="muted" style={{ fontSize: 13 }}>
                  / {c.max}
                </span>
              </div>

              <div
                style={{
                  height: 6,
                  borderRadius: 999,
                  background: 'var(--hairline)',
                  overflow: 'hidden',
                  margin: '10px 0 6px',
                }}
              >
                <div
                  style={{
                    width: `${c.share * 100}%`,
                    height: '100%',
                    background: color,
                    borderRadius: 999,
                  }}
                />
              </div>
              <div className="tiny">{Math.round(c.share * 100)}% of this category's cap</div>

              <button
                type="button"
                className="btn"
                style={{ marginTop: 10, width: '100%', fontSize: 12.5, padding: '5px 8px' }}
                aria-expanded={expanded}
                onClick={() => setOpen(expanded ? null : c.key)}
              >
                {expanded ? 'Hide detail' : 'How this was earned'}
              </button>

              {expanded ? (
                <div style={{ marginTop: 10, borderTop: '1px solid var(--hairline)', paddingTop: 9 }}>
                  <p className="tiny" style={{ margin: '0 0 8px' }}>
                    {c.hint}
                  </p>
                  {c.contributions.map((e) => (
                    <div key={e.label} style={{ marginBottom: 7 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                        <span style={{ fontSize: 12.5 }}>{e.label}</span>
                        <span
                          style={{
                            fontSize: 12.5,
                            fontWeight: 650,
                            fontVariantNumeric: 'tabular-nums',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {e.points}/{e.max}
                        </span>
                      </div>
                      <div className="tiny">{e.detail}</div>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          )
        })}
      </div>

      {/* What's still available */}
      {data.opportunities.length ? (
        <section className="card panel">
          <h2 style={{ fontSize: 15 }}>Earn more</h2>
          <p className="muted" style={{ margin: '4px 0 12px' }}>
            {num.format(data.max_score - data.score)} points still available, worth{' '}
            {num.format((data.max_score - data.score) * data.miles_per_point)} miles.
          </p>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'grid', gap: 8 }}>
            {data.opportunities.map((o) => (
              <li
                key={o.label}
                style={{
                  background: 'var(--page)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '11px 13px',
                  display: 'flex',
                  gap: 12,
                  alignItems: 'center',
                  flexWrap: 'wrap',
                }}
              >
                <strong style={{ fontSize: 13.5 }}>{o.label}</strong>
                <span className="muted" style={{ fontSize: 13 }}>
                  {o.hint}
                </span>
                <span className="spacer" />
                <span
                  style={{
                    fontWeight: 650,
                    fontVariantNumeric: 'tabular-nums',
                    whiteSpace: 'nowrap',
                  }}
                >
                  +{o.available} pts
                </span>
              </li>
            ))}
          </ul>
          <p className="tiny" style={{ marginTop: 12 }}>
            <Link to="/route">Plan a safer route</Link> ·{' '}
            <Link to="/report">Report an incident</Link> ·{' '}
            <Link to="/profile">Update your profile</Link>
          </p>
        </section>
      ) : null}

      <p className="tiny" style={{ marginTop: 14, maxWidth: 760 }}>
        Camera and claims points come from your actual claims record. App activity and route
        points come from your engagement log. Your score never affects whether a claim is
        approved.
      </p>
    </>
  )
}

/* One ring, four arcs — a part-to-whole of where the score came from.
   Four segments is well inside the readable limit, and each is labelled in the
   cards below, so the ring is a summary rather than the only place to read it. */
function ScoreRing({ categories, score }) {
  const size = 150
  const stroke = 14
  const radius = (size - stroke) / 2
  const circumference = 2 * Math.PI * radius

  let offset = 0
  const arcs = categories.map((c) => {
    const length = (c.points / 100) * circumference
    const arc = { key: c.key, length, offset, color: CATEGORY_COLOR[c.key] || 'var(--accent)' }
    offset += length
    return arc
  })

  return (
    <div style={{ position: 'relative', width: size, height: size, flex: 'none' }}>
      <svg
        width={size}
        height={size}
        role="img"
        aria-label={`Safety score ${score} out of 100`}
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--hairline)"
          strokeWidth={stroke}
        />
        {arcs.map((a) => (
          <circle
            key={a.key}
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={a.color}
            strokeWidth={stroke}
            strokeDasharray={`${a.length} ${circumference - a.length}`}
            strokeDashoffset={-a.offset}
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
          />
        ))}
      </svg>
      <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center' }}>
        <div style={{ textAlign: 'center', lineHeight: 1.05 }}>
          <div style={{ fontSize: 38, fontWeight: 700, letterSpacing: '-0.02em' }}>
            {num.format(score)}
          </div>
          <div className="tiny">/ 100</div>
        </div>
      </div>
    </div>
  )
}
