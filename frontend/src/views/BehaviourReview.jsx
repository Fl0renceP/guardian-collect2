/* Behavioural review queue — reads live.
 *
 * The human review step between the two signals: facial recognition says who it
 * thinks this is, behavioural analysis says what they were doing, and a person
 * decides. Contract: BEHAVIOUR_REVIEW_API.md.
 *
 * WIRED (steps 1-3): the queue and each card read from
 * GET /api/v1/behaviour/review-queue[/{id}]. No mock data remains.
 *
 * NOT WIRED YET (step 4): confirm / deny. The buttons are disabled and say so.
 * They are not left clickable with a fake success message — a reviewer who
 * believes they have actioned a flag when nothing was recorded is worse off
 * than one who can see the feature is unfinished.
 *
 * NOT WIRED YET (step 5): clip and live video. The panel says which is missing
 * rather than showing a placeholder that implies footage exists.
 */

import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import { useSession } from '../session'

const TABS = [
  { key: 'pending', label: 'Awaiting review' },
  { key: 'confirmed', label: 'Confirmed' },
  { key: 'denied', label: 'False flags' },
]

const HEURISTIC_LABELS = {
  loitering: 'Loitering / casing',
  perimeter_probing: 'Perimeter probing',
  climbing_posture: 'Climbing posture',
  concealment_approach: 'Face not visible while approaching',
  crouched_near_vehicle: 'Crouched at a vehicle',
  tampering_motion: 'Tampering motion',
  group_coordination: 'Lookout + actor',
  fleeing: 'Sudden change of pace',
}

const LABEL_TEXT = {
  offender: 'Offender',
  suspect: 'Suspect',
  verified: 'Known resident',
}

const STATUS_PILL = {
  pending: { cls: 'pending', text: 'Awaiting review' },
  confirmed: { cls: 'approved', text: 'Confirmed' },
  denied: { cls: 'denied', text: 'False flag' },
}

const pct = (value) => (value == null ? null : Math.round(value * 100))

function riskTone(score) {
  if (score >= 0.5) return 'critical'
  if (score >= 0.3) return 'warning'
  return 'good'
}

function formatTime(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString()
}

/* Score bar. The number is always written out beside it — colour is never the
   only signal, matching StatusPill's reasoning for colourblind users. */
function ScoreBar({ label, value, tone, absentText }) {
  if (value == null) {
    return (
      <div style={{ marginBottom: 10 }}>
        <div className="tiny" style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>{label}</span>
          <em>{absentText || 'none'}</em>
        </div>
        <div style={{ height: 8, borderRadius: 4, background: 'var(--hairline)', marginTop: 4 }} />
      </div>
    )
  }
  return (
    <div style={{ marginBottom: 10 }}>
      <div className="tiny" style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span>{label}</span>
        <strong style={{ fontVariantNumeric: 'tabular-nums' }}>{pct(value)}%</strong>
      </div>
      <div
        style={{ height: 8, borderRadius: 4, background: 'var(--hairline)', overflow: 'hidden', marginTop: 4 }}
      >
        <i style={{ display: 'block', height: '100%', width: `${pct(value)}%`, background: `var(--${tone})` }} />
      </div>
    </div>
  )
}

/* The camera as it is RIGHT NOW, with the module's own annotations.
 *
 * This is not evidence of the flag and must never be mistaken for it. The card
 * describes something that happened at a fixed moment; this shows whoever is in
 * front of the camera at the moment you are reading. Frequently a different
 * person entirely. Hence the timestamp, the wording, and the deliberate visual
 * distance from the clip below — a reviewer who confuses the two is making an
 * identification about the wrong human being.
 */
function LiveFeed({ cameraId, occurredAt }) {
  const [live, setLive] = useState(null)
  const [nonce, setNonce] = useState(0)
  const [broken, setBroken] = useState(false)

  useEffect(() => {
    if (!cameraId) return undefined
    let alive = true

    const check = () =>
      api
        .behaviourLiveStatus(cameraId)
        .then((status) => {
          if (!alive) return
          setLive(status.live)
          // A feed that went away and came back needs a new stream URL, not the
          // dead one the browser is still holding.
          if (status.live && broken) {
            setBroken(false)
            setNonce((n) => n + 1)
          }
        })
        .catch(() => alive && setLive(false))

    check()
    const timer = setInterval(check, 5000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [cameraId, broken])

  if (live === null) return null

  if (!live || broken) {
    return (
      <div
        className="tiny"
        style={{
          border: '1px dashed var(--hairline)',
          borderRadius: 'var(--radius-sm)',
          padding: '10px 12px',
          marginBottom: 12,
          opacity: 0.85,
        }}
      >
        <strong>No live feed from {cameraId || 'this camera'}.</strong> Nothing is analysing
        this camera at the moment. The recorded clip below is unaffected — it is what was
        captured when the flag fired.
      </div>
    )
  }

  const flaggedAt = occurredAt ? formatTime(occurredAt) : null

  return (
    <div style={{ marginBottom: 14 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginBottom: 6,
          flexWrap: 'wrap',
        }}
      >
        <span className="pill pill-denied">
          <span className="dot" aria-hidden="true" />
          LIVE — now
        </span>
        <strong className="tiny">{cameraId}</strong>
      </div>

      <img
        key={nonce}
        src={api.behaviourLiveUrl(cameraId, nonce)}
        alt={`Live annotated feed from ${cameraId}, showing detection boxes, pose skeletons and zones`}
        onError={() => setBroken(true)}
        style={{
          width: '100%',
          aspectRatio: '4/3',
          objectFit: 'contain',
          borderRadius: 'var(--radius-sm)',
          border: '2px solid var(--critical)',
          background: '#000',
          display: 'block',
        }}
      />

      <p className="tiny" style={{ margin: '6px 0 0' }}>
        <strong>This is the camera now, not the flagged moment.</strong> The flag was raised
        {flaggedAt ? ` at ${flaggedAt}` : ' earlier'} — whoever is in shot here may have nothing
        to do with it. Boxes, skeletons and zones are the module's own working, shown so you can
        see what it is measuring.
      </p>
    </div>
  )
}

function IdentityPanel({ identity }) {
  if (!identity?.attached) {
    return (
      <div className="card" style={{ background: 'var(--warning-wash)', borderColor: 'var(--warning)' }}>
        <strong style={{ fontSize: 14.5 }}>No facial match</strong>
        <p className="tiny" style={{ margin: '6px 0 0' }}>
          The face was either not visible, not in the registry, or could not be tied to this
          person’s body. <strong>This flag rests on movement alone.</strong> Nothing is known or
          claimed about who this person is.
        </p>
      </div>
    )
  }

  const isResident = identity.label === 'verified'
  const automatic = identity.source === 'automatic'
  const evidence = identity.link_evidence

  return (
    <div
      className="card"
      style={{
        background: isResident ? 'var(--good-wash)' : 'var(--critical-wash)',
        borderColor: isResident ? 'var(--good)' : 'var(--critical)',
      }}
    >
      <div className="card-head" style={{ marginBottom: 6 }}>
        <strong style={{ fontSize: 15 }}>{identity.full_name || 'Matched person'}</strong>
        <span className={`pill pill-${isResident ? 'approved' : 'denied'}`}>
          <span className="dot" aria-hidden="true" />
          {LABEL_TEXT[identity.label] || identity.label}
        </span>
      </div>
      <p className="tiny" style={{ margin: 0 }}>
        {identity.confidence != null
          ? `Facial match ${pct(identity.confidence)}% confident · `
          : ''}
        {identity.first_seen_label}
      </p>

      {automatic ? (
        // The join is geometry, not proof. Saying so — with the evidence — is
        // what keeps a reviewer weighing it rather than reading it as fact.
        <p className="tiny" style={{ margin: '8px 0 0', opacity: 0.9 }}>
          <strong>Attached automatically.</strong> The face was matched by the recognition
          module and tied to this body because the face sat inside it
          {evidence?.time_delta_ms != null ? `, ${evidence.time_delta_ms}ms apart` : ''}
          {evidence?.bodies_considered > 1
            ? `, with ${evidence.bodies_considered} bodies in frame`
            : ''}
          . That is a positional inference, not a confirmed identification — check the footage
          before trusting the name.
        </p>
      ) : null}

      <p className="tiny" style={{ margin: '6px 0 0', opacity: 0.85 }}>
        Identified by the facial recognition module. The behavioural module never sees this name —
        it receives a confidence score only.
      </p>
    </div>
  )
}

function DecisionTrail({ trail }) {
  if (!trail?.length) return null
  const VERB = { confirm: 'Confirmed', deny: 'Recorded as a false flag', reopen: 'Reopened' }
  return (
    <details style={{ marginTop: 10 }}>
      <summary className="tiny" style={{ cursor: 'pointer' }}>
        Decision history ({trail.length})
      </summary>
      <ul className="tiny" style={{ margin: '8px 0 0', paddingLeft: 18 }}>
        {trail.map((d, i) => (
          <li key={i} style={{ marginBottom: 4 }}>
            <strong>{VERB[d.decision] || d.decision}</strong> by {d.reviewer_id} ·{' '}
            {formatTime(d.decided_at)}
            {d.label ? ` · as ${LABEL_TEXT[d.label] || d.label}` : ''}
            {d.reason ? ` · “${d.reason}”` : ''}
            {d.note ? ` · note: “${d.note}”` : ''}
            {d.alerts_sent?.length ? ` · alerted ${d.alerts_sent.join(', ')}` : ''}
          </li>
        ))}
      </ul>
      <p className="tiny" style={{ margin: '6px 0 0', opacity: 0.75 }}>
        Append-only. Undoing a decision adds a line — it never removes one.
      </p>
    </details>
  )
}

/* The registry photo of whoever the face module matched, under the footage.
 *
 * It sits BELOW the video deliberately. The question a reviewer is answering is
 * "is the person in this footage the person in this photograph" — putting the
 * name and face first invites them to read the footage looking for confirmation
 * of a name they have already been given, which is how a wrong match survives
 * review. Footage first, then the claim being made about it.
 */
function MatchedFace({ identity }) {
  const [imageFailed, setImageFailed] = useState(false)

  if (!identity?.attached) return null

  const isResident = identity.label === 'verified'
  const photo = identity.reference_image_url

  return (
    <div
      className="card"
      style={{
        marginTop: 14,
        display: 'flex',
        gap: 12,
        alignItems: 'flex-start',
        background: isResident ? 'var(--good-wash)' : 'var(--critical-wash)',
        borderColor: isResident ? 'var(--good)' : 'var(--critical)',
      }}
    >
      {photo && !imageFailed ? (
        <img
          src={photo}
          alt={`Registry photograph of ${identity.full_name || 'the matched person'}`}
          onError={() => setImageFailed(true)}
          style={{
            width: 96,
            height: 96,
            objectFit: 'cover',
            borderRadius: 'var(--radius-sm)',
            border: '2px solid var(--hairline)',
            flexShrink: 0,
          }}
        />
      ) : (
        <div
          className="tiny"
          style={{
            width: 96,
            height: 96,
            display: 'grid',
            placeItems: 'center',
            textAlign: 'center',
            borderRadius: 'var(--radius-sm)',
            border: '1px dashed var(--hairline)',
            flexShrink: 0,
            padding: 6,
          }}
        >
          No photo on file
        </div>
      )}

      <div style={{ minWidth: 0 }}>
        <div className="card-head" style={{ marginBottom: 4 }}>
          <strong style={{ fontSize: 15 }}>{identity.full_name || 'Matched person'}</strong>
          <span className={`pill pill-${isResident ? 'approved' : 'denied'}`}>
            <span className="dot" aria-hidden="true" />
            {LABEL_TEXT[identity.label] || identity.label}
          </span>
        </div>
        <p className="tiny" style={{ margin: 0 }}>
          Registry photograph{identity.confidence != null
            ? ` · facial match ${pct(identity.confidence)}% confident`
            : ''}
          . <strong>Compare it against the footage above.</strong> The match is what put this
          card in front of you; it is not what settles it.
        </p>
      </div>
    </div>
  )
}

function ReviewDetail({ review, trail, busy, onConfirm, onDeny, onReopen, error }) {
  const { identity, behaviour, decision } = review
  const tone = riskTone(behaviour.composite_risk_score)
  const [denying, setDenying] = useState(false)
  const [reason, setReason] = useState('')
  const [note, setNote] = useState('')

  return (
    <div style={{ marginTop: 14, borderTop: '1px solid var(--hairline)', paddingTop: 14 }}>
      {/* Only shown when nothing is attached — which the conditional filter now
          prevents from reaching the queue at all, so it means a card raised
          before that rule existed. Kept, because those cards are still real. */}
      {!identity.attached ? <IdentityPanel identity={identity} /> : null}

      <h2 style={{ fontSize: 14, margin: identity.attached ? '0 0 8px' : '18px 0 8px' }}>
        1 · What they were doing
      </h2>
      <div className="behaviour-split">
        <div>
          <LiveFeed cameraId={review.camera_id} occurredAt={review.occurred_at || review.opened_at} />

          <h3 className="tiny" style={{ margin: '0 0 6px', textTransform: 'uppercase' }}>
            Recorded — the flagged moment
          </h3>
          {behaviour.clip_url ? (
            <>
              <video
                key={behaviour.clip_url}
                src={behaviour.clip_url}
                controls
                loop
                playsInline
                style={{
                  width: '100%',
                  aspectRatio: '16/10',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--hairline)',
                  background: '#000',
                  objectFit: 'contain',
                }}
              />
              <p className="tiny" style={{ margin: '6px 0 0', opacity: 0.85 }}>
                The seconds around the flag, cut from before it fired. Served over a link that
                expires shortly — it is not a shareable URL, and the footage is deleted on a
                retention schedule.
              </p>
            </>
          ) : (
            <div
              style={{
                position: 'relative',
                aspectRatio: '16/10',
                borderRadius: 'var(--radius-sm)',
                border: '1px dashed var(--hairline)',
                background: 'var(--page)',
                display: 'grid',
                placeItems: 'center',
                padding: 16,
              }}
            >
              <span className="tiny" style={{ textAlign: 'center' }}>
                <strong>No footage attached.</strong>
                <br />
                Either the camera was not buffering, or the upload did not reach storage. The
                explanations beside this panel remain the record of what was observed.
              </span>
            </div>
          )}

          {/* The face goes under the footage, not above it — see MatchedFace. */}
          <MatchedFace identity={identity} />
        </div>

        <div>
          {/* The summary leads, because it is the only part of this card a
              reviewer can disagree with. Scores rank cards; sentences are what
              a person actually weighs. */}
          <h3 className="tiny" style={{ margin: '0 0 6px', textTransform: 'uppercase' }}>
            What was observed
          </h3>
          {behaviour.triggered_heuristics.map((h, i) => (
            <div
              key={h.type}
              style={{
                marginBottom: 12,
                paddingLeft: 10,
                borderLeft: `3px solid var(--${i === 0 ? riskTone(h.confidence ?? 0) : 'hairline'})`,
              }}
            >
              <strong style={{ fontSize: 14 }}>
                {HEURISTIC_LABELS[h.type] || h.type}{' '}
                <span className="tiny" style={{ fontWeight: 400 }}>({pct(h.confidence)}%)</span>
              </strong>
              <p className="tiny" style={{ margin: '3px 0 0' }}>{h.explanation}</p>
            </div>
          ))}

          {!behaviour.triggered_heuristics?.length ? (
            <p className="tiny" style={{ marginTop: 0 }}>
              No heuristic explanation was recorded with this event.
            </p>
          ) : null}

          <div style={{ marginTop: 16 }}>
            <ScoreBar
              label="Behaviour alone"
              value={behaviour.behavioural_risk_score}
              tone={riskTone(behaviour.behavioural_risk_score)}
            />
            <ScoreBar
              label="Face match"
              value={identity.confidence}
              tone="accent"
              absentText="no match"
            />
            <ScoreBar label="Composite" value={behaviour.composite_risk_score} tone={tone} />
          </div>

          {/* Why this card exists at all, given the filter suppresses most. */}
          <p className="tiny" style={{ margin: '12px 0 0', opacity: 0.85 }}>
            This reached you because the movement crossed its threshold <em>and</em> the body was
            already matched to a person flagged as{' '}
            <strong>{LABEL_TEXT[identity.label]?.toLowerCase() || 'high risk'}</strong>. Movement
            alone does not open a card, and neither does a face match alone.
          </p>
        </div>
      </div>

      {behaviour.reasoning?.length ? (
        <details style={{ marginTop: 12 }}>
          <summary className="tiny" style={{ cursor: 'pointer' }}>How the score was reached</summary>
          <ul className="tiny" style={{ margin: '8px 0 0', paddingLeft: 18 }}>
            {behaviour.reasoning.map((step, i) => (
              <li key={i} style={{ marginBottom: 4 }}>{step}</li>
            ))}
          </ul>
        </details>
      ) : null}

      <h2 style={{ fontSize: 14, margin: '18px 0 8px' }}>2 · Your decision</h2>

      {error ? (
        <div className="banner banner-bad" role="alert" style={{ marginBottom: 10 }}>
          {error}
        </div>
      ) : null}

      {review.status === 'pending' ? (
        denying ? (
          <div className="field">
            <label htmlFor={`reason-${review.review_id}`}>
              Why is this not a genuine flag?
            </label>
            <textarea
              id={`reason-${review.review_id}`}
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g. Resident fetching something from their own car."
            />
            <span className="hint">
              Required. A denied flag is a measured false positive — the reason is how these
              thresholds get tuned.
            </span>
            <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
              <button
                type="button"
                className="btn btn-danger"
                disabled={busy}
                onClick={() => onDeny(reason)}
              >
                {busy ? 'Recording…' : 'Record as false flag'}
              </button>
              <button type="button" className="btn" onClick={() => setDenying(false)}>
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="field" style={{ marginBottom: 12 }}>
              <label htmlFor={`note-${review.review_id}`}>Reviewer note (optional)</label>
              <input
                id={`note-${review.review_id}`}
                type="text"
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="e.g. Seen trying door handles."
              />
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                type="button"
                className="btn btn-danger"
                disabled={busy}
                onClick={() => onConfirm(note, identity.label)}
              >
                {busy
                  ? 'Recording…'
                  : identity.attached && identity.label && identity.label !== 'verified'
                  ? `Confirm as ${LABEL_TEXT[identity.label].toLowerCase()}`
                  : 'Confirm the flag'}
              </button>
              <button type="button" className="btn btn-good" onClick={() => setDenying(true)}>
                Not a concern…
              </button>
            </div>
            <p className="tiny" style={{ marginTop: 8 }}>
              <strong>Confirming is your identification, not the system’s.</strong>{' '}
              {decision.confirm_effect} It is recorded against your name and can be undone for 24
              hours. It does <strong>not</strong> add anyone to the face registry — that stays a
              separate, deliberate step.
            </p>
            {!identity.attached ? (
              <p className="tiny" style={{ marginTop: 6, opacity: 0.85 }}>
                No identity is attached, so confirming says the <em>behaviour</em> was real. It
                makes no claim about who this person is, and alerts nobody.
              </p>
            ) : null}
          </>
        )
      ) : (
        <>
          <p className="tiny" style={{ margin: 0 }}>
            {review.status === 'confirmed' ? 'Confirmed' : 'Recorded as a false flag'}
            {decision.decided_by ? ` by ${decision.decided_by}` : ''}
            {decision.decided_at ? ` on ${formatTime(decision.decided_at)}` : ''}.
            {decision.denial_reason ? ` Reason: “${decision.denial_reason}”` : ''}
            {decision.decision_note ? ` Note: “${decision.decision_note}”` : ''}
          </p>
          <button
            type="button"
            className="btn"
            style={{ marginTop: 10 }}
            disabled={busy}
            onClick={() => onReopen()}
          >
            {busy ? 'Reopening…' : 'Undo this decision'}
          </button>
          <p className="tiny" style={{ marginTop: 6, opacity: 0.85 }}>
            Undoing returns this to the queue. The original decision stays on the record.
          </p>
        </>
      )}

      <DecisionTrail trail={trail} />
    </div>
  )
}

export default function BehaviourReview() {
  const { employee } = useSession()
  const [tab, setTab] = useState('pending')
  const [reviews, setReviews] = useState([])
  const [counts, setCounts] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [openId, setOpenId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [detailError, setDetailError] = useState(null)
  const [trail, setTrail] = useState([])
  const [busy, setBusy] = useState(false)
  const [decisionError, setDecisionError] = useState(null)
  const [banner, setBanner] = useState(null)

  const load = useCallback(() => {
    setLoading(true)
    api
      .behaviourQueue({ status: tab })
      .then((data) => {
        setReviews(data.reviews || [])
        setCounts(data.counts || {})
        setError(null)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [tab])

  useEffect(load, [load])

  // New flags should surface without the reviewer reloading.
  useEffect(() => {
    const timer = setInterval(load, 15000)
    return () => clearInterval(timer)
  }, [load])

  const loadDetail = useCallback((reviewId) => {
    setDetailError(null)
    setDecisionError(null)
    Promise.all([api.behaviourReview(reviewId), api.behaviourHistory(reviewId)])
      .then(([card, trailData]) => {
        setDetail(card)
        setTrail(trailData.decisions || [])
      })
      .catch((err) => setDetailError(err.message))
  }, [])

  function openDetail(reviewId) {
    if (openId === reviewId) {
      setOpenId(null)
      setDetail(null)
      setTrail([])
      return
    }
    setOpenId(reviewId)
    setDetail(null)
    setTrail([])
    loadDetail(reviewId)
  }

  const reviewerName = employee?.name || 'this reviewer'
  // Stands in for authentication — session.jsx supplies it and the API trusts
  // it. On an identification decision this is the audit trail's only signature.
  const reviewerId = employee?.employee_id || 'unauthenticated-demo-user'

  async function submitDecision(reviewId, decision, payload, successText) {
    setBusy(true)
    setDecisionError(null)
    try {
      await api.behaviourDecide(reviewId, decision, { reviewer_id: reviewerId, ...payload })
      setBanner({ kind: 'good', text: successText })
      loadDetail(reviewId)
      load()
    } catch (err) {
      setDecisionError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <h1>Behavioural review</h1>
      <p className="muted" style={{ margin: '2px 0 14px' }}>
        Reviewing as <strong>{reviewerName}</strong>. Each card pairs what the facial recognition
        module concluded with what the person was <em>doing</em>. Nothing here has acted on its own —
        these are prompts to look.
      </p>

      {error ? (
        <div className="banner banner-bad" role="alert" style={{ marginBottom: 16 }}>
          {error}
        </div>
      ) : null}

      {banner ? (
        <div className={`banner banner-${banner.kind}`} role="status" style={{ marginBottom: 16 }}>
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
            {counts[t.key] ? (
              <span style={{ marginLeft: 6, fontVariantNumeric: 'tabular-nums', opacity: 0.85 }}>
                {counts[t.key]}
              </span>
            ) : null}
          </button>
        ))}
      </div>

      {loading && !reviews.length ? <p className="muted">Loading…</p> : null}

      {!loading && !reviews.length && !error ? (
        <div className="card empty">
          <p style={{ margin: 0 }}>
            {tab === 'pending' ? 'Nothing is waiting for review.' : `No ${tab} reviews.`}
          </p>
          {tab === 'pending' ? (
            <p className="tiny" style={{ margin: '8px 0 0' }}>
              A card appears here when the behavioural module emits an event that needs a human —
              either a composite score over the threshold, or strong behaviour with no facial match
              at all. Run it with <code>--push</code> to feed this queue.
            </p>
          ) : null}
        </div>
      ) : null}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {reviews.map((review) => {
          const open = openId === review.review_id
          const tone = riskTone(review.composite_risk_score)
          const pill = STATUS_PILL[review.status] || STATUS_PILL.pending

          return (
            <article key={review.review_id} className="card list-card">
              <div className="card-head">
                <span className={`pill pill-${pill.cls}`}>
                  <span className="dot" aria-hidden="true" />
                  {pill.text}
                </span>
                <strong style={{ fontSize: 14.5 }}>
                  {HEURISTIC_LABELS[review.top_heuristic] || review.top_heuristic || 'Behavioural flag'}
                </strong>
                <span className="tiny">
                  {review.face.attached ? LABEL_TEXT[review.face.label] || review.face.label : 'no face match'}
                  {review.trigger_count > 1 ? ` · +${review.trigger_count - 1} more` : ''}
                </span>
                <span className="spacer" />
                <span
                  className="tiny"
                  style={{ color: `var(--${tone})`, fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}
                >
                  risk {pct(review.composite_risk_score)}%
                </span>
                <button type="button" className="btn" onClick={() => openDetail(review.review_id)}>
                  {open ? 'Hide' : 'Review'}
                </button>
              </div>

              <p className="tiny" style={{ margin: '8px 0 0' }}>
                {review.camera_id || 'unknown camera'} · {formatTime(review.occurred_at)} ·{' '}
                {review.review_id}
              </p>
              {review.headline ? (
                <p className="tiny" style={{ margin: '3px 0 0' }}>{review.headline}.</p>
              ) : null}

              {open ? (
                detailError ? (
                  <div className="banner banner-bad" style={{ marginTop: 12 }} role="alert">
                    {detailError}
                  </div>
                ) : detail && detail.review_id === review.review_id ? (
                  <ReviewDetail
                    review={detail}
                    trail={trail}
                    busy={busy}
                    error={decisionError}
                    onConfirm={(note, label) =>
                      submitDecision(
                        review.review_id,
                        'confirm',
                        { note: note || undefined, label: label || undefined },
                        `${review.review_id} confirmed. Recorded against ${reviewerName}; undoable for 24 hours.`
                      )
                    }
                    onDeny={(reason) =>
                      submitDecision(
                        review.review_id,
                        'deny',
                        { reason },
                        `${review.review_id} recorded as a false flag. No alert sent.`
                      )
                    }
                    onReopen={() =>
                      submitDecision(
                        review.review_id,
                        'reopen',
                        {},
                        `${review.review_id} reopened and returned to the queue.`
                      )
                    }
                  />
                ) : (
                  <p className="muted" style={{ marginTop: 12 }}>Loading…</p>
                )
              ) : null}
            </article>
          )
        })}
      </div>

      <style>{`
        .behaviour-split { display: grid; grid-template-columns: 1.25fr 1fr; gap: 14px; }
        .behaviour-split > * { min-width: 0; }
        @media (max-width: 860px) {
          .behaviour-split { grid-template-columns: 1fr; }
        }
      `}</style>
    </>
  )
}
