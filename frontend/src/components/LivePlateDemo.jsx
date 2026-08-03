import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Webcam from 'react-webcam'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

// The backend spends roughly 1.2s on a frame (locate + OCR + registry lookup).
// Firing faster than it can answer just queues frames that are already stale by
// the time they are read, so the interval is paced to the work, not to the
// camera's frame rate.
const SCAN_INTERVAL_MS = 1300
const CAPTURE_MAX_WIDTH = 720
const CAPTURE_QUALITY = 0.72

const VIDEO_CONSTRAINTS = { facingMode: 'environment', width: 1280, height: 720 }

/** Draw the current video frame to a canvas and hand back a JPEG blob. */
async function captureFrameBlob(video, quality = CAPTURE_QUALITY, maxWidth = CAPTURE_MAX_WIDTH) {
  if (!video || video.readyState !== 4) return null
  const scale = Math.min(1, maxWidth / video.videoWidth)
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(video.videoWidth * scale)
  canvas.height = Math.round(video.videoHeight * scale)
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality))
}

function statusTone(status, matchFound) {
  if (status === 'offender') return 'bad'
  if (status === 'suspect') return 'warn'
  if (status === 'verified') return 'ok'
  return matchFound ? 'ok' : 'muted'
}

export default function LivePlateDemo() {
  const webcamRef = useRef(null)
  const inFlightRef = useRef(false)

  // One id per mount. It partitions the backend's frame-to-frame vote so a
  // second operator scanning at the same time cannot vote in this session.
  const streamId = useMemo(
    () => `plate-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    [],
  )

  const [scanning, setScanning] = useState(false)
  const [webcamReady, setWebcamReady] = useState(false)
  const [webcamError, setWebcamError] = useState('')
  const [backendReady, setBackendReady] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [roundtripMs, setRoundtripMs] = useState(null)
  const [confirmed, setConfirmed] = useState([])

  useEffect(() => {
    let active = true
    const ping = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/health`)
        if (active) setBackendReady(res.ok)
      } catch {
        if (active) setBackendReady(false)
      }
    }
    ping()
    const id = setInterval(ping, 10000)
    return () => { active = false; clearInterval(id) }
  }, [])

  const sendFrame = useCallback(async () => {
    // One frame in flight at a time. Overlapping requests would interleave in
    // the vote and make "3 of 5 frames agree" a statement about nothing.
    if (inFlightRef.current) return
    const video = webcamRef.current?.video
    if (!video || !webcamReady) return

    inFlightRef.current = true
    const started = performance.now()
    try {
      const blob = await captureFrameBlob(video)
      if (!blob) return

      const form = new FormData()
      form.append('file', blob, 'frame.jpg')
      form.append('stream_id', streamId)

      const res = await fetch(`${API_BASE_URL}/api/v1/scan-plate-live`, {
        method: 'POST',
        body: form,
      })
      const data = await res.json().catch(() => null)
      setRoundtripMs(Math.round(performance.now() - started))

      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`)

      setResult(data)
      setError('')

      // Log only the frame the vote lands on, so a car held in view is one row
      // in the audit list rather than one row per frame.
      if (data?.stability?.newly_confirmed && data?.match_found) {
        setConfirmed((prev) => [
          {
            id: `${data.plate?.id}-${Date.now()}`,
            plateNumber: data.plate?.plate_number,
            status: data.status,
            ownerName: data.plate?.owner_name,
            alert: !!data.alert,
            matchType: data.match_type,
            at: new Date().toISOString(),
          },
          ...prev,
        ].slice(0, 12))
      }
    } catch (err) {
      setError(err?.message || 'Frame scan failed.')
    } finally {
      inFlightRef.current = false
    }
  }, [streamId, webcamReady])

  useEffect(() => {
    if (!scanning) return undefined
    const id = setInterval(sendFrame, SCAN_INTERVAL_MS)
    sendFrame()
    return () => clearInterval(id)
  }, [scanning, sendFrame])

  const resetStream = async () => {
    setResult(null)
    setError('')
    setConfirmed([])
    try {
      await fetch(`${API_BASE_URL}/api/v1/scan-plate-live/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stream_id: streamId }),
      })
    } catch {
      /* Reset is best-effort; entries expire on their own. */
    }
  }

  const stability = result?.stability
  const votePct = stability?.required_frames
    ? Math.min(100, ((stability.agreeing_frames || 0) / stability.required_frames) * 100)
    : 0
  const tone = statusTone(result?.status, result?.match_found)

  // Backend boxes are in captured-frame pixels; the <video> renders at a
  // different size, so scale by the ratio rather than assuming they match.
  const frameW = result?.frame?.width || 0
  const regions = result?.regions || []

  return (
    <div className="lp-wrap">
      <style>{`
        .lp-wrap { background: var(--bg); color: var(--fg); min-height: 100vh; padding: 24px 16px 64px; }
        .lp-shell { max-width: 1180px; margin: 0 auto; }
        .lp-sub { color: var(--muted); font-size: 13px; margin: 0 0 14px; }
        .lp-grid { display: grid; grid-template-columns: minmax(420px, 2fr) minmax(320px, 1fr); gap: 14px; }
        .lp-card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; }
        .lp-pad { padding: 16px; }
        .lp-camera { position: relative; width: 100%; aspect-ratio: 4/3; border-radius: 12px; overflow: hidden; border: 1px solid var(--line); background: #000; }
        .lp-camera video { width: 100%; height: 100%; object-fit: cover; }
        .lp-box { position: absolute; border: 2px solid var(--accent); border-radius: 4px; box-shadow: 0 0 0 9999px rgba(0,0,0,0.18); pointer-events: none; }
        .lp-overlay { position: absolute; top: 10px; left: 10px; display: grid; gap: 6px; z-index: 3; }
        .lp-chip { display: inline-block; padding: 4px 9px; border-radius: 999px; border: 1px solid var(--line); background: rgba(0,0,0,0.45); color: #fff; font-size: 11px; font-weight: 700; letter-spacing: .03em; }
        .lp-plate { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 30px; font-weight: 700; letter-spacing: .08em; margin: 6px 0; }
        .lp-tone-bad { color: var(--unknown); border-color: var(--unknown); }
        .lp-tone-warn { color: #d98324; border-color: #d98324; }
        .lp-tone-ok { color: var(--ok); border-color: var(--ok); }
        .lp-tone-muted { color: var(--muted); }
        .lp-vote { height: 6px; border-radius: 999px; background: var(--line); overflow: hidden; margin: 8px 0 4px; }
        .lp-vote > div { height: 100%; background: var(--accent); transition: width .2s ease; }
        .lp-meta { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 8px; font-size: 12px; color: var(--muted); margin-top: 12px; }
        .lp-controls { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
        .lp-controls button { padding: 8px 14px; border-radius: 8px; border: 1px solid var(--line); background: var(--panel); color: var(--fg); font-size: 13px; cursor: pointer; }
        .lp-controls button.primary { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 12%, var(--panel)); }
        .lp-ready { display: flex; gap: 8px; font-size: 12px; margin-bottom: 10px; flex-wrap: wrap; }
        .lp-ready > span { padding: 3px 8px; border-radius: 999px; border: 1px solid var(--line); }
        .lp-ready .ok { border-color: var(--ok); color: var(--ok); }
        .lp-ready .bad { border-color: var(--unknown); color: var(--unknown); }
        .lp-row { display: flex; justify-content: space-between; gap: 10px; padding: 8px 0; border-bottom: 1px solid var(--line); font-size: 13px; }
        .lp-row:last-child { border-bottom: 0; }
        .lp-alert { margin-top: 12px; padding: 10px 12px; border-radius: 9px; border: 1px solid var(--unknown); color: var(--unknown); font-weight: 600; font-size: 13px; }
        .lp-warn { margin-top: 12px; font-size: 13px; color: var(--unknown); }
        .lp-raw { font-family: ui-monospace, monospace; font-size: 12px; color: var(--muted); word-break: break-all; }
        @media (max-width: 980px) { .lp-grid { grid-template-columns: 1fr; } }
      `}</style>

      <div className="lp-shell">
        <h2 style={{ margin: '0 0 4px' }}>Live plate recognition</h2>
        <p className="lp-sub">
          Frames are read locally with EasyOCR. A plate must be read consistently across
          several frames before it raises an alert.
        </p>

        <div className="lp-ready">
          <span className={webcamReady ? 'ok' : 'bad'}>
            camera {webcamReady ? 'ready' : 'waiting'}
          </span>
          <span className={backendReady ? 'ok' : 'bad'}>
            backend {backendReady ? 'ready' : 'unreachable'}
          </span>
          {roundtripMs != null && <span>{roundtripMs}ms / frame</span>}
        </div>

        <div className="lp-controls">
          <button
            className={scanning ? '' : 'primary'}
            onClick={() => setScanning((s) => !s)}
            disabled={!webcamReady || !backendReady}
          >
            {scanning ? 'Stop scanning' : 'Start scanning'}
          </button>
          <button onClick={resetStream}>Reset</button>
        </div>

        <div className="lp-grid">
          <div className="lp-card lp-pad">
            <div className="lp-camera">
              <Webcam
                ref={webcamRef}
                audio={false}
                videoConstraints={VIDEO_CONSTRAINTS}
                onUserMedia={() => { setWebcamReady(true); setWebcamError('') }}
                onUserMediaError={(e) => {
                  setWebcamReady(false)
                  setWebcamError(e?.message || 'Camera permission denied.')
                }}
              />
              <div className="lp-overlay">
                <span className="lp-chip">{scanning ? 'SCANNING' : 'IDLE'}</span>
                {result?.localised && <span className="lp-chip">PLATE LOCATED</span>}
              </div>
              {frameW > 0 && regions.map((r, i) => (
                <div
                  key={i}
                  className="lp-box"
                  style={{
                    left: `${(r.x / frameW) * 100}%`,
                    top: `${(r.y / (result.frame.height || 1)) * 100}%`,
                    width: `${(r.w / frameW) * 100}%`,
                    height: `${(r.h / (result.frame.height || 1)) * 100}%`,
                  }}
                />
              ))}
            </div>
            {webcamError && <div className="lp-warn">Camera: {webcamError}</div>}
            {error && <div className="lp-warn">Scan: {error}</div>}
          </div>

          <div>
            <div className="lp-card lp-pad">
              <div style={{ fontSize: 12, textTransform: 'uppercase', color: 'var(--muted)' }}>
                Current read
              </div>
              <div className={`lp-plate lp-tone-${tone}`}>
                {result?.extracted_text || '—'}
              </div>

              {stability && (
                <>
                  <div className="lp-vote">
                    <div style={{ width: `${votePct}%` }} />
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--muted)' }}>
                    {stability.confirmed
                      ? 'Confirmed across frames'
                      : `${stability.agreeing_frames || 0} of ${stability.required_frames} frames agree`}
                  </div>
                </>
              )}

              {result?.match_found && (
                <div style={{ marginTop: 10, fontSize: 13 }}>
                  <div><strong>{result.plate?.plate_number}</strong></div>
                  <div style={{ color: 'var(--muted)' }}>{result.plate?.owner_name}</div>
                  <span className={`lp-chip lp-tone-${tone}`} style={{ marginTop: 6 }}>
                    {String(result.status || '').toUpperCase()}
                  </span>
                  {result.match_type === 'normalised' && (
                    <div style={{ color: 'var(--muted)', fontSize: 12, marginTop: 6 }}>
                      Resolved through OCR character confusion.
                    </div>
                  )}
                </div>
              )}

              {result?.alert && <div className="lp-alert">{result.message}</div>}
              {!result?.alert && result?.message && (
                <div style={{ marginTop: 10, fontSize: 13, color: 'var(--muted)' }}>
                  {result.message}
                </div>
              )}

              {result?.raw_text && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ fontSize: 11, textTransform: 'uppercase', color: 'var(--muted)' }}>
                    Raw OCR
                  </div>
                  <div className="lp-raw">{result.raw_text}</div>
                </div>
              )}

              {result?.timings_ms && (
                <div className="lp-meta">
                  <div>locate<br /><strong>{result.timings_ms.locate ?? '—'}ms</strong></div>
                  <div>ocr<br /><strong>{result.timings_ms.ocr ?? '—'}ms</strong></div>
                  <div>match<br /><strong>{result.timings_ms.match ?? '—'}ms</strong></div>
                </div>
              )}
            </div>

            <div className="lp-card lp-pad" style={{ marginTop: 14 }}>
              <div style={{ fontSize: 12, textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 6 }}>
                Confirmed this session
              </div>
              {confirmed.length === 0 && (
                <div style={{ fontSize: 13, color: 'var(--muted)' }}>Nothing confirmed yet.</div>
              )}
              {confirmed.map((c) => (
                <div className="lp-row" key={c.id}>
                  <span style={{ fontFamily: 'ui-monospace, monospace' }}>{c.plateNumber}</span>
                  <span className={`lp-tone-${statusTone(c.status, true)}`}>{c.status}</span>
                  <span style={{ color: 'var(--muted)' }}>
                    {new Date(c.at).toLocaleTimeString()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
