import React, { useMemo, useRef, useEffect, useState } from 'react'
import Webcam from 'react-webcam'
import { Link } from 'react-router-dom'
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''
const SCAN_INTERVAL_MS = 1500
// Plates get their own cadence because they are gated differently: the face
// scan only fires when MediaPipe sees a face, so on a frame with a car and no
// visible driver the face path is idle and the plate path is the only one
// working. Slower than the face interval because plate OCR costs ~1.2s and
// there is no point queueing frames the backend cannot reach.
const PLATE_SCAN_INTERVAL_MS = 2000
const HISTORY_LIMIT = 30
const MODEL_LABEL = 'Facenet512 / Cosine'
const MEDIAPIPE_WASM_ROOT =
  import.meta.env.VITE_MEDIAPIPE_WASM_ROOT || 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
const MEDIAPIPE_MODEL_PATH =
  import.meta.env.VITE_MEDIAPIPE_MODEL_PATH ||
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))

const formatTimestamp = (iso) => {
  if (!iso) return '-'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '-' : d.toLocaleString()
}

const toneClassFromResult = ({ hasFace, isAnalyzing, matchResult }) => {
  if (matchResult?.isAlert) return 'alert'
  if (matchResult?.isKnownUser) return 'ok'
  if (!hasFace) return 'unknown'
  if (isAnalyzing) return 'unknown'
  return 'unknown'
}

const toneLabelFromResult = ({ hasFace, isAnalyzing, matchResult }) => {
  if (matchResult?.isAlert) return 'FLAGGED'
  if (matchResult?.isKnownUser) return 'RECOGNISED'
  if (!hasFace) return 'NO TARGET'
  if (isAnalyzing) return 'SCANNING'
  return 'UNREGISTERED'
}

const auditTone = (entry) => {
  const status = (entry?.status || '').toLowerCase()
  if (status === 'offender' || status === 'suspect' || entry?.isAlert) return 'alert'
  if (entry?.isKnownUser || status === 'verified') return 'ok'
  return 'unknown'
}

const toConfidencePct = (distance) => {
  if (typeof distance !== 'number') return 0
  return clamp(Math.round((1 - distance / 0.6) * 100), 0, 100)
}

const parsePoseFromUrl = (url) => {
  if (!url) return '-'
  const name = url.split('/').pop() || ''
  const stripped = name.replace(/\.[a-zA-Z0-9]+$/, '')
  const parts = stripped.split('_').filter(Boolean)
  return parts.length < 2 ? '-' : parts.slice(1).join(' ')
}

const captureCrop = (video, bbox) => {
  if (!video || !bbox || !video.videoWidth || !video.videoHeight) return null

  const padX = bbox.w * 0.2
  const padY = bbox.h * 0.25
  const sx = clamp(Math.floor(bbox.x - padX), 0, video.videoWidth - 1)
  const sy = clamp(Math.floor(bbox.y - padY), 0, video.videoHeight - 1)
  const ex = clamp(Math.ceil(bbox.x + bbox.w + padX), 1, video.videoWidth)
  const ey = clamp(Math.ceil(bbox.y + bbox.h + padY), 1, video.videoHeight)
  const sw = Math.max(1, ex - sx)
  const sh = Math.max(1, ey - sy)

  const c = document.createElement('canvas')
  c.width = 180
  c.height = 180
  const g = c.getContext('2d')
  g.drawImage(video, sx, sy, sw, sh, 0, 0, c.width, c.height)
  return c.toDataURL('image/jpeg', 0.8)
}

const captureFrameBlob = (video, quality = 0.72, maxWidth = 720) => {
  if (!video || !video.videoWidth || !video.videoHeight) return Promise.resolve(null)

  const ratio = Math.min(1, maxWidth / video.videoWidth)
  const width = Math.max(1, Math.round(video.videoWidth * ratio))
  const height = Math.max(1, Math.round(video.videoHeight * ratio))

  const c = document.createElement('canvas')
  c.width = width
  c.height = height
  const g = c.getContext('2d')
  g.drawImage(video, 0, 0, width, height)

  return new Promise((resolve) => {
    c.toBlob((blob) => resolve(blob), 'image/jpeg', quality)
  })
}

const drawReticle = (ctx, box, toneClass, scorePct) => {
  const color = toneClass === 'alert' ? '#b3261e' : toneClass === 'ok' ? '#186b3c' : '#7a5c00'
  const dashed = toneClass === 'unknown'

  ctx.save()
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = 2.5
  ctx.setLineDash(dashed ? [7, 5] : [])
  ctx.strokeRect(box.x, box.y, box.w, box.h)

  const tick = Math.max(10, Math.min(box.w, box.h) * 0.18)
  const x2 = box.x + box.w
  const y2 = box.y + box.h

  const ticks = [
    [box.x, box.y + tick, box.x, box.y, box.x + tick, box.y],
    [x2 - tick, box.y, x2, box.y, x2, box.y + tick],
    [box.x, y2 - tick, box.x, y2, box.x + tick, y2],
    [x2 - tick, y2, x2, y2, x2, y2 - tick],
  ]

  ctx.lineWidth = 3
  for (const t of ticks) {
    ctx.beginPath()
    ctx.moveTo(t[0], t[1])
    ctx.lineTo(t[2], t[3])
    ctx.lineTo(t[4], t[5])
    ctx.stroke()
  }

  ctx.setLineDash([])
  ctx.font = '700 12px ui-monospace, SFMono-Regular, Menlo, monospace'
  ctx.fillText(`${scorePct}%`, box.x, Math.max(14, box.y - 6))
  ctx.restore()
}

export default function LiveScanDemo() {
  const webcamRef = useRef(null)
  const canvasRef = useRef(null)
  const detectorRef = useRef(null)
  const lastVideoTimeRef = useRef(-1)
  const lastFrameAtRef = useRef(0)
  const faceBoxRef = useRef(null)
  const failureCountRef = useRef(0)
  const nextScanAllowedAtRef = useRef(0)

  const [modelState, setModelState] = useState('loading')
  const [modelError, setModelError] = useState('')
  const [backendReady, setBackendReady] = useState(false)
  const [matchResult, setMatchResult] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [scanError, setScanError] = useState('')
  const [webcamReady, setWebcamReady] = useState(false)
  const [webcamError, setWebcamError] = useState('')
  const [hasFaceInFrame, setHasFaceInFrame] = useState(false)
  const [faceScore, setFaceScore] = useState(0)
  const [fps, setFps] = useState(0)
  const [memberCount, setMemberCount] = useState(null)
  const [liveCrop, setLiveCrop] = useState(null)
  const [roundtripMs, setRoundtripMs] = useState(null)
  const [history, setHistory] = useState([])
  const [selectedHistoryId, setSelectedHistoryId] = useState(null)
  const [dbImageBroken, setDbImageBroken] = useState(false)

  const [plateResult, setPlateResult] = useState(null)
  const [plateError, setPlateError] = useState('')
  const [plateSightings, setPlateSightings] = useState([])

  // Face and plate requests are serialised through this. The dev server runs
  // both concurrently if asked, but they contend for the same CPU — TensorFlow
  // and EasyOCR each want it — so overlapping them makes both slower than
  // running them in turn.
  const scanBusyRef = useRef(false)
  const plateStreamIdRef = useRef(
    `livescan-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  )

  const isModelLoaded = modelState === 'ready'

  useEffect(() => {
    let active = true

    const pingBackend = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/health`)
        if (active) setBackendReady(res.ok)
      } catch {
        if (active) setBackendReady(false)
      }
    }

    pingBackend()
    const id = setInterval(pingBackend, 10000)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  useEffect(() => {
    const fetchMemberCount = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/members`)
        if (!res.ok) return
        const data = await res.json()
        const count = Array.isArray(data)
          ? data.length
          : Array.isArray(data?.members)
          ? data.members.length
          : null
        setMemberCount(count)
      } catch {
        setMemberCount(null)
      }
    }
    fetchMemberCount()
  }, [])

  useEffect(() => {
    const loadDetector = async () => {
      try {
        setModelState('loading')
        setModelError('')

        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT)

        detectorRef.current = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MEDIAPIPE_MODEL_PATH,
          },
          runningMode: 'VIDEO',
          minDetectionConfidence: 0.52,
        })

        setModelState('ready')
      } catch (err) {
        setModelState('failed')
        setModelError(`Failed to load MediaPipe detector assets. ${err?.message || ''}`.trim())
      }
    }

    loadDetector()
    return () => {
      detectorRef.current?.close?.()
      detectorRef.current = null
    }
  }, [])

  const toneClass = toneClassFromResult({ hasFace: hasFaceInFrame, isAnalyzing, matchResult })
  const toneLabel = toneLabelFromResult({ hasFace: hasFaceInFrame, isAnalyzing, matchResult })

  useEffect(() => {
    let frameId

    const run = async () => {
      const video = webcamRef.current?.video
      const canvas = canvasRef.current
      const detector = detectorRef.current

      if (
        video &&
        canvas &&
        detector &&
        webcamReady &&
        isModelLoaded &&
        video.readyState === 4
      ) {
        const now = performance.now()
        if (lastFrameAtRef.current > 0) {
          const elapsed = now - lastFrameAtRef.current
          if (elapsed > 0) setFps(Math.round(1000 / elapsed))
        }
        lastFrameAtRef.current = now

        const videoTime = video.currentTime
        if (videoTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = videoTime

          const vw = video.videoWidth
          const vh = video.videoHeight
          const rw = video.clientWidth || vw
          const rh = video.clientHeight || vh

          const results = detector.detectForVideo(video, now)
          const detections = results?.detections || []

          canvas.width = rw
          canvas.height = rh
          const ctx = canvas.getContext('2d')
          ctx.clearRect(0, 0, rw, rh)

          if (detections.length > 0) {
            const best = [...detections].sort(
              (a, b) => (b.categories?.[0]?.score || 0) - (a.categories?.[0]?.score || 0)
            )[0]

            const b = best.boundingBox
            const sx = rw / vw
            const sy = rh / vh
            const box = {
              x: b.originX * sx,
              y: b.originY * sy,
              w: b.width * sx,
              h: b.height * sy,
            }
            const score = best.categories?.[0]?.score || 0

            faceBoxRef.current = { x: b.originX, y: b.originY, w: b.width, h: b.height }
            setHasFaceInFrame(true)
            setFaceScore(score)

            drawReticle(ctx, box, toneClass, Math.round(score * 100))
          } else {
            faceBoxRef.current = null
            setHasFaceInFrame(false)
            setFaceScore(0)
          }
        }
      }

      frameId = requestAnimationFrame(run)
    }

    if (isModelLoaded) run()
    return () => cancelAnimationFrame(frameId)
  }, [isModelLoaded, webcamReady, toneClass])

  useEffect(() => {
    if (!isModelLoaded || !webcamReady || !backendReady) return undefined

    const id = setInterval(() => {
      if (!isAnalyzing && !scanBusyRef.current && hasFaceInFrame) {
        sendFrameToBackend()
      }
    }, SCAN_INTERVAL_MS)

    return () => clearInterval(id)
  }, [isModelLoaded, webcamReady, backendReady, isAnalyzing, hasFaceInFrame])

  const sendPlateFrame = async () => {
    const video = webcamRef.current?.video
    if (!video || scanBusyRef.current) return

    scanBusyRef.current = true
    try {
      const blob = await captureFrameBlob(video, 0.72, 720)
      if (!blob) return

      const formData = new FormData()
      formData.append('file', blob, 'live_plate.jpg')
      formData.append('stream_id', plateStreamIdRef.current)

      const response = await fetch(`${API_BASE_URL}/api/v1/scan-plate-live`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json().catch(() => null)
      if (!response.ok) throw new Error(data?.error || `HTTP ${response.status}`)

      setPlateResult(data)
      setPlateError('')

      // Only the frame the vote lands on. A car parked in view would otherwise
      // add a row every couple of seconds for as long as it sits there.
      if (data?.stability?.newly_confirmed && data?.match_found) {
        setPlateSightings((prev) => [
          {
            id: `plate-${Date.now()}`,
            plateNumber: data.plate?.plate_number,
            status: data.status,
            ownerName: data.plate?.owner_name,
            alert: !!data.alert,
            at: new Date().toISOString(),
          },
          ...prev,
        ].slice(0, 10))
      }
    } catch (err) {
      setPlateError(err?.message || 'Plate scan failed.')
    } finally {
      scanBusyRef.current = false
    }
  }

  useEffect(() => {
    if (!webcamReady || !backendReady) return undefined

    const id = setInterval(() => {
      // Yields to the face scan: a face in frame is the higher-value read, and
      // the plate is usually still there a beat later.
      if (!isAnalyzing && !scanBusyRef.current) {
        sendPlateFrame()
      }
    }, PLATE_SCAN_INTERVAL_MS)

    return () => clearInterval(id)
  }, [webcamReady, backendReady, isAnalyzing])

  const addHistory = (entry) => {
    setHistory((prev) => {
      const next = [entry, ...prev].slice(0, HISTORY_LIMIT)
      if (!selectedHistoryId) setSelectedHistoryId(entry.id)
      return next
    })
  }

  const sendFrameToBackend = async () => {
    const webcam = webcamRef.current
    const video = webcam?.video
    if (!webcam || !video) return
    if (Date.now() < nextScanAllowedAtRef.current) return
    if (!webcamReady || !backendReady) {
      setScanError('System not ready: waiting on camera and backend health.')
      return
    }

    const localCrop = captureCrop(video, faceBoxRef.current)

    try {
      scanBusyRef.current = true
      setIsAnalyzing(true)
      setScanError('')

      const started = performance.now()
      const blob = await captureFrameBlob(video, 0.72, 720)
      if (!blob) throw new Error('Camera frame unavailable.')

      const formData = new FormData()
      formData.append('file', blob, 'live_scan.jpg')

      const response = await fetch(`${API_BASE_URL}/api/v1/scan-face`, {
        method: 'POST',
        body: formData,
      })

      let data = null
      try {
        data = await response.json()
      } catch {
        data = null
      }

      const durationMs = Math.round(performance.now() - started)
      setRoundtripMs(durationMs)

      if (!response.ok) {
        throw new Error(data?.error || `Scan failed with HTTP ${response.status}`)
      }

      const distance = typeof data?.match_distance === 'number' ? data.match_distance : null
      const confidence = toConfidencePct(distance)
      const capture = Array.isArray(data?.supporting_captures) ? data.supporting_captures[0] : null
      const poseLabel = parsePoseFromUrl(capture?.image_url || data?.person?.image_url)
      const ts = new Date().toISOString()

      const normalized = {
        id: `scan-${Date.now()}`,
        success: !!data?.success,
        isKnownUser: !!data?.is_known_user,
        isAlert: !!data?.alert,
        status: data?.status || null,
        fullName: data?.person?.full_name || 'Unknown Identity',
        matchDistance: distance,
        matchConfidence: confidence,
        error: data?.error || null,
        message: data?.message || '',
        timestamp: ts,
        poseLabel,
        personImageUrl: data?.person?.image_url || null,
        sourceImageUrl: capture?.image_url || data?.person?.image_url || null,
        sourceText: capture?.source || 'Registry Seed',
        liveCrop: localCrop,
        latencyMs: durationMs,
        stageTiming: data?.timings_ms || null,
      }

      setDbImageBroken(false)
      setMatchResult(normalized)
      setLiveCrop(localCrop)
      addHistory(normalized)
      failureCountRef.current = 0
      nextScanAllowedAtRef.current = 0
    } catch (err) {
      const nextFailureCount = failureCountRef.current + 1
      failureCountRef.current = nextFailureCount
      const cooldownMs = Math.min(5000, 500 * Math.pow(2, nextFailureCount - 1))
      nextScanAllowedAtRef.current = Date.now() + cooldownMs
      setScanError(err?.message || `Scan request failed. Retrying in ${Math.round(cooldownMs / 1000)}s.`)
    } finally {
      setIsAnalyzing(false)
      scanBusyRef.current = false
    }
  }

  const activeEntry = useMemo(() => {
    if (!history.length) return null
    return history.find((h) => h.id === selectedHistoryId) || history[0]
  }, [history, selectedHistoryId])

  const dbImageUrl = activeEntry?.sourceImageUrl || activeEntry?.personImageUrl || matchResult?.sourceImageUrl || matchResult?.personImageUrl || null

  const readiness = [
    { label: 'Camera', ok: webcamReady },
    { label: 'Detector', ok: isModelLoaded },
    { label: 'Backend', ok: backendReady },
  ]

  const confidencePct = typeof matchResult?.matchConfidence === 'number' ? matchResult.matchConfidence : 0

  return (
    <div className="ls-wrap">
      <style>{`
        :root {
          --bg:#f6f7f9; --panel:#ffffff; --fg:#16181d; --muted:#666c78; --line:#e2e5ea;
          --alert:#b3261e; --alert-bg:#fdeceb; --ok:#186b3c; --ok-bg:#e9f6ee;
          --unknown:#7a5c00; --unknown-bg:#fdf5e0; --accent:#1c4fd8;
        }
        @media (prefers-color-scheme: dark) {
          :root {
            --bg:#131519; --panel:#1b1e24; --fg:#e9eaee; --muted:#9aa1ad; --line:#2b3038;
            --alert:#ff9a92; --alert-bg:#3a1c1a; --ok:#7fd4a0; --ok-bg:#14301f;
            --unknown:#e8c86a; --unknown-bg:#312713; --accent:#8fb0ff;
          }
        }
        .ls-wrap { background: var(--bg); color: var(--fg); min-height: 100vh; padding: 24px 16px 64px; width: 100%; max-width: 100%; overflow-x: clip; }
        .ls-shell { max-width: 1180px; margin: 0 auto; width: 100%; max-width: 100%; }
        .ls-sub { color: var(--muted); font-size: 13px; margin: 0 0 14px; }
        .ls-grid { display: grid; grid-template-columns: minmax(420px, 2fr) minmax(320px, 1fr); gap: 14px; }
        .ls-grid > * { min-width: 0; }
        .ls-card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; min-width: 0; }
        .ls-pad { padding: 16px; }
        .ls-meta { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; font-size: 12px; color: var(--muted); }
        .ls-camera { position: relative; width: 100%; aspect-ratio: 4/3; border-radius: 12px; overflow: hidden; border: 1px solid var(--line); background: #000; }
        .ls-chip { display: inline-block; padding: 4px 9px; border-radius: 999px; border: 1px solid var(--line); background: rgba(0,0,0,0.35); color: #fff; font-size: 11px; font-weight: 700; letter-spacing: .03em; }
        .ls-overlay-top-left { position: absolute; top: 10px; left: 10px; z-index: 4; display: grid; gap: 6px; }
        .ls-overlay-top-right { position: absolute; top: 10px; right: 10px; z-index: 4; }
        .ls-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; min-width: 0; }
        .ls-row > * { min-width: 0; overflow-wrap: anywhere; }
        .ls-verdict { border-radius: 12px; padding: 14px; border: 1px solid; margin-bottom: 12px; }
        .ls-verdict.alert { background: var(--alert-bg); border-color: var(--alert); color: var(--alert); }
        .ls-verdict.ok { background: var(--ok-bg); border-color: var(--ok); color: var(--ok); }
        .ls-verdict.unknown { background: var(--unknown-bg); border-color: var(--unknown); color: var(--unknown); }
        .ls-kv { display: flex; justify-content: space-between; gap: 14px; font-size: 13px; padding: 6px 0; border-bottom: 1px solid var(--line); min-width: 0; }
        .ls-kv:last-child { border-bottom: 0; }
        .ls-kv span:first-child { color: var(--muted); }
        .ls-kv span:last-child { min-width: 0; max-width: 62%; text-align: right; overflow-wrap: anywhere; word-break: break-word; }
        .ls-meter { height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; margin: 8px 0 10px; }
        .ls-meter > i { display: block; height: 100%; }
        .ls-ready { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
        .ls-ready > span { font-size: 12px; padding: 3px 8px; border-radius: 999px; border: 1px solid var(--line); }
        .ls-ready .ok { border-color: var(--ok); color: var(--ok); }
        .ls-ready .bad { border-color: var(--unknown); color: var(--unknown); }
        .ls-side-by-side { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .ls-thumb-wrap { border: 1px solid var(--line); border-radius: 9px; overflow: hidden; background: var(--bg); aspect-ratio: 1/1; }
        .ls-thumb-wrap img { width: 100%; height: 100%; object-fit: cover; }
        .ls-caption { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
        .ls-audit { max-height: 220px; overflow-y: auto; display: grid; gap: 6px; }
        .ls-audit button { text-align: left; border-radius: 9px; border: 1px solid var(--line); background: var(--panel); color: var(--fg); padding: 8px 9px; cursor: pointer; font-size: 12px; }
        .ls-audit button.active { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 8%, var(--panel)); }
        .ls-audit button.alert { border-color: var(--alert); background: color-mix(in srgb, var(--alert) 14%, var(--panel)); color: var(--alert); }
        .ls-audit button.ok { border-color: var(--ok); background: color-mix(in srgb, var(--ok) 12%, var(--panel)); color: var(--ok); }
        .ls-audit button.unknown { border-color: var(--unknown); background: color-mix(in srgb, var(--unknown) 12%, var(--panel)); color: var(--unknown); }
        .ls-audit button.alert.active,
        .ls-audit button.ok.active,
        .ls-audit button.unknown.active { box-shadow: inset 0 0 0 1px var(--accent); }
        .ls-warn { margin-top: 12px; font-size: 13px; color: var(--unknown); }
        @media (max-width: 980px) {
          .ls-grid { grid-template-columns: 1fr; }
          .ls-meta { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .ls-side-by-side { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 640px) {
          .ls-wrap { padding: 14px 10px 42px; }
          .ls-meta { grid-template-columns: 1fr; }
          .ls-side-by-side { grid-template-columns: 1fr; }
          .ls-kv { flex-direction: column; align-items: flex-start; gap: 4px; }
          .ls-kv span:last-child { max-width: 100%; text-align: left; }
          .ls-audit-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>

      <div className="ls-shell">
        <h1>Live Scan — Faces &amp; Plates</h1>
        <p className="ls-sub">
          One camera, both registries. Faces are scanned when a face is in frame;
          plates are read continuously and must agree across several frames before
          they raise an alert.{' '}
          <Link to="/">Return home</Link>
        </p>

        <div className="ls-card ls-pad" style={{ marginBottom: 14 }}>
          <div className="ls-meta">
            <div>Match Latency: {roundtripMs ?? '--'} ms</div>
            <div>Model: {MODEL_LABEL}</div>
            <div>Indexed Identities: {memberCount ?? '--'}</div>
            <div>Scan Interval: {SCAN_INTERVAL_MS} ms</div>
          </div>
          <div className="ls-ready">
            {readiness.map((item) => (
              <span key={item.label} className={item.ok ? 'ok' : 'bad'}>
                {item.label}: {item.ok ? 'Ready' : 'Waiting'}
              </span>
            ))}
          </div>
        </div>

        <div className="ls-grid">
          <div className="ls-card ls-pad">
            <div className="ls-camera">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                screenshotQuality={0.72}
                videoConstraints={{ width: 960, height: 720, facingMode: 'user' }}
                onUserMedia={() => {
                  setWebcamReady(true)
                  setWebcamError('')
                  setScanError('')
                }}
                onUserMediaError={(err) => {
                  setWebcamReady(false)
                  setWebcamError(err?.message || 'Unable to access camera stream.')
                }}
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'fill',
                  zIndex: 1,
                }}
              />

              <canvas
                ref={canvasRef}
                style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', zIndex: 2 }}
              />

              <div className="ls-overlay-top-left">
                <span className="ls-chip">REC / LIVE {fps || 30} FPS</span>
                <span className="ls-chip">Face: {(faceScore * 100).toFixed(1)}% • {toneLabel}</span>
                {plateResult?.extracted_text && (
                  <span className="ls-chip">
                    Plate: {plateResult.extracted_text}
                    {plateResult.stability && !plateResult.stability.confirmed
                      ? ` • ${plateResult.stability.agreeing_frames}/${plateResult.stability.required_frames}`
                      : plateResult.match_found ? ' • CONFIRMED' : ''}
                  </span>
                )}
              </div>

              {/* Plate boxes come back in captured-frame pixels, so they are
                  scaled to percentages rather than assumed to match the video's
                  rendered size. */}
              {plateResult?.frame?.width > 0 && (plateResult.regions || []).map((r, i) => (
                <div
                  key={i}
                  style={{
                    position: 'absolute',
                    left: `${(r.x / plateResult.frame.width) * 100}%`,
                    top: `${(r.y / plateResult.frame.height) * 100}%`,
                    width: `${(r.w / plateResult.frame.width) * 100}%`,
                    height: `${(r.h / plateResult.frame.height) * 100}%`,
                    border: `2px solid ${plateResult.alert ? '#b3261e' : '#2f6feb'}`,
                    borderRadius: 4,
                    zIndex: 3,
                    pointerEvents: 'none',
                  }}
                />
              ))}

              <div className="ls-overlay-top-right">
                <span className="ls-chip">{formatTimestamp(matchResult?.timestamp || new Date().toISOString())}</span>
              </div>
            </div>

            <div className="ls-card ls-pad" style={{ marginTop: 12 }}>
              <h3 style={{ margin: '0 0 8px', fontSize: 12, textTransform: 'uppercase', color: 'var(--muted)' }}>
                Vehicle plate
              </h3>
              <div style={{
                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                fontSize: 24, fontWeight: 700, letterSpacing: '.08em',
                color: plateResult?.alert ? 'var(--unknown)' : 'var(--fg)',
              }}>
                {plateResult?.extracted_text || '—'}
              </div>

              {plateResult?.stability && (
                <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4 }}>
                  {plateResult.stability.confirmed
                    ? 'Confirmed across frames'
                    : `${plateResult.stability.agreeing_frames || 0} of ${plateResult.stability.required_frames} frames agree`}
                </div>
              )}

              {plateResult?.match_found && (
                <div style={{ fontSize: 13, marginTop: 8 }}>
                  <strong>{plateResult.plate?.plate_number}</strong>
                  {' — '}
                  <span style={{ textTransform: 'uppercase' }}>{plateResult.status}</span>
                  <div style={{ color: 'var(--muted)' }}>{plateResult.plate?.owner_name}</div>
                </div>
              )}

              {plateResult?.message && (
                <div style={{
                  fontSize: 13, marginTop: 8,
                  color: plateResult.alert ? 'var(--unknown)' : 'var(--muted)',
                  fontWeight: plateResult.alert ? 600 : 400,
                }}>
                  {plateResult.message}
                </div>
              )}

              {plateSightings.length > 0 && (
                <div style={{ marginTop: 10, borderTop: '1px solid var(--line)', paddingTop: 8 }}>
                  <div style={{ fontSize: 11, textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 4 }}>
                    Plates seen this session
                  </div>
                  {plateSightings.map((s) => (
                    <div key={s.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '3px 0' }}>
                      <span style={{ fontFamily: 'ui-monospace, monospace' }}>{s.plateNumber}</span>
                      <span>{s.status}</span>
                      <span style={{ color: 'var(--muted)' }}>{new Date(s.at).toLocaleTimeString()}</span>
                    </div>
                  ))}
                </div>
              )}

              {plateError && <div className="ls-warn">Plate: {plateError}</div>}
            </div>

            <div className="ls-warn">
              {hasFaceInFrame
                ? 'Target lock acquired. Scanning against identity registry.'
                : 'No face in frame. Move closer and improve front lighting.'}
            </div>
          </div>

          <div className="ls-card ls-pad">
            <div className={`ls-verdict ${toneClass}`}>
              <h2 style={{ margin: 0 }}>
                {matchResult?.isAlert
                  ? 'Flagged identity detected'
                  : matchResult?.isKnownUser
                  ? 'Recognised identity'
                  : 'Awaiting confident match'}
              </h2>
              <p style={{ margin: '8px 0 0', color: 'var(--fg)', fontWeight: 600 }}>{matchResult?.fullName || '-'}</p>
              <p style={{ margin: '6px 0 0', color: 'var(--muted)', fontSize: 13 }}>{matchResult?.message || 'Live scan running.'}</p>
            </div>

            <h3 style={{ margin: '0 0 8px', fontSize: 12, textTransform: 'uppercase', color: 'var(--muted)' }}>How certain</h3>
            <div className="ls-meter">
              <i
                style={{
                  width: `${confidencePct}%`,
                  background:
                    toneClass === 'alert' ? 'var(--alert)' : toneClass === 'ok' ? 'var(--ok)' : 'var(--unknown)',
                }}
              />
            </div>

            <div className="ls-kv"><span>Confidence</span><span>{confidencePct}%</span></div>
            <div className="ls-kv"><span>Cosine distance</span><span>{typeof matchResult?.matchDistance === 'number' ? matchResult.matchDistance.toFixed(4) : '--'}</span></div>
            <div className="ls-kv"><span>Status</span><span>{(matchResult?.status || 'unknown').toUpperCase()}</span></div>
            <div className="ls-kv"><span>Pose</span><span>{matchResult?.poseLabel || '-'}</span></div>
            <div className="ls-kv"><span>Round trip</span><span>{roundtripMs ?? '--'} ms</span></div>
            <div className="ls-kv"><span>Server stages</span><span>{matchResult?.stageTiming ? JSON.stringify(matchResult.stageTiming) : '--'}</span></div>
          </div>
        </div>

        <div className="ls-card ls-pad" style={{ marginTop: 14 }}>
          <h3 style={{ margin: '0 0 10px', fontSize: 12, textTransform: 'uppercase', color: 'var(--muted)' }}>
            Reference comparison
          </h3>
          <div className="ls-side-by-side">
            <div>
              <div className="ls-caption">Live crop</div>
              <div className="ls-thumb-wrap">{liveCrop ? <img src={liveCrop} alt="Live crop" /> : null}</div>
            </div>
            <div>
              <div className="ls-caption">DB seed photo</div>
              <div className="ls-thumb-wrap">
                {dbImageUrl && !dbImageBroken ? (
                  <img
                    src={dbImageUrl}
                    alt="Database reference"
                    onError={() => setDbImageBroken(true)}
                  />
                ) : null}
              </div>
            </div>
          </div>
          <div className="ls-row" style={{ marginTop: 10, color: 'var(--muted)', fontSize: 12 }}>
            <span>Source: {matchResult?.sourceText || '-'}</span>
            <span>Pose: {matchResult?.poseLabel || '-'}</span>
          </div>
          {dbImageBroken ? (
            <div className="ls-warn" style={{ marginTop: 8 }}>
              Stored reference photo could not be loaded from blob storage.
            </div>
          ) : null}
        </div>

        <div className="ls-card ls-pad" style={{ marginTop: 14 }}>
          <h3 style={{ margin: '0 0 10px', fontSize: 12, textTransform: 'uppercase', color: 'var(--muted)' }}>Audit log</h3>
          <div className="ls-audit-grid" style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 10 }}>
            <div className="ls-audit">
              {history.map((entry) => (
                <button
                  key={entry.id}
                  className={`${auditTone(entry)}${selectedHistoryId === entry.id ? ' active' : ''}`}
                  onClick={() => {
                    setSelectedHistoryId(entry.id)
                    setDbImageBroken(false)
                  }}
                >
                  {formatTimestamp(entry.timestamp)} - {entry.fullName} ({(entry.status || 'unknown').toUpperCase()}) -{' '}
                  {typeof entry.matchConfidence === 'number' ? `${entry.matchConfidence.toFixed(1)}%` : '--'}
                </button>
              ))}
              {!history.length ? <div className="ls-sub">No detection events yet.</div> : null}
            </div>

            <div className="ls-card ls-pad">
              <div style={{ fontWeight: 700 }}>{activeEntry?.fullName || 'Select an event'}</div>
              <div className="ls-kv"><span>Status</span><span>{(activeEntry?.status || 'unknown').toUpperCase()}</span></div>
              <div className="ls-kv"><span>Distance</span><span>{typeof activeEntry?.matchDistance === 'number' ? activeEntry.matchDistance.toFixed(4) : '--'}</span></div>
              <div className="ls-kv"><span>Latency</span><span>{activeEntry?.latencyMs ?? '--'} ms</span></div>
              {activeEntry?.liveCrop ? <img src={activeEntry.liveCrop} alt="Event crop" style={{ width: '100%', borderRadius: 8, marginTop: 10 }} /> : null}
            </div>
          </div>
        </div>

        {webcamError ? <div className="ls-warn">Camera unavailable: {webcamError}</div> : null}
        {scanError ? <div className="ls-warn">{scanError}</div> : null}
        {modelState === 'loading' ? <div className="ls-sub">Initializing detector assets...</div> : null}
        {modelState === 'failed' ? <div className="ls-warn">Model Load Failed: {modelError || 'Unknown error.'}</div> : null}
      </div>
    </div>
  )
}
