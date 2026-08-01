import React, { useMemo, useRef, useEffect, useState } from 'react'
import Webcam from 'react-webcam'
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''
const SCAN_INTERVAL_MS = 1500
const HISTORY_LIMIT = 30
const MODEL_LABEL = 'Facenet512 / Cosine'

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))

const toTitle = (value) =>
  (value || '')
    .toLowerCase()
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (ch) => ch.toUpperCase())

const toConfidencePct = (distance) => {
  if (typeof distance !== 'number') return null
  return clamp((1 - distance) * 100, 0, 100)
}

const parsePoseFromUrl = (url) => {
  if (!url) return 'Unknown Pose'
  const name = url.split('/').pop() || ''
  const stripped = name.replace(/\.[a-zA-Z0-9]+$/, '')
  const parts = stripped.split('_').filter(Boolean)
  if (parts.length < 2) return 'Unknown Pose'
  return toTitle(parts.slice(1).join(' '))
}

const formatTimestamp = (iso) => {
  if (!iso) return '-'
  const d = new Date(iso)
  const yyyy = d.getFullYear()
  const mm = String(d.getMonth() + 1).padStart(2, '0')
  const dd = String(d.getDate()).padStart(2, '0')
  const hh = String(d.getHours()).padStart(2, '0')
  const mi = String(d.getMinutes()).padStart(2, '0')
  const ss = String(d.getSeconds()).padStart(2, '0')
  const ms = String(d.getMilliseconds()).padStart(3, '0')
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss}.${ms}`
}

const toneFromMatch = ({ hasFace, isAnalyzing, matchResult }) => {
  if (!hasFace) return 'idle'
  if (isAnalyzing) return 'scanning'
  if (matchResult?.isKnownUser && matchResult?.isAlert) return 'flagged'
  if (matchResult?.isKnownUser && matchResult?.status === 'verified') return 'verified'
  if (matchResult?.isKnownUser) return 'monitored'
  if (matchResult?.success && matchResult?.isKnownUser === false) return 'unregistered'
  return 'scanning'
}

const toneStyle = (tone) => {
  switch (tone) {
    case 'verified':
      return { color: '#22c55e', dashed: false, label: 'VERIFIED MEMBER' }
    case 'flagged':
      return { color: '#ef4444', dashed: false, label: 'FLAGGED' }
    case 'monitored':
      return { color: '#f97316', dashed: false, label: 'MONITORED' }
    case 'unregistered':
      return { color: '#f59e0b', dashed: true, label: 'UNREGISTERED' }
    case 'idle':
      return { color: '#64748b', dashed: true, label: 'NO TARGET' }
    default:
      return { color: '#22d3ee', dashed: true, label: 'SCANNING' }
  }
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
  return c.toDataURL('image/jpeg', 0.84)
}

const drawReticle = (ctx, box, color, dashed, scorePct) => {
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

  const cx = box.x + box.w / 2
  const cy = box.y + box.h / 2
  ctx.lineWidth = 1.5
  ctx.setLineDash([])
  ctx.beginPath()
  ctx.moveTo(cx - 10, cy)
  ctx.lineTo(cx + 10, cy)
  ctx.moveTo(cx, cy - 10)
  ctx.lineTo(cx, cy + 10)
  ctx.stroke()

  ctx.font = '700 12px ui-monospace, SFMono-Regular, Menlo, monospace'
  const label = `${scorePct}%`
  ctx.fillText(label, box.x, Math.max(14, box.y - 6))
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

  const isModelLoaded = modelState === 'ready'

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

        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        )

        detectorRef.current = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite',
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

  const tone = toneFromMatch({ hasFace: hasFaceInFrame, isAnalyzing, matchResult })
  const toneMeta = toneStyle(tone)

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

            drawReticle(ctx, box, toneMeta.color, toneMeta.dashed, Math.round(score * 100))
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
  }, [isModelLoaded, webcamReady, toneMeta.color, toneMeta.dashed])

  useEffect(() => {
    if (!isModelLoaded || !webcamReady) return undefined

    const id = setInterval(() => {
      if (!isAnalyzing && hasFaceInFrame) {
        sendFrameToBackend()
      }
    }, SCAN_INTERVAL_MS)

    return () => clearInterval(id)
  }, [isModelLoaded, webcamReady, isAnalyzing, hasFaceInFrame])

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
    if (!webcamReady) {
      setScanError('Camera stream is not ready yet.')
      return
    }

    const imageSrc = webcam.getScreenshot()
    if (!imageSrc) {
      setScanError('Camera frame unavailable. Check camera permissions and device access.')
      return
    }

    const localCrop = captureCrop(video, faceBoxRef.current)

    try {
      setIsAnalyzing(true)
      setScanError('')

      const started = performance.now()
      const blob = await (await fetch(imageSrc)).blob()
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
        sourceText: capture?.source ? toTitle(capture.source) : 'Registry Seed',
        liveCrop: localCrop,
        latencyMs: durationMs,
      }

      if (normalized.success === false) {
        setScanError(normalized.error || 'Scan completed but no usable face was detected.')
      }

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
      setScanError(`Scan request failed. Retrying in ${Math.round(cooldownMs / 1000)}s.`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const activeEntry = useMemo(() => {
    if (!history.length) return null
    return history.find((h) => h.id === selectedHistoryId) || history[0]
  }, [history, selectedHistoryId])

  const confidenceText =
    typeof matchResult?.matchConfidence === 'number' ? `${matchResult.matchConfidence.toFixed(2)}%` : '--'

  const distanceText =
    typeof matchResult?.matchDistance === 'number' ? matchResult.matchDistance.toFixed(4) : '--'

  return (
    <div
      style={{
        minHeight: '100vh',
        padding: '24px 18px 28px',
        color: '#e2e8f0',
        background:
          'radial-gradient(1600px 700px at 15% -10%, #052e37 0%, #020617 45%, #02040b 100%)',
        fontFamily: 'Sora, Segoe UI, system-ui, sans-serif',
      }}
    >
      <div style={{ maxWidth: 1280, margin: '0 auto', display: 'grid', gap: 14 }}>
        <div
          style={{
            border: '1px solid rgba(45,212,191,0.25)',
            borderRadius: 12,
            padding: '10px 12px',
            background: 'rgba(2,6,23,0.65)',
            display: 'grid',
            gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
            gap: 8,
          }}
        >
          <div style={{ fontSize: 12, color: '#7dd3fc' }}>Match Latency: {roundtripMs ?? '--'} ms</div>
          <div style={{ fontSize: 12, color: '#7dd3fc' }}>Model: {MODEL_LABEL}</div>
          <div style={{ fontSize: 12, color: '#7dd3fc' }}>Indexed Identities: {memberCount ?? '--'}</div>
          <div style={{ fontSize: 12, color: '#7dd3fc' }}>Scan Interval: {SCAN_INTERVAL_MS} ms</div>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(420px, 2fr) minmax(320px, 1fr)',
            gap: 14,
          }}
        >
          <div
            style={{
              borderRadius: 16,
              border: '1px solid rgba(34,211,238,0.28)',
              background: 'linear-gradient(180deg, rgba(3,7,18,0.85), rgba(2,6,23,0.75))',
              padding: 12,
            }}
          >
            <div
              style={{
                position: 'relative',
                width: '100%',
                maxWidth: 760,
                aspectRatio: '4 / 3',
                overflow: 'hidden',
                borderRadius: 12,
                border: '1px solid rgba(100,116,139,0.45)',
                background: '#000',
              }}
            >
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                screenshotQuality={0.85}
                videoConstraints={{ width: 1280, height: 960, facingMode: 'user' }}
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

              <div
                style={{
                  position: 'absolute',
                  top: 12,
                  left: 12,
                  display: 'grid',
                  gap: 6,
                  zIndex: 3,
                }}
              >
                <div
                  style={{
                    fontSize: 11,
                    letterSpacing: 0.8,
                    fontWeight: 700,
                    color: '#f8fafc',
                    padding: '5px 10px',
                    borderRadius: 999,
                    border: '1px solid rgba(248,113,113,0.7)',
                    background: 'rgba(185,28,28,0.4)',
                  }}
                >
                  REC / LIVE {fps || 30} FPS
                </div>
                <div
                  style={{
                    fontSize: 11,
                    letterSpacing: 0.5,
                    fontWeight: 600,
                    color: '#e2e8f0',
                    padding: '5px 10px',
                    borderRadius: 999,
                    border: `1px solid ${toneMeta.color}`,
                    background: 'rgba(2,6,23,0.82)',
                  }}
                >
                  Face Detected: {(faceScore * 100).toFixed(1)}% · {toneMeta.label}
                </div>
              </div>

              <div
                style={{
                  position: 'absolute',
                  right: 12,
                  top: 12,
                  zIndex: 3,
                  color: '#cbd5e1',
                  fontSize: 11,
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                  background: 'rgba(2,6,23,0.68)',
                  border: '1px solid rgba(100,116,139,0.45)',
                  borderRadius: 8,
                  padding: '6px 8px',
                }}
              >
                {formatTimestamp(matchResult?.timestamp || new Date().toISOString())}
              </div>
            </div>

            <div style={{ marginTop: 10, fontSize: 12, color: '#94a3b8' }}>
              {hasFaceInFrame
                ? 'Target lock acquired. Scanning against identity registry.'
                : 'No face in frame. Move closer and increase front lighting.'}
            </div>
          </div>

          <div
            style={{
              borderRadius: 16,
              border: '1px solid rgba(56,189,248,0.28)',
              background: 'linear-gradient(180deg, rgba(4,13,29,0.9), rgba(2,6,23,0.78))',
              padding: 12,
              display: 'grid',
              gridTemplateRows: 'auto auto 1fr',
              gap: 12,
            }}
          >
            <div style={{ display: 'grid', gap: 6 }}>
              <div style={{ fontSize: 11, color: '#67e8f9', letterSpacing: 1, textTransform: 'uppercase' }}>
                Target Profile
              </div>
              <div style={{ fontSize: 28, fontWeight: 700, color: '#f8fafc', lineHeight: 1.05 }}>
                {matchResult?.fullName || 'Awaiting Match'}
              </div>
              <div
                style={{
                  display: 'inline-block',
                  width: 'fit-content',
                  padding: '4px 10px',
                  borderRadius: 999,
                  border: `1px solid ${toneMeta.color}`,
                  color: toneMeta.color,
                  fontSize: 12,
                  fontWeight: 700,
                  letterSpacing: 0.6,
                }}
              >
                {toneMeta.label}
              </div>
            </div>

            <div style={{ display: 'grid', gap: 8 }}>
              <div style={{ fontSize: 12, color: '#a5b4fc' }}>Match Confidence: {confidenceText}</div>
              <div
                style={{
                  height: 10,
                  borderRadius: 999,
                  background: 'rgba(71,85,105,0.42)',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${matchResult?.matchConfidence || 0}%`,
                    height: '100%',
                    background: `linear-gradient(90deg, ${toneMeta.color}, #22d3ee)`,
                  }}
                />
              </div>

              <div style={{ display: 'grid', gap: 6, fontSize: 12, color: '#cbd5e1' }}>
                <div>Pose Angle: {matchResult?.poseLabel || '-'}</div>
                <div>Cosine Dist: {distanceText}</div>
                <div>Timestamp: {formatTimestamp(matchResult?.timestamp)}</div>
                <div>Status: {(matchResult?.status || 'unknown').toUpperCase()}</div>
              </div>
            </div>

            <div style={{ display: 'grid', gap: 10 }}>
              <div style={{ fontSize: 11, color: '#7dd3fc', letterSpacing: 1, textTransform: 'uppercase' }}>
                Reference Comparison
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <div style={{ display: 'grid', gap: 4 }}>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>Live Crop</div>
                  <div
                    style={{
                      border: '1px solid rgba(100,116,139,0.5)',
                      borderRadius: 10,
                      overflow: 'hidden',
                      aspectRatio: '1 / 1',
                      background: '#020617',
                    }}
                  >
                    {liveCrop ? (
                      <img src={liveCrop} alt="Live crop" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    ) : null}
                  </div>
                </div>

                <div style={{ display: 'grid', gap: 4 }}>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>DB Seed Photo</div>
                  <div
                    style={{
                      border: '1px solid rgba(100,116,139,0.5)',
                      borderRadius: 10,
                      overflow: 'hidden',
                      aspectRatio: '1 / 1',
                      background: '#020617',
                    }}
                  >
                    {matchResult?.sourceImageUrl ? (
                      <img
                        src={matchResult.sourceImageUrl}
                        alt="Database reference"
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    ) : null}
                  </div>
                </div>
              </div>

              <div style={{ fontSize: 11, color: '#94a3b8' }}>
                Source: {matchResult?.sourceText || '-'} · Pose: {matchResult?.poseLabel || '-'}
              </div>
            </div>
          </div>
        </div>

        <div
          style={{
            border: '1px solid rgba(56,189,248,0.28)',
            borderRadius: 12,
            background: 'rgba(2,6,23,0.72)',
            padding: 12,
            display: 'grid',
            gap: 10,
          }}
        >
          <div style={{ fontSize: 11, color: '#67e8f9', letterSpacing: 1, textTransform: 'uppercase' }}>
            Audit Log
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 10 }}>
            <div style={{ maxHeight: 170, overflowY: 'auto', display: 'grid', gap: 6 }}>
              {history.map((entry) => (
                <button
                  key={entry.id}
                  onClick={() => setSelectedHistoryId(entry.id)}
                  style={{
                    textAlign: 'left',
                    borderRadius: 10,
                    border:
                      selectedHistoryId === entry.id
                        ? '1px solid rgba(34,211,238,0.75)'
                        : '1px solid rgba(100,116,139,0.45)',
                    background: selectedHistoryId === entry.id ? 'rgba(14,116,144,0.22)' : 'rgba(15,23,42,0.45)',
                    color: '#e2e8f0',
                    padding: '8px 10px',
                    cursor: 'pointer',
                    fontSize: 12,
                  }}
                >
                  {formatTimestamp(entry.timestamp)} - {entry.fullName} ({(entry.status || 'unknown').toUpperCase()}) -{' '}
                  {typeof entry.matchConfidence === 'number' ? `${entry.matchConfidence.toFixed(1)}%` : '--'}
                </button>
              ))}
              {!history.length ? <div style={{ color: '#94a3b8', fontSize: 12 }}>No detection events yet.</div> : null}
            </div>

            <div
              style={{
                border: '1px solid rgba(100,116,139,0.45)',
                borderRadius: 10,
                padding: 10,
                fontSize: 12,
                color: '#cbd5e1',
                display: 'grid',
                gap: 8,
              }}
            >
              <div style={{ fontWeight: 700, color: '#e2e8f0' }}>{activeEntry?.fullName || 'Select an event'}</div>
              <div>Status: {(activeEntry?.status || 'unknown').toUpperCase()}</div>
              <div>Distance: {typeof activeEntry?.matchDistance === 'number' ? activeEntry.matchDistance.toFixed(4) : '--'}</div>
              <div>Latency: {activeEntry?.latencyMs ?? '--'} ms</div>
              {activeEntry?.liveCrop ? (
                <img src={activeEntry.liveCrop} alt="Event crop" style={{ width: '100%', borderRadius: 8 }} />
              ) : null}
            </div>
          </div>
        </div>

        {webcamError ? (
          <div style={{ color: '#fecaca', fontSize: 13 }}>
            Camera unavailable: {webcamError}. Ensure camera is connected, free, and permission is granted.
          </div>
        ) : null}

        {scanError ? <div style={{ color: '#fcd34d', fontSize: 13 }}>{scanError}</div> : null}

        {modelState === 'loading' ? <div style={{ color: '#94a3b8', fontSize: 13 }}>Initializing detector...</div> : null}

        {modelState === 'failed' ? (
          <div style={{ color: '#fda4af', fontSize: 13 }}>Model Load Failed: {modelError || 'Unknown model loading error.'}</div>
        ) : null}
      </div>
    </div>
  )
}