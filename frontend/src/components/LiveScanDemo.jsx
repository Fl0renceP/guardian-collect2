import React, { useMemo, useRef, useEffect, useState } from 'react'
import Webcam from 'react-webcam'
import { Link } from 'react-router-dom'
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''
const SCAN_INTERVAL_MS = 1500
const HISTORY_LIMIT = 50
const MODEL_LABEL = 'Facenet512 / Cosine'

// One person standing in front of the camera is one sighting, not forty. The
// feed scans every 1.5s and now resolves every face at once, so without this the
// audit log would hold about five seconds of the same handful of people and an
// operator would never scroll back to a real event. Matches the server-side
// alert window in spirit; the log is deliberately the shorter of the two.
const AUDIT_DEDUPE_MS = 30000

// How long the identities from one scan keep annotating the live boxes. Long
// enough to bridge the gap between scans (plus the round trip), short enough
// that a name never lingers on a face after the person has walked out.
const LABEL_TTL_MS = 4500
// Overlap below which a returned identity is assumed to belong to a different
// face than the one currently being drawn, and so is not attached to it.
const MIN_LABEL_IOU = 0.25

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

const toneForFace = (face) => {
  if (!face) return 'unknown'
  if (face.isAlert) return 'alert'
  if (face.isKnownUser) return 'ok'
  return 'unknown'
}

const labelForFace = (face) => {
  if (!face) return 'UNREGISTERED'
  if (face.isAlert) return 'FLAGGED'
  if (face.isKnownUser) return 'RECOGNISED'
  return 'UNREGISTERED'
}

// A frame's tone is its worst face: one flagged person among four verified ones
// is still a flagged person, and the header must not average that away.
const TONE_SEVERITY = { unknown: 0, ok: 1, alert: 2 }
const frameTone = (faces) =>
  faces.reduce((worst, face) => {
    const tone = toneForFace(face)
    return TONE_SEVERITY[tone] > TONE_SEVERITY[worst] ? tone : worst
  }, 'unknown')

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

const boxIoU = (a, b) => {
  if (!a || !b) return 0
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.w, b.x + b.w)
  const y2 = Math.min(a.y + a.h, b.y + b.h)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  if (!inter) return 0
  return inter / (a.w * a.h + b.w * b.h - inter)
}

// One frozen frame, kept as a canvas as well as a blob. The crops have to come
// from the exact image the server saw: cropping the live video instead would
// use whatever the subjects are doing a round trip later, which in a group is
// reliably the wrong faces in the wrong places.
const captureFrameSnapshot = (video, quality = 0.72, maxWidth = 720) => {
  if (!video || !video.videoWidth || !video.videoHeight) return Promise.resolve(null)

  const ratio = Math.min(1, maxWidth / video.videoWidth)
  const width = Math.max(1, Math.round(video.videoWidth * ratio))
  const height = Math.max(1, Math.round(video.videoHeight * ratio))

  const c = document.createElement('canvas')
  c.width = width
  c.height = height
  c.getContext('2d').drawImage(video, 0, 0, width, height)

  return new Promise((resolve) => {
    c.toBlob(
      (blob) => resolve(blob ? { blob, canvas: c, width, height } : null),
      'image/jpeg',
      quality
    )
  })
}

const cropFromCanvas = (source, bbox) => {
  if (!source || !bbox || !bbox.w || !bbox.h) return null

  const padX = bbox.w * 0.2
  const padY = bbox.h * 0.25
  const sx = clamp(Math.floor(bbox.x - padX), 0, source.width - 1)
  const sy = clamp(Math.floor(bbox.y - padY), 0, source.height - 1)
  const ex = clamp(Math.ceil(bbox.x + bbox.w + padX), 1, source.width)
  const ey = clamp(Math.ceil(bbox.y + bbox.h + padY), 1, source.height)
  const sw = Math.max(1, ex - sx)
  const sh = Math.max(1, ey - sy)

  const c = document.createElement('canvas')
  c.width = 180
  c.height = 180
  c.getContext('2d').drawImage(source, sx, sy, sw, sh, 0, 0, c.width, c.height)
  return c.toDataURL('image/jpeg', 0.8)
}

const drawReticle = (ctx, box, toneClass, caption) => {
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

  // With several boxes on screen at once, a bare percentage is unreadable — each
  // reticle has to say who it is. Chip behind the text so a name stays legible
  // against a bright shirt or a window.
  if (caption) {
    ctx.font = '700 12px ui-monospace, SFMono-Regular, Menlo, monospace'
    const padding = 5
    const textWidth = ctx.measureText(caption).width
    const chipH = 18
    const above = box.y - chipH - 4 >= 0
    const chipX = clamp(box.x, 0, Math.max(0, ctx.canvas.width - textWidth - padding * 2))
    const chipY = above ? box.y - chipH - 4 : box.y + 4

    ctx.fillStyle = color
    ctx.fillRect(chipX, chipY, textWidth + padding * 2, chipH)
    ctx.fillStyle = '#ffffff'
    ctx.fillText(caption, chipX + padding, chipY + chipH - 5)
  }

  ctx.restore()
}

export default function LiveScanDemo() {
  const webcamRef = useRef(null)
  const canvasRef = useRef(null)
  const detectorRef = useRef(null)
  const lastVideoTimeRef = useRef(-1)
  const lastFrameAtRef = useRef(0)
  const failureCountRef = useRef(0)
  const nextScanAllowedAtRef = useRef(0)
  // Identities from the most recent scan, in video-pixel coordinates, re-attached
  // to whatever the detector is seeing right now on every animation frame.
  const labelsRef = useRef(null)
  // face key -> when it last earned an audit entry. See AUDIT_DEDUPE_MS.
  const lastLoggedRef = useRef(new Map())

  const [modelState, setModelState] = useState('loading')
  const [modelError, setModelError] = useState('')
  const [backendReady, setBackendReady] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [scanError, setScanError] = useState('')
  const [webcamReady, setWebcamReady] = useState(false)
  const [webcamError, setWebcamError] = useState('')
  const [faceCount, setFaceCount] = useState(0)
  const [fps, setFps] = useState(0)
  const [memberCount, setMemberCount] = useState(null)
  const [roundtripMs, setRoundtripMs] = useState(null)
  const [frameFaces, setFrameFaces] = useState([])
  const [truncatedFaces, setTruncatedFaces] = useState(0)
  const [selectedFaceKey, setSelectedFaceKey] = useState(null)
  const [history, setHistory] = useState([])
  const [selectedHistoryId, setSelectedHistoryId] = useState(null)
  const [dbImageBroken, setDbImageBroken] = useState(false)

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
          // BlazeFace returns every face it finds — there is no count to raise.
          // What did limit us was suppression: adjacent people in a group get
          // overlapping boxes, and the 0.3 default merges the pair into one.
          minSuppressionThreshold: 0.45,
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

  // The overlay loop reads identities out of a ref rather than state, so adding
  // a face to the frame doesn't tear down and rebuild the animation loop.
  useEffect(() => {
    let frameId

    const findLabel = (videoBox, now) => {
      const labels = labelsRef.current
      if (!labels || now - labels.at > LABEL_TTL_MS) return null

      let best = null
      let bestScore = MIN_LABEL_IOU
      for (const face of labels.faces) {
        const score = boxIoU(videoBox, face.box)
        if (score > bestScore) {
          bestScore = score
          best = face
        }
      }
      return best
    }

    const run = async () => {
      const video = webcamRef.current?.video
      const canvas = canvasRef.current
      const detector = detectorRef.current

      if (video && canvas && detector && webcamReady && isModelLoaded && video.readyState === 4) {
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

          const sx = rw / vw
          const sy = rh / vh
          const wallClock = Date.now()

          for (const detection of detections) {
            const b = detection.boundingBox
            const videoBox = { x: b.originX, y: b.originY, w: b.width, h: b.height }
            const drawBox = { x: b.originX * sx, y: b.originY * sy, w: b.width * sx, h: b.height * sy }
            const score = detection.categories?.[0]?.score || 0

            const label = findLabel(videoBox, wallClock)
            const caption = label
              ? `${label.fullName} · ${label.matchConfidence}%`
              : `${Math.round(score * 100)}%`

            drawReticle(ctx, drawBox, toneForFace(label), caption)
          }

          setFaceCount(detections.length)
        }
      }

      frameId = requestAnimationFrame(run)
    }

    if (isModelLoaded) run()
    return () => cancelAnimationFrame(frameId)
  }, [isModelLoaded, webcamReady])

  useEffect(() => {
    if (!isModelLoaded || !webcamReady || !backendReady) return undefined

    const id = setInterval(() => {
      if (!isAnalyzing && faceCount > 0) {
        sendFrameToBackend()
      }
    }, SCAN_INTERVAL_MS)

    return () => clearInterval(id)
  }, [isModelLoaded, webcamReady, backendReady, isAnalyzing, faceCount])

  const addHistory = (entries) => {
    if (!entries.length) return
    setHistory((prev) => {
      const next = [...entries, ...prev].slice(0, HISTORY_LIMIT)
      if (!selectedHistoryId) setSelectedHistoryId(entries[0].id)
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

    try {
      setIsAnalyzing(true)
      setScanError('')

      const started = performance.now()
      const snapshot = await captureFrameSnapshot(video, 0.72, 720)
      if (!snapshot) throw new Error('Camera frame unavailable.')

      const formData = new FormData()
      formData.append('file', snapshot.blob, 'live_scan.jpg')

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

      // `faces` is the multi-face response. Fall back to the flat single-face
      // shape so this component still works against an older backend.
      const rawFaces = Array.isArray(data?.faces)
        ? data.faces
        : data?.success
        ? [{ ...data, index: 0, is_primary: true }]
        : []

      // Boxes come back in the coordinate space of the image that was uploaded,
      // which is the snapshot, not the full-resolution video.
      const serverWidth = data?.image_size?.width || snapshot.width
      const cropScale = snapshot.width / serverWidth
      const videoScale = (video.videoWidth || snapshot.width) / serverWidth

      const ts = new Date().toISOString()
      const stamp = Date.now()
      const usedKeys = new Set()

      const faces = rawFaces.map((face, i) => {
        const area = face.facial_area || {}
        const hasBox = typeof area.x === 'number' && area.w > 0 && area.h > 0
        const scaleBox = (k) =>
          hasBox
            ? { x: area.x * k, y: area.y * k, w: area.w * k, h: area.h * k }
            : null

        const distance = typeof face.match_distance === 'number' ? face.match_distance : null
        const capture = Array.isArray(face.supporting_captures) ? face.supporting_captures[0] : null
        const person = face.person || null

        // Keyed on the identity where there is one, so selecting a person keeps
        // that person selected across scans even as the face order shifts. Two
        // faces can legitimately resolve to the same identity — a person and a
        // photo of them on a wall — so fall back to the slot to stay unique.
        let key = person?.id ? `person-${person.id}` : `slot-${face.index ?? i}`
        if (usedKeys.has(key)) key = `${key}-${face.index ?? i}`
        usedKeys.add(key)

        return {
          key,
          id: `scan-${stamp}-${face.index ?? i}`,
          index: face.index ?? i,
          isPrimary: !!face.is_primary,
          isKnownUser: !!face.is_known_user,
          isAlert: !!face.alert,
          status: face.status || null,
          fullName: person?.full_name || 'Unknown Identity',
          matchDistance: distance,
          matchConfidence: toConfidencePct(distance),
          message: face.message || '',
          timestamp: ts,
          poseLabel: parsePoseFromUrl(capture?.image_url || person?.image_url),
          personImageUrl: person?.image_url || null,
          sourceImageUrl: capture?.image_url || person?.image_url || null,
          sourceText: capture?.source || 'Registry Seed',
          faceConfidence: face.face_confidence ?? null,
          qualityScore: face.capture_quality?.quality_score ?? null,
          matchedAgainstPhotos: face.matched_against_photos ?? null,
          agreeingCaptures: face.agreeing_captures ?? null,
          marginToNext: face.margin_to_next_person ?? null,
          box: scaleBox(videoScale),
          crop: hasBox ? cropFromCanvas(snapshot.canvas, scaleBox(cropScale)) : null,
          latencyMs: durationMs,
          stageTiming: data?.timings_ms || null,
        }
      })

      labelsRef.current = { at: stamp, faces: faces.filter((f) => f.box) }

      // The panel shows every face every scan; the audit log records each one
      // once per appearance, so it stays a log rather than a ticker.
      const lastLogged = lastLoggedRef.current
      const fresh = faces.filter((face) => {
        const seenAt = lastLogged.get(face.key)
        if (seenAt && stamp - seenAt < AUDIT_DEDUPE_MS) return false
        lastLogged.set(face.key, stamp)
        return true
      })
      for (const [key, seenAt] of lastLogged) {
        if (stamp - seenAt >= AUDIT_DEDUPE_MS) lastLogged.delete(key)
      }

      setDbImageBroken(false)
      setFrameFaces(faces)
      setTruncatedFaces(data?.faces_truncated || 0)
      addHistory(fresh)
      failureCountRef.current = 0
      nextScanAllowedAtRef.current = 0

      // A 200 with success:false means the server's detector found nothing in a
      // frame the browser detector did see — an angle RetinaFace won't take, or
      // a face too small once the frame was scaled down for upload. Say so
      // rather than leaving a stale panel that looks like a live reading.
      if (data?.success === false) {
        setScanError(data?.error || 'Server could not resolve any face in the frame.')
      }
    } catch (err) {
      const nextFailureCount = failureCountRef.current + 1
      failureCountRef.current = nextFailureCount
      const cooldownMs = Math.min(5000, 500 * Math.pow(2, nextFailureCount - 1))
      nextScanAllowedAtRef.current = Date.now() + cooldownMs
      setScanError(err?.message || `Scan request failed. Retrying in ${Math.round(cooldownMs / 1000)}s.`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const primaryFace = useMemo(
    () => frameFaces.find((f) => f.isPrimary) || frameFaces[0] || null,
    [frameFaces]
  )

  const selectedFace = useMemo(
    () => frameFaces.find((f) => f.key === selectedFaceKey) || primaryFace,
    [frameFaces, selectedFaceKey, primaryFace]
  )

  const flaggedCount = frameFaces.filter((f) => f.isAlert).length
  const knownCount = frameFaces.filter((f) => f.isKnownUser).length
  const unknownCount = frameFaces.length - knownCount

  const toneClass = frameTone(frameFaces)
  const toneLabel = frameFaces.length ? labelForFace(primaryFace) : faceCount ? 'SCANNING' : 'NO TARGET'

  const activeEntry = useMemo(() => {
    if (!history.length) return null
    return history.find((h) => h.id === selectedHistoryId) || history[0]
  }, [history, selectedHistoryId])

  const dbImageUrl = selectedFace?.sourceImageUrl || selectedFace?.personImageUrl || null

  const readiness = [
    { label: 'Camera', ok: webcamReady },
    { label: 'Detector', ok: isModelLoaded },
    { label: 'Backend', ok: backendReady },
  ]

  const confidencePct = typeof selectedFace?.matchConfidence === 'number' ? selectedFace.matchConfidence : 0

  const verdictHeadline = () => {
    if (flaggedCount) return `${flaggedCount} flagged ${flaggedCount === 1 ? 'identity' : 'identities'} in frame`
    if (knownCount) return `${knownCount} recognised ${knownCount === 1 ? 'identity' : 'identities'}`
    if (frameFaces.length) return `${frameFaces.length} ${frameFaces.length === 1 ? 'face' : 'faces'}, none matched`
    return 'Awaiting confident match'
  }

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
        .ls-faces { display: grid; gap: 6px; margin-bottom: 12px; max-height: 250px; overflow-y: auto; }
        .ls-faces button { display: flex; align-items: center; gap: 9px; text-align: left; width: 100%; border-radius: 9px; border: 1px solid var(--line); background: var(--panel); color: var(--fg); padding: 7px 9px; cursor: pointer; font-size: 12px; }
        .ls-faces button.alert { border-color: var(--alert); background: color-mix(in srgb, var(--alert) 12%, var(--panel)); }
        .ls-faces button.ok { border-color: var(--ok); background: color-mix(in srgb, var(--ok) 10%, var(--panel)); }
        .ls-faces button.unknown { border-color: var(--unknown); background: color-mix(in srgb, var(--unknown) 10%, var(--panel)); }
        .ls-faces button.active { box-shadow: inset 0 0 0 2px var(--accent); }
        .ls-faces img { width: 34px; height: 34px; border-radius: 7px; object-fit: cover; flex: 0 0 auto; }
        .ls-face-name { font-weight: 700; overflow-wrap: anywhere; }
        .ls-face-sub { color: var(--muted); font-size: 11px; }
        .ls-audit { max-height: 220px; overflow-y: auto; display: grid; gap: 6px; }
        .ls-audit button { text-align: left; border-radius: 9px; border: 1px solid var(--line); background: var(--panel); color: var(--fg); padding: 8px 9px; cursor: pointer; font-size: 12px; }
        .ls-audit button.active { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 8%, var(--panel)); }
        .ls-audit button.alert { border-color: var(--alert); background: color-mix(in srgb, var(--alert) 14%, var(--panel)); color: var(--alert); }
        .ls-audit button.ok { border-color: var(--ok); background: color-mix(in srgb, var(--ok) 12%, var(--panel)); color: var(--ok); }
        .ls-audit button.unknown { border-color: var(--unknown); background: color-mix(in srgb, var(--unknown) 12%, var(--panel)); color: var(--unknown); }
        .ls-audit button.alert.active,
        .ls-audit button.ok.active,
        .ls-audit button.unknown.active { box-shadow: inset 0 0 0 1px var(--accent); }
        .ls-side-by-side { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .ls-thumb-wrap { border: 1px solid var(--line); border-radius: 9px; overflow: hidden; background: var(--bg); aspect-ratio: 1/1; }
        .ls-thumb-wrap img { width: 100%; height: 100%; object-fit: cover; }
        .ls-caption { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
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
        <h1>Face Scan (Live)</h1>
        <p className="ls-sub">
          Live camera scan against registry — every face in frame is tracked and matched independently.{' '}
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
                <span className="ls-chip">
                  Faces: {faceCount} • {knownCount} known • {flaggedCount} flagged
                </span>
                <span className="ls-chip">{toneLabel}</span>
              </div>

              <div className="ls-overlay-top-right">
                <span className="ls-chip">{formatTimestamp(primaryFace?.timestamp || new Date().toISOString())}</span>
              </div>
            </div>

            <div className="ls-warn">
              {faceCount
                ? `Tracking ${faceCount} ${faceCount === 1 ? 'face' : 'faces'}. Each is matched against the identity registry independently.`
                : 'No face in frame. Move closer and improve front lighting.'}
            </div>
            {truncatedFaces > 0 ? (
              <div className="ls-warn">
                {truncatedFaces} further {truncatedFaces === 1 ? 'face was' : 'faces were'} detected but not
                identified — the frame is over the per-scan cap. The smallest faces are dropped first.
              </div>
            ) : null}
          </div>

          <div className="ls-card ls-pad">
            <div className={`ls-verdict ${toneClass}`}>
              <h2 style={{ margin: 0, fontSize: 18 }}>{verdictHeadline()}</h2>
              <p style={{ margin: '8px 0 0', color: 'var(--fg)', fontSize: 13 }}>
                {frameFaces.length
                  ? `${frameFaces.length} matched this frame — ${knownCount} known, ${unknownCount} unregistered.`
                  : 'Live scan running.'}
              </p>
            </div>

            <h3 style={{ margin: '0 0 8px', fontSize: 12, textTransform: 'uppercase', color: 'var(--muted)' }}>
              Faces in frame {frameFaces.length ? `(${frameFaces.length})` : ''}
            </h3>
            <div className="ls-faces">
              {frameFaces.map((face) => (
                <button
                  key={face.key}
                  className={`${toneForFace(face)}${selectedFace?.key === face.key ? ' active' : ''}`}
                  onClick={() => {
                    setSelectedFaceKey(face.key)
                    setDbImageBroken(false)
                  }}
                >
                  {face.crop ? <img src={face.crop} alt="" /> : null}
                  <span>
                    <span className="ls-face-name">{face.fullName}</span>
                    <br />
                    <span className="ls-face-sub">
                      {labelForFace(face)} • {(face.status || 'unregistered').toUpperCase()} •{' '}
                      {face.matchConfidence}%
                    </span>
                  </span>
                </button>
              ))}
              {!frameFaces.length ? <div className="ls-sub">No identities resolved yet.</div> : null}
            </div>

            <h3 style={{ margin: '0 0 8px', fontSize: 12, textTransform: 'uppercase', color: 'var(--muted)' }}>
              How certain {selectedFace ? `— ${selectedFace.fullName}` : ''}
            </h3>
            <div className="ls-meter">
              <i
                style={{
                  width: `${confidencePct}%`,
                  background:
                    toneForFace(selectedFace) === 'alert'
                      ? 'var(--alert)'
                      : toneForFace(selectedFace) === 'ok'
                      ? 'var(--ok)'
                      : 'var(--unknown)',
                }}
              />
            </div>

            <div className="ls-kv"><span>Confidence</span><span>{confidencePct}%</span></div>
            <div className="ls-kv"><span>Cosine distance</span><span>{typeof selectedFace?.matchDistance === 'number' ? selectedFace.matchDistance.toFixed(4) : '--'}</span></div>
            <div className="ls-kv"><span>Status</span><span>{(selectedFace?.status || 'unknown').toUpperCase()}</span></div>
            <div className="ls-kv"><span>Margin to next person</span><span>{typeof selectedFace?.marginToNext === 'number' ? selectedFace.marginToNext.toFixed(4) : '--'}</span></div>
            <div className="ls-kv"><span>Pose</span><span>{selectedFace?.poseLabel || '-'}</span></div>
            <div className="ls-kv"><span>Round trip</span><span>{roundtripMs ?? '--'} ms</span></div>
            <div className="ls-kv"><span>Server stages</span><span>{selectedFace?.stageTiming ? JSON.stringify(selectedFace.stageTiming) : '--'}</span></div>
          </div>
        </div>

        <div className="ls-card ls-pad" style={{ marginTop: 14 }}>
          <h3 style={{ margin: '0 0 10px', fontSize: 12, textTransform: 'uppercase', color: 'var(--muted)' }}>
            Reference comparison {selectedFace ? `— ${selectedFace.fullName}` : ''}
          </h3>
          <div className="ls-side-by-side">
            <div>
              <div className="ls-caption">Live crop</div>
              <div className="ls-thumb-wrap">{selectedFace?.crop ? <img src={selectedFace.crop} alt="Live crop" /> : null}</div>
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
            <span>Source: {selectedFace?.sourceText || '-'}</span>
            <span>Pose: {selectedFace?.poseLabel || '-'}</span>
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
              {activeEntry?.crop ? <img src={activeEntry.crop} alt="Event crop" style={{ width: '100%', borderRadius: 8, marginTop: 10 }} /> : null}
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
