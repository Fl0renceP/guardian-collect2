import React, { useMemo, useRef, useEffect, useState } from 'react'
import Webcam from 'react-webcam'
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision'

import { detectPlateRegions, cropRegionBlob, unionOfRegions } from '../lib/plateRegions'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''
const SCAN_INTERVAL_MS = 1500
// Which camera this stream represents. Echoed back on every scan and carried
// into behavioural events, so the two signals can be correlated per location.
const CAMERA_ID = import.meta.env.VITE_CAMERA_ID || 'live_scan_demo'
const HISTORY_LIMIT = 30

// Plate pacing. Face recognition runs locally and is free; plate OCR is a
// metered Azure call on a 20-per-minute free tier, so it fires on a slower
// clock and only when the in-browser detector has actually seen something
// plate-shaped. These intervals stretch further as the budget runs down.
const PLATE_SCAN_INTERVAL_MS = 3500
const PLATE_SCAN_INTERVAL_LOW_MS = 7000
const PLATE_SCAN_INTERVAL_CRITICAL_MS = 14000
// With a person at the door but no plate-like region resolved, sweep the whole
// frame occasionally: a vehicle can be too distant for the browser's detector
// while still being readable once the server deskews and upscales it.
const PLATE_SWEEP_INTERVAL_MS = 15000
// Region detection every Nth animation frame — it is cheap, but not free, and
// the face reticle has to stay smooth.
const PLATE_DETECT_EVERY_N_FRAMES = 6
// How long a server-confirmed plate box stays drawn on the overlay.
const PLATE_BOX_TTL_MS = 4000
// Repeat reads of one plate update the existing audit row rather than stacking
// up — a car sitting in the driveway should be one event, not forty.
const PLATE_DEDUPE_WINDOW_MS = 30000

const MEDIAPIPE_WASM_ROOT =
  import.meta.env.VITE_MEDIAPIPE_WASM_ROOT || 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
const MEDIAPIPE_MODEL_PATH =
  import.meta.env.VITE_MEDIAPIPE_MODEL_PATH ||
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'

// Requesting 1280 wide rather than the previous 960: a plate down a driveway is
// only tens of pixels across, and the server can upscale a crop but cannot
// recover detail the capture never had.
const VIDEO_CONSTRAINTS = { width: 1280, height: 720, facingMode: 'user' }

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))

const formatTimestamp = (iso) => {
  if (!iso) return '-'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '-' : d.toLocaleString()
}

const formatPlate = (text) => {
  if (!text) return '-'
  // Display "CA 123-456" rather than the normalised "CA123456" the matcher uses.
  const m = /^([A-Z]{2,3})(\d{3})(\d{3})$/.exec(text)
  if (m) return `${m[1]} ${m[2]}-${m[3]}`
  const short = /^([A-Z]{2,3})(\d{4,5})$/.exec(text)
  if (short) return `${short[1]} ${short[2]}`
  return text
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

const plateToneFromResult = (plateResult) => {
  if (!plateResult) return 'unknown'
  const status = (plateResult.status || '').toLowerCase()
  if (plateResult.matchFound && (status === 'offender' || status === 'suspect')) return 'alert'
  if (plateResult.matchFound) return 'ok'
  return 'unknown'
}

const plateToneLabel = (plateResult, hasCandidate, isScanning) => {
  if (plateResult?.matchFound) {
    return plateResult.matchConfidence === 'probable' ? 'PROBABLE MATCH' : 'PLATE FLAGGED'
  }
  if (plateResult?.plateDetected) return 'PLATE READ'
  if (isScanning) return 'READING PLATE'
  if (hasCandidate) return 'PLATE IN FRAME'
  return 'NO PLATE'
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

/** Wide thumbnail of a plate region, for the evidence panel. */
const capturePlateCrop = (video, crop) => {
  if (!video || !crop || !video.videoWidth) return null
  const c = document.createElement('canvas')
  c.width = 320
  c.height = Math.max(1, Math.round((320 * crop.sh) / Math.max(crop.sw, 1)))
  c.getContext('2d').drawImage(video, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, c.width, c.height)
  return c.toDataURL('image/jpeg', 0.85)
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

/**
 * Plate overlay. Deliberately unlike the face reticle: a candidate the browser
 * is considering is a thin dashed box, and a plate the server has actually
 * read is a solid box carrying the text. The operator can tell at a glance
 * which boxes are guesses.
 */
const drawPlateBox = (ctx, box, { confirmed = false, tone = 'unknown', label = '' } = {}) => {
  const color = confirmed
    ? tone === 'alert' ? '#b3261e' : tone === 'ok' ? '#186b3c' : '#1c4fd8'
    : '#9aa1ad'

  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = confirmed ? 3 : 1.5
  ctx.setLineDash(confirmed ? [] : [5, 4])
  ctx.strokeRect(box.x, box.y, box.w, box.h)

  if (label) {
    ctx.setLineDash([])
    ctx.font = '700 13px ui-monospace, SFMono-Regular, Menlo, monospace'
    const width = ctx.measureText(label).width + 12
    const ly = box.y + box.h + 20 > ctx.canvas.height ? box.y - 6 : box.y + box.h + 18
    ctx.fillStyle = color
    ctx.fillRect(box.x, ly - 14, width, 18)
    ctx.fillStyle = '#ffffff'
    ctx.fillText(label, box.x + 6, ly)
  }
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

  // Plate scanning state that must not trigger re-renders.
  const plateRegionsRef = useRef([])
  const plateFrameCounterRef = useRef(0)
  const platePendingRef = useRef(false)
  const nextPlateScanAtRef = useRef(0)
  const lastPlateSweepAtRef = useRef(0)
  const plateBudgetRef = useRef({ remaining: null, limit: null })
  const confirmedPlateBoxRef = useRef(null)
  const plateToneRef = useRef('unknown')

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
  const [liveCrop, setLiveCrop] = useState(null)
  const [history, setHistory] = useState([])
  const [selectedHistoryId, setSelectedHistoryId] = useState(null)
  const [dbImageBroken, setDbImageBroken] = useState(false)

  const [plateResult, setPlateResult] = useState(null)
  const [isPlateAnalyzing, setIsPlateAnalyzing] = useState(false)
  const [plateCandidateInFrame, setPlateCandidateInFrame] = useState(false)
  const [plateScanError, setPlateScanError] = useState('')
  const [plateCrop, setPlateCrop] = useState(null)
  const [plateBudget, setPlateBudget] = useState({
    remaining: null,
    limit: null,
    azureConfigured: null,
    offlineFallback: false,
  })

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

  // Poll the OCR call budget. No longer shown on this page, but the live loop
  // paces itself off it, and that pacing has to survive a page left open
  // across a quota window.
  useEffect(() => {
    let active = true

    const fetchBudget = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/v1/plate-scan-budget`)
        if (!res.ok) return
        const data = await res.json()
        if (!active) return
        setPlateBudget({
          remaining: data.remaining,
          limit: data.limit_per_minute,
          azureConfigured: data.azure_configured,
          offlineFallback: data.offline_fallback,
        })
      } catch {
        /* budget display is advisory — a failed poll changes nothing */
      }
    }

    fetchBudget()
    const id = setInterval(fetchBudget, 20000)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  useEffect(() => {
    plateBudgetRef.current = plateBudget
  }, [plateBudget])

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
  const plateTone = plateToneFromResult(plateResult)
  const plateLabel = plateToneLabel(plateResult, plateCandidateInFrame, isPlateAnalyzing)

  useEffect(() => {
    plateToneRef.current = plateTone
  }, [plateTone])

  useEffect(() => {
    let frameId

    const run = async () => {
      const video = webcamRef.current?.video
      const canvas = canvasRef.current

      // Gated on the camera only, not on the face model. Plate detection and
      // its overlay live in this loop, and they must keep working when the
      // MediaPipe assets are slow to arrive off the CDN or fail outright —
      // otherwise a face-model problem silently takes plate scanning with it.
      if (video && canvas && webcamReady && video.readyState === 4) {
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
          const sx = rw / vw
          const sy = rh / vh

          const detector = detectorRef.current
          const detections =
            detector && isModelLoaded ? detector.detectForVideo(video, now)?.detections || [] : []

          canvas.width = rw
          canvas.height = rh
          const ctx = canvas.getContext('2d')
          ctx.clearRect(0, 0, rw, rh)

          if (detections.length > 0) {
            const best = [...detections].sort(
              (a, b) => (b.categories?.[0]?.score || 0) - (a.categories?.[0]?.score || 0)
            )[0]

            const b = best.boundingBox
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

          // Plate region search runs on its own slower clock — it costs no API
          // budget, but it does cost frame time.
          plateFrameCounterRef.current += 1
          if (plateFrameCounterRef.current % PLATE_DETECT_EVERY_N_FRAMES === 0) {
            const regions = detectPlateRegions(video, { maxRegions: 4 })
            plateRegionsRef.current = regions
            const hasCandidate = regions.length > 0
            setPlateCandidateInFrame((prev) => (prev === hasCandidate ? prev : hasCandidate))
          }

          for (const region of plateRegionsRef.current) {
            drawPlateBox(ctx, {
              x: region.x * sx,
              y: region.y * sy,
              w: region.w * sx,
              h: region.h * sy,
            })
          }

          const confirmed = confirmedPlateBoxRef.current
          if (confirmed && Date.now() - confirmed.at < PLATE_BOX_TTL_MS) {
            drawPlateBox(
              ctx,
              {
                x: confirmed.x * sx,
                y: confirmed.y * sy,
                w: confirmed.w * sx,
                h: confirmed.h * sy,
              },
              { confirmed: true, tone: plateToneRef.current, label: confirmed.label }
            )
          }
        }
      }

      frameId = requestAnimationFrame(run)
    }

    if (webcamReady) run()
    return () => cancelAnimationFrame(frameId)
  }, [isModelLoaded, webcamReady, toneClass])

  useEffect(() => {
    if (!isModelLoaded || !webcamReady || !backendReady) return undefined

    const id = setInterval(() => {
      if (!isAnalyzing && hasFaceInFrame) {
        sendFrameToBackend()
      }
    }, SCAN_INTERVAL_MS)

    return () => clearInterval(id)
  }, [isModelLoaded, webcamReady, backendReady, isAnalyzing, hasFaceInFrame])

  // Plate scan scheduler. Separate from the face loop and much more reluctant:
  // it fires only on a candidate, and it stretches its own interval as the
  // minute's OCR budget runs down.
  useEffect(() => {
    if (!webcamReady || !backendReady) return undefined

    const id = setInterval(() => {
      if (platePendingRef.current) return
      if (Date.now() < nextPlateScanAtRef.current) return

      const regions = plateRegionsRef.current
      if (regions && regions.length) {
        sendPlateToBackend(regions)
        return
      }

      const now = Date.now()
      if (hasFaceInFrame && now - lastPlateSweepAtRef.current > PLATE_SWEEP_INTERVAL_MS) {
        lastPlateSweepAtRef.current = now
        sendPlateToBackend(null)
      }
    }, 1000)

    return () => clearInterval(id)
  }, [webcamReady, backendReady, hasFaceInFrame])

  const addHistory = (entry) => {
    setHistory((prev) => {
      const next = [entry, ...prev].slice(0, HISTORY_LIMIT)
      if (!selectedHistoryId) setSelectedHistoryId(entry.id)
      return next
    })
  }

  /** Collapse a repeat read of the same plate into the existing audit row. */
  const addPlateHistory = (entry) => {
    setHistory((prev) => {
      const recent = prev.find(
        (h) =>
          h.type === 'vehicle' &&
          h.plateText === entry.plateText &&
          Date.now() - new Date(h.timestamp).getTime() < PLATE_DEDUPE_WINDOW_MS
      )
      if (recent) {
        return prev.map((h) =>
          h === recent
            ? { ...entry, id: recent.id, seenCount: (recent.seenCount || 1) + 1 }
            : h
        )
      }
      const next = [entry, ...prev].slice(0, HISTORY_LIMIT)
      if (!selectedHistoryId) setSelectedHistoryId(entry.id)
      return next
    })
  }

  const plateIntervalForBudget = () => {
    const remaining = plateBudgetRef.current?.remaining
    if (typeof remaining !== 'number') return PLATE_SCAN_INTERVAL_MS
    if (remaining <= 3) return PLATE_SCAN_INTERVAL_CRITICAL_MS
    if (remaining <= 7) return PLATE_SCAN_INTERVAL_LOW_MS
    return PLATE_SCAN_INTERVAL_MS
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

    // Snapshot the box now, alongside the crop. The detection loop runs on
    // requestAnimationFrame and will have moved it on by the time the upload
    // finishes, and the box has to describe the frame we actually send.
    const faceBox = faceBoxRef.current
    const localCrop = captureCrop(video, faceBox)

    try {
      setIsAnalyzing(true)
      setScanError('')

      const started = performance.now()
      const blob = await captureFrameBlob(video, 0.72, 720)
      if (!blob) throw new Error('Camera frame unavailable.')

      const formData = new FormData()
      formData.append('file', blob, 'live_scan.jpg')

      // WHERE the face was, not just that there was one. Without this a match
      // cannot be attached to a tracked body once two people are in shot, and
      // the behavioural signal has nothing to fuse with. See
      // BEHAVIOUR_REVIEW_API.md §1.
      //
      // The box is in SOURCE video pixels while the uploaded JPEG is
      // downscaled, so the source dimensions go with it — the backend
      // normalises against those, and normalised coordinates are the part that
      // survives every resolution change between here and the analysis.
      if (faceBox && video.videoWidth && video.videoHeight) {
        formData.append('face_box', JSON.stringify(faceBox))
        formData.append('frame_width', String(video.videoWidth))
        formData.append('frame_height', String(video.videoHeight))
      }
      formData.append('camera_id', CAMERA_ID)
      formData.append('captured_at', new Date().toISOString())

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

      // Still measured and carried onto the audit-log entry below; just no
      // longer surfaced on the identity card.
      const durationMs = Math.round(performance.now() - started)

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
        type: 'person',
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
        // Echoed back by the backend. This is what a behavioural track gets
        // joined on — displayed so the round-trip is visible while wiring it.
        faceBoxNormalised: data?.face_box_normalised || null,
        faceBoxCentre: data?.face_box_centre || null,
        cameraId: data?.camera_id || null,
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
    }
  }

  /**
   * Send the plate candidates for OCR as a single cropped frame.
   *
   * `regions` empty or null means a whole-frame sweep — used when someone is at
   * the door but the local detector found nothing plate-shaped.
   *
   * With more than one candidate we send the box covering all of them and no
   * ROI hint, so the server searches the whole crop. The alternative — crop to
   * our best guess — reliably posts the model badge instead of the plate,
   * because a badge and a plate look identical to an edge detector.
   */
  const sendPlateToBackend = async (regions) => {
    const video = webcamRef.current?.video
    if (!video || platePendingRef.current) return
    if (!webcamReady || !backendReady) return

    const list = Array.isArray(regions) ? regions : regions ? [regions] : []
    const single = list.length === 1
    let target = single ? list[0] : unionOfRegions(list)

    // A union spanning most of the frame is not a crop, it is the frame.
    if (target && video.videoWidth && video.videoHeight) {
      const coverage = (target.w * target.h) / (video.videoWidth * video.videoHeight)
      if (coverage > 0.6) target = null
    }

    platePendingRef.current = true
    setIsPlateAnalyzing(true)

    try {
      const captured = await cropRegionBlob(video, target, { pad: single ? 1.0 : 0.35 })
      if (!captured) throw new Error('Camera frame unavailable.')

      const formData = new FormData()
      formData.append('file', captured.blob, 'plate_scan.jpg')
      if (single && captured.roi) formData.append('roi', JSON.stringify(captured.roi))

      const started = performance.now()
      const response = await fetch(`${API_BASE_URL}/api/v1/scan-plate-live`, {
        method: 'POST',
        body: formData,
      })

      let data = null
      try {
        data = await response.json()
      } catch {
        data = null
      }

      const latencyMs = Math.round(performance.now() - started)

      if (typeof data?.azure_calls_remaining === 'number') {
        setPlateBudget((prev) => ({ ...prev, remaining: data.azure_calls_remaining }))
      }

      if (!response.ok) {
        throw new Error(data?.error || `Plate scan failed with HTTP ${response.status}`)
      }

      // Not a failure: the server refused the call to stay inside the free
      // tier. Wait out its window rather than hammering it.
      if (data?.throttled) {
        nextPlateScanAtRef.current = Date.now() + Math.max(2000, (data.retry_after_seconds || 20) * 1000)
        setPlateScanError(data.message || 'Pacing plate scans to stay inside the Azure Vision free tier.')
        return
      }

      setPlateScanError('')
      nextPlateScanAtRef.current = Date.now() + plateIntervalForBudget()

      const status = data?.plate?.status || null
      const normalized = {
        id: `plate-${Date.now()}`,
        type: 'vehicle',
        plateDetected: !!data?.plate_detected,
        plateText: data?.extracted_text || null,
        rawText: data?.raw_text || null,
        detectionKind: data?.detection_kind || null,
        matchFound: !!data?.match_found,
        matchConfidence: data?.match_confidence || null,
        matchReason: data?.match_reason || null,
        registryPlate: data?.registry_plate || null,
        status,
        isAlert: !!data?.match_found && (status === 'offender' || status === 'suspect'),
        ownerName: data?.plate?.owner_name || null,
        plateImageUrl: data?.plate?.image_url || null,
        ignoredText: Array.isArray(data?.ignored_text) ? data.ignored_text : [],
        ocrConfidence: typeof data?.ocr_confidence === 'number' ? data.ocr_confidence : null,
        engine: data?.engine || null,
        passesUsed: data?.passes_used || null,
        alertSuppressed: data?.alert_suppressed || null,
        message: data?.message || '',
        timestamp: new Date().toISOString(),
        latencyMs,
        stageTiming: data?.timings_ms || null,
        seenCount: 1,
      }

      setPlateResult(normalized)

      if (normalized.plateDetected) {
        const crop = capturePlateCrop(video, captured.crop)
        normalized.liveCrop = crop
        setPlateCrop(crop)

        // The server reports the plate box in the coordinates of the crop it
        // received, so shift it back into the full frame before drawing.
        const boxNorm = data?.plate_box_norm
        if (boxNorm && captured.crop) {
          confirmedPlateBoxRef.current = {
            x: captured.crop.sx + boxNorm.x * captured.crop.sw,
            y: captured.crop.sy + boxNorm.y * captured.crop.sh,
            w: boxNorm.w * captured.crop.sw,
            h: boxNorm.h * captured.crop.sh,
            label: formatPlate(normalized.plateText),
            at: Date.now(),
          }
        }
        addPlateHistory(normalized)
      }
    } catch (err) {
      nextPlateScanAtRef.current = Date.now() + Math.max(4000, plateIntervalForBudget())
      setPlateScanError(err?.message || 'Plate scan request failed.')
    } finally {
      platePendingRef.current = false
      setIsPlateAnalyzing(false)
    }
  }

  const activeEntry = useMemo(() => {
    if (!history.length) return null
    return history.find((h) => h.id === selectedHistoryId) || history[0]
  }, [history, selectedHistoryId])

  const isVehicleEntry = activeEntry?.type === 'vehicle'
  const dbImageUrl = isVehicleEntry
    ? activeEntry?.plateImageUrl || null
    : activeEntry?.sourceImageUrl ||
      activeEntry?.personImageUrl ||
      matchResult?.sourceImageUrl ||
      matchResult?.personImageUrl ||
      null
  const evidenceCrop = isVehicleEntry ? activeEntry?.liveCrop || plateCrop : activeEntry?.liveCrop || liveCrop

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
        .ls-grid { display: grid; grid-template-columns: minmax(420px, 2fr) minmax(320px, 1fr); gap: 14px; align-items: stretch; }
        .ls-grid > * { min-width: 0; }
        .ls-card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; min-width: 0; }
        /* Both columns are driven by the grid row height. The camera card keeps
           its 4:3 frame and pushes its caption to its own bottom edge; the two
           info cards split whatever that height is. Result: the columns end on
           the same line without either being pinned to a hardcoded height. */
        .ls-cam-card { display: flex; flex-direction: column; }
        .ls-cam-caption { margin-top: auto; padding-top: 12px; }
        .ls-side { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
        /* 1 1 auto, not 1 1 0: forcing exact halves clips the vehicle card,
           which legitimately carries more rows than the identity card. Sizing
           to content first and then sharing the leftover space keeps them
           near-equal, keeps both bottoms on the grid row's baseline, and can
           never cut a line off. */
        .ls-side > .ls-card { flex: 1 1 auto; display: flex; flex-direction: column; }
        .ls-pad { padding: 16px; }
        .ls-meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; font-size: 12px; color: var(--muted); }
        .ls-camera { position: relative; width: 100%; aspect-ratio: 4/3; border-radius: 12px; overflow: hidden; border: 1px solid var(--line); background: #000; }
        .ls-chip { display: inline-block; padding: 4px 9px; border-radius: 999px; border: 1px solid var(--line); background: rgba(0,0,0,0.35); color: #fff; font-size: 11px; font-weight: 700; letter-spacing: .03em; }
        .ls-overlay-top-left { position: absolute; top: 10px; left: 10px; z-index: 4; display: grid; gap: 6px; justify-items: start; }
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
        /* Portrait, and identical for both frames — a live crop and a registry
           photo can only be compared fairly at the same size and ratio. These
           sources are typically phone-in-hand or vertical registry shots, which
           a wide frame wastes width on. */
        /* Capped, or a 3:4.6 frame in a full-width row renders ~860px tall and
           swamps the page. The cap applies to both equally, so the pair stays
           the same size and ratio as each other. */
        .ls-thumb-wrap { border: 1px solid var(--line); border-radius: 9px; overflow: hidden; background: var(--bg); aspect-ratio: 3 / 4.6; max-width: 300px; }
        .ls-thumb-wrap img { width: 100%; height: 100%; object-fit: cover; }
        /* The plate crop is a wide strip; contain keeps the number readable
           inside the portrait frame instead of cropping its ends off. */
        .ls-thumb-wrap.plate img { object-fit: contain; }
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
        .ls-plate-text { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 26px; font-weight: 700; letter-spacing: .06em; margin: 6px 0 0; color: var(--fg); }
        .ls-tag { display: inline-block; font-size: 10px; font-weight: 800; letter-spacing: .06em; padding: 2px 6px; border-radius: 4px; border: 1px solid currentColor; margin-right: 6px; vertical-align: 1px; }
        .ls-ignored { font-size: 12px; color: var(--muted); margin-top: 10px; line-height: 1.5; }
        .ls-ignored code { background: var(--bg); border: 1px solid var(--line); border-radius: 4px; padding: 1px 5px; margin-right: 4px; display: inline-block; text-decoration: line-through; }
        @media (max-width: 980px) {
          .ls-grid { grid-template-columns: 1fr; }
          .ls-side-by-side { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 640px) {
          .ls-wrap { padding: 14px 10px 42px; }
          .ls-side-by-side { grid-template-columns: 1fr; }
          .ls-kv { flex-direction: column; align-items: flex-start; gap: 4px; }
          .ls-kv span:last-child { max-width: 100%; text-align: left; }
          .ls-audit-grid { grid-template-columns: 1fr !important; }
          .ls-plate-text { font-size: 21px; }
        }
      `}</style>

      <div className="ls-shell">
        <h1>Doorbell Scan (Live)</h1>

        <div className="ls-grid">
          <div className="ls-card ls-pad ls-cam-card">
            <div className="ls-camera">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                screenshotQuality={0.72}
                videoConstraints={VIDEO_CONSTRAINTS}
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
                <span className="ls-chip">Plate: {plateLabel}</span>
              </div>

              <div className="ls-overlay-top-right">
                <span className="ls-chip">{formatTimestamp(matchResult?.timestamp || new Date().toISOString())}</span>
              </div>
            </div>

            <div className="ls-cam-caption">
              <div className="ls-warn" style={{ marginTop: 0 }}>
                {hasFaceInFrame
                  ? 'Target lock acquired. Scanning against identity registry.'
                  : 'No face in frame. Move closer and improve front lighting.'}
              </div>
              <div className="ls-warn" style={{ marginTop: 4 }}>
                {plateCandidateInFrame
                  ? 'Plate-shaped region locked. Cropping at native resolution for OCR.'
                  : 'No plate-shaped region yet. Scans are held back rather than spending OCR budget on an empty frame.'}
              </div>
            </div>
          </div>

          <div className="ls-side">
            <div className="ls-card ls-pad">
              <div className={`ls-verdict ${toneClass}`}>
                <h2 style={{ margin: 0, fontSize: 17 }}>
                  <span className="ls-tag">PERSON</span>
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

              {/* Cosine distance, pose and round-trip are pipeline diagnostics,
                  not something a person watching the feed acts on. They are
                  still returned by the API and kept on each audit-log entry. */}
              <div className="ls-kv"><span>Confidence</span><span>{confidencePct}%</span></div>
              <div className="ls-kv"><span>Status</span><span>{(matchResult?.status || 'unknown').toUpperCase()}</span></div>
            </div>

            <div className="ls-card ls-pad">
              <div className={`ls-verdict ${plateTone}`}>
                <h2 style={{ margin: 0, fontSize: 17 }}>
                  <span className="ls-tag">VEHICLE</span>
                  {plateResult?.matchFound
                    ? plateResult.matchConfidence === 'probable'
                      ? 'Probable registry match'
                      : 'Plate matched in registry'
                    : plateResult?.plateDetected
                    ? 'Plate read, not in registry'
                    : 'Awaiting plate'}
                </h2>
                <p className="ls-plate-text">{formatPlate(plateResult?.plateText)}</p>
                <p style={{ margin: '6px 0 0', color: 'var(--muted)', fontSize: 13 }}>
                  {plateResult?.message || 'Waiting for a plate-shaped region in frame.'}
                </p>
              </div>

              {/* Which OCR engine ran, how confident the raw read was and how
                  long it took are pipeline internals — they don't help anyone
                  decide whether to act. Still in the API response and the
                  audit-log entry for debugging. */}
              <div className="ls-kv"><span>Registry plate</span><span>{formatPlate(plateResult?.registryPlate) || '--'}</span></div>
              <div className="ls-kv"><span>Match</span><span>{(plateResult?.matchConfidence || 'none').toUpperCase()}</span></div>
              <div className="ls-kv"><span>Status</span><span>{(plateResult?.status || 'unknown').toUpperCase()}</span></div>
              <div className="ls-kv"><span>Owner / flag</span><span>{plateResult?.ownerName || '-'}</span></div>

              {plateResult?.matchConfidence === 'probable' ? (
                <div className="ls-warn">
                  Tolerant match ({(plateResult.matchReason || '').replace(/_/g, ' ')}) — shown to the
                  operator but deliberately not escalated to the alert feed without a confirmed read.
                </div>
              ) : null}

              {plateResult?.ignoredText?.length ? (
                <div className="ls-ignored">
                  Text rejected as non-plate:{' '}
                  {plateResult.ignoredText.slice(0, 6).map((text) => (
                    <code key={text}>{text}</code>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        </div>

        <div className="ls-card ls-pad" style={{ marginTop: 14 }}>
          <h3 style={{ margin: '0 0 10px', fontSize: 12, textTransform: 'uppercase', color: 'var(--muted)' }}>
            Reference comparison
          </h3>
          <div className="ls-side-by-side">
            <div>
              <div className="ls-caption">{isVehicleEntry ? 'Live plate crop' : 'Live crop'}</div>
              <div className={`ls-thumb-wrap${isVehicleEntry ? ' plate' : ''}`}>
                {evidenceCrop ? <img src={evidenceCrop} alt={isVehicleEntry ? 'Live plate crop' : 'Live crop'} /> : null}
              </div>
            </div>
            <div>
              <div className="ls-caption">{isVehicleEntry ? 'Registry plate photo' : 'DB seed photo'}</div>
              <div className={`ls-thumb-wrap${isVehicleEntry ? ' plate' : ''}`}>
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
            <span>
              Source: {isVehicleEntry ? activeEntry?.engine === 'easyocr' ? 'EasyOCR (offline)' : 'Azure Vision READ' : matchResult?.sourceText || '-'}
            </span>
            <span>{isVehicleEntry ? `Read: ${activeEntry?.detectionKind || '-'}` : `Pose: ${matchResult?.poseLabel || '-'}`}</span>
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
                  {entry.type === 'vehicle' ? (
                    <>
                      <span className="ls-tag">VEHICLE</span>
                      {formatTimestamp(entry.timestamp)} - {formatPlate(entry.plateText)} (
                      {(entry.status || 'unlisted').toUpperCase()}) - {(entry.matchConfidence || 'no match').toUpperCase()}
                      {entry.seenCount > 1 ? ` - seen ${entry.seenCount}x` : ''}
                    </>
                  ) : (
                    <>
                      <span className="ls-tag">PERSON</span>
                      {formatTimestamp(entry.timestamp)} - {entry.fullName} ({(entry.status || 'unknown').toUpperCase()}) -{' '}
                      {typeof entry.matchConfidence === 'number' ? `${entry.matchConfidence.toFixed(1)}%` : '--'}
                    </>
                  )}
                </button>
              ))}
              {!history.length ? <div className="ls-sub">No detection events yet.</div> : null}
            </div>

            <div className="ls-card ls-pad">
              {isVehicleEntry ? (
                <>
                  <div style={{ fontWeight: 700 }}>{formatPlate(activeEntry?.plateText)}</div>
                  <div className="ls-kv"><span>Registry</span><span>{formatPlate(activeEntry?.registryPlate) || 'not listed'}</span></div>
                  <div className="ls-kv"><span>Status</span><span>{(activeEntry?.status || 'unlisted').toUpperCase()}</span></div>
                  <div className="ls-kv"><span>Match</span><span>{(activeEntry?.matchConfidence || 'none').toUpperCase()}</span></div>
                  <div className="ls-kv"><span>Owner / flag</span><span>{activeEntry?.ownerName || '-'}</span></div>
                  <div className="ls-kv"><span>Latency</span><span>{activeEntry?.latencyMs ?? '--'} ms</span></div>
                  {activeEntry?.liveCrop ? (
                    <img src={activeEntry.liveCrop} alt="Plate crop" style={{ width: '100%', borderRadius: 8, marginTop: 10 }} />
                  ) : null}
                </>
              ) : (
                <>
                  <div style={{ fontWeight: 700 }}>{activeEntry?.fullName || 'Select an event'}</div>
                  <div className="ls-kv"><span>Status</span><span>{(activeEntry?.status || 'unknown').toUpperCase()}</span></div>
                  <div className="ls-kv"><span>Distance</span><span>{typeof activeEntry?.matchDistance === 'number' ? activeEntry.matchDistance.toFixed(4) : '--'}</span></div>
                  <div className="ls-kv"><span>Latency</span><span>{activeEntry?.latencyMs ?? '--'} ms</span></div>
                  {activeEntry?.liveCrop ? <img src={activeEntry.liveCrop} alt="Event crop" style={{ width: '100%', borderRadius: 8, marginTop: 10 }} /> : null}
                </>
              )}
            </div>
          </div>
        </div>

        {webcamError ? <div className="ls-warn">Camera unavailable: {webcamError}</div> : null}
        {scanError ? <div className="ls-warn">Face scan: {scanError}</div> : null}
        {plateScanError ? <div className="ls-warn">Plate scan: {plateScanError}</div> : null}
        {plateBudget.azureConfigured === false && !plateBudget.offlineFallback ? (
          <div className="ls-warn">
            Azure Vision is not configured on the server — set AZURE_VISION_KEY and
            AZURE_VISION_ENDPOINT to enable plate reading.
          </div>
        ) : null}
        {modelState === 'loading' ? <div className="ls-sub">Initializing detector assets...</div> : null}
        {modelState === 'failed' ? <div className="ls-warn">Model Load Failed: {modelError || 'Unknown error.'}</div> : null}
      </div>
    </div>
  )
}
