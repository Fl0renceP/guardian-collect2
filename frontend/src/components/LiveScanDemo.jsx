import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Webcam from 'react-webcam'
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''
const SCAN_INTERVAL_MS = 1500
// Plates get their own cadence because they are gated differently: the face
// scan only fires when MediaPipe sees a face, so on a frame with a car and no
// visible driver the face path is idle and the plate path is the only one
// working. Slower than the face interval because plate OCR costs ~1.2s and
// there is no point queueing frames the backend cannot reach.
const PLATE_SCAN_INTERVAL_MS = 2000
const HISTORY_LIMIT = 30
// The live half of the reference comparison is refreshed from the local
// MediaPipe box, not from a scan response, so the tile fills as soon as a face
// is in frame — including when the backend is unreachable.
const LIVE_CROP_INTERVAL_MS = 700
const MODEL_LABEL = 'Facenet512 / Cosine'
const MEDIAPIPE_WASM_ROOT =
  import.meta.env.VITE_MEDIAPIPE_WASM_ROOT || 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
// FaceLandmarker rather than FaceDetector: it returns the same face position
// AND per-frame blendshapes, which is what the blink check reads. Running a
// separate detector alongside it would pay for face-finding twice.
const MEDIAPIPE_MODEL_PATH =
  import.meta.env.VITE_MEDIAPIPE_MODEL_PATH ||
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'

// ---------------------------------------------------------------------------
// Liveness (build guide §5).
//
// A photo held up to the camera does not blink. Tracking eye-closure across
// frames is the cheapest check that separates a live person from a printed
// face or a phone screen, and it is the difference between "any face in frame
// = match" and something defensible.
//
// Deliberately NOT a hard gate on scanning: an un-blinked face still gets
// checked, it is just reported as unverified. Blocking the scan outright would
// mean a still, staring person — exactly what a doorbell camera catches — goes
// unrecognised, which is a worse failure than a flagged spoof.
// ---------------------------------------------------------------------------
const BLINK_CLOSED_SCORE = 0.45   // above this, the eye counts as shut
const BLINK_OPEN_SCORE = 0.2      // back below this completes the blink
const LIVENESS_WINDOW_MS = 12000  // a blink counts as recent for this long

const LIVENESS = {
  PENDING: 'pending',
  LIVE: 'live',
  UNAVAILABLE: 'unavailable',
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))

const formatTimestamp = (iso) => {
  if (!iso) return '-'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '-' : d.toLocaleString()
}

const formatClock = (iso) => {
  if (!iso) return null
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? null : d.toLocaleTimeString()
}

// ---------------------------------------------------------------------------
// The four-state status system (design spec §7).
//
// Every result on this screen — face or plate — resolves to exactly one of
// these. It is the single semantic language of the page: the user learns the
// traffic light once and reads both panels the same way. Anything that needs a
// fifth state is a design change, not a local special case.
// ---------------------------------------------------------------------------
const STATE = {
  SCANNING: 'scanning',
  UNKNOWN: 'unknown',
  ALERT: 'alert',
  CLEAR: 'clear',
}

// Registry status -> screen state. The registry owns the vocabulary; this map
// is the only place it is translated, so a new registry status needs one line
// here rather than a redesign.
const registryStateFor = (status) => {
  const s = (status || '').toLowerCase()
  if (s === 'offender' || s === 'suspect') return STATE.ALERT
  if (s === 'verified') return STATE.CLEAR
  return STATE.UNKNOWN
}

// ---------------------------------------------------------------------------
// Reference URLs are SAS-signed, so everything from the '?' on is a token.
// Strip it before parsing or the entire signature gets rendered on screen.
// ---------------------------------------------------------------------------
const parsePoseFromUrl = (url) => {
  if (!url) return '-'
  const path = String(url).split(/[?#]/)[0]
  let name = path.split('/').pop() || ''
  try {
    name = decodeURIComponent(name)
  } catch {
    // Leave the raw segment if it is not valid percent-encoding.
  }
  const stripped = name.replace(/\.[a-zA-Z0-9]+$/, '')
  const parts = stripped.split('_').filter(Boolean)
  return parts.length < 2 ? '-' : parts.slice(1).join(' ')
}

// Per-stage server timings, for the System status panel only. JSON.stringify
// dumps an unreadable blob into a table cell.
const formatStageTiming = (timings) => {
  if (!timings || typeof timings !== 'object') return '--'
  const parts = Object.entries(timings)
    .filter(([, v]) => typeof v === 'number')
    .map(([k, v]) => `${k.replace(/_/g, ' ')} ${Math.round(v)}ms`)
  return parts.length ? parts.join(' · ') : '--'
}

const toConfidencePct = (distance) => {
  if (typeof distance !== 'number') return 0
  return clamp(Math.round((1 - distance / 0.6) * 100), 0, 100)
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

// The video is rendered with object-fit: cover, so it is scaled uniformly and
// centre-cropped. Overlay boxes have to undergo the same transform or they sit
// beside the thing they are meant to be marking. Previously the video used
// object-fit: fill and the boxes assumed a per-axis stretch, which put every
// box slightly off whenever the camera's aspect ratio was not exactly 4:3.
const coverTransform = (srcW, srcH, dstW, dstH) => {
  if (!srcW || !srcH || !dstW || !dstH) return { scale: 1, offX: 0, offY: 0 }
  const scale = Math.max(dstW / srcW, dstH / srcH)
  return {
    scale,
    offX: (dstW - srcW * scale) / 2,
    offY: (dstH - srcH * scale) / 2,
  }
}

const STATUS_HEX = {
  [STATE.SCANNING]: '#E3A438',
  [STATE.UNKNOWN]: '#7C8299',
  [STATE.ALERT]: '#E14F4F',
  [STATE.CLEAR]: '#3FBF7F',
}

const drawReticle = (ctx, box, state, scorePct) => {
  const color = STATUS_HEX[state] || STATUS_HEX[STATE.UNKNOWN]
  const dashed = state === STATE.UNKNOWN || state === STATE.SCANNING

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
  // Only the real match confidence is drawn. The landmarker exposes no
  // detection-confidence score, and a made-up number on the face is worse
  // than no number.
  if (typeof scorePct === 'number') {
    ctx.font = '700 12px ui-monospace, SFMono-Regular, Menlo, monospace'
    ctx.fillText(`${scorePct}%`, box.x, Math.max(14, box.y - 6))
  }
  ctx.restore()
}

// Status is never colour-only: the word travels with the dot so the screen
// still reads for colourblind users and in the black-and-white printouts
// judges take notes on (spec §8).
function StatusPill({ state, label }) {
  return (
    <span className={`ls-pill ls-pill--${state}`}>
      <i className={`ls-dot${state === STATE.SCANNING ? ' ls-dot--pulse' : ''}`} aria-hidden="true" />
      {label}
    </span>
  )
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
  const statusBtnRef = useRef(null)
  // Blink state machine: eyes must go shut and then open again to count.
  const eyesClosedRef = useRef(false)
  const lastBlinkAtRef = useRef(0)

  const [modelState, setModelState] = useState('loading')
  const [modelError, setModelError] = useState('')
  const [backendReady, setBackendReady] = useState(false)
  const [matchResult, setMatchResult] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [scanError, setScanError] = useState('')
  const [webcamReady, setWebcamReady] = useState(false)
  const [webcamError, setWebcamError] = useState('')
  const [hasFaceInFrame, setHasFaceInFrame] = useState(false)
  const [liveness, setLiveness] = useState(LIVENESS.PENDING)
  const [blinkCount, setBlinkCount] = useState(0)
  const [fps, setFps] = useState(null)
  const [memberCount, setMemberCount] = useState(null)
  const [liveCrop, setLiveCrop] = useState(null)
  const [liveCropAt, setLiveCropAt] = useState(null)
  const [roundtripMs, setRoundtripMs] = useState(null)
  const [history, setHistory] = useState([])
  const [selectedHistoryId, setSelectedHistoryId] = useState(null)
  const [dbImageBroken, setDbImageBroken] = useState(false)
  const [systemPanelOpen, setSystemPanelOpen] = useState(false)
  const [videoGeom, setVideoGeom] = useState(null)

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

        detectorRef.current = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MEDIAPIPE_MODEL_PATH,
          },
          runningMode: 'VIDEO',
          numFaces: 1,
          // The blink check reads eyeBlinkLeft / eyeBlinkRight from here.
          outputFaceBlendshapes: true,
          minFaceDetectionConfidence: 0.52,
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

  // ------------------------------------------------------------------
  // Face verdict, expressed in the four-state language.
  // ------------------------------------------------------------------
  const faceVerdict = useMemo(() => {
    if (matchResult?.isKnownUser || matchResult?.status) {
      const state = registryStateFor(matchResult.status)
      if (state === STATE.ALERT) {
        return {
          state,
          label: 'SUSPECT',
          name: matchResult.fullName,
          // The registry owns this copy. Where it has no reason field we say
          // what we actually know rather than inventing an offence.
          detail:
            matchResult.reason ||
            (matchResult.status ? `Flagged on the registry as ${matchResult.status}` : 'Flagged on the registry'),
        }
      }
      if (state === STATE.CLEAR) {
        return {
          state,
          label: 'RESIDENT',
          name: matchResult.fullName,
          detail: matchResult.reason || 'Registered community member',
        }
      }
    }
    if (isAnalyzing) {
      return { state: STATE.SCANNING, label: 'SCANNING', name: null, detail: 'Checking this face against the registry' }
    }
    if (matchResult) {
      // Name is deliberately not rendered for UNKNOWN (spec §4.2).
      return { state: STATE.UNKNOWN, label: 'UNKNOWN', name: null, detail: 'Not in the safety registry' }
    }
    if (hasFaceInFrame) {
      return { state: STATE.SCANNING, label: 'SCANNING', name: null, detail: 'Face detected — reading' }
    }
    return { state: STATE.UNKNOWN, label: 'NO FACE', name: null, detail: 'Step into frame to begin a scan' }
  }, [matchResult, isAnalyzing, hasFaceInFrame])

  // ------------------------------------------------------------------
  // Plate verdict, same four states so both panels read identically.
  // ------------------------------------------------------------------
  const plateVerdict = useMemo(() => {
    const text = plateResult?.extracted_text
    const stability = plateResult?.stability

    if (plateResult?.match_found) {
      const state = registryStateFor(plateResult.status)
      const owner = plateResult.plate?.owner_name
      if (state === STATE.ALERT) {
        return {
          state,
          label: 'FLAGGED',
          plate: plateResult.plate?.plate_number || text,
          detail: plateResult.reason || owner || 'Flagged on the vehicle registry',
        }
      }
      return {
        state: STATE.CLEAR,
        label: 'REGISTERED',
        plate: plateResult.plate?.plate_number || text,
        detail: owner ? `Registered to ${owner}` : 'On the vehicle registry',
      }
    }

    if (text && stability && !stability.confirmed) {
      return {
        state: STATE.SCANNING,
        label: 'SCANNING',
        plate: text,
        detail: `Reading plate — ${stability.agreeing_frames || 0} of ${stability.required_frames} frames agree`,
      }
    }

    if (text) {
      return { state: STATE.UNKNOWN, label: 'UNKNOWN', plate: text, detail: 'Not in the vehicle registry' }
    }

    // Never a bare em-dash: an empty state has to read as "working", not
    // "broken" (spec §5).
    return { state: STATE.SCANNING, label: 'SCANNING', plate: null, detail: 'Looking for a plate in frame' }
  }, [plateResult])

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

          setVideoGeom((prev) =>
            prev && prev.vw === vw && prev.vh === vh && prev.rw === rw && prev.rh === rh
              ? prev
              : { vw, vh, rw, rh },
          )

          const results = detector.detectForVideo(video, now)
          const landmarks = results?.faceLandmarks?.[0]

          canvas.width = rw
          canvas.height = rh
          const ctx = canvas.getContext('2d')
          ctx.clearRect(0, 0, rw, rh)

          if (landmarks && landmarks.length) {
            // Landmarks are normalised 0..1; the box is their extent.
            let minX = 1, minY = 1, maxX = 0, maxY = 0
            for (const p of landmarks) {
              if (p.x < minX) minX = p.x
              if (p.y < minY) minY = p.y
              if (p.x > maxX) maxX = p.x
              if (p.y > maxY) maxY = p.y
            }
            const src = {
              x: minX * vw,
              y: minY * vh,
              w: (maxX - minX) * vw,
              h: (maxY - minY) * vh,
            }

            const { scale, offX, offY } = coverTransform(vw, vh, rw, rh)
            const box = {
              x: src.x * scale + offX,
              y: src.y * scale + offY,
              w: src.w * scale,
              h: src.h * scale,
            }

            faceBoxRef.current = src
            setHasFaceInFrame(true)

            // ---- blink / liveness ----
            const shapes = results?.faceBlendshapes?.[0]?.categories
            if (shapes && shapes.length) {
              let left = 0
              let right = 0
              for (const c of shapes) {
                if (c.categoryName === 'eyeBlinkLeft') left = c.score
                else if (c.categoryName === 'eyeBlinkRight') right = c.score
              }
              // Both eyes, so a wink or a one-sided shadow does not count.
              const closed = Math.min(left, right)
              if (!eyesClosedRef.current && closed > BLINK_CLOSED_SCORE) {
                eyesClosedRef.current = true
              } else if (eyesClosedRef.current && closed < BLINK_OPEN_SCORE) {
                // Shut then open again — one complete blink.
                eyesClosedRef.current = false
                lastBlinkAtRef.current = Date.now()
                setBlinkCount((n) => n + 1)
              }
              setLiveness(
                Date.now() - lastBlinkAtRef.current < LIVENESS_WINDOW_MS
                  ? LIVENESS.LIVE
                  : LIVENESS.PENDING,
              )
            } else {
              // Blendshapes absent (older model asset / unsupported build).
              // Say so rather than reporting a liveness result we do not have.
              setLiveness(LIVENESS.UNAVAILABLE)
            }

            drawReticle(
              ctx,
              box,
              faceVerdict.state,
              typeof matchResult?.matchConfidence === 'number' ? matchResult.matchConfidence : null,
            )
          } else {
            faceBoxRef.current = null
            setHasFaceInFrame(false)
            eyesClosedRef.current = false
          }
        }
      }

      frameId = requestAnimationFrame(run)
    }

    if (isModelLoaded) run()
    return () => cancelAnimationFrame(frameId)
  }, [isModelLoaded, webcamReady, faceVerdict.state, matchResult?.matchConfidence])

  const sendFrameToBackend = useCallback(async () => {
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
      // Read from the ref, not state: this callback is memoised and a state
      // read here would be a stale closure.
      const livenessConfirmed = Date.now() - lastBlinkAtRef.current < LIVENESS_WINDOW_MS
      formData.append('liveness_confirmed', livenessConfirmed ? 'true' : 'false')

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
        // Rendered verbatim when present; never synthesised (spec §7).
        reason: data?.person?.reason || data?.reason || null,
        matchDistance: distance,
        matchConfidence: confidence,
        error: data?.error || null,
        message: data?.message || '',
        timestamp: ts,
        poseLabel,
        personImageUrl: data?.person?.image_url || null,
        sourceImageUrl: capture?.image_url || data?.person?.image_url || null,
        sourceText: capture?.source || 'Registry Seed',
        capturedAt: capture?.captured_at || null,
        liveCrop: localCrop,
        latencyMs: durationMs,
        stageTiming: data?.timings_ms || null,
      }

      setDbImageBroken(false)
      setMatchResult(normalized)
      // Prefer the frame the match was actually made against, but never blank
      // the tile if this particular frame had no usable box.
      if (localCrop) {
        setLiveCrop(localCrop)
        setLiveCropAt(ts)
      }
      setHistory((prev) => {
        const next = [normalized, ...prev].slice(0, HISTORY_LIMIT)
        return next
      })
      setSelectedHistoryId((prev) => prev || normalized.id)
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
  }, [webcamReady, backendReady])

  useEffect(() => {
    if (!isModelLoaded || !webcamReady || !backendReady) return undefined

    const id = setInterval(() => {
      if (!isAnalyzing && !scanBusyRef.current && hasFaceInFrame) {
        sendFrameToBackend()
      }
    }, SCAN_INTERVAL_MS)

    return () => clearInterval(id)
  }, [isModelLoaded, webcamReady, backendReady, isAnalyzing, hasFaceInFrame, sendFrameToBackend])

  const sendPlateFrame = useCallback(async () => {
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
  }, [])

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
  }, [webcamReady, backendReady, isAnalyzing, sendPlateFrame])

  // Keep the live comparison tile current off the local detector. Cheap: one
  // 180x180 canvas draw, and only while a face is actually in frame.
  useEffect(() => {
    if (!webcamReady || !hasFaceInFrame) return undefined

    const capture = () => {
      const video = webcamRef.current?.video
      if (!video || !faceBoxRef.current) return
      const crop = captureCrop(video, faceBoxRef.current)
      if (crop) {
        setLiveCrop(crop)
        setLiveCropAt(new Date().toISOString())
      }
    }

    capture()
    const id = setInterval(capture, LIVE_CROP_INTERVAL_MS)
    return () => clearInterval(id)
  }, [webcamReady, hasFaceInFrame])

  // Esc closes the diagnostics slide-over and returns focus to its trigger.
  useEffect(() => {
    if (!systemPanelOpen) return undefined
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setSystemPanelOpen(false)
        statusBtnRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [systemPanelOpen])

  const activeEntry = useMemo(() => {
    if (!history.length) return null
    return history.find((h) => h.id === selectedHistoryId) || history[0]
  }, [history, selectedHistoryId])

  const dbImageUrl =
    activeEntry?.sourceImageUrl ||
    activeEntry?.personImageUrl ||
    matchResult?.sourceImageUrl ||
    matchResult?.personImageUrl ||
    null

  const confidencePct = typeof matchResult?.matchConfidence === 'number' ? matchResult.matchConfidence : 0

  const plateBoxes =
    videoGeom && plateResult?.frame?.width > 0
      ? (plateResult.regions || []).map((r, i) => {
          // Region coords are in captured-frame pixels; the capture is a
          // uniformly scaled copy of the video, so convert to video pixels
          // first, then apply the same cover transform the video gets.
          const toVideo = videoGeom.vw / plateResult.frame.width
          const { scale, offX, offY } = coverTransform(
            videoGeom.vw,
            videoGeom.vh,
            videoGeom.rw,
            videoGeom.rh,
          )
          return {
            key: i,
            left: r.x * toVideo * scale + offX,
            top: r.y * toVideo * scale + offY,
            width: r.w * toVideo * scale,
            height: r.h * toVideo * scale,
          }
        })
      : []

  return (
    <div className="ls-wrap">
      <style>{`
        /* Tokens are scoped to .ls-wrap, not :root. The previous version
           declared them globally, which collided with the app theme's
           --accent and re-coloured the surrounding chrome. */
        .ls-wrap {
          --bg-canvas:#0B0F1A; --bg-panel:#151B2C; --bg-panel-elevated:#111624;
          --bg-panel-inset:#0D111C; --border-hairline:#252C40;
          --accent-brand-a:#2FD5C8; --accent-brand-b:#B14CE0; --accent-line:#E23B8C;
          --text-primary:#F4F6FB; --text-secondary:#8B93A7;
          --status-unknown:#7C8299; --status-caution:#E3A438;
          --status-alert:#E14F4F; --status-clear:#3FBF7F;

          background: var(--bg-canvas);
          color: var(--text-primary);
          min-height: 100%;
          padding: 24px 16px 64px;
          width: 100%;
          overflow-x: clip;
          font-synthesis-weight: none;
        }
        .ls-shell { max-width: 1180px; margin: 0 auto; }

        .ls-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
        .ls-title { font-size: 28px; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
        .ls-sysbtn {
          display: inline-flex; align-items: center; gap: 7px;
          background: transparent; border: 1px solid var(--border-hairline);
          color: var(--text-secondary); border-radius: 999px;
          padding: 7px 13px; font-size: 13px; font-weight: 500; cursor: pointer;
        }
        .ls-sysbtn:hover { color: var(--text-primary); border-color: var(--accent-brand-a); }

        .ls-grid { display: grid; grid-template-columns: 62fr 38fr; gap: 14px; align-items: start; }
        .ls-grid > * { min-width: 0; }
        .ls-card { background: var(--bg-panel); border: 1px solid var(--border-hairline); border-radius: 14px; }
        .ls-pad { padding: 16px; }
        .ls-eyebrow {
          font-size: 12px; font-weight: 600; text-transform: uppercase;
          letter-spacing: .08em; color: var(--text-secondary); margin: 0 0 10px;
        }

        .ls-camera {
          position: relative; width: 100%; aspect-ratio: 4/3;
          border-radius: 12px; overflow: hidden;
          border: 1px solid var(--border-hairline); background: var(--bg-panel-inset);
        }
        .ls-chip {
          display: inline-block; padding: 4px 9px; border-radius: 999px;
          border: 1px solid rgba(255,255,255,.14); background: rgba(11,15,26,.62);
          color: #fff; font-size: 11px; font-weight: 700; letter-spacing: .03em;
          backdrop-filter: blur(3px);
        }
        .ls-overlay-top-left { position: absolute; top: 10px; left: 10px; z-index: 4; display: grid; gap: 6px; justify-items: start; }
        .ls-overlay-top-right { position: absolute; top: 10px; right: 10px; z-index: 4; }

        /* --- the four-state pill: colour AND word, never colour alone --- */
        .ls-pill {
          display: inline-flex; align-items: center; gap: 7px;
          font-size: 13px; font-weight: 700; text-transform: uppercase;
          letter-spacing: .06em; padding: 5px 11px; border-radius: 999px;
          border: 1px solid currentColor;
        }
        .ls-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; flex: none; }
        .ls-dot--pulse { animation: ls-pulse 1.4s ease-in-out infinite; }
        @keyframes ls-pulse { 0%,100% { opacity: 1 } 50% { opacity: .25 } }
        @media (prefers-reduced-motion: reduce) { .ls-dot--pulse { animation: none } }

        .ls-pill--scanning { color: var(--status-caution); background: rgba(227,164,56,.12); }
        .ls-pill--unknown  { color: var(--status-unknown);  background: rgba(124,130,153,.14); }
        .ls-pill--alert    { color: var(--status-alert);    background: rgba(225,79,79,.14); }
        .ls-pill--clear    { color: var(--status-clear);    background: rgba(63,191,127,.13); }

        .ls-verdict { display: grid; gap: 10px; }
        .ls-identity { font-size: 22px; font-weight: 600; margin: 0; overflow-wrap: anywhere; }
        .ls-explain { font-size: 14px; font-weight: 400; color: var(--text-secondary); margin: 0; overflow-wrap: anywhere; }
        .ls-confidence { font-size: 14px; margin: 2px 0 0; }
        .ls-confidence b { font-size: 22px; font-weight: 600; }

        .ls-plateno {
          display: inline-block; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: 22px; font-weight: 600; letter-spacing: .12em;
          padding: 7px 14px; border-radius: 8px;
          border: 1px solid var(--border-hairline); background: var(--bg-panel-inset);
        }

        .ls-compare { display: grid; grid-template-columns: 1fr auto 1fr; gap: 12px; align-items: center; }
        .ls-tile { border-radius: 10px; overflow: hidden; aspect-ratio: 1/1; background: var(--bg-panel-inset); border: 1px solid var(--border-hairline); }
        .ls-tile img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .ls-tile--empty {
          border: 1px dashed var(--status-unknown); display: grid; place-items: center;
          text-align: center; color: var(--status-unknown); font-size: 12px; padding: 10px; gap: 6px;
        }
        .ls-swap { color: var(--text-secondary); font-size: 20px; user-select: none; }
        .ls-caption { color: var(--text-secondary); font-size: 12px; margin-top: 6px; }
        .ls-similarity { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-hairline); font-size: 14px; }

        .ls-audit { max-height: 240px; overflow-y: auto; display: grid; gap: 6px; align-content: start; }
        .ls-audit button {
          text-align: left; border-radius: 9px; border: 1px solid var(--border-hairline);
          background: var(--bg-panel-elevated); color: var(--text-primary);
          padding: 9px 10px; cursor: pointer; font-family: inherit; font-size: 12px;
        }
        .ls-audit button.active { border-color: var(--accent-brand-a); }
        .ls-audit button.alert { border-left: 3px solid var(--status-alert); }
        .ls-audit button.clear { border-left: 3px solid var(--status-clear); }
        .ls-audit button.unknown { border-left: 3px solid var(--status-unknown); }
        .ls-audit-time { color: var(--text-secondary); }

        .ls-kv { display: flex; justify-content: space-between; gap: 14px; font-size: 13px; padding: 7px 0; border-bottom: 1px solid var(--border-hairline); }
        .ls-kv:last-child { border-bottom: 0; }
        .ls-kv span:first-child { color: var(--text-secondary); }
        .ls-kv span:last-child { max-width: 62%; text-align: right; overflow-wrap: anywhere; }

        .ls-warn { margin-top: 12px; font-size: 13px; color: var(--status-caution); }

        /* diagnostics slide-over */
        .ls-scrim { position: fixed; inset: 0; background: rgba(4,6,12,.55); z-index: 40; }
        .ls-slide {
          position: fixed; top: 0; right: 0; bottom: 0; width: min(380px, 92vw);
          background: var(--bg-panel); border-left: 1px solid var(--border-hairline);
          z-index: 41; padding: 20px; overflow-y: auto;
          --text-primary:#F4F6FB; --text-secondary:#8B93A7; --border-hairline:#252C40;
          --bg-panel-inset:#0D111C; --accent-brand-a:#2FD5C8;
          --status-caution:#E3A438; --status-clear:#3FBF7F; --status-alert:#E14F4F;
          color: var(--text-primary);
        }
        .ls-slide h2 { font-size: 16px; margin: 0 0 4px; }
        .ls-slide .ls-sub { color: var(--text-secondary); font-size: 13px; margin: 0 0 16px; }
        .ls-close { position: absolute; top: 16px; right: 16px; background: transparent; border: 0; color: var(--text-secondary); font-size: 20px; cursor: pointer; line-height: 1; }
        .ls-ready { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 16px; }
        .ls-ready > span { font-size: 12px; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--border-hairline); color: var(--text-secondary); }
        .ls-ready .on { color: var(--status-clear); border-color: var(--status-clear); }
        .ls-ready .off { color: var(--status-caution); border-color: var(--status-caution); }

        .ls-wrap :focus-visible, .ls-slide :focus-visible {
          outline: 2px solid var(--accent-brand-a); outline-offset: 2px; border-radius: 6px;
        }

        @media (max-width: 980px) {
          .ls-grid { grid-template-columns: 1fr; }
        }
        @media (max-width: 640px) {
          .ls-wrap { padding: 16px 12px 48px; }
          .ls-title { font-size: 24px; }
          .ls-kv { flex-direction: column; align-items: flex-start; gap: 3px; }
          .ls-kv span:last-child { max-width: 100%; text-align: left; }
          .ls-compare { grid-template-columns: 1fr; }
          .ls-swap { justify-self: center; transform: rotate(90deg); }
        }
      `}</style>

      <div className="ls-shell">
        <header className="ls-head">
          <h1 className="ls-title">Live scan</h1>
          <button
            type="button"
            ref={statusBtnRef}
            className="ls-sysbtn"
            aria-expanded={systemPanelOpen}
            onClick={() => setSystemPanelOpen((v) => !v)}
          >
            <span aria-hidden="true">◍</span> System status
          </button>
        </header>

        <div className="ls-grid">
          {/* ---------------- video ---------------- */}
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
                  objectFit: 'cover',
                  zIndex: 1,
                }}
              />

              <canvas
                ref={canvasRef}
                style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', zIndex: 2 }}
              />

              {plateBoxes.map((b) => (
                <div
                  key={b.key}
                  style={{
                    position: 'absolute',
                    left: b.left,
                    top: b.top,
                    width: b.width,
                    height: b.height,
                    border: `2px solid ${STATUS_HEX[plateVerdict.state]}`,
                    borderRadius: 4,
                    zIndex: 3,
                    pointerEvents: 'none',
                  }}
                />
              ))}

              <div className="ls-overlay-top-left">
                <span className="ls-chip">REC / LIVE{fps ? ` ${fps} FPS` : ''}</span>
                {hasFaceInFrame ? (
                  <span className="ls-chip">
                    {liveness === LIVENESS.LIVE
                      ? '✓ LIVE PERSON'
                      : liveness === LIVENESS.UNAVAILABLE
                      ? 'LIVENESS N/A'
                      : 'BLINK TO VERIFY'}
                  </span>
                ) : null}
              </div>

              <div className="ls-overlay-top-right">
                <span className="ls-chip">{formatClock(matchResult?.timestamp) || formatClock(new Date().toISOString())}</span>
              </div>
            </div>
          </div>

          {/* ---------------- face verdict ---------------- */}
          <div className="ls-card ls-pad">
            <p className="ls-eyebrow">Face registry</p>
            <div className="ls-verdict">
              <StatusPill state={faceVerdict.state} label={faceVerdict.label} />
              {faceVerdict.name ? <p className="ls-identity">{faceVerdict.name}</p> : null}
              <p className="ls-explain">{faceVerdict.detail}</p>
              {faceVerdict.state === STATE.ALERT || faceVerdict.state === STATE.CLEAR ? (
                <p className="ls-confidence">
                  Confidence: <b>{confidencePct}%</b>
                </p>
              ) : null}

              {/* Liveness is reported separately from the match, never folded
                  into it — "we recognised this face" and "this was a live
                  person" are two different claims (build guide §5). */}
              {hasFaceInFrame ? (
                <p className="ls-explain" style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                  <i
                    className="ls-dot"
                    aria-hidden="true"
                    style={{
                      color:
                        liveness === LIVENESS.LIVE
                          ? 'var(--status-clear)'
                          : liveness === LIVENESS.UNAVAILABLE
                          ? 'var(--status-unknown)'
                          : 'var(--status-caution)',
                    }}
                  />
                  {liveness === LIVENESS.LIVE
                    ? 'Liveness confirmed — blink detected'
                    : liveness === LIVENESS.UNAVAILABLE
                    ? 'Liveness check unavailable on this device'
                    : 'Liveness unconfirmed — a photo does not blink'}
                </p>
              ) : null}
            </div>
          </div>
        </div>

        {/* ---------------- plate registry ---------------- */}
        <div className="ls-card ls-pad" style={{ marginTop: 14 }}>
          <p className="ls-eyebrow">License plate</p>
          <div className="ls-verdict">
            <StatusPill state={plateVerdict.state} label={plateVerdict.label} />
            {plateVerdict.plate ? <span className="ls-plateno">{plateVerdict.plate}</span> : null}
            <p className="ls-explain">{plateVerdict.detail}</p>
          </div>

          {plateSightings.length > 0 && (
            <div style={{ marginTop: 14, borderTop: '1px solid var(--border-hairline)', paddingTop: 10 }}>
              <p className="ls-eyebrow" style={{ marginBottom: 6 }}>Plates seen this session</p>
              {plateSightings.map((s) => (
                <div key={s.id} className="ls-kv">
                  <span style={{ fontFamily: 'ui-monospace, monospace' }}>{s.plateNumber}</span>
                  <span>{(s.status || 'unknown').toUpperCase()} · {new Date(s.at).toLocaleTimeString()}</span>
                </div>
              ))}
            </div>
          )}

          {plateError ? <div className="ls-warn">Plate reader: {plateError}</div> : null}
        </div>

        {/* ---------------- reference comparison ----------------
            Always rendered, directly below the plate panel. Both halves have
            an explicit labelled placeholder so an un-run comparison reads as
            "waiting", never as a broken empty box. */}
        <div className="ls-card ls-pad" style={{ marginTop: 14 }}>
          <p className="ls-eyebrow">Reference comparison</p>
          <div className="ls-compare">
            <div>
              {liveCrop ? (
                <div className="ls-tile">
                  <img src={liveCrop} alt="Face captured from the live camera" />
                </div>
              ) : (
                <div className="ls-tile ls-tile--empty">
                  <span aria-hidden="true" style={{ fontSize: 18 }}>◌</span>
                  <span>Waiting for a face in frame</span>
                </div>
              )}
              <div className="ls-caption">
                {liveCropAt ? `Live camera · captured ${formatClock(liveCropAt)}` : 'Live camera'}
              </div>
            </div>

            <div className="ls-swap" aria-hidden="true">⇄</div>

            <div>
              {dbImageUrl && !dbImageBroken ? (
                <div className="ls-tile">
                  <img
                    src={dbImageUrl}
                    alt="Reference photo held on the identity registry"
                    onError={() => setDbImageBroken(true)}
                  />
                </div>
              ) : (
                <div className="ls-tile ls-tile--empty">
                  <span aria-hidden="true" style={{ fontSize: 18 }}>◌</span>
                  <span>{dbImageBroken ? 'Reference photo unavailable' : 'No reference on file'}</span>
                </div>
              )}
              <div className="ls-caption">
                {activeEntry?.sourceText ? `On file · ${activeEntry.sourceText}` : 'Identity registry'}
              </div>
            </div>
          </div>

          <div className="ls-similarity">
            {typeof matchResult?.matchDistance === 'number' ? (
              <>Similarity: <b>{confidencePct}% match</b></>
            ) : (
              <span style={{ color: 'var(--text-secondary)' }}>
                No comparison yet — waiting for a registry match.
              </span>
            )}
          </div>
        </div>

        {/* ---------------- audit log ---------------- */}
        <div className="ls-card ls-pad" style={{ marginTop: 14 }}>
          <p className="ls-eyebrow">Audit log</p>
          <div className="ls-audit">
            {history.map((entry) => {
              const state = registryStateFor(entry.status)
              return (
                <button
                  key={entry.id}
                  type="button"
                  className={`${state}${selectedHistoryId === entry.id ? ' active' : ''}`}
                  onClick={() => {
                    setSelectedHistoryId(entry.id)
                    setDbImageBroken(false)
                  }}
                >
                  <span className="ls-audit-time">{formatClock(entry.timestamp)}</span>
                  {'  '}
                  {registryStateFor(entry.status) === STATE.UNKNOWN ? 'Unregistered face' : entry.fullName}
                  {'  ·  '}
                  {entry.matchConfidence}%
                </button>
              )
            })}
            {!history.length ? <p className="ls-explain">No scans yet.</p> : null}
          </div>
        </div>

        {webcamError ? <div className="ls-warn">Camera unavailable: {webcamError}</div> : null}
        {scanError ? <div className="ls-warn">{scanError}</div> : null}
        {modelState === 'failed' ? <div className="ls-warn">Detector failed to load: {modelError || 'Unknown error.'}</div> : null}
      </div>

      {/* ---------------- diagnostics slide-over ---------------- */}
      {systemPanelOpen ? (
        <>
          <div className="ls-scrim" onClick={() => setSystemPanelOpen(false)} />
          <aside className="ls-slide" role="dialog" aria-modal="true" aria-label="System status">
            <button
              type="button"
              className="ls-close"
              aria-label="Close system status"
              onClick={() => {
                setSystemPanelOpen(false)
                statusBtnRef.current?.focus()
              }}
            >
              ✕
            </button>
            <h2>System status</h2>
            <p className="ls-sub">Diagnostics for QA — not part of the operator view.</p>

            <div className="ls-ready">
              <span className={webcamReady ? 'on' : 'off'}>Camera: {webcamReady ? 'ready' : 'waiting'}</span>
              <span className={isModelLoaded ? 'on' : 'off'}>Detector: {isModelLoaded ? 'ready' : 'waiting'}</span>
              <span className={backendReady ? 'on' : 'off'}>Backend: {backendReady ? 'ready' : 'waiting'}</span>
            </div>

            <div className="ls-kv"><span>Model</span><span>{MODEL_LABEL}</span></div>
            <div className="ls-kv"><span>Indexed identities</span><span>{memberCount ?? '--'}</span></div>
            <div className="ls-kv"><span>Scan interval</span><span>{SCAN_INTERVAL_MS} ms</span></div>
            <div className="ls-kv"><span>Plate interval</span><span>{PLATE_SCAN_INTERVAL_MS} ms</span></div>
            <div className="ls-kv"><span>Detector FPS</span><span>{fps ?? '--'}</span></div>
            <div className="ls-kv"><span>Liveness</span><span>{liveness}</span></div>
            <div className="ls-kv"><span>Blinks seen</span><span>{blinkCount}</span></div>
            <div className="ls-kv"><span>Round trip</span><span>{roundtripMs ?? '--'} ms</span></div>
            <div className="ls-kv"><span>Cosine distance</span><span>{typeof matchResult?.matchDistance === 'number' ? matchResult.matchDistance.toFixed(4) : '--'}</span></div>
            <div className="ls-kv"><span>Registry status</span><span>{(matchResult?.status || 'none').toUpperCase()}</span></div>
            <div className="ls-kv"><span>Pose</span><span>{matchResult?.poseLabel || '-'}</span></div>
            <div className="ls-kv"><span>Server stages</span><span>{formatStageTiming(matchResult?.stageTiming)}</span></div>
            <div className="ls-kv"><span>Last message</span><span>{matchResult?.message || '--'}</span></div>

            {activeEntry ? (
              <>
                <p className="ls-eyebrow" style={{ marginTop: 18 }}>Selected audit event</p>
                <div className="ls-kv"><span>Name</span><span>{activeEntry.fullName}</span></div>
                <div className="ls-kv"><span>Distance</span><span>{typeof activeEntry.matchDistance === 'number' ? activeEntry.matchDistance.toFixed(4) : '--'}</span></div>
                <div className="ls-kv"><span>Latency</span><span>{activeEntry.latencyMs ?? '--'} ms</span></div>
                {activeEntry.liveCrop ? (
                  <img src={activeEntry.liveCrop} alt="Crop from the selected audit event" style={{ width: '100%', borderRadius: 8, marginTop: 10 }} />
                ) : null}
              </>
            ) : null}
          </aside>
        </>
      ) : null}
    </div>
  )
}
