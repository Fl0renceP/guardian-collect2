/**
 * In-browser licence-plate region finder.
 *
 * Why this exists: the Azure Vision free tier allows 20 OCR calls a minute,
 * total. A live camera at even one frame per second would exhaust that in
 * under half a minute, and almost every one of those calls would be spent on a
 * frame containing no vehicle at all.
 *
 * So the browser answers the cheap question first — "is there anything in this
 * frame shaped like a plate, and where?" — using nothing but canvas pixels. A
 * call is only spent when the answer is yes, and it is spent on a crop around
 * the candidate rather than the whole frame, which is also what makes a plate
 * at the end of a driveway legible once the server upscales it.
 *
 * This deliberately does not try to decide *which* candidate is the plate. It
 * cannot: a model badge is also a row of high-contrast characters. It returns
 * a shortlist and lets the server's plate grammar rule.
 */

const WORK_WIDTH = 400

// Characters within a plate sit a few pixels apart at this working width, so
// closing horizontally over this radius fuses them into one blob while leaving
// genuinely separate objects apart.
const CLOSE_RADIUS_X = 9
const CLOSE_RADIUS_Y = 1

let workCanvas = null

const getWorkCanvas = (w, h) => {
  if (!workCanvas) workCanvas = document.createElement('canvas')
  if (workCanvas.width !== w || workCanvas.height !== h) {
    workCanvas.width = w
    workCanvas.height = h
  }
  return workCanvas
}

/** Horizontal dilate/erode via a running window sum — O(n) per row. */
const morphH = (src, w, h, radius, erode) => {
  const out = new Uint8Array(w * h)
  const span = radius * 2 + 1
  for (let y = 0; y < h; y++) {
    const row = y * w
    let sum = 0
    for (let x = 0; x <= radius && x < w; x++) sum += src[row + x]
    for (let x = 0; x < w; x++) {
      const need = erode ? Math.min(span, Math.min(x + radius, w - 1) - Math.max(x - radius, 0) + 1) : 1
      out[row + x] = (erode ? sum >= need : sum > 0) ? 1 : 0
      const drop = x - radius
      const add = x + radius + 1
      if (drop >= 0) sum -= src[row + drop]
      if (add < w) sum += src[row + add]
    }
  }
  return out
}

const dilateV = (src, w, h, radius) => {
  if (radius <= 0) return src
  const out = new Uint8Array(w * h)
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let on = 0
      for (let dy = -radius; dy <= radius && !on; dy++) {
        const yy = y + dy
        if (yy >= 0 && yy < h && src[yy * w + x]) on = 1
      }
      out[y * w + x] = on
    }
  }
  return out
}

/**
 * Connected components, 4-connected, iterative so a large blob cannot blow the
 * call stack on a low-end phone.
 */
const components = (binary, w, h) => {
  const seen = new Uint8Array(w * h)
  const stack = new Int32Array(w * h)
  const boxes = []

  for (let i = 0; i < binary.length; i++) {
    if (!binary[i] || seen[i]) continue
    let top = 0
    stack[top++] = i
    seen[i] = 1
    let minX = w, minY = h, maxX = 0, maxY = 0, area = 0

    while (top > 0) {
      const p = stack[--top]
      const x = p % w
      const y = (p - x) / w
      area++
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y

      if (x > 0 && binary[p - 1] && !seen[p - 1]) { seen[p - 1] = 1; stack[top++] = p - 1 }
      if (x < w - 1 && binary[p + 1] && !seen[p + 1]) { seen[p + 1] = 1; stack[top++] = p + 1 }
      if (y > 0 && binary[p - w] && !seen[p - w]) { seen[p - w] = 1; stack[top++] = p - w }
      if (y < h - 1 && binary[p + w] && !seen[p + w]) { seen[p + w] = 1; stack[top++] = p + w }
    }

    boxes.push({ x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1, area })
  }
  return boxes
}

/**
 * Find plate-shaped regions in the current video frame.
 *
 * Returns boxes in the video's own pixel coordinates, best first, each with a
 * `score` and a normalised copy for sending to the API.
 */
export function detectPlateRegions(video, options = {}) {
  const { maxRegions = 4, minScore = 0.55 } = options
  if (!video || !video.videoWidth || !video.videoHeight) return []

  const vw = video.videoWidth
  const vh = video.videoHeight
  const scale = Math.min(1, WORK_WIDTH / vw)
  const w = Math.max(32, Math.round(vw * scale))
  const h = Math.max(32, Math.round(vh * scale))

  const canvas = getWorkCanvas(w, h)
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  ctx.drawImage(video, 0, 0, w, h)

  let pixels
  try {
    pixels = ctx.getImageData(0, 0, w, h).data
  } catch {
    return [] // tainted canvas — nothing we can do from here
  }

  const gray = new Float32Array(w * h)
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * pixels[p] + 0.587 * pixels[p + 1] + 0.114 * pixels[p + 2]
  }

  // Vertical-edge energy. Plate characters are stamped verticals; sky, road
  // and body panels are not.
  const edges = new Float32Array(w * h)
  let sum = 0
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x
      const value = Math.abs(
        gray[i - w - 1] + 2 * gray[i - 1] + gray[i + w - 1] -
        gray[i - w + 1] - 2 * gray[i + 1] - gray[i + w + 1]
      )
      edges[i] = value
      sum += value
    }
  }

  const mean = sum / (w * h)
  let variance = 0
  for (let i = 0; i < edges.length; i++) {
    const d = edges[i] - mean
    variance += d * d
  }
  const std = Math.sqrt(variance / edges.length)

  // Adaptive rather than fixed: a plate in porch light and a plate in noon sun
  // produce very different absolute gradients, but both sit well above their
  // own frame's mean.
  const threshold = mean + std * 1.1
  const binary = new Uint8Array(w * h)
  for (let i = 0; i < edges.length; i++) binary[i] = edges[i] > threshold ? 1 : 0

  // Close: fuse characters into a plate-sized blob.
  let mask = morphH(binary, w, h, CLOSE_RADIUS_X, false)
  mask = dilateV(mask, w, h, CLOSE_RADIUS_Y)
  mask = morphH(mask, w, h, CLOSE_RADIUS_X, true)

  const frameArea = w * h
  const scored = []

  for (const box of components(mask, w, h)) {
    if (box.w < 16 || box.h < 5) continue
    const ratio = box.w / box.h
    if (ratio < 1.8 || ratio > 8.5) continue
    const coverage = (box.w * box.h) / frameArea
    if (coverage < 0.0012 || coverage > 0.55) continue
    const fill = box.area / (box.w * box.h)
    if (fill < 0.4) continue

    // An SA long plate is about 4.7:1; the square two-row type about 2:1.
    const ratioFit = 1 - Math.min(
      Math.min(Math.abs(ratio - 4.7) / 4.7, Math.abs(ratio - 2.0) / 2.0),
      1
    )
    const score = fill * 0.45 + ratioFit * 0.45 + Math.min(coverage * 6, 1) * 0.1
    if (score < minScore) continue

    scored.push({
      score,
      x: box.x / scale,
      y: box.y / scale,
      w: box.w / scale,
      h: box.h / scale,
      norm: { x: box.x / w, y: box.y / h, w: box.w / w, h: box.h / h },
    })
  }

  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, maxRegions)
}

/**
 * Smallest box covering every candidate, or null.
 *
 * Sending the top-scoring candidate alone is the obvious move and the wrong
 * one. Measured against rendered vehicle frames, the plate is reliably found —
 * and reliably ranked *third*, behind the maker's name and the model badge,
 * because all three are horizontal bands of high-contrast characters and this
 * detector cannot read. Cropping to the winner would post the badge to Azure
 * and never look at the plate. The union costs a little resolution and gets
 * the plate into the frame the server actually searches.
 */
export function unionOfRegions(regions) {
  if (!regions || !regions.length) return null
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  for (const r of regions) {
    x0 = Math.min(x0, r.x)
    y0 = Math.min(y0, r.y)
    x1 = Math.max(x1, r.x + r.w)
    y1 = Math.max(y1, r.y + r.h)
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0, score: regions[0].score }
}

/**
 * Crop a padded window around a candidate at the camera's native resolution.
 *
 * Native resolution matters: the server can upscale, but it cannot recover
 * detail that downscaling already threw away. The padding matters too — the
 * detector locks onto the character block, not the plate housing, and the
 * server needs the surrounding edges to deskew against.
 */
export function cropRegionBlob(video, region, { pad = 1.0, quality = 0.9, maxWidth = 1280 } = {}) {
  if (!video || !video.videoWidth) return Promise.resolve(null)

  const vw = video.videoWidth
  const vh = video.videoHeight
  const px = region ? region.w * pad : 0
  const py = region ? region.h * pad * 1.6 : 0

  const sx = region ? Math.max(0, Math.floor(region.x - px)) : 0
  const sy = region ? Math.max(0, Math.floor(region.y - py)) : 0
  const ex = region ? Math.min(vw, Math.ceil(region.x + region.w + px)) : vw
  const ey = region ? Math.min(vh, Math.ceil(region.y + region.h + py)) : vh
  const sw = Math.max(1, ex - sx)
  const sh = Math.max(1, ey - sy)

  const outScale = Math.min(1, maxWidth / sw)
  const cw = Math.max(1, Math.round(sw * outScale))
  const ch = Math.max(1, Math.round(sh * outScale))

  const canvas = document.createElement('canvas')
  canvas.width = cw
  canvas.height = ch
  canvas.getContext('2d').drawImage(video, sx, sy, sw, sh, 0, 0, cw, ch)

  // Where the candidate sits inside the crop we are sending, so the server
  // refines our guess instead of searching the crop from scratch.
  const roi = region
    ? {
        x: (region.x - sx) / sw,
        y: (region.y - sy) / sh,
        w: region.w / sw,
        h: region.h / sh,
      }
    : null

  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob ? { blob, roi, crop: { sx, sy, sw, sh } } : null), 'image/jpeg', quality)
  })
}
