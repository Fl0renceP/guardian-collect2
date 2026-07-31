import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import { api, num } from '../api'
import { useSession } from '../session'

/* Safer-route planning.

   Colour reasoning: the risk cells keep the sequential magenta->violet ramp used
   by the hot-spot map, because they encode the same thing (magnitude) and should
   read the same way. The routes therefore must not share that hue family —
   a magenta line over magenta cells would be unreadable, which is why the
   recommended route stays orange even though magenta is now the brand accent.
   Beyond that the routes use **emphasis**: the recommended one is drawn thick
   and saturated, the alternative thin and grey. That matches what the screen is
   actually saying (one of these is advised) and keeps identity off colour alone,
   since both are labelled in the legend and the comparison table. */

const MODES = [
  { id: 'auto', label: 'Driving' },
  { id: 'bicycle', label: 'Cycling' },
  { id: 'pedestrian', label: 'Walking' },
]

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

// A trip where the trade-off is real, so the page shows something meaningful
// on first load rather than an empty map.
const DEFAULT_ORIGIN = { lat: -25.7479, lng: 28.2293, label: 'Pretoria' }
const DEFAULT_DEST = { lat: -26.2041, lng: 28.0473, label: 'Johannesburg CBD' }

const RISK_STEPS = [
  { min: 0.85, color: '#a3126b' },
  { min: 0.7, color: '#7a3fe0' },
  { min: 0.55, color: '#5b8ede' },
  { min: 0.4, color: '#8fd3dd' },
  { min: 0.0, color: '#d3f0ec' },
]
const riskColor = (score) => (RISK_STEPS.find((s) => score >= s.min) || RISK_STEPS[4]).color

function useThemeName() {
  const [theme, setTheme] = useState(
    () => document.documentElement.getAttribute('data-theme') || 'light',
  )
  useEffect(() => {
    const observer = new MutationObserver(() =>
      setTheme(document.documentElement.getAttribute('data-theme') || 'light'),
    )
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => observer.disconnect()
  }, [])
  return theme
}

export default function SafeRoute() {
  const theme = useThemeName()
  const { member } = useSession()
  const now = useMemo(() => new Date(), [])

  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const tileRef = useRef(null)
  const riskLayerRef = useRef(null)
  const routeLayerRef = useRef(null)
  const pinLayerRef = useRef(null)

  const [origin, setOrigin] = useState(DEFAULT_ORIGIN)
  const [destination, setDestination] = useState(DEFAULT_DEST)
  const [picking, setPicking] = useState(null) // 'origin' | 'destination' | null
  const [mode, setMode] = useState('auto')
  const [hour, setHour] = useState(now.getHours())
  const [weekday, setWeekday] = useState((now.getDay() + 6) % 7) // JS Sun=0 -> Mon=0
  const [result, setResult] = useState(null)
  const [risk, setRisk] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showRisk, setShowRisk] = useState(true)
  const [usedHome, setUsedHome] = useState(false)

  // Start from home when the member has opted in — that's the trip they're most
  // likely planning. Only once, so it never fights a manually placed pin.
  useEffect(() => {
    if (usedHome || !member?.share_location || member.home_lat == null) return
    setOrigin({ lat: member.home_lat, lng: member.home_lng, label: 'Home' })
    setUsedHome(true)
  }, [member, usedHome])

  /* ---------- map ---------- */
  useEffect(() => {
    // Guard on the node, not just on mapRef: if anything above this ever
    // early-returns before the container renders, Leaflet throws
    // "Map container not found" and takes the whole view down with it.
    if (mapRef.current || !containerRef.current) return undefined
    const map = L.map(containerRef.current, {
      center: [-26.0, 28.1],
      zoom: 9,
      minZoom: 4,
      preferCanvas: true,
    })
    mapRef.current = map
    riskLayerRef.current = L.layerGroup().addTo(map)
    routeLayerRef.current = L.layerGroup().addTo(map)
    pinLayerRef.current = L.layerGroup().addTo(map)
    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (tileRef.current) map.removeLayer(tileRef.current)
    const url =
      theme === 'dark'
        ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
        : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
    tileRef.current = L.tileLayer(url, {
      maxZoom: 18,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    }).addTo(map)
    tileRef.current.bringToBack()
  }, [theme])

  // Click to set the start or end point.
  useEffect(() => {
    const map = mapRef.current
    if (!map) return undefined
    const onClick = (event) => {
      if (!picking) return
      const point = { lat: event.latlng.lat, lng: event.latlng.lng, label: 'Dropped pin' }
      if (picking === 'origin') setOrigin(point)
      else setDestination(point)
      setPicking(null)
    }
    map.on('click', onClick)
    return () => map.off('click', onClick)
  }, [picking])

  /* ---------- data ---------- */
  const loadRisk = useCallback(() => {
    api
      .risk({ hour, weekday, min_score: 0.4, limit: 900 })
      .then(setRisk)
      .catch(() => setRisk(null))
  }, [hour, weekday])

  useEffect(loadRisk, [loadRisk])

  // Two routing calls per comparison, so dragging the hour slider leaves
  // several requests in flight. Without this token a slower earlier response
  // can land last and show a route for the wrong departure time.
  const requestRef = useRef(0)

  const compare = useCallback(() => {
    const token = ++requestRef.current
    setLoading(true)
    setError(null)
    api
      .compareRoutes({
        origin: [origin.lat, origin.lng],
        destination: [destination.lat, destination.lng],
        mode,
        hour,
        weekday,
      })
      .then((data) => {
        if (token !== requestRef.current) return
        setResult(data)
      })
      .catch((err) => {
        if (token !== requestRef.current) return
        setError(err.message)
        setResult(null)
      })
      .finally(() => {
        if (token === requestRef.current) setLoading(false)
      })
  }, [origin, destination, mode, hour, weekday])

  useEffect(compare, [compare])

  /* ---------- draw risk cells ---------- */
  useEffect(() => {
    const layer = riskLayerRef.current
    if (!layer) return
    layer.clearLayers()
    if (!risk || !showRisk) return

    risk.cells.forEach((cell) => {
      // Approximate the hex with a circle — drawing true boundaries for ~900
      // cells costs a lot of DOM for a backdrop the eye reads as a wash anyway.
      L.circleMarker([cell.lat, cell.lng], {
        radius: 9,
        stroke: false,
        fillColor: riskColor(cell.score),
        fillOpacity: 0.1 + 0.35 * cell.score,
        interactive: false,
      }).addTo(layer)
    })
  }, [risk, showRisk])

  /* ---------- draw routes + pins ---------- */
  useEffect(() => {
    const map = mapRef.current
    const layer = routeLayerRef.current
    const pins = pinLayerRef.current
    if (!map || !layer) return

    layer.clearLayers()
    pins.clearLayers()

    const accent = theme === 'dark' ? '#d95926' : '#eb6834'
    const muted = theme === 'dark' ? '#898781' : '#898781'

    const drawn = []
    if (result?.fastest) drawn.push({ ...result.fastest, key: 'fastest' })
    if (result?.safer) drawn.push({ ...result.safer, key: 'safer' })

    // Draw the non-recommended line first so the recommended one sits on top.
    drawn
      .slice()
      .sort((a) => (a.key === result?.recommendation ? 1 : -1))
      .forEach((route) => {
        const recommended = route.key === result?.recommendation
        L.polyline(route.coordinates, {
          color: recommended ? accent : muted,
          weight: recommended ? 6 : 3.5,
          opacity: recommended ? 0.95 : 0.75,
          dashArray: recommended ? null : '6 6',
        }).addTo(layer)
      })

    const pin = (point, label, color) =>
      L.circleMarker([point.lat, point.lng], {
        radius: 7,
        color: '#fff',
        weight: 2,
        fillColor: color,
        fillOpacity: 1,
      })
        .bindTooltip(label, { direction: 'top' })
        .addTo(pins)

    pin(origin, 'Start', '#0ca30c')
    pin(destination, 'Destination', '#d03b3b')

    if (drawn.length) {
      const bounds = L.latLngBounds(drawn.flatMap((r) => r.coordinates))
      if (bounds.isValid()) map.fitBounds(bounds, { padding: [40, 40] })
    }
  }, [result, theme, origin, destination])

  const swap = () => {
    setOrigin(destination)
    setDestination(origin)
  }

  const rec = result?.recommendation
  const advisedSafer = rec === 'safer'

  return (
    <>
      <h1>Plan a safer route</h1>
      <p className="muted" style={{ margin: '2px 0 16px' }}>
        Compares the fastest way there against one that avoids areas with more
        travel-related claims at your time of travel.
      </p>

      <div className="card filters">
        <div className="field" style={{ gap: 7 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Start &amp; destination
          </span>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
            <button
              type="button"
              className={`btn${picking === 'origin' ? ' btn-primary' : ''}`}
              onClick={() => setPicking(picking === 'origin' ? null : 'origin')}
            >
              {picking === 'origin' ? 'Click the map…' : `Start: ${origin.label}`}
            </button>
            <button type="button" className="btn" onClick={swap} title="Swap start and destination">
              ⇄
            </button>
            <button
              type="button"
              className={`btn${picking === 'destination' ? ' btn-primary' : ''}`}
              onClick={() => setPicking(picking === 'destination' ? null : 'destination')}
            >
              {picking === 'destination' ? 'Click the map…' : `To: ${destination.label}`}
            </button>
          </div>
        </div>

        <div className="field" style={{ gap: 7 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            How you're travelling
          </span>
          <div style={{ display: 'flex', gap: 6 }}>
            {MODES.map((m) => (
              <button
                key={m.id}
                type="button"
                className={`btn btn-chip${mode === m.id ? ' btn-primary' : ''}`}
                onClick={() => setMode(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <div className="field" style={{ gap: 7, minWidth: 260 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            When you're travelling
          </span>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <select value={weekday} onChange={(e) => setWeekday(Number(e.target.value))} style={{ width: 'auto' }}>
              {DAYS.map((d, i) => (
                <option key={d} value={i}>
                  {d}
                </option>
              ))}
            </select>
            <input
              type="range"
              min="0"
              max="23"
              value={hour}
              onChange={(e) => setHour(Number(e.target.value))}
              aria-label="Hour of departure"
              style={{ flex: 1, minWidth: 90 }}
            />
            <strong style={{ fontVariantNumeric: 'tabular-nums', minWidth: 48 }}>
              {String(hour).padStart(2, '0')}:00
            </strong>
          </div>
          {risk ? (
            <span className="hint">
              Risk at this time is {risk.hour_multiplier < 1 ? 'below' : 'above'} average
              (×{risk.hour_multiplier.toFixed(2)} for the hour, ×{risk.dow_multiplier.toFixed(2)} for the day)
            </span>
          ) : null}
        </div>

        <div className="field" style={{ gap: 7 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Overlay
          </span>
          <button type="button" className={`btn${showRisk ? ' btn-primary' : ''}`} onClick={() => setShowRisk((v) => !v)}>
            {showRisk ? 'Risk areas on' : 'Risk areas off'}
          </button>
        </div>
      </div>

      {error ? (
        <div className="banner banner-bad" style={{ marginBottom: 16 }} role="alert">
          {error}
        </div>
      ) : null}

      {result ? (
        <div
          className={`banner ${advisedSafer ? 'banner-good' : 'banner-info'}`}
          style={{ marginBottom: 16 }}
          role="status"
        >
          <strong>
            {advisedSafer ? 'A safer route is worth taking' : 'The fastest route is fine'}
          </strong>
          <p style={{ margin: '3px 0 0' }}>{result.reason}</p>
        </div>
      ) : null}

      <div className="card" style={{ overflow: 'hidden', marginBottom: 16 }}>
        <div
          ref={containerRef}
          className="map-canvas map-md"
          style={{
            opacity: loading ? 0.55 : 1,
            cursor: picking ? 'crosshair' : 'grab',
          }}
        />
        <div className="map-footer">
          <LegendLine color={theme === 'dark' ? '#d95926' : '#eb6834'} weight={6} label="Recommended" />
          <LegendLine color="#898781" weight={3} dashed label="Alternative" />
          <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <span style={{ display: 'flex', gap: 2 }}>
              {RISK_STEPS.slice().reverse().map((s) => (
                <span key={s.min} style={{ width: 16, height: 9, background: s.color, borderRadius: 2 }} />
              ))}
            </span>
            <span className="muted">Travel-claim density, low → high</span>
          </span>
          <span className="spacer" />
          <span className="tiny">
            {picking ? 'Click the map to place the point' : 'Tip: use the Start / To buttons to pick points on the map'}
          </span>
        </div>
      </div>

      {result ? (
        <div className="card">
          <p className="muted table-note" id="route-table-note">
            Exposure is the average travel-claim density along the route at this time, on a 0–1
            scale. It compares routes — it is not a probability of anything happening.
          </p>
          <div className="table-wrap">
            <table className="data-table" aria-describedby="route-table-note">
              <thead>
                <tr>
                  <Th>Route</Th>
                  <Th num>Distance</Th>
                  <Th num>Time</Th>
                  <Th num>Exposure</Th>
                  <Th num>Worst point</Th>
                  <Th num>Share in elevated areas</Th>
                </tr>
              </thead>
              <tbody>
                <RouteRow
                  label="Fastest"
                  route={result.fastest}
                  recommended={rec === 'fastest'}
                  accent={theme === 'dark' ? '#d95926' : '#eb6834'}
                />
                {result.safer ? (
                  <RouteRow
                    label={`Avoiding ${result.safer.avoided_cells} area${result.safer.avoided_cells === 1 ? '' : 's'}`}
                    route={result.safer}
                    recommended={rec === 'safer'}
                    accent={theme === 'dark' ? '#d95926' : '#eb6834'}
                  />
                ) : null}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      <p className="tiny" style={{ marginTop: 14, maxWidth: 760 }}>
        Based on {num.format(risk?.profile?.claims_used || 0)} historical travel-related Discovery
        Insure claims, located to suburb centres and grouped into ~
        {risk?.profile?.cell_km2 || 5} km² areas. It shows where claims have concentrated in the
        past, not where crime will happen — and it is not adjusted for how many members drive
        through each area, so busy areas will tend to show more claims.
      </p>
    </>
  )
}

function RouteRow({ label, route, recommended, accent }) {
  return (
    <tr style={{ background: recommended ? 'var(--page)' : 'transparent' }}>
      <Td>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              width: 14,
              height: recommended ? 4 : 2,
              borderRadius: 2,
              background: recommended ? accent : '#898781',
            }}
          />
          <span style={{ fontWeight: recommended ? 650 : 400 }}>{label}</span>
          {recommended ? <span className="tiny">recommended</span> : null}
        </span>
      </Td>
      <Td num>{route.distance_km} km</Td>
      <Td num>{route.duration_min} min</Td>
      <Td num>{route.risk.mean.toFixed(2)}</Td>
      <Td num>{route.risk.peak.toFixed(2)}</Td>
      <Td num>{Math.round(route.risk.high_share * 100)}%</Td>
    </tr>
  )
}

function LegendLine({ color, weight, dashed, label }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
      <span
        style={{
          width: 26,
          height: 0,
          borderTop: `${weight}px ${dashed ? 'dashed' : 'solid'} ${color}`,
          borderRadius: 2,
        }}
      />
      <span className="muted">{label}</span>
    </span>
  )
}

const Th = ({ children, num: isNum }) => <th className={isNum ? 'num' : undefined}>{children}</th>

const Td = ({ children, num: isNum }) => <td className={isNum ? 'num' : undefined}>{children}</td>
