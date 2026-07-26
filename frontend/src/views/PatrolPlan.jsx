import { useCallback, useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import { api, num } from '../api'
import { useSession } from '../session'

/* Patrol planning for a Crime Prevention Unit.

   Each vehicle gets its own loop, so this is the one screen in the app where a
   categorical palette is the right call — the colours identify vehicles, not
   magnitude. Slots are assigned in fixed order and capped at the first few, and
   every route is also labelled "Vehicle N" in the table, so identity never rests
   on hue alone. The risk cells stay on the sequential blue used everywhere else. */

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

// Categorical slots in fixed order, light and dark steps. The magenta slot was
// dropped when the brand moved to magenta — a vehicle line the same hue as the
// accent reads as chrome rather than data. Validated as a set: worst adjacent
// pair is yellow<->aqua at CVD dE 9.1 / normal dE 22.9, both above the gates.
// Aqua and yellow fall below 3:1 on the light surface, which is why every
// vehicle is also named in the legend and the table.
const VEHICLE_COLORS = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7', '#008300'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#9085e9', '#0ca30c'],
}

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

export default function PatrolPlan() {
  const { unit } = useSession()
  const theme = useThemeName()
  const now = new Date()

  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const tileRef = useRef(null)
  const layerRef = useRef(null)

  const [vehicles, setVehicles] = useState(null)
  const [hour, setHour] = useState(now.getHours())
  const [weekday, setWeekday] = useState((now.getDay() + 6) % 7)
  const [plan, setPlan] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Adopt each unit's own fleet size when the unit changes — carrying the
  // previous unit's vehicle count over silently plans for the wrong fleet.
  useEffect(() => {
    if (unit) setVehicles(unit.vehicles)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [unit?.unit_id])

  /* ---------- map ---------- */
  useEffect(() => {
    if (mapRef.current || !containerRef.current) return undefined
    const map = L.map(containerRef.current, {
      center: [-26.1, 28.05],
      zoom: 10,
      minZoom: 4,
      preferCanvas: true,
    })
    mapRef.current = map
    layerRef.current = L.layerGroup().addTo(map)
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

  /* ---------- data ---------- */
  // Planning takes several seconds (one routing call per vehicle), so changing
  // unit or fleet size mid-flight leaves two requests racing. Without this
  // token the slower earlier response lands last and wins, and you see the
  // previous unit's plan under the new unit's name.
  const requestRef = useRef(0)

  const build = useCallback(() => {
    if (!unit || vehicles === null) return
    const token = ++requestRef.current
    setLoading(true)
    setError(null)
    api
      .patrolPlan({ unit_id: unit.unit_id, vehicles, hour, weekday })
      .then((result) => {
        if (token !== requestRef.current) return
        setPlan(result)
      })
      .catch((err) => {
        if (token !== requestRef.current) return
        setError(err.message)
        setPlan(null)
      })
      .finally(() => {
        if (token === requestRef.current) setLoading(false)
      })
  }, [unit, vehicles, hour, weekday])

  useEffect(build, [build])

  /* ---------- draw ---------- */
  useEffect(() => {
    const map = mapRef.current
    const layer = layerRef.current
    if (!map || !layer || !plan) return

    layer.clearLayers()
    const palette = VEHICLE_COLORS[theme]

    plan.routes.forEach((route, index) => {
      const color = palette[index % palette.length]
      if (route.coordinates.length) {
        L.polyline(route.coordinates, { color, weight: 4, opacity: 0.85 }).addTo(layer)
      }
      route.stops.forEach((stop, order) => {
        L.circleMarker([stop.lat, stop.lng], {
          radius: 7,
          color: '#fff',
          weight: 2,
          fillColor: color,
          fillOpacity: 0.95,
        })
          .bindTooltip(
            `Vehicle ${route.vehicle} · stop ${order + 1} — risk ${stop.score.toFixed(2)}`,
            { direction: 'top' },
          )
          .addTo(layer)
      })
    })

    // The base, where every loop starts and ends.
    L.circleMarker([plan.unit.base_lat, plan.unit.base_lng], {
      radius: 9,
      color: '#fff',
      weight: 3,
      fillColor: '#0b0b0b',
      fillOpacity: 1,
    })
      .bindTooltip(`${plan.unit.name} — base`, { direction: 'top' })
      .addTo(layer)

    const points = plan.routes.flatMap((r) => r.coordinates)
    if (points.length) {
      const bounds = L.latLngBounds(points)
      if (bounds.isValid()) map.fitBounds(bounds, { padding: [40, 40] })
    } else {
      map.setView([plan.unit.base_lat, plan.unit.base_lng], 11)
    }
  }, [plan, theme])

  if (!unit) return <p className="muted">Loading unit…</p>

  const coverage = plan?.coverage
  const palette = VEHICLE_COLORS[theme]

  return (
    <>
      <h1>Patrol planning</h1>
      <p className="muted" style={{ margin: '2px 0 16px' }}>
        {unit.name} · allocates vehicles across the highest-risk areas within {unit.radius_km} km of
        base, for the shift you're planning.
      </p>

      <div
        className="card"
        style={{ padding: 14, marginBottom: 16, display: 'flex', gap: 24, flexWrap: 'wrap' }}
      >
        <div className="field" style={{ gap: 7 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Vehicles on shift
          </span>
          <div style={{ display: 'flex', gap: 6 }}>
            {[1, 2, 3, 4, 5, 6].map((n) => (
              <button
                key={n}
                type="button"
                className={`btn${vehicles === n ? ' btn-primary' : ''}`}
                style={{ borderRadius: 999, padding: '4px 13px' }}
                onClick={() => setVehicles(n)}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        <div className="field" style={{ gap: 7, minWidth: 280 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Shift start
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
              aria-label="Shift start hour"
              style={{ flex: 1, minWidth: 90 }}
            />
            <strong style={{ fontVariantNumeric: 'tabular-nums', minWidth: 48 }}>
              {String(hour).padStart(2, '0')}:00
            </strong>
          </div>
        </div>
      </div>

      {error ? (
        <div className="banner banner-bad" role="alert" style={{ marginBottom: 16 }}>
          {error}
        </div>
      ) : null}

      {plan?.reason ? (
        <div className="banner banner-info" style={{ marginBottom: 16 }} role="status">
          {plan.reason}
        </div>
      ) : null}

      {coverage && coverage.cells ? (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 12,
            marginBottom: 16,
            opacity: loading ? 0.55 : 1,
            transition: 'opacity 120ms ease',
          }}
          className="grid-4-tiles"
        >
          <Tile k="Risk covered" v={`${Math.round(coverage.share * 100)}%`} m={`${coverage.cells} areas visited`} />
          <Tile k="Distance" v={`${num.format(Math.round(coverage.total_km))} km`} m="across all vehicles" />
          <Tile
            k="Risk per km"
            v={coverage.risk_per_km.toFixed(3)}
            m="higher is more efficient"
          />
          <Tile k="Vehicles" v={plan.vehicles} m={`${unit.vehicles} available`} />
        </div>
      ) : null}

      <div className="card" style={{ overflow: 'hidden', marginBottom: 16 }}>
        <div
          ref={containerRef}
          style={{
            height: 520,
            background: 'var(--page)',
            opacity: loading ? 0.55 : 1,
            transition: 'opacity 120ms ease',
          }}
        />
        <div
          style={{
            display: 'flex',
            gap: 16,
            flexWrap: 'wrap',
            alignItems: 'center',
            borderTop: '1px solid var(--hairline)',
            padding: '11px 15px',
          }}
        >
          {(plan?.routes || []).map((route, index) => (
            <span key={route.vehicle} style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <span
                style={{
                  width: 22,
                  height: 4,
                  borderRadius: 2,
                  background: palette[index % palette.length],
                }}
              />
              <span className="muted">Vehicle {route.vehicle}</span>
            </span>
          ))}
          <span className="spacer" />
          <span className="tiny">Black marker is the unit's base — every loop starts and ends there</span>
        </div>
      </div>

      {plan?.routes?.length ? (
        <div className="card" style={{ overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 13.5 }}>
            <caption className="muted" style={{ textAlign: 'left', padding: '14px 15px' }}>
              Stops are ordered into a loop from base and back. Distances and times are real road
              distances from OpenStreetMap, not straight lines.
            </caption>
            <thead>
              <tr>
                <Th>Vehicle</Th>
                <Th num>Stops</Th>
                <Th num>Distance</Th>
                <Th num>Driving time</Th>
                <Th num>Risk covered</Th>
              </tr>
            </thead>
            <tbody>
              {plan.routes.map((route, index) => (
                <tr key={route.vehicle}>
                  <Td>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                      <span
                        style={{
                          width: 14,
                          height: 4,
                          borderRadius: 2,
                          background: palette[index % palette.length],
                        }}
                      />
                      Vehicle {route.vehicle}
                    </span>
                  </Td>
                  <Td num>{route.stops.length}</Td>
                  <Td num>{route.distance_km != null ? `${route.distance_km} km` : '—'}</Td>
                  <Td num>{route.duration_min != null ? `${Math.round(route.duration_min)} min` : '—'}</Td>
                  <Td num>{route.risk_covered.toFixed(2)}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <p className="tiny" style={{ marginTop: 14, maxWidth: 780 }}>
        Stops are the highest-risk areas for this hour, split geographically across the vehicles and
        ordered into a loop (nearest-neighbour with 2-opt). This is a good heuristic, not a proven
        optimum — a full vehicle-routing solver (VROOM or OR-Tools) is the upgrade once shift
        lengths and time windows matter. Risk is based on historical claim density and is not
        adjusted for how many members travel through each area.
      </p>
    </>
  )
}

function Tile({ k, v, m }) {
  return (
    <div className="card" style={{ padding: '13px 15px' }}>
      <div
        className="tiny"
        style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 5 }}
      >
        {k}
      </div>
      <div style={{ fontSize: 27, fontWeight: 650, letterSpacing: '-0.02em', lineHeight: 1.15 }}>{v}</div>
      {m ? <div className="muted" style={{ fontSize: 12.5, marginTop: 2 }}>{m}</div> : null}
    </div>
  )
}

const Th = ({ children, num: isNum }) => (
  <th
    style={{
      textAlign: isNum ? 'right' : 'left',
      padding: '8px 14px',
      borderBottom: '1px solid var(--hairline)',
      fontSize: 11,
      fontWeight: 650,
      letterSpacing: '0.05em',
      textTransform: 'uppercase',
      color: 'var(--ink-muted)',
      whiteSpace: 'nowrap',
    }}
  >
    {children}
  </th>
)

const Td = ({ children, num: isNum }) => (
  <td
    style={{
      textAlign: isNum ? 'right' : 'left',
      padding: '9px 14px',
      borderBottom: '1px solid var(--hairline)',
      fontVariantNumeric: isNum ? 'tabular-nums' : 'normal',
    }}
  >
    {children}
  </td>
)
