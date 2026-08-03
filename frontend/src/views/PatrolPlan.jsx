import { useCallback, useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import { api, num } from '../api'
import LoadingGraphic from '../components/LoadingGraphic'
import { useSession } from '../session'

/* Patrol planning for a Crime Prevention Unit.

   Each vehicle gets its own loop, so this is the one screen in the app where a
   categorical palette is the right call — the colours identify vehicles, not
   magnitude. Slots are assigned in fixed order and capped at the first few, and
   every route is also labelled "Vehicle N" in the table, so identity never rests
   on hue alone. The risk cells stay on the sequential blue used everywhere else.

   Laid under those is the comparison layer: the plain fastest loop through the
   same stops, in red, the route a standard navigation app would give a driver
   handed that list. Red is reserved for it here and on the member's route
   screen, so "red" means the same thing in both places — the standard route,
   not ours. It is drawn thin and dashed beneath the vehicle colours because it
   is a reference, not a plan, and it can be switched off outright. */

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

// The standard-navigation comparison loop. Deliberately the same red as the
// member screen's fastest route.
const FASTEST_COLOR = { light: '#d92b2b', dark: '#f2635f' }

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

  // A callback ref, not useRef: this view returns early until the unit loads,
  // so on the first render the map container isn't in the DOM at all. A mount
  // effect would run once against a null node and never fire again, leaving a
  // permanently blank map. This re-runs the moment the node actually appears.
  const [container, setContainer] = useState(null)
  const mapRef = useRef(null)
  const tileRef = useRef(null)
  const layerRef = useRef(null)
  const fittedRef = useRef(null)

  const [vehicles, setVehicles] = useState(null)
  const [hour, setHour] = useState(now.getHours())
  const [weekday, setWeekday] = useState((now.getDay() + 6) % 7)
  const [plan, setPlan] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showFastest, setShowFastest] = useState(true)

  // Adopt each unit's own fleet size when the unit changes — carrying the
  // previous unit's vehicle count over silently plans for the wrong fleet.
  useEffect(() => {
    if (unit) setVehicles(unit.vehicles)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [unit?.unit_id])

  /* ---------- map ---------- */
  useEffect(() => {
    if (mapRef.current || !container) return undefined
    const map = L.map(container, {
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
  }, [container])

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

    // Reference layer first, so every vehicle's own loop draws on top of it.
    if (showFastest) {
      plan.routes.forEach((route) => {
        if (!route.detoured || !route.fastest?.coordinates?.length) return
        L.polyline(route.fastest.coordinates, {
          color: FASTEST_COLOR[theme],
          weight: 3,
          opacity: 0.8,
          dashArray: '7 6',
        })
          .bindTooltip(
            `Vehicle ${route.vehicle} — fastest route: ${route.fastest.distance_km} km, ` +
              `${Math.round(route.fastest.duration_min)} min`,
            { direction: 'top', sticky: true },
          )
          .addTo(layer)
      })
    }

    plan.routes.forEach((route, index) => {
      const color = palette[index % palette.length]
      if (route.coordinates.length) {
        L.polyline(route.coordinates, { color, weight: 4, opacity: 0.85 }).addTo(layer)
      }
      // Where the loop was pulled off the fastest line, and why.
      route.via_points.forEach((via) => {
        L.circleMarker([via.lat, via.lng], {
          radius: 5,
          color,
          weight: 2,
          fillColor: theme === 'dark' ? '#101021' : '#ffffff',
          fillOpacity: 1,
          dashArray: '2 2',
        })
          .bindTooltip(
            `Vehicle ${route.vehicle} · extra sweep — risk ${via.score.toFixed(2)}`,
            { direction: 'top' },
          )
          .addTo(layer)
      })
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

    // Re-fit only for a new plan. Toggling the comparison layer or the theme
    // redraws, and snapping the viewport back would throw away wherever the
    // controller had panned to.
    if (fittedRef.current !== plan) {
      fittedRef.current = plan
      const points = plan.routes.flatMap((r) => r.coordinates)
      if (points.length) {
        const bounds = L.latLngBounds(points)
        if (bounds.isValid()) map.fitBounds(bounds, { padding: [40, 40] })
      } else {
        map.setView([plan.unit.base_lat, plan.unit.base_lng], 11)
      }
    }
  }, [plan, theme, showFastest])

  if (!unit) return <LoadingGraphic label="Loading unit…" />

  const coverage = plan?.coverage
  const comparison = plan?.comparison
  const palette = VEHICLE_COLORS[theme]
  // What the fastest route already covers per kilometre — the number the extra
  // kilometres have to beat for the detours to be worth running.
  const fastestRiskPerKm = comparison?.fastest.distance_km
    ? (comparison.fastest.risk_seen / comparison.fastest.distance_km).toFixed(3)
    : '—'

  return (
    <>
      <h1>Patrol planning</h1>
      <p className="muted" style={{ margin: '2px 0 16px' }}>
        {unit.name} · allocates vehicles across the highest-risk areas within {unit.radius_km} km of
        base, for the shift you're planning — and puts each loop next to the plain fastest route
        through the same stops, so the cost of sweeping the extra areas is visible.
      </p>

      <div className="card filters">
        <div className="field" style={{ gap: 7 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Vehicles on shift
          </span>
          <div style={{ display: 'flex', gap: 6 }}>
            {[1, 2, 3, 4, 5, 6].map((n) => (
              <button
                key={n}
                type="button"
                className={`btn btn-chip${vehicles === n ? ' btn-primary' : ''}`}
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

        <div className="field" style={{ gap: 7 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Comparison
          </span>
          <button
            type="button"
            className={`btn${showFastest ? ' btn-primary' : ''}`}
            aria-pressed={showFastest}
            onClick={() => setShowFastest((v) => !v)}
          >
            {showFastest ? 'Fastest route shown' : 'Fastest route hidden'}
          </button>
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

      {comparison ? (
        <div className="banner banner-info" style={{ marginBottom: 16 }} role="status">
          {comparison.detoured_vehicles ? (
            <>
              <strong>
                {comparison.detoured_vehicles} of {plan.routes.length} loop
                {plan.routes.length === 1 ? '' : 's'} detour off the fastest route
              </strong>
              <p style={{ margin: '3px 0 0' }}>
                Sweeping {comparison.extra_cells} more area
                {comparison.extra_cells === 1 ? '' : 's'} costs {comparison.extra_km} km and about{' '}
                {Math.round(comparison.extra_min)} more minutes across the fleet — {' '}
                {comparison.risk_per_extra_km} risk covered per extra kilometre, against{' '}
                {fastestRiskPerKm} on the fastest route itself.
              </p>
            </>
          ) : (
            <>
              <strong>No detour was worth taking</strong>
              <p style={{ margin: '3px 0 0' }}>
                Every elevated-risk area in range is already a stop, or already on the fastest route
                between the stops. The patrol loops and the fastest loops are the same here.
              </p>
            </>
          )}
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
          ref={setContainer}
          className="map-canvas map-md"
          style={{ opacity: loading ? 0.55 : 1 }}
        />
        <div className="map-footer">
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
          {showFastest && comparison?.detoured_vehicles ? (
            <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <span
                style={{
                  width: 22,
                  height: 0,
                  borderTop: `3px dashed ${FASTEST_COLOR[theme]}`,
                }}
              />
              <span className="muted">Fastest route</span>
            </span>
          ) : null}
          <span className="spacer" />
          <span className="tiny map-note">
            Black marker is the unit's base — every loop starts and ends there. Hollow markers are
            the extra areas each loop detours to sweep.
          </span>
        </div>
      </div>

      {plan?.routes?.length ? (
        <div className="card">
          <p className="muted table-note" id="patrol-table-note">
            Stops are ordered into a loop from base and back. Distances and times are real road
            distances from OpenStreetMap, not straight lines. &ldquo;Areas swept&rdquo; counts every
            risk area the loop actually drives through, not just the stops — that is the figure the
            detours move, and the only one that can compare two routes through the same stops.
          </p>
          <div className="table-wrap">
            <table className="data-table" aria-describedby="patrol-table-note">
              <thead>
                <tr>
                  <Th>Vehicle</Th>
                  <Th num>Stops</Th>
                  <Th num>Distance</Th>
                  <Th num>Driving time</Th>
                  <Th num>Areas swept</Th>
                  <Th num>vs fastest route</Th>
                  <Th num>Extra areas swept</Th>
                </tr>
              </thead>
              <tbody>
                {plan.routes.map((route, index) => {
                  const fastest = route.fastest
                  const extraKm = fastest ? route.distance_km - fastest.distance_km : null
                  const extraMin = fastest ? route.duration_min - fastest.duration_min : null
                  const extraCells = fastest ? route.path.cells - fastest.path.cells : null
                  return (
                    <tr key={route.vehicle}>
                      <Td>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                          <span
                            style={{
                              width: 14,
                              height: 4,
                              borderRadius: 2,
                              flex: 'none',
                              background: palette[index % palette.length],
                            }}
                          />
                          Vehicle {route.vehicle}
                        </span>
                      </Td>
                      <Td num>{route.stops.length}</Td>
                      <Td num>{route.distance_km != null ? `${route.distance_km} km` : '—'}</Td>
                      <Td num>
                        {route.duration_min != null ? `${Math.round(route.duration_min)} min` : '—'}
                      </Td>
                      <Td num>{route.path ? route.path.cells : '—'}</Td>
                      <Td num>
                        {route.detoured
                          ? `+${extraKm.toFixed(1)} km · +${Math.round(extraMin)} min`
                          : 'same route'}
                      </Td>
                      <Td num>{route.detoured ? `+${extraCells}` : '—'}</Td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      <p className="tiny" style={{ marginTop: 14, maxWidth: 780 }}>
        Stops are the highest-risk areas for this hour, split geographically across the vehicles and
        ordered into a loop (nearest-neighbour with 2-opt). This is a good heuristic, not a proven
        optimum — a full vehicle-routing solver (VROOM or OR-Tools) is the upgrade once shift
        lengths and time windows matter. The fastest route is the same loop through the same stops
        with no risk detours: the plain shortest-time road route, as a standard navigation app
        would give it. Both come from OpenStreetMap routing, not from a commercial maps provider.
        Risk is based on historical claim density and is not adjusted for how many members travel
        through each area.
      </p>
    </>
  )
}

function Tile({ k, v, m }) {
  return (
    <div className="card tile">
      <div className="tiny tile-k">{k}</div>
      <div className="tile-v">{v}</div>
      {m ? <div className="muted tile-m">{m}</div> : null}
    </div>
  )
}

const Th = ({ children, num: isNum }) => <th className={isNum ? 'num' : undefined}>{children}</th>

const Td = ({ children, num: isNum }) => <td className={isNum ? 'num' : undefined}>{children}</td>
