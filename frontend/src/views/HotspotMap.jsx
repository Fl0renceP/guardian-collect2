import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import 'leaflet.heat'
import { api, money, num } from '../api'

/* Crime hot-spots.

   Leaflet is driven imperatively through refs rather than via react-leaflet:
   the heat layer is a canvas plugin that React shouldn't be diffing, and this
   keeps the render path identical to the plain-JS version it replaced.

   Colour: one sequential blue ramp for magnitude (never a rainbow), stepped
   separately for the dark surface. Identity is carried by labels, not hue. */

const SA_CENTER = [-29.0, 25.0]

const THEMES = {
  light: {
    url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    gradient: { 0.15: '#d3f0ec', 0.4: '#8fd3dd', 0.62: '#5b8ede', 0.82: '#7a3fe0', 1.0: '#a3126b' },
    stroke: '#ffffff',
    mark: '#7a3fe0',
  },
  dark: {
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    gradient: { 0.15: '#0e3147', 0.4: '#1a5c8e', 0.62: '#3d63cf', 0.82: '#b25ae0', 1.0: '#f6b8dc' },
    stroke: '#1b1b32',
    mark: '#b25ae0',
  },
}

// Claim counts are heavily right-skewed, so a linear ramp would render almost
// everything invisible. Square root compresses the top end; the legend states
// the scale and labels real counts, so nothing is hidden by the choice.
const intensity = (count, max) => (max > 0 ? Math.sqrt(count) / Math.sqrt(max) : 0)

function useTheme() {
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

function buildTooltip(spot) {
  // Suburb and peril names originate in the claims data — untrusted. Built as
  // DOM nodes with textContent, never innerHTML.
  const el = document.createElement('div')
  el.className = 'tip'

  const head = document.createElement('div')
  head.className = 't-sub'
  head.textContent = spot.suburb
  el.appendChild(head)

  const val = document.createElement('div')
  val.className = 't-val'
  val.textContent = num.format(spot.count)
  const unit = document.createElement('small')
  unit.textContent = spot.count === 1 ? 'claim' : 'claims'
  val.appendChild(unit)
  el.appendChild(val)

  const amt = document.createElement('div')
  amt.className = 't-amt'
  amt.textContent = `${money.format(spot.total_amount)} total value`
  el.appendChild(amt)

  if (spot.approximate) {
    const approx = document.createElement('div')
    approx.className = 't-approx'
    approx.textContent = 'Approximate location (nearest known place)'
    el.appendChild(approx)
  }

  const perils = Object.keys(spot.perils).slice(0, 4)
  if (perils.length) {
    const rows = document.createElement('div')
    rows.className = 't-rows'
    perils.forEach((name) => {
      const row = document.createElement('div')
      row.className = 't-row'
      const label = document.createElement('span')
      label.className = 't-name'
      label.textContent = name
      row.appendChild(label)
      const n = document.createElement('span')
      n.className = 't-n'
      n.textContent = num.format(spot.perils[name])
      row.appendChild(n)
      rows.appendChild(row)
    })
    el.appendChild(rows)
  }
  return el
}

export default function HotspotMap() {
  const theme = useTheme()
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const tileRef = useRef(null)
  const heatRef = useRef(null)
  const markersRef = useRef(null)
  const rendererRef = useRef(null)

  const [options, setOptions] = useState(null)
  const [data, setData] = useState(null)
  const [peril, setPeril] = useState('')
  const [itemType, setItemType] = useState('')
  const [range, setRange] = useState({ from: '', to: '' })
  const [rangeId, setRangeId] = useState('all')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showTable, setShowTable] = useState(false)

  /* ---------- map lifecycle ---------- */
  useEffect(() => {
    if (mapRef.current) return undefined
    const map = L.map(containerRef.current, {
      center: SA_CENTER,
      zoom: 5,
      minZoom: 4,
      preferCanvas: true,
      worldCopyJump: false,
    })
    mapRef.current = map
    // Dedicated pane above the heat canvas so hit targets stay hoverable.
    map.createPane('marks')
    map.getPane('marks').style.zIndex = 450
    rendererRef.current = L.canvas({ padding: 0.4, pane: 'marks' })
    markersRef.current = L.layerGroup().addTo(map)
    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (tileRef.current) map.removeLayer(tileRef.current)
    tileRef.current = L.tileLayer(THEMES[theme].url, {
      maxZoom: 18,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    }).addTo(map)
    tileRef.current.bringToBack()
  }, [theme])

  /* ---------- data ---------- */
  useEffect(() => {
    api
      .filters()
      .then(setOptions)
      .catch((err) => setError(err.message))
  }, [])

  const load = useCallback(() => {
    setLoading(true)
    api
      .hotspots({
        peril,
        item_type: itemType,
        date_from: range.from,
        date_to: range.to,
      })
      .then((result) => {
        setData(result)
        setError(null)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [peril, itemType, range])

  useEffect(load, [load])

  /* ---------- draw ---------- */
  useEffect(() => {
    const map = mapRef.current
    if (!map || !data) return

    const max = data.max_count || 1
    const palette = THEMES[theme]

    if (heatRef.current) map.removeLayer(heatRef.current)
    heatRef.current = L.heatLayer(
      data.hotspots.map((s) => [s.lat, s.lng, intensity(s.count, max)]),
      { radius: 22, blur: 18, max: 1.0, minOpacity: 0.32, maxZoom: 11, gradient: palette.gradient },
    ).addTo(map)

    // Invisible hit targets — the heat layer carries the visual, these carry
    // interaction, so thousands of rings never compete with it for ink.
    markersRef.current.clearLayers()
    data.hotspots.forEach((spot) => {
      const r = 5 + 11 * intensity(spot.count, max)
      const marker = L.circleMarker([spot.lat, spot.lng], {
        renderer: rendererRef.current,
        radius: Math.max(r, 12), // 24px minimum target, never a pinpoint
        bubblingMouseEvents: false,
        color: palette.stroke,
        weight: 1.5,
        opacity: 0,
        fillColor: palette.mark,
        fillOpacity: 0,
      })
      marker.bindTooltip(buildTooltip(spot), {
        className: 'gc-tip',
        direction: 'top',
        offset: [0, -6],
      })
      marker.on('mouseover', () => marker.setStyle({ fillOpacity: 0.3, opacity: 0.9 }))
      marker.on('mouseout', () => marker.setStyle({ fillOpacity: 0, opacity: 0 }))
      markersRef.current.addLayer(marker)
    })
  }, [data, theme])

  /* ---------- filter helpers ---------- */
  const presets = useMemo(() => {
    if (!options?.date_max) return []
    const shift = (days) => {
      const d = new Date(`${options.date_max}T00:00:00Z`)
      d.setUTCDate(d.getUTCDate() - days)
      return d.toISOString().slice(0, 10)
    }
    // Anchored to the newest incident in the data, not to today — the dataset
    // ends before the current date, so "last 30 days" from today is empty.
    return [
      { id: 'all', label: 'All time', from: '', to: '' },
      { id: '12m', label: 'Last 12 months', from: shift(365), to: options.date_max },
      { id: '90d', label: 'Last 90 days', from: shift(90), to: options.date_max },
      { id: '30d', label: 'Last 30 days', from: shift(30), to: options.date_max },
    ]
  }, [options])

  useEffect(() => {
    const selected = presets.find((p) => p.id === rangeId) || presets[0]
    if (!selected) return
    setRange({ from: selected.from, to: selected.to })
  }, [presets, rangeId])

  const stats = useMemo(() => {
    if (!data) return null
    const total = data.hotspots.reduce((a, s) => a + s.total_amount, 0)
    return { total, top: data.hotspots[0] }
  }, [data])

  return (
    <>
      <h1>Crime hot-spots</h1>
      <p className="muted" style={{ margin: '2px 0 16px' }}>
        {options
          ? `${num.format(options.total_claims)} Discovery Insure claims, ${options.date_min} to ${options.date_max} — where incidents cluster, and when.`
          : 'Loading claims…'}
      </p>

      {error ? (
        <div className="banner banner-bad" style={{ marginBottom: 16 }} role="alert">
          {error}
        </div>
      ) : null}

      {/* Filters: one row, above everything they scope. Date range first. */}
      <div className="card filters">
        <div className="field" style={{ gap: 7 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Date range
          </span>
          <select
            aria-label="Date range"
            value={rangeId}
            onChange={(e) => setRangeId(e.target.value)}
            style={{ minWidth: 200 }}
          >
            {presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </div>

        <div className="field" style={{ gap: 7 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Crime category
          </span>
          <select
            aria-label="Crime category"
            value={peril}
            onChange={(e) => setPeril(e.target.value)}
            style={{ minWidth: 260 }}
          >
            <option value="">All categories</option>
            {(options?.perils || []).map((p) => (
              <option key={p.value} value={p.value}>
                {p.value} ({num.format(p.count)})
              </option>
            ))}
          </select>
        </div>

        <div className="field" style={{ gap: 7 }}>
          <span className="tiny" style={{ fontWeight: 650, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Item type
          </span>
          <select
            aria-label="Item type"
            value={itemType}
            onChange={(e) => setItemType(e.target.value)}
            style={{ minWidth: 220 }}
          >
            <option value="">All item types</option>
            {(options?.item_types || []).map((t) => (
              <option key={t.value} value={t.value}>
                {t.value} ({num.format(t.count)})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Stat tiles — a headline number is not a one-bar chart. */}
      <div className="grid-4-tiles" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
        <Tile k="Incidents" v={data ? num.format(data.matched_claims) : '—'} m={data ? `${num.format(data.placed_claims)} mapped` : ''} stale={loading} />
        <Tile k="Suburbs affected" v={data ? num.format(data.hotspots.length) : '—'} m="with at least one mapped claim" stale={loading} />
        <Tile
          k="Busiest suburb"
          v={stats?.top ? stats.top.suburb : '—'}
          m={stats?.top ? `${num.format(stats.top.count)} claims · ${stats.top.top_peril}` : 'No data for this filter'}
          stale={loading}
        />
        <Tile k="Total claim value" v={stats ? money.format(stats.total) : '—'} m="mapped claims only" stale={loading} />
      </div>

      <div className="card" style={{ overflow: 'hidden' }}>
        <div ref={containerRef} className="map-canvas" style={{ opacity: loading ? 0.55 : 1 }} />
        <div className="map-footer">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3, minWidth: 210 }}>
            <div
              style={{
                height: 9,
                borderRadius: 3,
                background:
                  'linear-gradient(to right, var(--seq-100), var(--seq-250), var(--seq-400), var(--seq-550), var(--seq-700))',
              }}
            />
            <div className="tiny" style={{ display: 'flex', justifyContent: 'space-between', fontVariantNumeric: 'tabular-nums' }}>
              {[0, 0.25, 0.5, 0.75, 1].map((f) => (
                <span key={f}>{num.format(Math.round((data?.max_count || 0) * f * f))}</span>
              ))}
            </div>
          </div>
          <span className="muted">Claims per suburb · square-root scale</span>
          <span className="spacer" />
          <span className="tiny map-note" style={{ textAlign: 'right' }}>
            {data ? <Coverage data={data} /> : null}
          </span>
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <button type="button" className="btn" aria-expanded={showTable} onClick={() => setShowTable((v) => !v)}>
          {showTable ? 'Hide data table' : 'Show data table'}
        </button>
        {showTable && data ? (
          <div className="card" style={{ marginTop: 10 }}>
            <p className="muted table-note" id="hotspot-table-note">
              Top 100 suburbs for the current filters.
            </p>
            <div className="table-wrap">
              <table className="data-table" aria-describedby="hotspot-table-note">
                <thead>
                  <tr>
                    <Th>Suburb</Th>
                    <Th num>Claims</Th>
                    <Th num>Total claim value</Th>
                    <Th>Leading peril</Th>
                  </tr>
                </thead>
                <tbody>
                  {data.hotspots.slice(0, 100).map((s) => (
                    <tr key={s.suburb}>
                      <Td>{s.suburb}</Td>
                      <Td num>{num.format(s.count)}</Td>
                      <Td num>{money.format(s.total_amount)}</Td>
                      <Td>{s.top_peril}</Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}
      </div>
    </>
  )
}

function Tile({ k, v, m, stale }) {
  const valueClass = k === 'Busiest suburb' ? 'tile-v tile-v-suburb' : 'tile-v'
  return (
    <div className="card tile" style={{ opacity: stale ? 0.55 : 1, transition: 'opacity 120ms ease' }}>
      <div className="tiny tile-k">{k}</div>
      <div className={valueClass}>{v}</div>
      {m ? <div className="muted tile-m">{m}</div> : null}
    </div>
  )
}

function Coverage({ data }) {
  const b = data.unplaced_breakdown
  const parts = []
  if (b.unknown_suburb) parts.push(`${num.format(b.unknown_suburb)} with no suburb recorded`)
  if (b.not_geocoded)
    parts.push(`${num.format(b.not_geocoded)} in ${num.format(b.not_geocoded_suburbs)} suburbs that couldn't be located`)
  let text = parts.length ? `Not on map: ${parts.join('; ')}` : 'All matching claims are on the map'
  if (data.approximate_claims) text += ` · ${num.format(data.approximate_claims)} pinned approximately`
  return <>{text}</>
}

const Th = ({ children, num: isNum }) => <th className={isNum ? 'num' : undefined}>{children}</th>

const Td = ({ children, num: isNum }) => <td className={isNum ? 'num' : undefined}>{children}</td>
