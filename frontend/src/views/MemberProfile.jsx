import { useCallback, useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import { api } from '../api'
import LoadingGraphic from '../components/LoadingGraphic'
import { useSession } from '../session'

/* Optional home location.

   Three things this screen has to get right, because it's collecting a real
   person's home address:

     1. It is genuinely optional. Nothing on the rest of the app breaks without
        it — the member simply sees national alerts instead of nearby ones.
     2. Sharing is a separate, explicit switch. Storing coordinates is not
        permission to use them; the backend checks `share_location`, never the
        mere presence of a latitude.
     3. Turning it off deletes the coordinates rather than hiding them, and the
        screen says so before you do it. */

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

export default function MemberProfile() {
  const { member, refreshDirectory } = useSession()
  const theme = useThemeName()

  // A callback ref, not useRef: this view returns early until the member loads,
  // so the container isn't in the DOM on the first render. A mount effect would
  // run once against a null node and never again, leaving a blank map.
  const [container, setContainer] = useState(null)
  const mapRef = useRef(null)
  const tileRef = useRef(null)
  const pinRef = useRef(null)
  const ringRef = useRef(null)

  const [profile, setProfile] = useState(null)
  const [address, setAddress] = useState('')
  const [suburb, setSuburb] = useState('')
  const [point, setPoint] = useState(null)
  const [radius, setRadius] = useState(10)
  const [saving, setSaving] = useState(false)
  const [banner, setBanner] = useState(null)

  const load = useCallback(() => {
    if (!member) return
    api
      .user(member.member_id)
      .then(({ user }) => {
        const p = user.member_profile || {}
        setProfile(p)
        setAddress(p.home_address || '')
        setSuburb(p.home_suburb || '')
        setRadius(p.alert_radius_km || 10)
        setPoint(p.home_lat != null ? { lat: p.home_lat, lng: p.home_lng } : null)
      })
      .catch((err) => setBanner({ kind: 'bad', text: err.message }))
  }, [member])

  useEffect(load, [load])

  /* ---------- map ---------- */
  useEffect(() => {
    if (mapRef.current || !container) return undefined
    const map = L.map(container, { center: [-26.1, 28.05], zoom: 10, minZoom: 4 })
    mapRef.current = map
    map.on('click', (e) => setPoint({ lat: e.latlng.lat, lng: e.latlng.lng }))
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

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (pinRef.current) map.removeLayer(pinRef.current)
    if (ringRef.current) map.removeLayer(ringRef.current)
    if (!point) return

    const accent = theme === 'dark' ? '#5b9cf0' : '#2b73d4'
    ringRef.current = L.circle([point.lat, point.lng], {
      radius: radius * 1000,
      color: accent,
      weight: 1.5,
      fillColor: accent,
      fillOpacity: 0.08,
    }).addTo(map)
    pinRef.current = L.circleMarker([point.lat, point.lng], {
      radius: 8,
      color: '#fff',
      weight: 2,
      fillColor: accent,
      fillOpacity: 1,
    })
      .bindTooltip('Your home', { direction: 'top' })
      .addTo(map)
    map.fitBounds(ringRef.current.getBounds(), { padding: [30, 30] })
  }, [point, radius, theme])

  async function save(shareLocation) {
    setSaving(true)
    setBanner(null)
    try {
      const payload =
        shareLocation === false
          ? { share_location: false }
          : {
              address,
              suburb,
              lat: point?.lat,
              lng: point?.lng,
              share_location: true,
              alert_radius_km: radius,
            }
      const { user } = await api.updateLocation(member.member_id, payload)
      const p = user.member_profile || {}
      setProfile(p)
      setPoint(p.home_lat != null ? { lat: p.home_lat, lng: p.home_lng } : null)
      setAddress(p.home_address || '')
      setBanner({
        kind: 'good',
        text:
          shareLocation === false
            ? 'Location sharing turned off and your saved location deleted.'
            : 'Home location saved. Alerts and route planning will use it.',
      })
      refreshDirectory?.()
    } catch (err) {
      setBanner({ kind: 'bad', text: err.message })
    } finally {
      setSaving(false)
    }
  }

  if (!member) return <LoadingGraphic label="Loading profile…" />

  const sharing = !!profile?.share_location

  return (
    <>
      <h1>My profile</h1>
      <p className="muted" style={{ margin: '2px 0 18px' }}>
        {member.name} · policy {member.policy_number}
      </p>

      {banner ? (
        <div className={`banner banner-${banner.kind}`} role="status" style={{ marginBottom: 16 }}>
          {banner.text}
        </div>
      ) : null}

      <div
        className={`banner ${sharing ? 'banner-good' : 'banner-info'}`}
        style={{ marginBottom: 16 }}
      >
        <strong>{sharing ? 'Location sharing is on' : 'Location sharing is off'}</strong>
        <p style={{ margin: '3px 0 0' }}>
          {sharing
            ? `You'll see alerts within ${profile.alert_radius_km} km of home, and route planning starts from there.`
            : 'Everything still works — you just see alerts from across the country instead of near you. Adding a location is optional.'}
        </p>
      </div>

      <section className="card panel" style={{ marginBottom: 16 }}>
        <h2>Home location</h2>
        <p className="muted" style={{ margin: '4px 0 14px' }}>
          Click the map to place your home, or move the pin. We store the point you place — not a
          continuous track of where you are.
        </p>

        <div className="grid-2" style={{ marginBottom: 14 }}>
          <div className="field">
            <label htmlFor="address">Street address (optional)</label>
            <input
              id="address"
              type="text"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              placeholder="e.g. 14 Coleraine Drive"
            />
          </div>
          <div className="field">
            <label htmlFor="suburb">Suburb</label>
            <input
              id="suburb"
              type="text"
              value={suburb}
              onChange={(e) => setSuburb(e.target.value)}
              placeholder="e.g. BRYANSTON"
            />
          </div>
        </div>

        <div className="field" style={{ marginBottom: 14, maxWidth: 420 }}>
          <label htmlFor="radius">Alert me about incidents within</label>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <input
              id="radius"
              type="range"
              min="1"
              max="50"
              value={radius}
              onChange={(e) => setRadius(Number(e.target.value))}
              style={{ flex: 1 }}
            />
            <strong style={{ fontVariantNumeric: 'tabular-nums', minWidth: 52 }}>{radius} km</strong>
          </div>
        </div>

        <div
          ref={setContainer}
          className="map-canvas map-sm"
          style={{
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border)',
            cursor: 'crosshair',
          }}
        />
        <p className="tiny" style={{ marginTop: 8 }}>
          {point
            ? `Pin at ${point.lat.toFixed(4)}, ${point.lng.toFixed(4)}`
            : 'No location set — click the map to place one.'}
        </p>
      </section>

      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
        <button
          type="button"
          className="btn btn-primary"
          disabled={saving || !point}
          onClick={() => save(true)}
        >
          {saving ? 'Saving…' : sharing ? 'Update my location' : 'Save and turn sharing on'}
        </button>
        {sharing ? (
          <button type="button" className="btn btn-danger" disabled={saving} onClick={() => save(false)}>
            Turn off and delete my location
          </button>
        ) : null}
      </div>

      <p className="tiny" style={{ marginTop: 14, maxWidth: 720 }}>
        Turning sharing off deletes the coordinates and address we hold — it isn't just hidden. We
        never store a movement history, only the single point you place here.
      </p>
    </>
  )
}
