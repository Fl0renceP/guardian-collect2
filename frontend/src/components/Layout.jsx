import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { ROLES, useSession } from '../session'
import { api } from '../api'
import GradientText from './GradientText'
import { enablePushNotifications, pushSupported } from '../push'

const NAV_BY_ROLE = {
  member: [
    { to: '/', label: 'Hot-spots', end: true },
    { to: '/live-scan', label: 'Live scan' },
    { to: '/safety-score', label: 'Safety score' },
    { to: '/alerts', label: 'Alerts' },
    { to: '/route', label: 'Plan a route' },
    // Shortened from "Report an incident" — with seven member tabs the app bar
    // wraps otherwise, and the page heading carries the full wording.
    { to: '/report', label: 'Report' },
    { to: '/my-claims', label: 'My claims' },
    { to: '/profile', label: 'Profile' },
  ],
  employee: [
    { to: '/', label: 'Hot-spots', end: true },
    { to: '/live-scan', label: 'Live scan' },
    { to: '/review', label: 'Review queue', badge: 'pending' },
  ],
  cpu: [
    { to: '/', label: 'Hot-spots', end: true },
    { to: '/live-scan', label: 'Live scan' },
    { to: '/alerts', label: 'Alerts', badge: 'alerts' },
    { to: '/patrol', label: 'Patrol planning' },
  ],
}

function currentTheme() {
  const stamped = document.documentElement.getAttribute('data-theme')
  if (stamped) return stamped
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export default function Layout() {
  const {
    role,
    setRole,
    directory,
    memberId,
    setMemberId,
    employeeId,
    setEmployeeId,
    unitId,
    setUnitId,
  } = useSession()

  const [theme, setTheme] = useState(currentTheme)
  const [pending, setPending] = useState(null)
  const [alertCount, setAlertCount] = useState(null)
  const [pushStatus, setPushStatus] = useState('idle')
  // Only meaningful below the mobile breakpoint, where the nav and the identity
  // switcher collapse behind a menu button. Above it the panel is display:contents
  // and this flag has no effect.
  const [menuOpen, setMenuOpen] = useState(false)
  const { pathname } = useLocation()

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  // Landing on a new screen should leave the menu behind, not on top of it.
  useEffect(() => setMenuOpen(false), [pathname])

  // Keep the review-queue badge current so an employee sees new work arrive.
  useEffect(() => {
    if (role !== 'employee') return undefined
    let alive = true
    const load = () =>
      api
        .claimCounts()
        .then((counts) => alive && setPending(counts.pending ?? 0))
        .catch(() => {})
    load()
    const timer = setInterval(load, 15000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [role])

  // Same for a unit's alert count.
  useEffect(() => {
    if (role !== 'cpu' || !unitId) return undefined
    let alive = true
    const load = () =>
      api
        .alerts({ audience: 'cpu', unit_id: unitId, limit: 200 })
        .then((data) => alive && setAlertCount(data.summary.total))
        .catch(() => {})
    load()
    const timer = setInterval(load, 20000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [role, unitId])

  const nav = NAV_BY_ROLE[role] || NAV_BY_ROLE.member
  const badgeValue = { pending, alerts: alertCount }
  // Only member and cpu are ever in an alert's audience — see alerts_service.audience_for.
  const pushUserId = role === 'member' ? memberId : role === 'cpu' ? unitId : null

  const handleEnablePush = async () => {
    setPushStatus('requesting')
    try {
      const ok = await enablePushNotifications(pushUserId)
      setPushStatus(ok ? 'enabled' : 'denied')
    } catch {
      setPushStatus('error')
    }
  }

  return (
    <>
      <header className="appbar">
        <div className="appbar-inner">
          <NavLink to="/" className="brand" aria-label="Guardian Collective home">
            <span className="brand-badge" aria-hidden="true">
              GC
            </span>
            <span className="brand-text">
              <GradientText
                className="brand-word"
                colors={['#0fafb0', '#3286dd', '#6a57e1', '#3286dd', '#0fafb0']}
                animationSpeed={5}
                pauseOnHover={false}
                yoyo={true}
                showBorder={false}
              >
                Guardian Collective
              </GradientText>
              <span className="brand-bar" aria-hidden="true" />
            </span>
          </NavLink>

          <button
            type="button"
            className="btn menu-toggle"
            aria-expanded={menuOpen}
            aria-controls="app-menu"
            onClick={() => setMenuOpen((open) => !open)}
          >
            <span aria-hidden="true">{menuOpen ? '✕' : '☰'}</span>
            {menuOpen ? 'Close' : 'Menu'}
          </button>

          <div className="appbar-menu" id="app-menu" data-open={menuOpen ? 'true' : 'false'}>
            <div className="appbar-nav-row">
              <nav className="nav">
                {nav.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.end}
                    className={({ isActive }) => (isActive ? 'active' : undefined)}
                    // Tapping the tab you are already on doesn't change the route,
                    // so the effect above won't fire — close it here as well.
                    onClick={() => setMenuOpen(false)}
                  >
                    {item.label}
                    {item.badge && badgeValue[item.badge] ? (
                      <span className="badge">{badgeValue[item.badge]}</span>
                    ) : null}
                  </NavLink>
                ))}
              </nav>
            </div>

            <div className="appbar-controls-row">
              {/* Demo identity switcher. Stands in for authentication — see session.jsx. */}
              <label className="tiny viewing-label" htmlFor="role-select">
                Viewing as
              </label>
              <select id="role-select" value={role} onChange={(e) => setRole(e.target.value)}>
                {ROLES.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.label}
                  </option>
                ))}
              </select>

              {role === 'member' ? (
                <select
                  aria-label="Select member"
                  value={memberId || ''}
                  onChange={(e) => setMemberId(e.target.value)}
                >
                  {directory.members.map((m) => (
                    <option key={m.member_id} value={m.member_id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              ) : null}

              {role === 'employee' ? (
                <select
                  aria-label="Select employee"
                  value={employeeId || ''}
                  onChange={(e) => setEmployeeId(e.target.value)}
                >
                  {directory.employees.map((e) => (
                    <option key={e.employee_id} value={e.employee_id}>
                      {e.name}
                    </option>
                  ))}
                </select>
              ) : null}

              {role === 'cpu' ? (
                <select
                  aria-label="Select unit"
                  value={unitId || ''}
                  onChange={(e) => setUnitId(e.target.value)}
                >
                  {directory.units.map((u) => (
                    <option key={u.unit_id} value={u.unit_id}>
                      {u.name}
                    </option>
                  ))}
                </select>
              ) : null}

              {pushUserId && pushSupported() ? (
                <button
                  type="button"
                  className="btn"
                  disabled={pushStatus === 'requesting' || pushStatus === 'enabled'}
                  onClick={handleEnablePush}
                >
                  {pushStatus === 'enabled'
                    ? 'Alerts enabled'
                    : pushStatus === 'requesting'
                      ? 'Enabling…'
                      : 'Enable alerts'}
                </button>
              ) : null}

              <button
                type="button"
                className="btn"
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              >
                {theme === 'dark' ? 'Light' : 'Dark'}
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="wrap">
        <Outlet />
      </main>
    </>
  )
}
