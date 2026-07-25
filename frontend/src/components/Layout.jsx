import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { ROLES, useSession } from '../session'
import { api } from '../api'

const NAV_BY_ROLE = {
  member: [
    { to: '/', label: 'Hot-spots', end: true },
    { to: '/alerts', label: 'Alerts' },
    { to: '/route', label: 'Plan a route' },
    { to: '/report', label: 'Report an incident' },
    { to: '/my-claims', label: 'My claims' },
    { to: '/profile', label: 'My profile' },
  ],
  employee: [
    { to: '/', label: 'Hot-spots', end: true },
    { to: '/review', label: 'Review queue', badge: 'pending' },
  ],
  cpu: [
    { to: '/', label: 'Hot-spots', end: true },
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

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

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

  return (
    <>
      <header className="appbar">
        <div className="appbar-inner">
          <NavLink to="/" className="brand" aria-label="Guardian Collective home">
            <span className="brand-badge" aria-hidden="true">
              GC
            </span>
            <span className="brand-text">
              <span className="brand-word">Guardian Collective</span>
              <span className="brand-bar" aria-hidden="true" />
            </span>
          </NavLink>

          <nav className="nav">
            {nav.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) => (isActive ? 'active' : undefined)}
              >
                {item.label}
                {item.badge && badgeValue[item.badge] ? (
                  <span className="badge">{badgeValue[item.badge]}</span>
                ) : null}
              </NavLink>
            ))}
          </nav>

          <div className="spacer" />

          {/* Demo identity switcher. Stands in for authentication — see session.jsx. */}
          <label className="tiny viewing-label" htmlFor="role-select" style={{ marginRight: -4 }}>
            Viewing as
          </label>
          <select
            id="role-select"
            value={role}
            onChange={(e) => setRole(e.target.value)}
            style={{ width: 'auto' }}
          >
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
              style={{ width: 'auto' }}
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
              style={{ width: 'auto' }}
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
              style={{ width: 'auto' }}
            >
              {directory.units.map((u) => (
                <option key={u.unit_id} value={u.unit_id}>
                  {u.name}
                </option>
              ))}
            </select>
          ) : null}

          <button
            type="button"
            className="btn"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          >
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </div>
      </header>

      <main className="wrap">
        <Outlet />
      </main>
    </>
  )
}
