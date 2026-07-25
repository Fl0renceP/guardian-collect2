import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { useSession } from '../session'
import { api } from '../api'

const MEMBER_NAV = [
  { to: '/', label: 'Hot-spots', end: true },
  { to: '/report', label: 'Report an incident' },
  { to: '/my-claims', label: 'My claims' },
]

const EMPLOYEE_NAV = [
  { to: '/', label: 'Hot-spots', end: true },
  { to: '/review', label: 'Review queue', badge: 'pending' },
]

function currentTheme() {
  const stamped = document.documentElement.getAttribute('data-theme')
  if (stamped) return stamped
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export default function Layout() {
  const { role, setRole, directory, memberId, setMemberId, employeeId, setEmployeeId } =
    useSession()
  const [theme, setTheme] = useState(currentTheme)
  const [pending, setPending] = useState(null)

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

  const nav = role === 'member' ? MEMBER_NAV : EMPLOYEE_NAV

  return (
    <>
      <header className="appbar">
        <div className="appbar-inner">
          <span className="brand">Guardian Collective</span>

          <nav className="nav">
            {nav.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) => (isActive ? 'active' : undefined)}
              >
                {item.label}
                {item.badge === 'pending' && pending ? (
                  <span className="badge">{pending}</span>
                ) : null}
              </NavLink>
            ))}
          </nav>

          <div className="spacer" />

          {/* Demo identity switcher. Stands in for authentication — see session.jsx. */}
          <label className="tiny" htmlFor="role-select" style={{ marginRight: -8 }}>
            Viewing as
          </label>
          <select
            id="role-select"
            value={role}
            onChange={(e) => setRole(e.target.value)}
            style={{ width: 'auto' }}
          >
            <option value="member">Insurance member</option>
            <option value="employee">Discovery employee</option>
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
          ) : (
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
          )}

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
