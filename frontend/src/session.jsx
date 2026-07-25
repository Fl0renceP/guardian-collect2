import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { api } from './api'

/* Who is using the app right now.
   THIS IS NOT AUTHENTICATION — there is no login in the hackathon build. The
   role and identity are picked in the UI and trusted by the API. Swapping this
   for a real auth provider is the single change needed to make the rest of the
   app honest about identity.

   Three stakeholders, per PROJECT_CONTEXT §2:
     member   — Discovery Insure member (consumer)
     employee — Discovery employee (internal)
     cpu      — Crime Prevention Unit (armed response / SAPS) */

const SessionContext = createContext(null)

const STORAGE_KEY = 'gc.session'

export const ROLES = [
  { id: 'member', label: 'Insurance member' },
  { id: 'employee', label: 'Discovery employee' },
  { id: 'cpu', label: 'Crime Prevention Unit' },
]

function readStored() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}
  } catch {
    return {}
  }
}

export function SessionProvider({ children }) {
  const stored = readStored()
  const [role, setRole] = useState(stored.role || 'member')
  const [memberId, setMemberId] = useState(stored.memberId || null)
  const [employeeId, setEmployeeId] = useState(stored.employeeId || null)
  const [unitId, setUnitId] = useState(stored.unitId || null)
  const [directory, setDirectory] = useState({ members: [], employees: [], units: [] })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const [reloadKey, setReloadKey] = useState(0)
  const refreshDirectory = () => setReloadKey((k) => k + 1)

  useEffect(() => {
    Promise.all([api.members(), api.units()])
      .then(([people, units]) => {
        setDirectory({ ...people, units: units.units })
        // Default to the first identity of each kind so the app is usable
        // immediately rather than demanding a pick before anything renders.
        setMemberId((current) => current || people.members[0]?.member_id || null)
        setEmployeeId((current) => current || people.employees[0]?.employee_id || null)
        setUnitId((current) => current || units.units[0]?.unit_id || null)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [reloadKey])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ role, memberId, employeeId, unitId }))
  }, [role, memberId, employeeId, unitId])

  const value = useMemo(() => {
    const member = directory.members.find((m) => m.member_id === memberId) || null
    const employee = directory.employees.find((e) => e.employee_id === employeeId) || null
    const unit = directory.units.find((u) => u.unit_id === unitId) || null
    return {
      role,
      setRole,
      member,
      employee,
      unit,
      memberId,
      setMemberId,
      employeeId,
      setEmployeeId,
      unitId,
      setUnitId,
      directory,
      loading,
      error,
      refreshDirectory,
      identity: role === 'member' ? member : role === 'employee' ? employee : unit,
    }
  }, [role, memberId, employeeId, unitId, directory, loading, error])

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}

export function useSession() {
  const ctx = useContext(SessionContext)
  if (!ctx) throw new Error('useSession must be used inside a SessionProvider')
  return ctx
}
