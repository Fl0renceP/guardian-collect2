import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { api } from './api'

/* Who is using the app right now.
   THIS IS NOT AUTHENTICATION — there is no login in the hackathon build. The
   role and identity are picked in the UI and trusted by the API. Swapping this
   for a real auth provider is the single change needed to make the rest of the
   app honest about identity. */

const SessionContext = createContext(null)

const STORAGE_KEY = 'gc.session'

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
  const [directory, setDirectory] = useState({ members: [], employees: [] })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api
      .members()
      .then((data) => {
        setDirectory(data)
        // Default to the first identity of each kind so the app is usable
        // immediately rather than demanding a pick before anything renders.
        setMemberId((current) => current || data.members[0]?.member_id || null)
        setEmployeeId((current) => current || data.employees[0]?.employee_id || null)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ role, memberId, employeeId }))
  }, [role, memberId, employeeId])

  const value = useMemo(() => {
    const member = directory.members.find((m) => m.member_id === memberId) || null
    const employee = directory.employees.find((e) => e.employee_id === employeeId) || null
    return {
      role,
      setRole,
      member,
      employee,
      memberId,
      setMemberId,
      employeeId,
      setEmployeeId,
      directory,
      loading,
      error,
identity: role === 'member' ? member : employee,
    }
  }, [role, memberId, employeeId, directory, loading, error])

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}

export function useSession() {
  const ctx = useContext(SessionContext)
  if (!ctx) throw new Error('useSession must be used inside a SessionProvider')
  return ctx
}
