/* Claim status. Status colour is never the only signal — the label is always
   present, so the state reads the same for a colourblind user. */

const LABELS = {
  pending: 'Awaiting review',
  approved: 'Approved',
  denied: 'Declined',
}

export default function StatusPill({ status }) {
  const key = (status || 'pending').toLowerCase()
  return (
    <span className={`pill pill-${key}`}>
      <span className="dot" aria-hidden="true" />
      {LABELS[key] || status}
    </span>
  )
}
