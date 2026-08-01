export default function LoadingGraphic({ label = 'Loading…', compact = false }) {
  return (
    <div className={`loading-graphic${compact ? ' compact' : ''}`} role="status" aria-live="polite">
      <div className="loading-graphic-mark" aria-hidden="true">
        <span className="loading-ring loading-ring-a" />
        <span className="loading-ring loading-ring-b" />
        <span className="loading-core" />
      </div>
      <span className="loading-label">{label}</span>
    </div>
  )
}
