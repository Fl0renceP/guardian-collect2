import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, ApiError } from '../api'
import { useSession } from '../session'

const MAX_FILES = 6
const MAX_FILE_MB = 64

const EMPTY = {
  peril: '',
  suburb: '',
  item_type: 'Contents',
  vehicle_make: '',
  vehicle_model: '',
  vehicle_year: '',
  incident_date_time: '',
  description: '',
  claim_amount: '',
  camera_consent: false,
}

function Field({ label, hint, error, children, htmlFor }) {
  return (
    <div className={`field${error ? ' invalid' : ''}`}>
      <label htmlFor={htmlFor}>{label}</label>
      {children}
      {error ? (
        <span className="err" role="alert">
          {error}
        </span>
      ) : hint ? (
        <span className="hint">{hint}</span>
      ) : null}
    </div>
  )
}

export default function SubmitClaim() {
  const { member } = useSession()
  const navigate = useNavigate()

  const [form, setForm] = useState(EMPTY)
  const [files, setFiles] = useState([])
  const [perils, setPerils] = useState([])
  const [suburbs, setSuburbs] = useState([])
  const [errors, setErrors] = useState({})
  const [submitting, setSubmitting] = useState(false)
  const [banner, setBanner] = useState(null)
  const fileInput = useRef(null)

  const isVehicle = form.item_type.toLowerCase() === 'vehicle'

  useEffect(() => {
    api
      .filters()
      .then((data) => setPerils(data.perils.map((p) => p.value)))
      .catch(() => setPerils(['Theft', 'Hijack', 'Armed Robbery', 'Burglary']))
  }, [])

  // Suburb suggestions come from the geocoded set, so a claim filed against one
  // of them reaches the hot-spot map straight away.
  useEffect(() => {
    const q = form.suburb.trim()
    if (q.length < 2) {
      setSuburbs([])
      return undefined
    }
    let alive = true
    const timer = setTimeout(() => {
      api
        .suburbs(q)
        .then((data) => alive && setSuburbs(data.suburbs))
        .catch(() => {})
    }, 180)
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [form.suburb])

  const set = (key) => (event) => {
    const value =
      event.target.type === 'checkbox' ? event.target.checked : event.target.value
    setForm((prev) => ({ ...prev, [key]: value }))
    setErrors((prev) => (prev[key] ? { ...prev, [key]: undefined } : prev))
  }

  const totalMb = useMemo(
    () => files.reduce((sum, f) => sum + f.size, 0) / (1024 * 1024),
    [files],
  )

  function onPickFiles(event) {
    const picked = Array.from(event.target.files || [])
    const combined = [...files, ...picked].slice(0, MAX_FILES)
    setFiles(combined)
    setErrors((prev) => ({ ...prev, media: undefined }))
    // Reset so re-picking the same file still fires a change event.
    if (fileInput.current) fileInput.current.value = ''
  }

  function removeFile(index) {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  async function onSubmit(event) {
    event.preventDefault()
    if (!member) return

    if (totalMb > MAX_FILE_MB) {
      setErrors({ media: `Attachments total ${totalMb.toFixed(1)}MB — the limit is ${MAX_FILE_MB}MB.` })
      return
    }

    setSubmitting(true)
    setBanner(null)
    setErrors({})

    const body = new FormData()
    body.append('member_id', member.member_id)
    Object.entries(form).forEach(([key, value]) => {
      if (key === 'camera_consent') body.append(key, value ? 'true' : 'false')
      else if (value !== '') body.append(key, value)
    })
    files.forEach((file) => body.append('media', file))

    try {
      const { claim } = await api.submitClaim(body)
      navigate('/my-claims', { state: { justSubmitted: claim.Incident } })
    } catch (err) {
      if (err instanceof ApiError && err.fields) {
        setErrors(err.fields)
        setBanner({ kind: 'bad', text: 'Please fix the highlighted fields and try again.' })
      } else {
        setBanner({ kind: 'bad', text: err.message })
      }
      setSubmitting(false)
    }
  }

  if (!member) return <p className="muted">Loading your details…</p>

  return (
    <>
      <h1>Report an incident</h1>
      <p className="muted" style={{ margin: '2px 0 18px' }}>
        Reporting as <strong>{member.name}</strong> · policy {member.policy_number}. A Discovery
        assessor reviews every report before it is added to the claims record.
      </p>

      {banner ? (
        <div className={`banner banner-${banner.kind}`} style={{ marginBottom: 16 }} role="alert">
          {banner.text}
        </div>
      ) : null}

      <form onSubmit={onSubmit} noValidate>
        <section className="card panel" style={{ marginBottom: 16 }}>
          <h2>What happened</h2>
          <div className="grid-2" style={{ marginTop: 14 }}>
            <Field label="Type of crime" error={errors.peril} htmlFor="peril">
              <select id="peril" value={form.peril} onChange={set('peril')}>
                <option value="">Select…</option>
                {perils.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="What was affected" error={errors.item_type} htmlFor="item_type">
              <select id="item_type" value={form.item_type} onChange={set('item_type')}>
                <option value="Contents">Home contents</option>
                <option value="Vehicle">Vehicle</option>
              </select>
            </Field>

            <Field
              label="Suburb"
              hint="Start typing to see suburbs we can map."
              error={errors.suburb}
              htmlFor="suburb"
            >
              <input
                id="suburb"
                type="text"
                list="suburb-options"
                autoComplete="off"
                placeholder="e.g. BRYANSTON"
                value={form.suburb}
                onChange={set('suburb')}
              />
              <datalist id="suburb-options">
                {suburbs.map((s) => (
                  <option key={s} value={s} />
                ))}
              </datalist>
            </Field>

            <Field
              label="When did it happen"
              error={errors.incident_date_time}
              htmlFor="incident_date_time"
            >
              <input
                id="incident_date_time"
                type="datetime-local"
                value={form.incident_date_time}
                onChange={set('incident_date_time')}
              />
            </Field>
          </div>

          {isVehicle ? (
            <div className="grid-3" style={{ marginTop: 14 }}>
              <Field label="Vehicle make" htmlFor="vehicle_make">
                <input
                  id="vehicle_make"
                  type="text"
                  placeholder="e.g. Toyota"
                  value={form.vehicle_make}
                  onChange={set('vehicle_make')}
                />
              </Field>
              <Field label="Vehicle model" htmlFor="vehicle_model">
                <input
                  id="vehicle_model"
                  type="text"
                  placeholder="e.g. Hilux"
                  value={form.vehicle_model}
                  onChange={set('vehicle_model')}
                />
              </Field>
              <Field label="Year" htmlFor="vehicle_year">
                <input
                  id="vehicle_year"
                  type="number"
                  min="1950"
                  max="2100"
                  placeholder="e.g. 2021"
                  value={form.vehicle_year}
                  onChange={set('vehicle_year')}
                />
              </Field>
            </div>
          ) : null}

          <div style={{ marginTop: 14 }}>
            <Field
              label="Describe what took place"
              hint="Include times, how many people were involved, and anything you noticed."
              error={errors.description}
              htmlFor="description"
            >
              <textarea
                id="description"
                value={form.description}
                onChange={set('description')}
                placeholder="Around 19:45 two men blocked the driveway as I arrived home…"
              />
            </Field>
          </div>

          <div style={{ marginTop: 14, maxWidth: 320 }}>
            <Field
              label="Estimated value of the loss (ZAR)"
              error={errors.claim_amount}
              htmlFor="claim_amount"
            >
              <input
                id="claim_amount"
                type="number"
                min="0"
                step="0.01"
                placeholder="0.00"
                value={form.claim_amount}
                onChange={set('claim_amount')}
              />
            </Field>
          </div>
        </section>

        <section className="card panel" style={{ marginBottom: 16 }}>
          <h2>Photos or video</h2>
          <p className="muted" style={{ margin: '4px 0 12px' }}>
            Up to {MAX_FILES} files, {MAX_FILE_MB}MB total. Images and video only.
          </p>

          <input
            ref={fileInput}
            id="media"
            type="file"
            multiple
            accept="image/*,video/*"
            onChange={onPickFiles}
            style={{ fontSize: 13.5 }}
          />
          {errors.media ? (
            <p className="err" role="alert" style={{ marginTop: 8, color: 'var(--critical)', fontSize: 12.5 }}>
              {errors.media}
            </p>
          ) : null}

          {files.length ? (
            <ul style={{ listStyle: 'none', padding: 0, margin: '14px 0 0' }}>
              {files.map((file, index) => (
                <li
                  key={`${file.name}-${index}`}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    padding: '7px 0',
                    borderTop: '1px solid var(--hairline)',
                  }}
                >
                  <span style={{ flex: 1, fontSize: 13.5, wordBreak: 'break-all' }}>{file.name}</span>
                  <span className="tiny">{(file.size / (1024 * 1024)).toFixed(1)}MB</span>
                  <button type="button" className="btn" onClick={() => removeFile(index)}>
                    Remove
                  </button>
                </li>
              ))}
              <li className="tiny" style={{ paddingTop: 8 }}>
                {files.length} of {MAX_FILES} files · {totalMb.toFixed(1)}MB total
              </li>
            </ul>
          ) : null}
        </section>

        <section className="card panel" style={{ marginBottom: 20 }}>
          <h2>Door camera footage</h2>
          {/* Opt-in, never pre-ticked, and scoped to this one incident. The
              backend timestamps the consent alongside the claim. */}
          <label
            style={{ display: 'flex', gap: 11, alignItems: 'flex-start', marginTop: 12, cursor: 'pointer' }}
          >
            <input
              type="checkbox"
              checked={form.camera_consent}
              onChange={set('camera_consent')}
              style={{ width: 18, height: 18, marginTop: 2, flex: 'none' }}
            />
            <span>
              <strong style={{ fontSize: 14 }}>
                Allow Discovery to review my door camera footage for this incident
              </strong>
              <span className="muted" style={{ display: 'block', marginTop: 3 }}>
                Footage is used only to help confirm this claim. Permission covers this incident
                only, is recorded with the date you gave it, and you can withdraw it by contacting
                Discovery. Leaving this unticked will not affect your claim.
              </span>
            </span>
          </label>
        </section>

        {/* Wraps rather than shrinks: without this the note squeezes the submit
            button into two lines on a phone. */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <button type="submit" className="btn btn-primary" disabled={submitting}>
            {submitting ? 'Submitting…' : 'Submit report'}
          </button>
          <span className="tiny">
            Your report goes to a Discovery assessor. You'll see the outcome under “My claims”.
          </span>
        </div>
      </form>
    </>
  )
}
