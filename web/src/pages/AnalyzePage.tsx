import { useState } from 'react'
import { useAsync, useUtcNow, href } from '../hooks'

interface Job {
  id: string
  video: string
  state: 'queued' | 'running' | 'done' | 'failed'
  progress: number
  message: string
  session_id: number | null
  error: string | null
  created: string
  started: string | null
}

const LOCAL_ZONE = Intl.DateTimeFormat().resolvedOptions().timeZone

/** A datetime-local value in this computer's time zone or in UTC, as UTC ISO 8601. */
function toUtc(value: string, zone: 'local' | 'utc'): string | null {
  if (!value) return null
  const d = new Date(zone === 'utc' ? `${value}Z` : value) // no zone: read as local time
  return Number.isNaN(d.getTime()) ? null : d.toISOString()
}

function timeLeft(job: Job, now: number): string | null {
  if (job.state !== 'running' || !job.started || job.progress < 0.01) return null
  const elapsed = (now - Date.parse(job.started)) / 1000
  const left = (elapsed * (1 - job.progress)) / job.progress
  if (left < 90) return 'less than 2 min left'
  const h = Math.floor(left / 3600), m = Math.round((left % 3600) / 60)
  return `about ${h ? `${h} h ` : ''}${m} min left`
}

const getJobs = async (): Promise<Job[]> => (await fetch('/api/jobs')).json()

/** Upload with progress (fetch cannot report upload progress, XMLHttpRequest can). */
function upload(form: FormData, onProgress: (f: number) => void): Promise<Job> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/analyze')
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
    xhr.onload = () =>
      xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(JSON.parse(xhr.responseText).detail ?? xhr.statusText))
    xhr.onerror = () => reject(new Error('Upload failed'))
    xhr.send(form)
  })
}

export default function AnalyzePage() {
  const jobs = useAsync(getJobs, [], 1500)
  const [file, setFile] = useState<File | null>(null)
  const [start, setStart] = useState('')
  const [zone, setZone] = useState<'local' | 'utc'>('local')
  const now = useUtcNow()
  const utcStart = toUtc(start, zone)
  const [lat, setLat] = useState('')
  const [lon, setLon] = useState('')
  const [elevation, setElevation] = useState('')
  const [sky, setSky] = useState(true)
  const [sending, setSending] = useState<number | null>(null)
  const [error, setError] = useState<string>()
  const [inputKey, setInputKey] = useState(0) // remounting the file input clears it

  const submit = async (ev: React.FormEvent) => {
    ev.preventDefault()
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    const utc = toUtc(start, zone)
    if (utc) form.append('start_time', utc)
    if (lat && lon) {
      form.append('lat', lat)
      form.append('lon', lon)
      if (elevation) form.append('elevation', elevation)
    }
    form.append('sky', String(sky))
    setError(undefined)
    setSending(0)
    try {
      await upload(form, setSending)
      setFile(null)
      setInputKey((k) => k + 1)
      jobs.reload()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSending(null)
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Analyse a video</h1>
          <div className="muted">Upload a night-sky recording; SkySeer analyses it in the background and adds it as a session.</div>
        </div>
      </div>
      <div className="grid cols-2">
        <form className="panel form" onSubmit={submit}>
          <label className="field">
            <span>Video file</span>
            <input key={inputKey} type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <div className="field-row">
            <label className="field">
              <span>Recording started</span>
              <input type="datetime-local" step="1" value={start} onChange={(e) => setStart(e.target.value)} />
            </label>
            <label className="field">
              <span>Time zone</span>
              <select value={zone} onChange={(e) => setZone(e.target.value as 'local' | 'utc')}>
                <option value="local">{LOCAL_ZONE} (this computer)</option>
                <option value="utc">UTC</option>
              </select>
            </label>
          </div>
          <small className="muted">
            {utcStart ? `= ${utcStart.slice(0, 19).replace('T', ' ')} UTC. ` : ''}
            Needed to name satellites, aircraft and meteor showers. If empty, SkySeer uses the
            time saved in the video file, but only when it was night there then.
          </small>
          <div className="field-row">
            <label className="field"><span>Latitude °N</span><input type="text" inputMode="decimal" placeholder="40.42" value={lat} onChange={(e) => setLat(e.target.value)} /></label>
            <label className="field"><span>Longitude °E</span><input type="text" inputMode="decimal" placeholder="-3.70" value={lon} onChange={(e) => setLon(e.target.value)} /></label>
            <label className="field"><span>Elevation m</span><input type="text" inputMode="decimal" placeholder="650" value={elevation} onChange={(e) => setElevation(e.target.value)} /></label>
          </div>
          <label className="check">
            <input type="checkbox" checked={sky} onChange={(e) => setSky(e.target.checked)} />
            Recognise the stars to find sky positions (needs a clear view of stars)
          </label>
          <div>
            <button className="btn primary" type="submit" disabled={!file || sending !== null}>
              {sending !== null ? `Uploading ${Math.round(100 * sending)}%` : 'Upload and analyse'}
            </button>
          </div>
          {error && <div className="error">{error}</div>}
        </form>
        <section className="panel">
          <h3>Analyses</h3>
          {!jobs.data?.length ? (
            <p className="muted">Nothing analysed since the server started.</p>
          ) : (
            <div className="jobs">
              {jobs.data.map((j) => (
                <div key={j.id} className="job">
                  <div className="event-row">
                    <strong>{j.video.replace(/^\d{8}_\d{6}_/, '')}</strong>
                    <span className={`chip ${j.state}`}>{j.state}</span>
                  </div>
                  <div className="bar-track"><div className="bar-fill" style={{ width: `${100 * j.progress}%`, background: j.state === 'failed' ? 'var(--bad)' : 'var(--accent)' }} /></div>
                  <div className="muted small">
                    {j.error ?? j.message}
                    {timeLeft(j, now.getTime()) && <> · {timeLeft(j, now.getTime())}</>}
                  </div>
                  {j.session_id && (
                    <div className="small">
                      <a href={href(`report/${j.session_id}`)}>Report</a> · <a href={href('events', { session: j.session_id })}>Events</a> · <a href={href('sky', { session: j.session_id })}>Sky map</a>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </>
  )
}
