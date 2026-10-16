import { useState } from 'react'
import { useAsync, href } from '../hooks'

interface Job {
  id: string
  video: string
  state: 'queued' | 'running' | 'done' | 'failed'
  progress: number
  message: string
  session_id: number | null
  error: string | null
  created: string
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
    if (start) form.append('start_time', start)
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
          <label className="field">
            <span>Recording started (UTC)</span>
            <input type="datetime-local" step="1" value={start} onChange={(e) => setStart(e.target.value)} />
            <small className="muted">Needed for sky positions and satellite names. Leave empty if unknown.</small>
          </label>
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
                  <div className="muted small">{j.error ?? j.message}</div>
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
