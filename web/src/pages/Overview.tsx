import { useEffect, useRef, useState } from 'react'
import { api, duration, type Session, type SessionFiles, sessionTitle, utcDate, utcTime } from '../api'
import { EventCard, Failed, Loading, SimTag } from '../components/common'
import { EditIcon, TrashIcon } from '../components/Icons'
import { href, navigate, useAsync } from '../hooks'

const length = (s: Session) =>
  s.started && s.ended ? duration((Date.parse(s.ended) - Date.parse(s.started)) / 1000) : null

const size = (mb: number) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.max(mb, 0.1).toFixed(1)} MB`)

/** Asks before removing a session; can also delete its files (SkySeer's own only). */
function RemoveBox({ s, onDone, onCancel }: { s: Session; onDone: () => void; onCancel: () => void }) {
  const files = useAsync<SessionFiles>(() => api.sessionFiles(s.id), [s.id])
  const [withFiles, setWithFiles] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()
  const f = files.data
  const deletable = Boolean(f && (f.folder || f.upload))
  const remove = async () => {
    setBusy(true)
    try {
      await api.removeSession(s.id, withFiles && deletable)
      onDone()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }
  return (
    <div className="remove-box">
      <div>Remove <strong>{sessionTitle(s)}</strong> from SkySeer? It won't come back on its own.</div>
      {f && deletable && (
        <label className="check">
          <input type="checkbox" checked={withFiles} onChange={(e) => setWithFiles(e.target.checked)} />
          Also delete its files from this computer:{' '}
          {[f.folder && `clips and reports (${size(f.folder_mb)})`, f.upload && `the uploaded video (${size(f.upload_mb)})`]
            .filter(Boolean)
            .join(' and ')}
        </label>
      )}
      {f && !deletable && <div className="muted small">Its files are outside SkySeer's data folders, so they are kept.</div>}
      {error && <div className="error">{error}</div>}
      <div className="btn-group">
        <button className="btn small danger" disabled={busy || !f} onClick={remove}>
          {busy ? 'Removing…' : withFiles ? 'Remove and delete files' : 'Remove'}
        </button>
        <button className="btn small" disabled={busy} onClick={onCancel}>Cancel</button>
      </div>
    </div>
  )
}

/** A session as a card: pictures of its most striking events, its counts and actions. */
function SessionCard({
  s,
  selected,
  onSelect,
  onChanged,
}: {
  s: Session
  selected: boolean
  onSelect: () => void
  onChanged: (removed: boolean) => void
}) {
  const [renaming, setRenaming] = useState(false)
  const [draft, setDraft] = useState('')
  const [removing, setRemoving] = useState(false)
  const [error, setError] = useState<string>()
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (renaming) input.current?.select()
  }, [renaming])
  const pics = s.highlights ?? []
  const save = async () => {
    try {
      await api.renameSession(s.id, draft)
      setRenaming(false)
      setError(undefined)
      onChanged(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }
  const len = length(s)
  return (
    <div className={`session-card${selected ? ' selected' : ''}`}>
      <button className={`session-cover${pics.length === 1 ? ' single' : ''}`} onClick={onSelect}
        title={pics.length ? pics.map((p) => `${p.label}${p.name ? `: ${p.name}` : ''}`).join(' · ') : 'Show this session'}>
        {pics.length === 0 && <div className="empty-cover">No clips</div>}
        {pics.map((p, i) => (
          <img key={p.id} className={i === 0 ? 'main' : ''} src={p.thumbnail} alt={`${p.label} ${p.name ?? ''}`} loading="lazy" />
        ))}
        <span className="tags">
          {s.meta.simulated && <span className="sim-tag">SIM</span>}
          {s.kind === 'live' && <span className="chip">live</span>}
        </span>
      </button>
      <div className="session-body">
        <div className="session-title">
          {renaming ? (
            <>
              <input ref={input} value={draft} maxLength={80} aria-label="Session name"
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') save()
                  if (e.key === 'Escape') setRenaming(false)
                }} />
              <button className="btn small" onClick={save}>Save</button>
            </>
          ) : (
            <>
              <button className="link" onClick={onSelect} title={s.title ? `Folder: ${s.name}` : undefined}>{sessionTitle(s)}</button>
              <button className="icon-btn" title="Rename" aria-label="Rename"
                onClick={() => { setDraft(sessionTitle(s)); setRenaming(true) }}><EditIcon /></button>
              <button className="icon-btn danger" title="Remove" aria-label="Remove"
                onClick={() => setRemoving(true)}><TrashIcon /></button>
            </>
          )}
        </div>
        {error && <div className="error small">{error}</div>}
        <div className="muted small">
          {s.started ? `${utcDate(s.started)} ${utcTime(s.started)} UTC` : 'date unknown'}
          {len && ` · ${len}`}
        </div>
        <div className="session-counts mono">
          <span className="c-Satellite">{s.satellites} satellites</span>
          <span className="c-Meteor">{s.meteors} meteors</span>
          <span className="c-Plane">{s.planes} aircraft</span>
        </div>
      </div>
      {removing && (
        <RemoveBox s={s} onCancel={() => setRemoving(false)} onDone={() => { setRemoving(false); onChanged(true) }} />
      )}
    </div>
  )
}

/** One session at a time (by default the one added last), or every session together. */
export default function Overview({ query }: { query: URLSearchParams }) {
  const sessions = useAsync(api.sessions, [])
  const param = query.get('session')
  const all = [...(sessions.data ?? [])].sort((a, b) => b.id - a.id) // newest first
  const selected: Session | undefined =
    param === 'all' ? undefined : param ? all.find((s) => s.id === Number(param)) : all[0]
  const loaded = sessions.data !== undefined
  const recent = useAsync(
    () => (loaded ? api.events({ session_id: selected?.id, limit: 10 }) : Promise.resolve([])),
    [loaded, selected?.id],
  )
  if (sessions.error) return <Failed error={sessions.error} />
  if (!sessions.data) return <Loading what="sessions" />
  const titles = new Map(all.map((s) => [s.id, sessionTitle(s)]))
  const shown = selected ? [selected] : all
  const totals = shown.reduce(
    (t, s) => ({ sat: t.sat + s.satellites, met: t.met + s.meteors, pla: t.pla + s.planes }),
    { sat: 0, met: 0, pla: 0 },
  )
  const choose = (id: number | undefined) => navigate(href('', { session: id ?? 'all' }))
  const changed = (removedId?: number) => {
    if (removedId !== undefined && removedId === selected?.id) navigate(href(''))
    sessions.reload()
    recent.reload()
  }
  const selectedLength = selected ? length(selected) : null

  return (
    <>
      <div className="page-head">
        <div>
          <h1>
            Overview{selected && <span className="muted">: {sessionTitle(selected)}</span>}
            {selected?.kind === 'live' && <> <span className="chip">live</span></>}
            {selected?.meta.simulated && <> <SimTag /></>}
          </h1>
          <div className="muted">
            {selected ? (
              <>
                {selected.started ? `${utcDate(selected.started)} ${utcTime(selected.started)} UTC` : 'date unknown'}
                {selectedLength && ` · ${selectedLength}`}
                {selected.source && ` · ${selected.source.split(/[\\/]/).pop()}`}
                {' · '}
                <a href={href('events', { session: selected.id })}>Events</a> ·{' '}
                <a href={href('sky', { session: selected.id })}>Sky map</a> ·{' '}
                <a href={href(`report/${selected.id}`)}>Night report</a>
              </>
            ) : (
              `All ${all.length} session${all.length === 1 ? '' : 's'} together`
            )}
          </div>
        </div>
      </div>

      <div className="grid cols-4">
        <div className="panel stat"><span className="stat-value c-Satellite">{totals.sat}</span><span className="stat-label">Satellites</span></div>
        <div className="panel stat"><span className="stat-value c-Meteor">{totals.met}</span><span className="stat-label">Meteors</span></div>
        <div className="panel stat"><span className="stat-value c-Plane">{totals.pla}</span><span className="stat-label">Aircraft</span></div>
        <div className="panel stat"><span className="stat-value">{shown.length}</span><span className="stat-label">{shown.length === 1 ? 'Session' : 'Sessions'}</span></div>
      </div>

      <section className="panel">
        <div className="page-head" style={{ marginBottom: 12 }}>
          <h2>Latest detections{selected ? '' : ', all sessions'}</h2>
          <div className="spacer" />
          <a href={href('events', selected ? { session: selected.id } : undefined)}>All events →</a>
        </div>
        {recent.data && recent.data.length > 0 ? (
          <div className="card-grid">
            {recent.data.map((e) => (
              <EventCard key={e.id} e={e} session={selected ? undefined : titles.get(e.session_id)} />
            ))}
          </div>
        ) : (
          <div className="empty">No detections yet. Analyse a video or start the live camera.</div>
        )}
      </section>

      <section className="panel">
        <div className="page-head" style={{ marginBottom: 12 }}>
          <h2>Sessions</h2>
          <div className="spacer" />
          {all.length > 1 && (
            <button className="btn small" aria-pressed={!selected} onClick={() => choose(undefined)}>All sessions together</button>
          )}
        </div>
        {all.length === 0 ? (
          <div className="empty">
            Nothing recorded yet. <a href={href('analyze')}>Analyse a video</a> or start the live camera.
          </div>
        ) : (
          <div className="session-grid">
            {all.map((s) => (
              <SessionCard key={s.id} s={s} selected={s.id === selected?.id}
                onSelect={() => choose(s.id)} onChanged={(removed) => changed(removed ? s.id : undefined)} />
            ))}
          </div>
        )}
      </section>
    </>
  )
}
