import { useEffect, useRef, useState } from 'react'
import { api, isTest, type Session, type SessionFiles, sessionLength, sessionTitle, utcDate, utcTime } from '../api'
import { Failed, Loading } from '../components/common'
import { EditIcon, TrashIcon } from '../components/Icons'
import { href, navigate, useAsync } from '../hooks'
import type { Scope } from '../scope'

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
  onChanged: () => void
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
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }
  const len = sessionLength(s)
  return (
    <div className={`session-card${selected ? ' selected' : ''}`}>
      <button className={`session-cover${pics.length === 1 ? ' single' : ''}`} onClick={onSelect}
        title={pics.length ? pics.map((p) => `${p.label}${p.name ? `: ${p.name}` : ''}`).join(' · ') : 'Show this session'}>
        {pics.length === 0 && <div className="empty-cover">No clips</div>}
        {pics.map((p, i) => (
          <img key={p.id} className={i === 0 ? 'main' : ''} src={p.thumbnail} alt={`${p.label} ${p.name ?? ''}`} loading="lazy" />
        ))}
        <span className="tags">
          {isTest(s) && <span className="sim-tag">TEST</span>}
          {s.kind === 'live' && <span className="chip">live</span>}
          {s.meta.interrupted && (
            <span className="chip" title="The station stopped mid-night (a crash or power cut); the session ends at its last recorded frame.">
              cut off
            </span>
          )}
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
        <RemoveBox s={s} onCancel={() => setRemoving(false)} onDone={() => { setRemoving(false); onChanged() }} />
      )}
    </div>
  )
}

type Kind = 'all' | 'file' | 'live'
type Sort = 'added' | 'detections' | 'longest'
const length = (s: Session) => (s.started && s.ended ? Date.parse(s.ended) - Date.parse(s.started) : 0)
const ORDER: Record<Sort, (a: Session, b: Session) => number> = {
  added: (a, b) => b.id - a.id,
  detections: (a, b) => b.satellites + b.meteors + b.planes - (a.satellites + a.meteors + a.planes),
  longest: (a, b) => length(b) - length(a),
}

/** Every session as a card: find, rename and remove them, and pick the one to show. */
export default function SessionsPage({ scope }: { scope: Scope }) {
  const [search, setSearch] = useState('')
  const [kind, setKind] = useState<Kind>('all')
  const [sort, setSort] = useState<Sort>('added')
  if (scope.error) return <Failed error={scope.error} />
  if (!scope.loaded) return <Loading what="sessions" />
  const words = search.toLowerCase().split(/\s+/).filter(Boolean)
  const shown = scope.visible
    .filter((s) => kind === 'all' || s.kind === kind)
    .filter((s) => {
      const text = `${sessionTitle(s)} ${s.name} ${s.source ?? ''}`.toLowerCase()
      return words.every((w) => text.includes(w))
    })
    .sort(ORDER[sort])
  const hidden = scope.all.length - scope.visible.length
  const open = (id: number) => {
    scope.choose(id)
    navigate(href(''))
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Sessions</h1>
          <div className="muted">Each analysed recording or live night. Pick one to show it on every page.</div>
        </div>
      </div>
      <div className="toolbar">
        <input type="search" placeholder="Search names and files" value={search} aria-label="Search sessions"
          onChange={(e) => setSearch(e.target.value)} />
        <div className="segmented" role="radiogroup" aria-label="Kind of session">
          {(['all', 'file', 'live'] as Kind[]).map((k) => (
            <button key={k} type="button" role="radio" aria-checked={kind === k}
              className={kind === k ? 'active' : ''} onClick={() => setKind(k)}>
              {k === 'all' ? 'All' : k === 'file' ? 'Recordings' : 'Live nights'}
            </button>
          ))}
        </div>
        <select value={sort} onChange={(e) => setSort(e.target.value as Sort)} aria-label="Sort by">
          <option value="added">Added last first</option>
          <option value="detections">Most detections</option>
          <option value="longest">Longest</option>
        </select>
        <div className="spacer" />
        <label className="check" title="Sessions made by SkySeer's simulators, not a real sky">
          <input type="checkbox" checked={scope.showTests} onChange={(e) => scope.setShowTests(e.target.checked)} />
          Show test sessions{!scope.showTests && hidden ? ` (${hidden} hidden)` : ''}
        </label>
      </div>
      {scope.all.length === 0 ? (
        <div className="empty">
          Nothing recorded yet. <a href={href('analyze')}>Analyse a video</a> or start the live camera.
        </div>
      ) : shown.length === 0 ? (
        <div className="empty">
          {scope.visible.length === 0
            ? `Only test sessions so far (${hidden}): tick "Show test sessions" to see them.`
            : 'No session matches.'}
        </div>
      ) : (
        <div className="session-grid">
          {shown.map((s) => (
            <SessionCard key={s.id} s={s} selected={s.id === scope.session?.id}
              onSelect={() => open(s.id)} onChanged={scope.reload} />
          ))}
        </div>
      )}
    </>
  )
}
