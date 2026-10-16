import type { EventSummary, ObjectClass, Session } from '../api'
import { duration, mag, utcDate, utcTime } from '../api'
import { href } from '../hooks'

export function ClassBadge({ label }: { label: ObjectClass }) {
  return (
    <span className={`badge c-${label}`}>
      <span className={`dot bg-${label}`} />
      {label}
    </span>
  )
}

/** One detection: its trail image (the whole path in one picture) and key facts. */
export function EventCard({ e }: { e: EventSummary }) {
  return (
    <a className="event-card" href={href(`event/${e.id}`)} style={{ borderTopColor: `var(--${e.label.toLowerCase()})` }}>
      <div className="event-thumb">
        {e.media ? <img src={e.media.thumbnail} alt={`${e.label} trail`} loading="lazy" /> : <div className="no-thumb">no image</div>}
        {e.simulated && <span className="sim-tag sim-corner" title="From SkySeer's simulator, not a real sky">SIM</span>}
        {e.review && (
          <span className={`review-flag ${e.review === e.label ? 'agree' : 'disagree'}`}
            title={e.review === e.label ? 'Confirmed by a person' : `A person says: ${e.review}`}>
            {e.review === e.label ? '✓' : `→ ${e.review}`}
          </span>
        )}
      </div>
      <div className="event-body">
        <div className="event-row">
          <ClassBadge label={e.label} />
          <span className="mono faint">#{e.event_id}</span>
        </div>
        <div className="event-name">{e.name ?? (e.label === 'Satellite' ? 'Unidentified' : '')}</div>
        <div className="event-row mono muted">
          <span>{utcTime(e.start_time)}</span>
          <span>{duration(e.duration_s)}</span>
          <span title="Peak brightness (magnitude: lower is brighter)">{mag(e.magnitude)}</span>
        </div>
      </div>
    </a>
  )
}

export function SessionSelect({
  sessions,
  value,
  onChange,
}: {
  sessions: Session[]
  value: number | undefined
  onChange: (id: number | undefined) => void
}) {
  return (
    <select value={value ?? ''} onChange={(ev) => onChange(ev.target.value ? Number(ev.target.value) : undefined)}>
      <option value="">All sessions</option>
      {sessions.map((s) => (
        <option key={s.id} value={s.id}>
          {s.started ? `${utcDate(s.started)} · ` : ''}{s.name}
        </option>
      ))}
    </select>
  )
}

export function SimTag() {
  return <span className="sim-tag" title="Made by SkySeer's simulator from real star, orbit and shower data: not a real recording">SIMULATION</span>
}

export function Loading({ what }: { what: string }) {
  return <div className="empty">Loading {what}…</div>
}

export function Failed({ error }: { error: string }) {
  return <div className="empty error">Could not load: {error}</div>
}
