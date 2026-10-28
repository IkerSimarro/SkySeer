import type { EventSummary, ObjectClass } from '../api'
import { duration, isTest, mag, sessionTitle, utcDate, utcTime } from '../api'
import { href } from '../hooks'
import type { Scope } from '../scope'

export function ClassBadge({ label }: { label: ObjectClass }) {
  return (
    <span className={`badge c-${label}`}>
      <span className={`dot bg-${label}`} />
      {label}
    </span>
  )
}

/** One detection: its trail image (the whole path in one picture) and key facts.
 * ``session`` names the session it belongs to (when cards from several are shown). */
export function EventCard({ e, session }: { e: EventSummary; session?: string }) {
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
        {session && <div className="event-session muted" title="Session">{session}</div>}
      </div>
    </a>
  )
}

/** The top bar's session choice: every page shows the session picked here. */
export function SessionSwitcher({ scope }: { scope: Scope }) {
  const { session, visible } = scope
  if (!scope.all.length) return null
  // A test session chosen by a link stays listed even while tests are hidden.
  const options = session && !visible.includes(session) ? [session, ...visible] : visible
  return (
    <label className="session-switcher" title="Every page shows this session">
      <span className="faint">Session</span>
      <select value={session?.id ?? 'all'}
        onChange={(ev) => scope.choose(ev.target.value === 'all' ? 'all' : Number(ev.target.value))}>
        {options.map((s) => (
          <option key={s.id} value={s.id}>
            {sessionTitle(s)}{s.started ? ` · ${utcDate(s.started)}` : ''}{isTest(s) ? ' · test' : ''}
          </option>
        ))}
        {options.length > 1 && <option value="all">All sessions together</option>}
      </select>
    </label>
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
