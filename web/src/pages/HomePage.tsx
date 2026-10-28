import { api, isTest, type Session, sessionLength, sessionTitle, utcDate, utcTime } from '../api'
import { EventCard, Failed, Loading, SimTag } from '../components/common'
import { href, useAsync } from '../hooks'
import type { Scope } from '../scope'

function Counts({ satellites, meteors, planes }: Pick<Session, 'satellites' | 'meteors' | 'planes'>) {
  return (
    <div className="hero-counts">
      <div><span className="stat-value c-Satellite">{satellites}</span><span className="stat-label">Satellites</span></div>
      <div><span className="stat-value c-Meteor">{meteors}</span><span className="stat-label">Meteors</span></div>
      <div><span className="stat-value c-Plane">{planes}</span><span className="stat-label">Aircraft</span></div>
    </div>
  )
}

/** The chosen session at a glance: its best moments, what it found, where to go next. */
function SessionHero({ s }: { s: Session }) {
  const pics = s.highlights ?? []
  const len = sessionLength(s)
  const file = s.source?.split(/[\\/]/).pop()
  return (
    <section className="panel hero">
      <div className={`hero-pics n${Math.min(pics.length, 3)}`}>
        {pics.length === 0 ? (
          <div className="empty-cover">No clips in this session</div>
        ) : (
          pics.slice(0, 3).map((p, i) => (
            <a key={p.id} href={href(`event/${p.id}`)} className={i === 0 ? 'main' : ''}
              title={`${p.label}${p.name ? `: ${p.name}` : ''}`}>
              <img src={p.thumbnail} alt={`${p.label} ${p.name ?? ''}`} />
              <span className={`hero-pic-label c-${p.label}`}>{p.name ?? p.label}</span>
            </a>
          ))
        )}
      </div>
      <div className="hero-info">
        <div className="hero-kicker">
          {s.kind === 'live' ? 'Live night' : 'Recording'}
          {isTest(s) && <> <SimTag /></>}
        </div>
        <h1>{sessionTitle(s)}</h1>
        <div className="muted">
          {s.started ? `${utcDate(s.started)}, ${utcTime(s.started)} UTC` : 'Date unknown'}
          {len && ` · ${len}`}
        </div>
        {file && <div className="faint small mono hero-file" title={s.source ?? undefined}>{file}</div>}
        <Counts {...s} />
        <div className="btn-group">
          <a className="btn" href={href('events')}>Events</a>
          <a className="btn" href={href('sky')}>Sky map</a>
          <a className="btn" href={href('report')}>Night report</a>
        </div>
      </div>
    </section>
  )
}

/** Home: the session chosen in the top bar (by default the one added last). */
export default function HomePage({ scope }: { scope: Scope }) {
  const { session } = scope
  const realOnly = !session && !scope.showTests
  const recent = useAsync(
    () =>
      scope.loaded
        ? api.events({ session_id: session?.id, simulated: realOnly ? false : undefined, limit: 8 })
        : Promise.resolve([]),
    [scope.loaded, session?.id, realOnly],
  )
  if (scope.error) return <Failed error={scope.error} />
  if (!scope.loaded) return <Loading what="sessions" />
  if (scope.all.length === 0) {
    return (
      <div className="welcome panel">
        <h1>Welcome to SkySeer</h1>
        <p className="muted">Nothing recorded yet. Analyse a recording of the night sky, or start the live camera.</p>
        <div className="btn-group">
          <a className="btn primary" href={href('analyze')}>Analyse a video</a>
          <a className="btn" href={href('live')}>Live camera</a>
        </div>
      </div>
    )
  }
  const titles = new Map(scope.all.map((s) => [s.id, sessionTitle(s)]))
  const hidden = scope.all.length - scope.visible.length
  const totals = scope.visible.reduce(
    (t, s) => ({ satellites: t.satellites + s.satellites, meteors: t.meteors + s.meteors, planes: t.planes + s.planes }),
    { satellites: 0, meteors: 0, planes: 0 },
  )

  return (
    <>
      {session ? (
        <SessionHero s={session} />
      ) : scope.visible.length === 0 ? (
        <div className="welcome panel">
          <h1>No real recordings yet</h1>
          <p className="muted">
            There {hidden === 1 ? 'is 1 test session' : `are ${hidden} test sessions`} (from SkySeer's simulators), hidden
            on the <a href={href('sessions')}>Sessions</a> page. Analyse a recording of the night sky to see it here.
          </p>
          <div className="btn-group"><a className="btn primary" href={href('analyze')}>Analyse a video</a></div>
        </div>
      ) : (
        <section className="panel hero all">
          <div className="hero-info">
            <div className="hero-kicker">All sessions together</div>
            <h1>{scope.visible.length} session{scope.visible.length === 1 ? '' : 's'}</h1>
            {hidden > 0 && <div className="muted">{hidden} test session{hidden === 1 ? '' : 's'} left out</div>}
            <Counts {...totals} />
            <div className="btn-group"><a className="btn" href={href('sessions')}>Browse sessions</a></div>
          </div>
        </section>
      )}

      {(session || scope.visible.length > 0) && (
        <section className="section">
          <div className="section-head">
            <h2>Latest detections</h2>
            <a href={href('events')}>All events →</a>
          </div>
          {recent.error ? (
            <Failed error={recent.error} />
          ) : !recent.data ? (
            <Loading what="detections" />
          ) : recent.data.length > 0 ? (
            <div className="card-grid">
              {recent.data.map((e) => (
                <EventCard key={e.id} e={e} session={session ? undefined : titles.get(e.session_id)} />
              ))}
            </div>
          ) : (
            <div className="empty">Nothing was detected in this session.</div>
          )}
        </section>
      )}
    </>
  )
}
