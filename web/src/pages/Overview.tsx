import { api, duration, utcDate, utcTime } from '../api'
import { EventCard, Failed, Loading, SimTag } from '../components/common'
import { href, useAsync } from '../hooks'

export default function Overview() {
  const sessions = useAsync(api.sessions, [])
  const recent = useAsync(() => api.events({ limit: 8 }), [])
  if (sessions.error) return <Failed error={sessions.error} />
  if (!sessions.data) return <Loading what="sessions" />
  const all = sessions.data
  const latest = all[0]
  const totals = all.reduce(
    (t, s) => ({ sat: t.sat + s.satellites, met: t.met + s.meteors, pla: t.pla + s.planes }),
    { sat: 0, met: 0, pla: 0 },
  )
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Overview</h1>
          <div className="muted">
            {all.length} session{all.length === 1 ? '' : 's'} recorded
            {latest?.started ? ` · latest ${utcDate(latest.started)}` : ''}
          </div>
        </div>
      </div>

      <div className="grid cols-4">
        <div className="panel stat"><span className="stat-value c-Satellite">{totals.sat}</span><span className="stat-label">Satellites</span></div>
        <div className="panel stat"><span className="stat-value c-Meteor">{totals.met}</span><span className="stat-label">Meteors</span></div>
        <div className="panel stat"><span className="stat-value c-Plane">{totals.pla}</span><span className="stat-label">Aircraft</span></div>
        <div className="panel stat"><span className="stat-value">{all.length}</span><span className="stat-label">Sessions</span></div>
      </div>

      <section className="panel">
        <div className="page-head" style={{ marginBottom: 12 }}>
          <h2>Latest detections</h2>
          <div className="spacer" />
          <a href={href('events')}>All events →</a>
        </div>
        {recent.data && recent.data.length > 0 ? (
          <div className="card-grid">{recent.data.map((e) => <EventCard key={e.id} e={e} />)}</div>
        ) : (
          <div className="empty">No detections yet. Analyse a video or start the live camera.</div>
        )}
      </section>

      <section className="panel">
        <h2 style={{ marginBottom: 10 }}>Sessions</h2>
        {all.length === 0 ? (
          <div className="empty">
            Nothing recorded yet. Run <span className="mono">skyseer analyze VIDEO</span> or start the live camera.
          </div>
        ) : (
          <table className="data">
            <thead>
              <tr>
                <th>Date (UTC)</th><th>Session</th><th>Source</th><th className="num">Length</th>
                <th className="num">Satellites</th><th className="num">Meteors</th><th className="num">Aircraft</th><th />
              </tr>
            </thead>
            <tbody>
              {all.map((s) => {
                const seconds = s.started && s.ended ? (Date.parse(s.ended) - Date.parse(s.started)) / 1000 : null
                return (
                  <tr key={s.id}>
                    <td className="mono">{s.started ? `${utcDate(s.started)} ${utcTime(s.started)}` : '—'}</td>
                    <td>{s.name} {s.kind === 'live' && <span className="chip">live</span>} {s.meta.simulated && <SimTag />}</td>
                    <td className="muted">{s.source?.split(/[\\/]/).pop() ?? '—'}</td>
                    <td className="num mono">{seconds !== null ? duration(seconds) : '—'}</td>
                    <td className="num mono c-Satellite">{s.satellites}</td>
                    <td className="num mono c-Meteor">{s.meteors}</td>
                    <td className="num mono c-Plane">{s.planes}</td>
                    <td className="num">
                      <a href={href('events', { session: s.id })}>Events</a> ·{' '}
                      <a href={href('sky', { session: s.id })}>Sky</a> ·{' '}
                      <a href={href(`report/${s.id}`)}>Report</a>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </section>
    </>
  )
}
