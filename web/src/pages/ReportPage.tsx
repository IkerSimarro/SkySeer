import { api, duration, mag, utcDate, utcTime } from '../api'
import { Timeline } from '../components/charts'
import { Failed, Loading, SessionSelect, SimTag } from '../components/common'
import { DownloadIcon } from '../components/Icons'
import { href, navigate, useAsync } from '../hooks'

function fmtSeconds(s: number): string {
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}

const TIME_SOURCE: Record<string, string> = {
  given: 'given when it was analysed',
  'video file': 'from the video file (checked: night at that place)',
  unknown: 'unknown: satellites, aircraft and showers were not identified',
}

export default function ReportPage({ id }: { id?: number }) {
  const sessions = useAsync(api.sessions, [])
  const session = id ?? sessions.data?.[0]?.id
  const report = useAsync(
    () => (session === undefined ? Promise.resolve(undefined) : api.report(session)),
    [session],
  )
  if (sessions.error) return <Failed error={sessions.error} />
  if (!sessions.data) return <Loading what="sessions" />
  if (session === undefined) return <div className="empty">Nothing recorded yet.</div>
  const r = report.data
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Night report {sessions.data.find((s) => s.id === session)?.meta.simulated && <SimTag />}</h1>
          {r && (
            <div className="muted mono">
              {r.session.name} · {r.start ? `${utcDate(r.start)} ${utcTime(r.start)}` : ''}
              {r.end ? `–${utcTime(r.end)} UTC` : ''} · {duration(r.duration_s)}
            </div>
          )}
        </div>
        <div className="spacer" />
        <div className="no-print toolbar" style={{ margin: 0 }}>
          <SessionSelect sessions={sessions.data} value={session}
            onChange={(sid) => sid !== undefined && navigate(href(`report/${sid}`))} />
          <a className="btn" href={api.exportUrl(session, 'csv')}><DownloadIcon /> CSV</a>
          <a className="btn" href={api.exportUrl(session, 'json')}><DownloadIcon /> JSON</a>
          <button className="btn" onClick={() => window.print()}>Print / PDF</button>
        </div>
      </div>
      {report.error ? <Failed error={report.error} /> : !r ? <Loading what="report" /> : (
        <>
          <div className="grid cols-4">
            <div className="panel stat"><span className="stat-value c-Satellite">{r.counts.Satellite}</span>
              <span className="stat-label">Satellites{r.identified_satellites ? ` · ${r.identified_satellites} named` : ''}</span></div>
            <div className="panel stat"><span className="stat-value c-Meteor">{r.counts.Meteor}</span>
              <span className="stat-label">Meteors{r.brightest_meteor ? ` · brightest ${mag(r.brightest_meteor.magnitude)}` : ''}</span></div>
            <div className="panel stat"><span className="stat-value c-Plane">{r.counts.Plane}</span>
              <span className="stat-label">Aircraft{r.identified_aircraft ? ` · ${r.identified_aircraft} named` : ''}</span></div>
            <div className="panel stat"><span className="stat-value">{r.reviewed}</span>
              <span className="stat-label">Checked by you (<a href={href('events', { session, review: 'unreviewed' })}>review more</a>)</span></div>
          </div>

          <section className="panel">
            <h3>Activity through the night</h3>
            <Timeline timeline={r.timeline} />
          </section>

          <div className="grid cols-2">
            <section className="panel">
              <h3>Satellites</h3>
              {r.satellites.length === 0 ? <p className="muted">None.</p> : (
                <table className="data">
                  <thead><tr><th>Time (UTC)</th><th>Identified as</th><th className="num">Mag</th></tr></thead>
                  <tbody>
                    {r.satellites.map((s) => (
                      <tr key={s.id}>
                        <td className="mono"><a href={href(`event/${s.id}`)}>{utcTime(s.time)}</a></td>
                        <td>{s.name ?? <span className="muted">not identified</span>}</td>
                        <td className="num mono">{mag(s.magnitude)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>
            <section className="panel">
              <h3>Meteors, brightest first</h3>
              {r.meteors.length === 0 ? <p className="muted">None.</p> : (
                <table className="data">
                  <thead><tr><th>Time (UTC)</th><th>Shower</th><th className="num">Mag</th></tr></thead>
                  <tbody>
                    {r.meteors.map((m) => (
                      <tr key={m.id}>
                        <td className="mono"><a href={href(`event/${m.id}`)}>{utcTime(m.time)}</a></td>
                        <td>{m.shower ?? <span className="muted">needs calibration</span>}{m.fireball && <span className="chip" style={{ marginLeft: 6 }}>fireball</span>}</td>
                        <td className="num mono">{mag(m.magnitude)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {Object.keys(r.showers).length > 0 && (
                <p className="muted" style={{ marginTop: 10 }}>
                  {Object.entries(r.showers).map(([code, n]) => `${code}: ${n}`).join(' · ')}
                </p>
              )}
            </section>
          </div>

          {r.aircraft.length > 0 && (
            <section className="panel">
              <h3>Aircraft</h3>
              <table className="data">
                <thead><tr><th>Time (UTC)</th><th>Flight</th><th className="num">Height</th></tr></thead>
                <tbody>
                  {r.aircraft.map((a) => (
                    <tr key={a.id}>
                      <td className="mono"><a href={href(`event/${a.id}`)}>{utcTime(a.time)}</a></td>
                      <td>{a.name ?? <span className="muted">not identified</span>}</td>
                      <td className="num mono">{a.altitude_m === null ? '—' : `${Math.round(a.altitude_m).toLocaleString()} m`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          <section className="panel">
            <h3>Station</h3>
            <dl className="kv">
              <dt>Location</dt>
              <dd>{r.observer ? `${r.observer.lat.toFixed(3)}°, ${r.observer.lon.toFixed(3)}°` : 'not given'}</dd>
              {r.time_source && (
                <>
                  <dt>Recording time</dt>
                  <dd>{TIME_SOURCE[r.time_source] ?? r.time_source}</dd>
                </>
              )}
              <dt>Camera pointing</dt>
              <dd>{r.pointing ? `alt ${r.pointing.alt.toFixed(1)}°, az ${r.pointing.az.toFixed(1)}°` : 'not calibrated'}</dd>
              <dt>Reviewed by a person</dt><dd>{r.reviewed} event{r.reviewed === 1 ? '' : 's'}</dd>
              <dt>Camera moves ignored</dt>
              <dd>{r.disturbances.length === 0 ? 'none' : `${r.disturbances.length} (${r.disturbances.slice(0, 6).map(([a]) => fmtSeconds(a)).join(', ')}${r.disturbances.length > 6 ? ', …' : ''})`}</dd>
            </dl>
          </section>
        </>
      )}
    </>
  )
}
