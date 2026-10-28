import { api, REAL_CLASSES, sessionTitle } from '../api'
import { SkyMap } from '../components/charts'
import { ClassBadge, Failed, Loading, SimTag } from '../components/common'
import { useAsync } from '../hooks'
import type { Scope } from '../scope'

export default function SkyPage({ scope }: { scope: Scope }) {
  // One session at a time (each has its own pointing): with "all sessions", the last
  // one added that has a star calibration.
  const current = scope.session ?? scope.visible.find((s) => s.meta.pointing)
  const session = current?.id
  const events = useAsync(
    () => (session === undefined ? Promise.resolve([]) : api.events({ session_id: session, limit: 1000 })),
    [session],
  )
  if (scope.error) return <Failed error={scope.error} />
  if (!scope.loaded) return <Loading what="sessions" />
  const placed = (events.data ?? []).filter((e) => e.sky?.path)
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Sky map{current && <span className="muted">: {sessionTitle(current)}</span>} {current?.meta.simulated && <SimTag />}</h1>
          <div className="muted">
            Where each object crossed the sky, from the star calibration. Click a path to open it.
            {!scope.session && current && ' The map shows one session at a time: choose another in the top bar.'}
          </div>
        </div>
      </div>
      <div className="grid sky-grid">
        <section className="panel sky-panel">
          {events.data ? <SkyMap events={placed} pointing={current?.meta.pointing} /> : <Loading what="events" />}
        </section>
        <section className="panel">
          <h3>Legend</h3>
          <div className="legend">
            {REAL_CLASSES.map((c) => (
              <div key={c}><ClassBadge label={c} /> <span className="mono muted">{placed.filter((e) => e.label === c).length}</span></div>
            ))}
            <div className="muted">⌖ where the camera points</div>
          </div>
          {current?.meta.observer && (
            <dl className="kv" style={{ marginTop: 16 }}>
              <dt>Location</dt>
              <dd>{current.meta.observer.lat.toFixed(3)}°, {current.meta.observer.lon.toFixed(3)}°</dd>
              {current.meta.pointing && (
                <>
                  <dt>Pointing</dt>
                  <dd>alt {current.meta.pointing.alt.toFixed(1)}°, az {current.meta.pointing.az.toFixed(1)}°</dd>
                </>
              )}
            </dl>
          )}
          {events.data && placed.length === 0 && (
            <p className="muted" style={{ marginTop: 16 }}>
              No sky positions in this session. They need a star calibration: enough stars in the video,
              plus the recording time and location (<span className="mono">--start-time --lat --lon</span>).
            </p>
          )}
        </section>
      </div>
    </>
  )
}
