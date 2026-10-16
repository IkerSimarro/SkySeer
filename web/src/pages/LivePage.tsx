import { useEffect, useRef, useState } from 'react'
import { api, duration, type EventSummary, type LiveStatus, utcTime } from '../api'
import { SkyMap } from '../components/charts'
import { EventCard } from '../components/common'
import { href, useUtcNow } from '../hooks'

interface Feed {
  connected: boolean
  status: LiveStatus | null | undefined // undefined = not heard yet; null = no live station
  events: EventSummary[]
}

/** Listens to the station's WebSocket: status every 2 s and each event the moment it ends. */
function useLiveFeed(): Feed {
  const [feed, setFeed] = useState<Feed>({ connected: false, status: undefined, events: [] })
  const retry = useRef<number | undefined>(undefined)
  useEffect(() => {
    let ws: WebSocket | undefined
    let closed = false
    const connect = () => {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${window.location.host}/ws/live`)
      ws.onopen = () => setFeed((f) => ({ ...f, connected: true }))
      ws.onmessage = (msg) => {
        const data = JSON.parse(msg.data as string)
        if (data.type === 'status') setFeed((f) => ({ ...f, status: data.live }))
        if (data.type === 'event') setFeed((f) => ({ ...f, events: [data.event, ...f.events].slice(0, 60) }))
      }
      ws.onclose = () => {
        setFeed((f) => ({ ...f, connected: false }))
        if (!closed) retry.current = window.setTimeout(connect, 3000)
      }
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(retry.current)
      ws?.close()
    }
  }, [])
  return feed
}

function gpsText(gps: NonNullable<LiveStatus['gps']>): string {
  const fix = gps.fix
  if (!fix || fix.mode < 2) return gps.using_cached_location ? 'no fix yet (using the saved location)' : gps.error ?? 'searching for satellites…'
  const clock = fix.clock_offset_s === null ? '' : `, clock ${fix.clock_offset_s >= 0 ? '+' : ''}${fix.clock_offset_s.toFixed(2)} s`
  return `${fix.mode === 3 ? '3-D' : '2-D'} fix, ${fix.satellites} satellites${clock}`
}

export default function LivePage() {
  const feed = useLiveFeed()
  const now = useUtcNow()
  const s = feed.status
  const [earlier, setEarlier] = useState<EventSummary[]>([])
  useEffect(() => {
    // Events of tonight's session that happened before this page was opened.
    if (s?.session_id) api.events({ session_id: s.session_id, limit: 60 }).then(setEarlier).catch(() => {})
  }, [s?.session_id])
  const ids = new Set(feed.events.map((e) => e.id))
  const events = [...feed.events, ...earlier.filter((e) => !ids.has(e.id))]

  if (s === undefined) return <div className="empty">Connecting to the station…</div>
  if (s === null)
    return (
      <div className="empty" style={{ textAlign: 'left' }}>
        <h2 style={{ marginBottom: 8 }}>No live camera is running</h2>
        <p>Start the station with the dashboard:</p>
        <pre className="mono">skyseer serve --camera 0 --lat 40.42 --lon -3.70</pre>
        <p>or, without a camera, replay a recording as if it were live:</p>
        <pre className="mono">skyseer serve --replay data/sim/demo_sats.mp4 --start-time 2026-10-09T19:45:00 --lat 40.42 --lon -3.70</pre>
      </div>
    )
  if (s.mode === 'waiting' && !s.running)
    return (
      <div className="empty" style={{ textAlign: 'left' }}>
        <h2 style={{ marginBottom: 8 }}>Waiting for darkness</h2>
        <p>
          The Sun is at <span className="mono">{s.sun_altitude?.toFixed(1)}°</span>; the station observes once it is below{' '}
          <span className="mono">{s.dark_below?.toFixed(0)}°</span>
          {s.next_change ? <> — tonight from <span className="mono">{utcTime(s.next_change)} UTC</span>.</> : '.'}
        </p>
        <p className="muted">{s.nights ? `${s.nights} night${s.nights === 1 ? '' : 's'} observed since the station started.` : ''}</p>
        {s.station_error && <p className="error">{s.station_error}</p>}
      </div>
    )
  const counts = s.counts ?? {}
  const started = s.started ? (now.getTime() - Date.parse(s.started)) / 1000 : 0
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Live</h1>
          <div className="muted">{s.camera} · {s.running ? `running for ${duration(started)}` : 'stopped'}</div>
        </div>
        <div className="spacer" />
        <span className="chip"><span className={`dot ${feed.connected ? 'ok' : 'warn'}`} />{feed.connected ? 'Connected' : 'Reconnecting…'}</span>
        {s.session_id && <a className="btn" href={href(`report/${s.session_id}`)}>Tonight's report</a>}
      </div>
      {s.error && <div className="panel error">The station stopped: {s.error}</div>}
      {s.clock_warning && <div className="panel error">{s.clock_warning}</div>}
      <div className="grid detail-grid">
        <section className="panel media-panel">
          {s.running ? (
            <img className="live-view" src="/api/live/preview.mjpg" alt="Live camera view" />
          ) : (
            <div className="empty">The camera is not running.</div>
          )}
          <div className="muted media-note">Tracked objects are bracketed; finished events are drawn in their class colour.</div>
        </section>
        <div>
          <section className="panel">
            <h3>Station</h3>
            <dl className="kv">
              <dt>Speed</dt><dd>{s.fps.toFixed(1)} frames/s</dd>
              <dt>Frames</dt><dd>{s.frames.toLocaleString()}</dd>
              <dt>Star calibration</dt>
              <dd>{s.calibrated ? (s.pointing ? `pointing alt ${s.pointing.alt.toFixed(1)}°, az ${s.pointing.az.toFixed(1)}°` : 'solved (sky coordinates only)') : 'waiting for enough stars…'}</dd>
              <dt>Location</dt><dd>{s.observer ? `${s.observer.lat.toFixed(3)}°, ${s.observer.lon.toFixed(3)}°` : 'not set: satellites cannot be named'}</dd>
              {s.gps && <><dt>GPS</dt><dd>{gpsText(s.gps)}</dd></>}
              <dt>Tonight</dt>
              <dd>
                <span className="c-Satellite">{counts.Satellite ?? 0} satellites</span> ·{' '}
                <span className="c-Meteor">{counts.Meteor ?? 0} meteors</span> ·{' '}
                <span className="c-Plane">{counts.Plane ?? 0} aircraft</span>
              </dd>
              <dt>Rejected</dt><dd>{(counts.Junk ?? 0) + (counts.Star ?? 0)} tracks (noise, clouds, stars)</dd>
            </dl>
          </section>
          <section className="panel">
            <h3>Tonight's sky</h3>
            <SkyMap events={events} pointing={s.pointing} size={420} />
          </section>
        </div>
      </div>
      <section className="panel">
        <h2 style={{ marginBottom: 10 }}>Detections {events.length > 0 && <span className="muted">· latest {utcTime(events[0].start_time)} UTC</span>}</h2>
        {events.length ? (
          <div className="card-grid">{events.map((e) => <EventCard key={e.id} e={e} />)}</div>
        ) : (
          <div className="empty">Watching… detections appear here the moment each object leaves the view.</div>
        )}
      </section>
    </>
  )
}
