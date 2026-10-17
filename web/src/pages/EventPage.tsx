import { useState } from 'react'
import { ALL_CLASSES, api, duration, mag, type EventDetail, type ObjectClass, utcDate, utcTime } from '../api'
import { LightCurve } from '../components/charts'
import { ClassBadge, Failed, Loading, SimTag } from '../components/common'
import { BackIcon } from '../components/Icons'
import { href, useAsync } from '../hooks'

const SOURCE: Record<string, string> = {
  rules: 'hand-written rules',
  model: 'learned model',
  orbit: 'matched to a satellite orbit',
  starfield: 'moves with the stars',
}

function ReviewPanel({ e, onChange }: { e: EventDetail; onChange: (label: ObjectClass | null) => void }) {
  const [saving, setSaving] = useState(false)
  const set = async (label: ObjectClass | null) => {
    setSaving(true)
    try {
      await api.review(e.id, label)
      onChange(label)
    } finally {
      setSaving(false)
    }
  }
  return (
    <section className="panel">
      <h3>Your verdict</h3>
      <p className="muted" style={{ margin: '6px 0 10px' }}>
        Is this really a {e.label.toLowerCase()}? Reviews become real training data for the learned classifier.
      </p>
      <div className="btn-group">
        {ALL_CLASSES.map((c) => (
          <button key={c} className={`btn small c-${c}`} disabled={saving} aria-pressed={e.review === c} onClick={() => set(c)}>
            {c === e.label ? `✓ ${c}` : c}
          </button>
        ))}
        {e.review && <button className="btn small" disabled={saving} onClick={() => set(null)}>Clear</button>}
      </div>
    </section>
  )
}

function Scores({ e }: { e: EventDetail }) {
  const entries = Object.entries(e.scores)
  const total = entries.reduce((t, [, v]) => t + v, 0) || 1
  if (!entries.length) return null
  return (
    <section className="panel">
      <h3>{e.label_source === 'model' ? 'Model probabilities' : 'Rule scores (share)'}</h3>
      <div className="bars">
        {entries.sort((a, b) => b[1] - a[1]).map(([k, v]) => (
          <div key={k} className="bar-row">
            <span className={`c-${k}`}>{k}</span>
            <div className="bar-track"><div className={`bar-fill bg-${k}`} style={{ width: `${(100 * v) / total}%` }} /></div>
            <span className="mono muted">{((100 * v) / total).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </section>
  )
}

export default function EventPage({ id }: { id: number }) {
  const ev = useAsync(() => api.event(id), [id])
  const [review, setReview] = useState<ObjectClass | null | undefined>(undefined)
  if (ev.error) return <Failed error={ev.error} />
  if (!ev.data) return <Loading what="event" />
  const e = { ...ev.data, review: review === undefined ? ev.data.review : review }
  const ident = e.identification
  const shower = e.shower
  const sky = e.sky
  return (
    <>
      <div className="page-head">
        <a className="btn small" href={href('events', { session: e.session_id })}><BackIcon /> Events</a>
        <div>
          <h1>
            <ClassBadge label={e.label} /> {ident?.name ?? (typeof shower === 'object' ? `${shower.name} meteor` : `Event #${e.event_id}`)}
            {e.fireball && <span className="chip" style={{ marginLeft: 8, color: 'var(--meteor)' }}>Fireball</span>}
            {e.simulated && <> <SimTag /></>}
          </h1>
          <div className="muted mono">
            {utcDate(e.start_time)} {utcTime(e.start_time)} UTC · {duration(e.duration_s)} · {Math.round(100 * e.confidence)}% confident ({SOURCE[e.label_source] ?? e.label_source})
          </div>
        </div>
      </div>

      <div className="grid detail-grid">
        <div>
          <section className="panel media-panel">
            {e.media ? (
              <>
                <video src={e.media.clip} controls autoPlay loop muted playsInline />
                <div className="muted media-note">
                  Zoomed clip{e.media.speedup > 1 ? `, ${e.media.speedup}× faster than real time` : ', real time'}
                  {e.media.partial ? ' (its start was too long ago to keep)' : ''}
                </div>
              </>
            ) : <div className="empty">No clip was recorded for this event.</div>}
          </section>
          {e.media && (
            <section className="panel">
              <h3>Where in the view</h3>
              <img className="context-img" src={e.media.context} alt="Whole view with the path marked" />
            </section>
          )}
          <section className="panel">
            <h3>Brightness over time</h3>
            {e.photometry?.light_curve?.some((m) => m !== null) ? (
              <LightCurve magnitudes={e.photometry.light_curve} track={e.track} seconds={e.duration_s} />
            ) : (
              <p className="muted">Needs a star calibration (stars, location and time) to measure magnitudes.</p>
            )}
          </section>
        </div>

        <div>
          <ReviewPanel e={e} onChange={setReview} />
          {ident && (
            <section className="panel">
              <h3>Satellite identification</h3>
              <dl className="kv">
                <dt>Name</dt><dd>{ident.name}</dd>
                <dt>NORAD ID</dt><dd><a href={`https://celestrak.org/satcat/table-satcat.php?CATNR=${ident.norad_id}`} target="_blank" rel="noreferrer">{ident.norad_id}</a></dd>
                <dt>Altitude</dt><dd>{ident.altitude_km.toFixed(0)} km</dd>
                <dt>Distance</dt><dd>{ident.range_km.toFixed(0)} km</dd>
                <dt>Speed</dt><dd>{ident.speed_km_s.toFixed(2)} km/s</dd>
                <dt>Match error</dt><dd>{(ident.cross_track_deg * 60).toFixed(1)}′ across the path</dd>
                <dt>Camera clock</dt><dd>{ident.clock_offset_s >= 0 ? '+' : ''}{ident.clock_offset_s.toFixed(2)} s off</dd>
                {ident.runner_up && <><dt>Runner-up</dt><dd>{ident.runner_up}</dd></>}
              </dl>
            </section>
          )}
          {shower && (
            <section className="panel">
              <h3>Meteor shower</h3>
              {shower === 'sporadic' ? (
                <p className="muted">Sporadic: its path does not point back to any active shower's radiant.</p>
              ) : (
                <dl className="kv">
                  <dt>Shower</dt><dd>{shower.name} ({shower.code})</dd>
                  <dt>Off radiant</dt><dd>{shower.radiant_offset_deg.toFixed(1)}°</dd>
                  <dt>Confidence</dt><dd>{Math.round(100 * shower.confidence)}%</dd>
                </dl>
              )}
            </section>
          )}
          <section className="panel">
            <h3>Measurements</h3>
            <dl className="kv">
              {e.photometry && <><dt>Peak brightness</dt><dd>{e.photometry.saturated ? 'brighter than ' : ''}{mag(e.photometry.peak_magnitude)} mag</dd></>}
              {sky?.start.alt !== undefined && sky.end.alt !== undefined && (
                <>
                  <dt>From</dt><dd>alt {sky.start.alt.toFixed(1)}°, az {sky.start.az?.toFixed(1)}°</dd>
                  <dt>To</dt><dd>alt {sky.end.alt.toFixed(1)}°, az {sky.end.az?.toFixed(1)}°</dd>
                </>
              )}
              {sky && <><dt>Angular speed</dt><dd>{sky.angular_speed_deg_s.toFixed(2)} °/s</dd></>}
              <dt>Speed on screen</dt><dd>{(100 * e.features.speed_frac_per_s).toFixed(1)}% of the frame per second</dd>
              <dt>Straightness</dt><dd>{(100 * e.features.path_deviation_max).toFixed(2)}% max deviation</dd>
              <dt>Seen in</dt><dd>{Math.round(100 * e.features.detection_rate)}% of frames</dd>
              <dt>Strobe rhythm</dt><dd>{e.features.flash_score > 0 ? `${e.features.flash_rate_hz.toFixed(2)} Hz (score ${e.features.flash_score.toFixed(2)})` : 'none'}</dd>
              {e.periodicity && <><dt>Blinks every</dt><dd>{e.periodicity.period_s.toFixed(2)} s (tumbling?)</dd></>}
              <dt>Detections</dt><dd>{e.track.length} (frames {e.first_frame}–{e.last_frame})</dd>
            </dl>
          </section>
          <Scores e={e} />
        </div>
      </div>
    </>
  )
}
