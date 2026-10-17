// Hand-drawn SVG charts: small, fast on a Raspberry Pi, and styled like the rest of the UI.
import type { EventSummary, Report } from '../api'
import { mag, utcTime } from '../api'
import { href } from '../hooks'

const CLASS_VAR: Record<string, string> = {
  Satellite: 'var(--satellite)',
  Meteor: 'var(--meteor)',
  Plane: 'var(--plane)',
  Junk: 'var(--junk)',
  Star: 'var(--star)',
}

/**
 * All-sky map as seen lying on your back: zenith in the centre, horizon at the edge,
 * North up and East on the LEFT (a map of the sky seen from below is mirrored compared
 * with a map of the ground). Azimuthal equidistant: distance from centre = 90° - altitude.
 */
export function SkyMap({
  events,
  pointing,
  size = 560,
}: {
  events: EventSummary[]
  pointing?: { alt: number; az: number } | null
  size?: number
}) {
  const c = size / 2
  const R = size / 2 - 28
  const xy = (alt: number, az: number): [number, number] => {
    const r = ((90 - Math.max(alt, -5)) / 90) * R
    const a = (az * Math.PI) / 180
    return [c - r * Math.sin(a), c - r * Math.cos(a)]
  }
  const withPath = events.filter((e) => e.sky?.path && e.sky.path.length > 1)
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="skymap" role="img" aria-label="All-sky map of detections">
      <defs>
        <radialGradient id="skyglow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#0d1a33" />
          <stop offset="100%" stopColor="#070b14" />
        </radialGradient>
      </defs>
      <circle cx={c} cy={c} r={R} fill="url(#skyglow)" stroke="var(--line-2)" />
      {[30, 60].map((alt) => (
        <circle key={alt} cx={c} cy={c} r={((90 - alt) / 90) * R} fill="none" stroke="var(--line)" strokeDasharray="3 4" />
      ))}
      {[0, 45, 90, 135].map((az) => {
        const [x1, y1] = xy(0, az)
        const [x2, y2] = xy(0, az + 180)
        return <line key={az} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--line)" />
      })}
      {(['N', 'E', 'S', 'W'] as const).map((d, i) => {
        const [x, y] = xy(-9, i * 90)
        return <text key={d} x={x} y={y} className="skymap-dir" textAnchor="middle" dominantBaseline="middle">{d}</text>
      })}
      {[30, 60].map((alt) => {
        const [x, y] = xy(alt, 180)
        return <text key={alt} x={x + 4} y={y - 3} className="skymap-alt">{alt}°</text>
      })}
      {pointing && (() => {
        const [x, y] = xy(pointing.alt, pointing.az)
        return (
          <g className="skymap-pointing">
            <circle cx={x} cy={y} r={9} />
            <path d={`M${x - 15} ${y}h8M${x + 7} ${y}h8M${x} ${y - 15}v8M${x} ${y + 7}v8`} />
            <title>Camera points here: alt {pointing.alt.toFixed(1)}°, az {pointing.az.toFixed(1)}°</title>
          </g>
        )
      })()}
      {withPath.map((e) => {
        const pts = e.sky!.path!.map(([alt, az]) => xy(alt, az))
        const [ex, ey] = pts[pts.length - 1]
        return (
          <a key={e.id} href={href(`event/${e.id}`)} className="skymap-track">
            <polyline points={pts.map((p) => p.join(',')).join(' ')} stroke={CLASS_VAR[e.label]} fill="none" strokeWidth={2} />
            <circle cx={ex} cy={ey} r={3} fill={CLASS_VAR[e.label]} />
            <title>{`${e.label}${e.name ? ` · ${e.name}` : ''} · ${utcTime(e.start_time)} UTC · mag ${mag(e.magnitude)}`}</title>
          </a>
        )
      })}
    </svg>
  )
}

/** Detections per time bin, stacked by class. */
export function Timeline({ timeline }: { timeline: Report['timeline'] }) {
  const bins = timeline.bins
  if (!bins.length) return <p className="muted">Nothing detected.</p>
  const classes = ['Satellite', 'Meteor', 'Plane']
  const W = 760, H = 180, padL = 30, padB = 24
  const max = Math.max(1, ...bins.map((b) => classes.reduce((t, c) => t + (b[c] ?? 0), 0)))
  const bw = (W - padL) / bins.length
  const padT = 10
  const y = (v: number) => padT + (H - padB - padT) * (1 - v / max)
  const ticks = [...new Set([0, Math.round(max / 2), max])]
  const every = Math.ceil(bins.length / 8)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Detections over time">
      {ticks.map((v) => (
        <g key={v}>
          <line x1={padL} x2={W} y1={y(v)} y2={y(v)} stroke="var(--line)" />
          <text x={padL - 6} y={y(v) + 4} className="axis" textAnchor="end">{v}</text>
        </g>
      ))}
      {bins.map((b, i) => {
        let base = 0
        return (
          <g key={b.start}>
            {classes.map((c) => {
              const v = b[c] ?? 0
              if (!v) return null
              const top = y(base + v)
              const rect = <rect key={c} x={padL + i * bw + 1} y={top} width={Math.max(bw - 2, 1)} height={y(base) - top} fill={CLASS_VAR[c]} rx={1.5}><title>{`${utcTime(b.start)}: ${v} ${c.toLowerCase()}${v === 1 ? '' : 's'}`}</title></rect>
              base += v
              return rect
            })}
            {i % every === 0 && <text x={padL + i * bw + bw / 2} y={H - 6} className="axis" textAnchor="middle">{b.start.slice(11, timeline.bin_minutes < 1 ? 19 : 16)}</text>}
          </g>
        )
      })}
    </svg>
  )
}

const MAX_POINTS = 200

/** Long tracks have thousands of measurements: draw the median of each time bin instead,
 * so the curve shows real changes rather than frame-to-frame noise. */
function binMedians(pts: { t: number; m: number }[], n: number): { t: number; m: number }[] {
  if (pts.length <= n) return pts
  const out: { t: number; m: number }[] = []
  for (let b = 0; b < n; b++) {
    const chunk = pts.slice(Math.floor((b * pts.length) / n), Math.floor(((b + 1) * pts.length) / n))
    if (!chunk.length) continue
    const ms = chunk.map((p) => p.m).sort((a, c) => a - c)
    out.push({ t: chunk[Math.floor(chunk.length / 2)].t, m: ms[Math.floor(ms.length / 2)] })
  }
  return out
}

/** Magnitude against time; brighter (smaller magnitude) is drawn higher. */
export function LightCurve({
  magnitudes,
  track,
  seconds,
}: {
  magnitudes: (number | null)[]
  track: { frame: number }[]
  seconds: number
}) {
  const W = 520, H = 160, pad = 32
  const f0 = track[0]?.frame ?? 0
  const f1 = track[track.length - 1]?.frame ?? 1
  const raw = magnitudes
    .map((m, i) => (m === null ? null : { t: ((track[i]?.frame - f0) / Math.max(f1 - f0, 1)) * seconds, m }))
    .filter((p): p is { t: number; m: number } => p !== null)
  const pts = binMedians(raw, MAX_POINTS)
  if (pts.length < 2) return <p className="muted">Too few measurements.</p>
  const lo = Math.min(...pts.map((p) => p.m)), hi = Math.max(...pts.map((p) => p.m))
  const span = Math.max(hi - lo, 0.5)
  const x = (t: number) => pad + (t / Math.max(seconds, 1e-6)) * (W - pad - 10)
  const y = (m: number) => 10 + ((m - (lo - 0.1 * span)) / (1.2 * span)) * (H - 10 - pad)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Brightness over time">
      <line x1={pad} x2={W - 10} y1={H - pad} y2={H - pad} stroke="var(--line-2)" />
      {[lo, hi].map((m) => (
        <text key={m} x={pad - 6} y={y(m) + 4} className="axis" textAnchor="end">{mag(m)}</text>
      ))}
      <text x={W - 10} y={H - pad + 16} className="axis" textAnchor="end">{seconds.toFixed(1)} s</text>
      <text x={pad} y={H - pad + 16} className="axis">0 s</text>
      <polyline fill="none" stroke="var(--accent)" strokeWidth={1.6} points={pts.map((p) => `${x(p.t)},${y(p.m)}`).join(' ')} />
    </svg>
  )
}
