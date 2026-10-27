// Typed access to SkySeer's HTTP API (see src/skyseer/api/app.py).

export type ObjectClass = 'Satellite' | 'Meteor' | 'Plane' | 'Star' | 'Junk'
export const REAL_CLASSES: ObjectClass[] = ['Satellite', 'Meteor', 'Plane']
export const ALL_CLASSES: ObjectClass[] = ['Satellite', 'Meteor', 'Plane', 'Junk', 'Star']

export interface Highlight {
  id: number
  label: ObjectClass
  name: string | null
  thumbnail: string
  magnitude: number | null
}

export interface Session {
  id: number
  name: string // ties it to its folder
  title: string | null // a name a person gave it
  highlights?: Highlight[] // its most striking events, best first
  kind: 'file' | 'live'
  source: string | null
  started: string | null
  ended: string | null
  folder: string
  events: number
  satellites: number
  meteors: number
  planes: number
  meta: {
    fps?: number
    frame_size?: [number, number]
    observer?: { lat: number; lon: number; elevation_m: number }
    pointing?: { alt: number; az: number; roll: number }
    calibration?: Record<string, number | string>
    disturbances?: [number, number][]
    discarded_tracks?: number
    simulated?: boolean
  }
}

/** What to call a session: the name a person gave it, else its folder's name. */
export const sessionTitle = (s: Pick<Session, 'name' | 'title'>) => s.title || s.name

export interface SessionFiles {
  folder: string | null // inside SkySeer's data folders, else null (never deleted)
  folder_mb: number
  upload: string | null
  upload_mb: number
}

export interface SkyPoint {
  ra: number
  dec: number
  alt?: number
  az?: number
}

export interface Media {
  clip: string
  thumbnail: string
  context: string
  speedup: number
  partial: boolean
  // The same clip without the box, trail and title (event detail only):
  clean?: string
  clean_status?: 'ready' | 'on_request' | 'unavailable'
  clean_reason?: string | null
}

export interface EventSummary {
  id: number
  session_id: number
  event_id: number
  label: ObjectClass
  label_source: string
  confidence: number
  review: ObjectClass | null
  start_time: string
  end_time: string
  duration_s: number
  fireball?: boolean
  media: Media | null
  name: string | null
  magnitude: number | null
  sky: {
    start: SkyPoint
    end: SkyPoint
    angular_speed_deg_s: number
    path?: [number, number][] // (alt, az) samples
  } | null
  simulated?: boolean // from one of SkySeer's simulators, not a real sky
}

export interface EventDetail extends EventSummary {
  first_frame: number
  last_frame: number
  features: Record<string, number>
  scores: Record<string, number>
  track: { frame: number; x: number; y: number; bbox: number[] }[]
  identification?: {
    norad_id: number
    name: string
    confidence: number
    cross_track_deg: number
    clock_offset_s: number
    range_km: number
    altitude_km: number
    speed_km_s: number
    sunlit: boolean
    runner_up?: string | null
  } | null
  shower?: { code: string; name: string; radiant_offset_deg: number; confidence: number } | 'sporadic'
  aircraft?: {
    icao24: string
    callsign: string | null
    name: string
    confidence: number
    cross_track_deg: number
    clock_offset_s: number | null // null: moves too slowly to time
    range_km: number
    altitude_m: number
    speed_m_s: number
    heading_deg: number
    vertical_m_s: number
    geometric_altitude: boolean
    broadcast_spacing_s: number
    source: 'opensky' | 'readsb' | 'log'
    country: string | null
    runner_up?: string | null
  } | null
  photometry?: { peak_magnitude: number | null; saturated: boolean; light_curve: (number | null)[] }
  periodicity?: { period_s: number; amplitude: number; false_alarm: number }
}

export interface Report {
  session: Pick<Session, 'id' | 'name' | 'title' | 'kind' | 'source' | 'started'>
  start: string | null
  end: string | null
  duration_s: number
  counts: Record<'Satellite' | 'Meteor' | 'Plane', number>
  rejected: number
  reviewed: number
  identified_satellites: number
  identified_aircraft: number
  timeline: { bin_minutes: number; bins: ({ start: string } & Record<string, number>)[] }
  satellites: { id: number; time: string; name: string | null; magnitude: number | null }[]
  meteors: { id: number; time: string; shower: string | null; magnitude: number | null; fireball: boolean }[]
  aircraft: { id: number; time: string; name: string | null; altitude_m: number | null }[]
  showers: Record<string, number>
  brightest_meteor: Report['meteors'][number] | null
  observer: Session['meta']['observer'] | null
  pointing: Session['meta']['pointing'] | null
  time_source?: 'given' | 'video file' | 'unknown' | null
  disturbances: [number, number][]
}

export interface Health {
  version: string
  uptime_s: number
  cpu_percent: number
  memory_percent: number
  temperature_c: number | null
  disk_free_gb: number
  disk_used_percent: number
  live: LiveStatus | null
}

export interface LiveStatus {
  running: boolean
  camera: string
  fps: number // frames processed per second
  camera_fps?: number // frames the camera delivers per second
  frames: number
  session_id: number | null
  calibrated: boolean
  pointing?: { alt: number; az: number; roll: number } | null
  observer?: { lat: number; lon: number } | null
  started?: string | null
  counts?: Record<string, number>
  error?: string | null
  // Present when the unattended station (`skyseer station`) is running:
  mode?: 'observing' | 'waiting'
  sun_altitude?: number | null
  dark_below?: number
  next_change?: string | null
  nights?: number
  station_error?: string | null
  location?: { lat: number; lon: number } | null
  gps?: {
    source: string
    fix: { mode: number; satellites: number; clock_offset_s: number | null } | null
    using_cached_location: boolean
    error: string | null
  } | null
  clock_warning?: string | null
  aircraft?: {
    source: string
    authenticated: boolean
    last_update: string | null
    aircraft: number
    requests_today: number
    error: string | null
  } | null
}

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} for ${url}`)
  return (await r.json()) as T
}

export interface EventQuery {
  session_id?: number
  label?: ObjectClass[]
  reviewed?: boolean
  include_junk?: boolean
  limit?: number
  offset?: number
}

export const api = {
  health: () => get<Health>('/api/health'),
  sessions: () => get<Session[]>('/api/sessions'),
  report: (id: number) => get<Report>(`/api/sessions/${id}/report`),
  event: (id: number) => get<EventDetail>(`/api/events/${id}`),
  events: (q: EventQuery = {}) => {
    const p = new URLSearchParams()
    if (q.session_id !== undefined) p.set('session_id', String(q.session_id))
    for (const l of q.label ?? []) p.append('label', l)
    if (q.reviewed !== undefined) p.set('reviewed', String(q.reviewed))
    if (q.include_junk) p.set('include_junk', 'true')
    p.set('limit', String(q.limit ?? 200))
    if (q.offset) p.set('offset', String(q.offset))
    return get<EventSummary[]>(`/api/events?${p}`)
  },
  review: async (id: number, label: ObjectClass | null) => {
    const r = await fetch(`/api/events/${id}/review`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    })
    if (!r.ok) throw new Error(`Review failed: ${r.status}`)
    return (await r.json()) as { id: number; review: ObjectClass | null }
  },
  /** Make the event's clip without markings (from the original recording) if needed. */
  cleanClip: async (id: number) => {
    const r = await fetch(`/api/events/${id}/clean-clip`, { method: 'POST' })
    if (!r.ok) {
      const body = (await r.json().catch(() => null)) as { detail?: string } | null
      throw new Error(body?.detail ?? `Could not make the clip: ${r.status}`)
    }
    return (await r.json()) as Media
  },
  renameSession: async (id: number, title: string) => {
    const r = await fetch(`/api/sessions/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    })
    if (!r.ok) throw new Error(((await r.json().catch(() => null)) as { detail?: string } | null)?.detail ?? `Rename failed: ${r.status}`)
    return (await r.json()) as Session
  },
  sessionFiles: (id: number) => get<SessionFiles>(`/api/sessions/${id}/files`),
  removeSession: async (id: number, files: boolean) => {
    const r = await fetch(`/api/sessions/${id}?files=${files}`, { method: 'DELETE' })
    if (!r.ok) throw new Error(((await r.json().catch(() => null)) as { detail?: string } | null)?.detail ?? `Remove failed: ${r.status}`)
    return (await r.json()) as { deleted: boolean; freed_mb: number }
  },
  exportUrl: (id: number, format: 'csv' | 'json') => `/api/sessions/${id}/export.${format}`,
}

// Formatting helpers shared by the pages.
export const utcTime = (iso: string) => iso.slice(11, 19)
export const utcDate = (iso: string) => iso.slice(0, 10)
export const mag = (m: number | null | undefined) =>
  m === null || m === undefined ? '—' : `${m >= 0 ? '+' : ''}${m.toFixed(1)}`
export function duration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`
  const m = Math.floor(seconds / 60)
  if (m < 60) return `${m} min ${Math.round(seconds % 60)} s`
  return `${Math.floor(m / 60)} h ${m % 60} min`
}
