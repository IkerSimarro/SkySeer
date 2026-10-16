import { useCallback, useEffect, useState } from 'react'

/** The current hash route, e.g. "#/events?session=3" -> { parts: ['events'], query }. */
export interface Route {
  parts: string[]
  query: URLSearchParams
}

function parse(): Route {
  const hash = window.location.hash.replace(/^#\/?/, '')
  const [path, qs] = hash.split('?')
  return { parts: path ? path.split('/').filter(Boolean) : [], query: new URLSearchParams(qs) }
}

export function useRoute(): Route {
  const [route, setRoute] = useState(parse)
  useEffect(() => {
    const onChange = () => setRoute(parse())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

export function href(path: string, query?: Record<string, string | number | undefined>): string {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(query ?? {})) if (v !== undefined && v !== '') p.set(k, String(v))
  const qs = p.toString()
  return `#/${path}${qs ? `?${qs}` : ''}`
}

/** Go to a dashboard route (a hash link from `href`). */
export function navigate(hashHref: string): void {
  window.location.assign(hashHref)
}

export interface Async<T> {
  data: T | undefined
  error: string | undefined
  loading: boolean
  reload: () => void
}

/** Load data from the API; re-runs when `deps` change, or every `pollMs` if given. */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[], pollMs?: number): Async<T> {
  const [data, setData] = useState<T>()
  const [error, setError] = useState<string>()
  const [loading, setLoading] = useState(true)
  const [tick, setTick] = useState(0)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const run = useCallback(load, deps)
  useEffect(() => {
    let alive = true
    setLoading(true)
    run()
      .then((d) => alive && (setData(d), setError(undefined)))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => alive && setLoading(false))
    return () => {
      alive = false
    }
  }, [run, tick])
  useEffect(() => {
    if (!pollMs) return
    const id = window.setInterval(() => setTick((t) => t + 1), pollMs)
    return () => window.clearInterval(id)
  }, [pollMs])
  return { data, error, loading, reload: () => setTick((t) => t + 1) }
}

/** The current UTC time, updated every second. */
export function useUtcNow(): Date {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000)
    return () => window.clearInterval(id)
  }, [])
  return now
}
