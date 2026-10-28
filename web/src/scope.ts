import { useCallback, useEffect } from 'react'
import { api, isTest, type Session } from './api'
import { href, navigate, type Route, useAsync, useStoredState } from './hooks'

/** Which session the dashboard is showing, shared by every page. */
export interface Scope {
  /** The session every page shows; undefined = all sessions together (or none yet). */
  session: Session | undefined
  all: Session[] // every session, the one added last first
  visible: Session[] // without test sessions (simulations), unless they are shown
  loaded: boolean
  error: string | undefined
  showTests: boolean
  setShowTests: (show: boolean) => void
  choose: (id: number | 'all') => void
  reload: () => void
}

/** The session is chosen in the top bar (remembered in this browser) or by a link with
 * ``?session=ID``. By default it is the one added last, leaving out test sessions unless
 * they are shown. */
export function useScope(route: Route): Scope {
  const sessions = useAsync(api.sessions, [], 30_000) // new analyses appear by themselves
  const [showTests, setShowTests] = useStoredState('skyseer.showTests', false)
  const [chosen, setChosen] = useStoredState<number | 'all' | null>('skyseer.session', null)
  const [page, id] = route.parts
  const linked = route.query.get('session') ?? (page === 'report' && id ? id : null)
  useEffect(() => {
    if (linked !== null) setChosen(linked === 'all' ? 'all' : Number(linked))
  }, [linked, setChosen])

  const all = [...(sessions.data ?? [])].sort((a, b) => b.id - a.id)
  const real = all.filter((s) => !isTest(s))
  const visible = showTests ? all : real
  const choice = linked !== null ? (linked === 'all' ? 'all' : Number(linked)) : chosen
  // Showing test sessions in the lists doesn't make one of them the default.
  const fallback = real[0] ?? visible[0]
  const session = choice === 'all' ? undefined : (all.find((s) => s.id === choice) ?? fallback)

  const choose = useCallback(
    (next: number | 'all') => {
      setChosen(next)
      if (linked === null) return
      // A link chose the session; the top bar now does, so drop it from the address.
      const query = Object.fromEntries([...route.query].filter(([k]) => k !== 'session'))
      navigate(href(page === 'report' ? 'report' : route.parts.join('/'), query))
    },
    [linked, page, route, setChosen],
  )
  return {
    session,
    all,
    visible,
    loaded: sessions.data !== undefined,
    error: sessions.error,
    showTests,
    setShowTests,
    choose,
    reload: sessions.reload,
  }
}
