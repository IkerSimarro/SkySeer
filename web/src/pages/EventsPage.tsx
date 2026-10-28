import { useState } from 'react'
import { ALL_CLASSES, api, REAL_CLASSES, sessionTitle, type ObjectClass } from '../api'
import { EventCard, Failed, Loading } from '../components/common'
import { href, navigate, useAsync } from '../hooks'
import type { Scope } from '../scope'

const PAGE = 120

export default function EventsPage({ query, scope }: { query: URLSearchParams; scope: Scope }) {
  const session = scope.session?.id
  const realOnly = session === undefined && !scope.showTests
  const classes = (query.get('classes')?.split(',').filter(Boolean) as ObjectClass[]) ?? []
  const shown = classes.length ? classes : REAL_CLASSES
  const review = query.get('review') ?? 'all'
  const [limit, setLimit] = useState(PAGE)

  const events = useAsync(
    () =>
      api.events({
        session_id: session,
        simulated: realOnly ? false : undefined,
        label: shown,
        reviewed: review === 'all' ? undefined : review === 'reviewed',
        limit,
      }),
    [session, realOnly, shown.join(), review, limit],
  )
  const titles = new Map(scope.all.map((s) => [s.id, sessionTitle(s)]))

  const go = (changes: Record<string, string | number | undefined>) => {
    navigate(href('events', {
      classes: classes.join(','),
      review: review === 'all' ? undefined : review,
      ...changes,
    }))
  }
  const toggle = (c: ObjectClass) => {
    const next = shown.includes(c) ? shown.filter((x) => x !== c) : [...shown, c]
    go({ classes: next.join(',') })
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Events{scope.session && <span className="muted">: {sessionTitle(scope.session)}</span>}</h1>
          <div className="muted">Every detection with its trail image. Open one to watch its clip and check it.</div>
        </div>
      </div>
      <div className="toolbar">
        <div className="btn-group">
          {ALL_CLASSES.map((c) => (
            <button key={c} className={`btn small c-${c}`} aria-pressed={shown.includes(c)} onClick={() => toggle(c)}>
              {c}
            </button>
          ))}
        </div>
        <select value={review} onChange={(ev) => go({ review: ev.target.value === 'all' ? undefined : ev.target.value })}
          title="Reviewed events have been checked by a person">
          <option value="all">Reviewed or not</option>
          <option value="unreviewed">Not reviewed yet</option>
          <option value="reviewed">Reviewed</option>
        </select>
        <div className="spacer" />
        {events.data && <span className="muted">{events.data.length} shown</span>}
      </div>
      {events.error ? (
        <Failed error={events.error} />
      ) : !events.data ? (
        <Loading what="events" />
      ) : events.data.length === 0 ? (
        <div className="empty">No events match these filters.</div>
      ) : (
        <>
          <div className="card-grid">
            {events.data.map((e) => (
              <EventCard key={e.id} e={e} session={session === undefined ? titles.get(e.session_id) : undefined} />
            ))}
          </div>
          {events.data.length >= limit && (
            <div style={{ textAlign: 'center', marginTop: 18 }}>
              <button className="btn" onClick={() => setLimit(limit + PAGE)}>Show more</button>
            </div>
          )}
        </>
      )}
    </>
  )
}
