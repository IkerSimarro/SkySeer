import { api } from './api'
import {
  AnalyzeIcon,
  EventsIcon,
  LiveIcon,
  OverviewIcon,
  ReportIcon,
  SkyIcon,
} from './components/Icons'
import { href, useAsync, useRoute, useUtcNow } from './hooks'
import AnalyzePage from './pages/AnalyzePage'
import EventPage from './pages/EventPage'
import EventsPage from './pages/EventsPage'
import LivePage from './pages/LivePage'
import Overview from './pages/Overview'
import ReportPage from './pages/ReportPage'
import SkyPage from './pages/SkyPage'

const NAV = [
  { path: '', label: 'Overview', icon: OverviewIcon },
  { path: 'events', label: 'Events', icon: EventsIcon },
  { path: 'sky', label: 'Sky map', icon: SkyIcon },
  { path: 'report', label: 'Night report', icon: ReportIcon },
  { path: 'live', label: 'Live', icon: LiveIcon },
  { path: 'analyze', label: 'Analyse video', icon: AnalyzeIcon },
]

function StationStatus() {
  const health = useAsync(api.health, [], 5000)
  const h = health.data
  if (health.error) return <span className="chip"><span className="dot warn" />API offline</span>
  if (!h) return null
  const live = h.live?.running
  return (
    <>
      <span className="chip">
        <span className={`dot ${live ? 'ok' : 'off'}`} />
        {live ? `Live · ${h.live?.fps.toFixed(0)} fps` : h.live?.mode === 'waiting' ? 'Waiting for dusk' : 'Camera idle'}
      </span>
      <span className="chip">CPU <span className="num">{h.cpu_percent.toFixed(0)}%</span></span>
      {h.temperature_c !== null && (
        <span className="chip">Temp <span className="num">{h.temperature_c.toFixed(0)}°C</span></span>
      )}
      <span className="chip">Disk free <span className="num">{h.disk_free_gb.toFixed(0)} GB</span></span>
    </>
  )
}

function UtcClock() {
  const now = useUtcNow()
  return (
    <span className="mono" title="Coordinated Universal Time: the time all observations use">
      {now.toISOString().slice(0, 10)} <strong>{now.toISOString().slice(11, 19)}</strong> UTC
    </span>
  )
}

export default function App() {
  const route = useRoute()
  const [page, id] = route.parts
  let body
  switch (page) {
    case 'events':
      body = <EventsPage query={route.query} />
      break
    case 'event':
      body = <EventPage id={Number(id)} />
      break
    case 'sky':
      body = <SkyPage query={route.query} />
      break
    case 'report':
      body = <ReportPage id={id ? Number(id) : undefined} />
      break
    case 'live':
      body = <LivePage />
      break
    case 'analyze':
      body = <AnalyzePage />
      break
    default:
      body = <Overview />
  }
  const active = page === 'event' ? 'events' : (page ?? '')
  return (
    <div className="shell">
      <nav className="sidebar">
        <div className="brand">
          <img className="brand-mark" src="/favicon.svg" alt="" />
          <div>
            <div className="brand-name">SKYSEER</div>
            <div className="brand-sub">Night-sky monitoring station</div>
          </div>
        </div>
        {NAV.map(({ path, label, icon: Icon }) => (
          <a key={path} href={href(path)} className={`nav-link ${active === path ? 'active' : ''}`}>
            <Icon />
            {label}
          </a>
        ))}
        <div className="sidebar-foot faint">
          <a href="/api/docs" className="faint">API documentation</a>
        </div>
      </nav>
      <div className="main">
        <header className="topbar">
          <UtcClock />
          <div className="spacer" />
          <StationStatus />
        </header>
        <main className="content">{body}</main>
      </div>
    </div>
  )
}
