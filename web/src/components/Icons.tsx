// Small line icons (stroke = currentColor) for navigation and buttons.
import type { ReactNode } from 'react'

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {children}
    </svg>
  )
}

export const OverviewIcon = () => (
  <Icon><rect x="3" y="3" width="7" height="9" rx="1.5" /><rect x="14" y="3" width="7" height="5" rx="1.5" />
    <rect x="14" y="12" width="7" height="9" rx="1.5" /><rect x="3" y="16" width="7" height="5" rx="1.5" /></Icon>
)
export const EventsIcon = () => (
  <Icon><path d="M4 6h16M4 12h16M4 18h10" /><circle cx="19" cy="18" r="2" /></Icon>
)
export const SkyIcon = () => (
  <Icon><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="4.5" /><path d="M12 3v2M12 19v2M3 12h2M19 12h2" /></Icon>
)
export const ReportIcon = () => (
  <Icon><path d="M6 3h9l4 4v14H6z" /><path d="M14 3v5h5M9 13h6M9 17h6M9 9h2" /></Icon>
)
export const LiveIcon = () => (
  <Icon><circle cx="12" cy="12" r="2.5" /><path d="M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M5 5a10 10 0 0 0 0 14M19 5a10 10 0 0 1 0 14" /></Icon>
)
export const AnalyzeIcon = () => (
  <Icon><path d="M12 15V4M8 8l4-4 4 4" /><path d="M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4" /></Icon>
)
export const DownloadIcon = () => (
  <Icon><path d="M12 4v11M8 11l4 4 4-4" /><path d="M5 20h14" /></Icon>
)
export const BackIcon = () => <Icon><path d="M15 6l-6 6 6 6" /></Icon>
