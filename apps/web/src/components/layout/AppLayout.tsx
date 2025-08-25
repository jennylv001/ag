import { makeStyles, tokens } from '@fluentui/react-components'
import { AppHeader } from './AppHeader'
import { AppSidebar } from './AppSidebar'

const useStyles = makeStyles({
  layout: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    overflow: 'hidden',
  },
  main: {
    display: 'flex',
    flex: 1,
    overflow: 'hidden',
  },
  content: {
    flex: 1,
    padding: tokens.spacingVerticalL,
    backgroundColor: tokens.colorNeutralBackground1,
    overflow: 'auto',
  },
  footer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: `${tokens.spacingVerticalS} ${tokens.spacingHorizontalL}`,
    backgroundColor: tokens.colorNeutralBackground2,
    borderTop: `1px solid ${tokens.colorNeutralStroke2}`,
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground2,
  },
})

interface AppLayoutProps {
  children: React.ReactNode
  activeRoute?: string
  onNavigate?: (route: string) => void
}

export function AppLayout({ children, activeRoute, onNavigate }: AppLayoutProps) {
  const styles = useStyles()

  return (
    <div className={styles.layout}>
      <AppHeader />
      <main className={styles.main}>
        <AppSidebar activeRoute={activeRoute} onNavigate={onNavigate} />
        <div className={styles.content}>
          {children}
        </div>
      </main>
      <footer className={styles.footer}>
        <span>Browser Use Control Center v0.1.0</span>
        <span>Status: Connected</span>
      </footer>
    </div>
  )
}
