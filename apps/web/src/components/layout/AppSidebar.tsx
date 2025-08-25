import { 
  makeStyles, 
  tokens,
  Tree,
  TreeItem,
  TreeItemLayout,
} from '@fluentui/react-components'
import {
  Home24Regular,
  Link24Regular,
  PeopleTeam24Regular,
  Settings24Regular,
  Bug24Regular,
  Wrench24Regular,
} from '@fluentui/react-icons'

const useStyles = makeStyles({
  sidebar: {
    width: '240px',
    backgroundColor: tokens.colorNeutralBackground2,
    borderRight: `1px solid ${tokens.colorNeutralStroke2}`,
    padding: tokens.spacingVerticalM,
    height: '100%',
    overflowY: 'auto',
  },
  nav: {
    '& .fui-TreeItem': {
      marginBottom: tokens.spacingVerticalXS,
    },
  },
})

const navigationItems = [
  { id: 'dashboard', label: 'Dashboard', icon: <Home24Regular />, path: '/' },
  { id: 'sessions', label: 'Sessions', icon: <Link24Regular />, path: '/sessions' },
  { id: 'profiles', label: 'Profiles', icon: <PeopleTeam24Regular />, 
    children: [
      { id: 'profiles-list', label: 'Profile List', path: '/profiles' },
      { id: 'profiles-new', label: 'Create New', path: '/profiles/new' },
    ]
  },
  { id: 'config', label: 'Config Manager', icon: <Settings24Regular />, path: '/config' },
  { id: 'observability', label: 'Observability', icon: <Bug24Regular />, path: '/observability' },
  { id: 'tools', label: 'Tools (MCP)', icon: <Wrench24Regular /> },
]

interface AppSidebarProps {
  activeRoute?: string
  onNavigate?: (route: string) => void
}

export function AppSidebar({ activeRoute = 'dashboard', onNavigate }: AppSidebarProps) {
  const styles = useStyles()

  const handleNavigation = (item: any) => {
    if (item.path) {
      // In a real implementation, would use router navigation
      window.location.hash = item.path;
    }
    onNavigate?.(item.id);
  };

  return (
    <nav className={styles.sidebar}>
      <Tree className={styles.nav} aria-label="Navigation">
        {navigationItems.map((item) => (
          <TreeItem 
            key={item.id}
            itemType="leaf"
            onClick={() => handleNavigation(item)}
            aria-current={activeRoute === item.id ? 'page' : undefined}
          >
            <TreeItemLayout 
              iconBefore={item.icon}
            >
              {item.label}
            </TreeItemLayout>
          </TreeItem>
        ))}
      </Tree>
    </nav>
  )
}
