import { 
  makeStyles, 
  tokens, 
  Button,
  Text,
  Badge,
  Dropdown,
  Option,
} from '@fluentui/react-components'
import { 
  Settings24Regular,
  Play24Filled,
} from '@fluentui/react-icons'

const useStyles = makeStyles({
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: `${tokens.spacingVerticalM} ${tokens.spacingHorizontalL}`,
    backgroundColor: tokens.colorNeutralBackground1,
    borderBottom: `1px solid ${tokens.colorNeutralStroke2}`,
    height: '60px',
    boxSizing: 'border-box',
  },
  left: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalL,
  },
  right: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalM,
  },
  profileSelector: {
    minWidth: '200px',
  },
})

export function AppHeader() {
  const styles = useStyles()

  return (
    <header className={styles.header}>
      <div className={styles.left}>
  <Text size={700} weight="semibold">Browser Use Control Center</Text>
        <Badge appearance="outline" color="informative">
          v0.1.0
        </Badge>
      </div>
      <div className={styles.right}>
        <Dropdown 
          className={styles.profileSelector}
          placeholder="Select Profile"
        >
          <Option key="default">Default Profile</Option>
          <Option key="stealth">Stealth Profile</Option>
        </Dropdown>
        <Button appearance="primary" icon={<Play24Filled />}>
          Run Agent
        </Button>
        <Button 
          appearance="secondary" 
          icon={<Settings24Regular />}
          aria-label="Settings"
        />
      </div>
    </header>
  )
}
