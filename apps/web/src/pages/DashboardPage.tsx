import { 
  Title1,
  Title2,
  Card,
  CardHeader,
  Body1,
  Button,
  makeStyles,
  tokens,
} from '@fluentui/react-components'
import { Play24Filled, Add24Regular, Link24Regular } from '@fluentui/react-icons'

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalL,
  },
  quickActions: {
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    marginTop: tokens.spacingVerticalM,
  },
  statusGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: tokens.spacingVerticalM,
  },
  statusCard: {
    padding: tokens.spacingVerticalM,
  },
})

export function DashboardPage() {
  const styles = useStyles()

  return (
    <div className={styles.container}>
      <div>
        <Title1>Dashboard</Title1>
        <Body1>Monitor and control your browser automation sessions</Body1>
        
        <div className={styles.quickActions}>
          <Button 
            appearance="primary" 
            icon={<Play24Filled />}
            size="large"
          >
            Run Agent
          </Button>
          <Button 
            appearance="secondary" 
            icon={<Add24Regular />}
            size="large"
          >
            New Profile
          </Button>
          <Button 
            appearance="secondary" 
            icon={<Link24Regular />}
            size="large"
          >
            Attach to Browser
          </Button>
        </div>
      </div>

      <div className={styles.statusGrid}>
        <Card className={styles.statusCard}>
          <CardHeader header={<Title2>Live Job Tracker</Title2>} />
          <Body1>No active jobs</Body1>
        </Card>

        <Card className={styles.statusCard}>
          <CardHeader header={<Title2>Recent Sessions</Title2>} />
          <Body1>No recent sessions</Body1>
        </Card>

        <Card className={styles.statusCard}>
          <CardHeader header={<Title2>Health & Alerts</Title2>} />
          <Body1>All systems operational</Body1>
        </Card>
      </div>
    </div>
  )
}
