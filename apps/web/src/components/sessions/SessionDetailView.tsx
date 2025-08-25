/**
 * Session Detail View - M3 Implementation
 * Split view with Launch/Connect tabs and session information
 */

import React from 'react';
import {
  Text,
  Tab,
  TabList,
  Button,
  Field,
  Input,
  Switch,
  Badge,
  Tag,
  TagGroup,
  Divider,
  Link,
  MessageBar,
  MessageBarBody,
  MessageBarTitle,
  Table,
  TableHeader,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  makeStyles,
  tokens,
} from '@fluentui/react-components';
import {
  Play24Regular,
  Stop24Regular,
  Camera24Regular,
  ArrowDownload24Regular,
  Delete24Regular,
  Circle12Regular,
  Info16Regular,
  Code24Regular,
} from '@fluentui/react-icons';
import { BrowserSession } from '../../types/session';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    gap: tokens.spacingVerticalM,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: tokens.spacingVerticalM,
    borderBottom: `1px solid ${tokens.colorNeutralStroke2}`,
  },
  headerLeft: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXS,
  },
  headerRight: {
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    alignItems: 'center',
  },
  tabContent: {
    flex: 1,
    overflow: 'auto',
    padding: tokens.spacingVerticalM,
  },
  configGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: tokens.spacingVerticalM,
  },
  configSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
  },
  connectionInfo: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
    padding: tokens.spacingVerticalM,
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusMedium,
  },
  stealthMatrix: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    gap: tokens.spacingVerticalS,
  },
  stealthCategory: {
    marginBottom: tokens.spacingVerticalM,
  },
  artifactsTable: {
    border: `1px solid ${tokens.colorNeutralStroke2}`,
    borderRadius: tokens.borderRadiusMedium,
  },
  statusIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXS,
  },
  emptyState: {
    textAlign: 'center',
    padding: tokens.spacingVerticalXL,
    color: tokens.colorNeutralForeground3,
  },
});

interface SessionDetailViewProps {
  session: BrowserSession;
  activeTab: 'launch' | 'connect' | 'pages' | 'artifacts';
  onTabChange: (tab: 'launch' | 'connect' | 'pages' | 'artifacts') => void;
  onStartSession: () => void;
  onStopSession: () => void;
  onOpenDevTools: () => void;
  onTakeScreenshot: () => void;
  onDownloadArtifact: (artifactId: string) => void;
  onDeleteArtifact: (artifactId: string) => void;
  isLoading?: boolean;
}

export const SessionDetailView: React.FC<SessionDetailViewProps> = ({
  session,
  activeTab,
  onTabChange,
  onStartSession,
  onStopSession,
  onOpenDevTools,
  onTakeScreenshot,
  onDownloadArtifact,
  onDeleteArtifact,
  isLoading = false,
}) => {
  const styles = useStyles();

  const getStatusColor = (status: BrowserSession['status']) => {
    switch (status) {
      case 'running': return tokens.colorPaletteGreenForeground1;
      case 'stopped': return tokens.colorNeutralForeground3;
      case 'launching': return tokens.colorPaletteYellowForeground1;
      case 'stopping': return tokens.colorPaletteYellowForeground1;
      case 'error': return tokens.colorPaletteRedForeground1;
      default: return tokens.colorNeutralForeground3;
    }
  };

  const getStealthFeaturesByCategory = () => {
    const categories = {
      detection: session.stealthFeatures.filter(f => f.category === 'detection'),
      fingerprint: session.stealthFeatures.filter(f => f.category === 'fingerprint'),
      behavior: session.stealthFeatures.filter(f => f.category === 'behavior'),
      network: session.stealthFeatures.filter(f => f.category === 'network'),
    };
    return categories;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <Text size={500} weight="semibold">{session.name}</Text>
          <div className={styles.statusIndicator}>
            <Circle12Regular style={{ color: getStatusColor(session.status) }} />
            <Text size={300}>{session.status.toUpperCase()}</Text>
            <Badge appearance="outline">{session.profileName}</Badge>
          </div>
        </div>
        <div className={styles.headerRight}>
          {session.status === 'running' && !session.launchConfig.headless && (
            <Button
              appearance="secondary"
              icon={<Code24Regular />}
              onClick={onOpenDevTools}
            >
              DevTools
            </Button>
          )}
          {session.status === 'running' && (
            <Button
              appearance="secondary"
              icon={<Camera24Regular />}
              onClick={onTakeScreenshot}
            >
              Screenshot
            </Button>
          )}
          {session.status === 'stopped' && (
            <Button
              appearance="primary"
              icon={<Play24Regular />}
              onClick={onStartSession}
              disabled={isLoading}
            >
              Start Session
            </Button>
          )}
          {session.status === 'running' && (
            <Button
              appearance="secondary"
              icon={<Stop24Regular />}
              onClick={onStopSession}
              disabled={isLoading}
            >
              Stop Session
            </Button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <TabList
        selectedValue={activeTab}
        onTabSelect={(_, data) => onTabChange(data.value as any)}
      >
        <Tab value="launch">Launch Config</Tab>
        <Tab value="connect">Connection Info</Tab>
        <Tab value="pages">Active Pages</Tab>
        <Tab value="artifacts">Artifacts</Tab>
      </TabList>

      {/* Tab Content */}
      <div className={styles.tabContent}>
        {activeTab === 'launch' && (
          <>
          <div className={styles.configGrid}>
            <div className={styles.configSection}>
              <Text size={400} weight="semibold">Browser Configuration</Text>
              
              <Field label="Channel">
                <Input value={session.launchConfig.channel || 'Default'} readOnly />
              </Field>
              
              <Field label="Executable Path">
                <Input 
                  value={session.launchConfig.executablePath || 'System Default'} 
                  readOnly 
                />
              </Field>
              
              <Field label="Timeout (ms)">
                <Input value={session.launchConfig.timeout.toString()} readOnly />
              </Field>
              
              <Field>
                <Switch 
                  checked={session.launchConfig.headless} 
                  label="Headless Mode" 
                  readOnly 
                />
              </Field>
              
              <Field>
                <Switch 
                  checked={session.launchConfig.devtools} 
                  label="DevTools Open" 
                  readOnly 
                />
              </Field>
            </div>

            <div className={styles.configSection}>
              <Text size={400} weight="semibold">Proxy Configuration</Text>
              
              {session.launchConfig.proxy ? (
                <>
                  <Field label="Proxy Server">
                    <Input value={session.launchConfig.proxy.server} readOnly />
                  </Field>
                  
                  <Field label="Username">
                    <Input 
                      value={session.launchConfig.proxy.username || 'None'} 
                      readOnly 
                    />
                  </Field>
                  
                  <Field label="Authentication">
                    <Badge appearance="filled" color="success">
                      {session.launchConfig.proxy.password ? 'Configured' : 'None'}
                    </Badge>
                  </Field>
                </>
              ) : (
                <MessageBar intent="info">
                  <MessageBarBody>
                    <MessageBarTitle>No Proxy Configured</MessageBarTitle>
                    Direct connection to the internet
                  </MessageBarBody>
                </MessageBar>
              )}
            </div>
          </div>

          {/* Stealth Feature Matrix */}
          <Divider style={{ margin: `${tokens.spacingVerticalL} 0` }} />
          
          <Text size={400} weight="semibold" style={{ marginBottom: tokens.spacingVerticalM }}>
            Stealth Feature Matrix
          </Text>
          
          {Object.entries(getStealthFeaturesByCategory()).map(([category, features]) => (
            <div key={category} className={styles.stealthCategory}>
              <Text size={300} weight="semibold" style={{ marginBottom: tokens.spacingVerticalS }}>
                {category.charAt(0).toUpperCase() + category.slice(1)} Features
              </Text>
              <TagGroup>
                {features.map(feature => (
                  <Tag
                    key={feature.name}
                    appearance={feature.enabled ? 'filled' : 'outline'}
                    disabled={!feature.enabled}
                  >
                    {feature.status === 'active' && (
                      <span style={{ display: 'inline-flex', alignItems: 'center', marginRight: tokens.spacingHorizontalXS }}>
                        <Circle12Regular style={{ color: tokens.colorPaletteGreenForeground1 }} />
                      </span>
                    )}
                    {feature.name}
                  </Tag>
                ))}
              </TagGroup>
            </div>
          ))}
          </>
        )}

        {activeTab === 'connect' && (
          session.connectionInfo ? (
            <div className={styles.connectionInfo}>
              <Text size={400} weight="semibold">Active Connection Details</Text>
              
              <Field label="WebSocket URL">
                <Input value={session.connectionInfo.wssUrl} readOnly />
              </Field>
              
              <Field label="Chrome DevTools Protocol URL">
                <Input value={session.connectionInfo.cdpUrl} readOnly />
              </Field>
              
              <Field label="Browser Process ID">
                <Input value={session.connectionInfo.browserPid.toString()} readOnly />
              </Field>
              
              {session.connectionInfo.debuggerUrl && (
                <Field label="Debugger URL">
                  <Link href={session.connectionInfo.debuggerUrl} target="_blank">
                    {session.connectionInfo.debuggerUrl}
                  </Link>
                </Field>
              )}
              
              <Divider />
              
              <Text size={300} weight="semibold">Runtime Metrics</Text>
              
              {session.runtime.memoryUsage && (
                <Field label="Memory Usage">
                  <Text>{Math.round(session.runtime.memoryUsage)} MB</Text>
                </Field>
              )}
              
              {session.runtime.cpuUsage && (
                <Field label="CPU Usage">
                  <Text>{Math.round(session.runtime.cpuUsage)}%</Text>
                </Field>
              )}
              
              <Field label="Active Pages">
                <Text>{session.runtime.activePages}</Text>
              </Field>
              
              <Field label="Total Requests">
                <Text>{session.runtime.totalRequests}</Text>
              </Field>
              
              <Field label="Errors">
                <Text>{session.runtime.errors}</Text>
              </Field>
            </div>
          ) : (
            <div className={styles.emptyState}>
              <Info16Regular />
              <Text>No connection information available. Session is not running.</Text>
            </div>
          )
        )}

        {activeTab === 'pages' && (
          <div className={styles.emptyState}>
            <Text>Active pages view - Coming in future update</Text>
            <Text size={200}>Will show real-time page navigation and status</Text>
          </div>
        )}

        {activeTab === 'artifacts' && (
          session.artifacts.length > 0 ? (
            <Table className={styles.artifactsTable}>
              <TableHeader>
                <TableRow>
                  <TableHeaderCell>Type</TableHeaderCell>
                  <TableHeaderCell>Name</TableHeaderCell>
                  <TableHeaderCell>Size</TableHeaderCell>
                  <TableHeaderCell>Status</TableHeaderCell>
                  <TableHeaderCell>Created</TableHeaderCell>
                  <TableHeaderCell>Actions</TableHeaderCell>
                </TableRow>
              </TableHeader>
              <TableBody>
                {session.artifacts.map(artifact => (
                  <TableRow key={artifact.id}>
                    <TableCell>
                      <Badge appearance="outline">
                        {artifact.type.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell>{artifact.name}</TableCell>
                    <TableCell>{formatFileSize(artifact.size)}</TableCell>
                    <TableCell>
                      <Badge 
                        appearance={artifact.status === 'recording' ? 'filled' : 'outline'}
                        color={artifact.status === 'recording' ? 'danger' : 'success'}
                      >
                        {artifact.status.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      {new Date(artifact.createdAt).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <div style={{ display: 'flex', gap: tokens.spacingHorizontalXS }}>
                        <Button
                          appearance="subtle"
                          icon={<ArrowDownload24Regular />}
                          size="small"
                          onClick={() => onDownloadArtifact(artifact.id)}
                          disabled={artifact.status === 'recording'}
                        />
                        <Button
                          appearance="subtle"
                          icon={<Delete24Regular />}
                          size="small"
                          onClick={() => onDeleteArtifact(artifact.id)}
                          disabled={artifact.status === 'recording'}
                        />
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className={styles.emptyState}>
              <Text>No artifacts recorded yet</Text>
              <Text size={200}>HAR files, videos, and traces will appear here when recorded</Text>
            </div>
          )
        )}
      </div>
    </div>
  );
};
