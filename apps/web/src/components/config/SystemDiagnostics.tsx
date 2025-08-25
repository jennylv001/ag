/**
 * System Diagnostics Component - M4 Implementation
 * Shows system diagnostics including IN_DOCKER, IS_IN_EVALS detection
 */

import React from 'react';
import {
  Card,
  CardHeader,
  Text,
  Badge,
  Button,
  ProgressBar,
  InfoLabel,
  mergeClasses,
  makeStyles,
  tokens,
  MessageBar,
  MessageBarTitle,
  MessageBarBody,
  DataGrid,
  DataGridHeader,
  DataGridRow,
  DataGridHeaderCell,
  DataGridCell,
  DataGridBody,
  TableColumnDefinition,
  createTableColumn,
} from '@fluentui/react-components';
import { InfoRegular, PlayRegular, StorageRegular, GlobeRegular } from '@fluentui/react-icons';
import { ConfigDiagnostics, ConfigValidationError } from '../../types/config';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalL,
  },
  diagnosticsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: tokens.spacingVerticalM,
  },
  metricCard: {
    height: '100%',
  },
  metricHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalS,
  },
  metricIcon: {
    fontSize: '20px',
  },
  metricValue: {
    fontSize: tokens.fontSizeBase600,
    fontWeight: tokens.fontWeightSemibold,
    marginTop: tokens.spacingVerticalS,
  },
  statusBadge: {
    marginLeft: 'auto',
  },
  progressContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXS,
    marginTop: tokens.spacingVerticalS,
  },
  progressLabel: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: tokens.fontSizeBase200,
  },
  envSection: {
    marginTop: tokens.spacingVerticalL,
  },
  envGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: tokens.spacingVerticalS,
    marginTop: tokens.spacingVerticalM,
  },
  envFlag: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: tokens.spacingVerticalS,
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusMedium,
  },
});

interface SystemDiagnosticsProps {
  diagnostics: ConfigDiagnostics;
  onRefresh: () => Promise<void>;
  isLoading: boolean;
  className?: string;
}

interface EnvironmentFlag {
  name: string;
  value: boolean;
  description: string;
  importance: 'critical' | 'important' | 'info';
}

const SystemDiagnostics: React.FC<SystemDiagnosticsProps> = ({
  diagnostics,
  onRefresh,
  isLoading,
  className,
}) => {
  const styles = useStyles();

  // Environment flags detection
  const environmentFlags: EnvironmentFlag[] = [
    {
      name: 'IN_DOCKER',
      value: diagnostics.inDocker,
      description: 'Running inside Docker container',
      importance: 'important',
    },
    {
      name: 'IS_IN_EVALS',
      value: diagnostics.isInEvals,
      description: 'Running in evaluation mode',
      importance: 'critical',
    },
    {
      name: 'DEBUG_MODE',
      value: diagnostics.environmentType === 'development',
      description: 'Development/debug mode active',
      importance: 'info',
    },
  ];

  const getStatusColor = (value: boolean, type: 'success' | 'error' | 'warning' = 'success') => {
    if (!value) return 'subtle';
    switch (type) {
      case 'error': return 'danger';
      case 'warning': return 'warning';
      default: return 'success';
    }
  };

  const formatMemory = (mb: number): string => {
    if (mb > 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb} MB`;
  };

  const formatDiskSpace = (mb: number): string => {
    if (mb > 1024 * 1024) {
      return `${(mb / (1024 * 1024)).toFixed(1)} TB`;
    }
    if (mb > 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb} MB`;
  };

  // Error table columns
  const errorColumns: TableColumnDefinition<ConfigValidationError>[] = [
    createTableColumn<ConfigValidationError>({
      columnId: 'severity',
      compare: (a, b) => a.severity.localeCompare(b.severity),
      renderHeaderCell: () => 'Severity',
      renderCell: (item) => (
        <Badge color={getStatusColor(true, item.severity === 'error' ? 'error' : item.severity === 'warning' ? 'warning' : 'success')}>
          {item.severity}
        </Badge>
      ),
    }),
    createTableColumn<ConfigValidationError>({
      columnId: 'path',
      compare: (a, b) => a.path.localeCompare(b.path),
      renderHeaderCell: () => 'Configuration Path',
      renderCell: (item) => (
        <Text weight="semibold">{item.path}</Text>
      ),
    }),
    createTableColumn<ConfigValidationError>({
      columnId: 'message',
      renderHeaderCell: () => 'Message',
      renderCell: (item) => (
        <Text>{item.message}</Text>
      ),
    }),
    createTableColumn<ConfigValidationError>({
      columnId: 'source',
      compare: (a, b) => a.source.localeCompare(b.source),
      renderHeaderCell: () => 'Source',
      renderCell: (item) => (
        <Badge appearance="outline">{item.source}</Badge>
      ),
    }),
  ];

  return (
    <div className={mergeClasses(styles.container, className)}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <Text size={500} weight="semibold">System Diagnostics</Text>
          <Text size={300} style={{ marginTop: tokens.spacingVerticalXS }}>
            System health, environment detection, and configuration validation
          </Text>
        </div>
        <Button
          appearance="secondary"
          icon={<PlayRegular />}
          onClick={onRefresh}
          disabled={isLoading}
        >
          {isLoading ? 'Running...' : 'Run Diagnostics'}
        </Button>
      </div>

      {/* Environment Flags */}
      <div className={styles.envSection}>
        <InfoLabel info="Critical environment flags that affect system behavior">
          <Text size={400} weight="semibold">Environment Detection</Text>
        </InfoLabel>
        <div className={styles.envGrid}>
          {environmentFlags.map(flag => (
            <div key={flag.name} className={styles.envFlag}>
              <div>
                <Text size={300} weight="semibold">{flag.name}</Text>
                <Text size={200}>{flag.description}</Text>
              </div>
              <Badge 
                color={getStatusColor(flag.value, flag.importance === 'critical' ? 'error' : 'success')}
                appearance={flag.value ? 'filled' : 'outline'}
              >
                {flag.value ? 'True' : 'False'}
              </Badge>
            </div>
          ))}
        </div>
      </div>

      {/* System Metrics */}
      <div className={styles.diagnosticsGrid}>
        {/* Platform Information */}
        <Card className={styles.metricCard}>
          <CardHeader
            header={
              <div className={styles.metricHeader}>
                <InfoRegular className={styles.metricIcon} />
                <Text weight="semibold">Platform</Text>
                <Badge 
                  appearance="filled" 
                  color="informative"
                  className={styles.statusBadge}
                >
                  {diagnostics.environmentType}
                </Badge>
              </div>
            }
          />
          <div>
            <Text className={styles.metricValue}>
              {diagnostics.platform} ({diagnostics.architecture})
            </Text>
            <Text size={300}>Node.js {diagnostics.nodeVersion}</Text>
          </div>
        </Card>

        {/* Memory Usage */}
        <Card className={styles.metricCard}>
          <CardHeader
            header={
              <div className={styles.metricHeader}>
                <StorageRegular className={styles.metricIcon} />
                <Text weight="semibold">Memory</Text>
                <Badge 
                  appearance="filled" 
                  color={diagnostics.availableMemory > 4096 ? 'success' : 'warning'}
                  className={styles.statusBadge}
                >
                  {diagnostics.availableMemory > 4096 ? 'OK' : 'Low'}
                </Badge>
              </div>
            }
          />
          <div>
            <Text className={styles.metricValue}>
              {formatMemory(diagnostics.availableMemory)}
            </Text>
            <div className={styles.progressContainer}>
              <div className={styles.progressLabel}>
                <Text size={200}>Available</Text>
                <Text size={200}>{(diagnostics.availableMemory / 16384 * 100).toFixed(1)}%</Text>
              </div>
              <ProgressBar 
                value={diagnostics.availableMemory / 16384} 
                color={diagnostics.availableMemory > 4096 ? 'success' : 'warning'}
              />
            </div>
          </div>
        </Card>

        {/* Disk Space */}
        <Card className={styles.metricCard}>
          <CardHeader
            header={
              <div className={styles.metricHeader}>
                <StorageRegular className={styles.metricIcon} />
                <Text weight="semibold">Disk Space</Text>
                <Badge 
                  appearance="filled" 
                  color={diagnostics.diskSpace > 10000 ? 'success' : 'warning'}
                  className={styles.statusBadge}
                >
                  {diagnostics.diskSpace > 10000 ? 'OK' : 'Low'}
                </Badge>
              </div>
            }
          />
          <div>
            <Text className={styles.metricValue}>
              {formatDiskSpace(diagnostics.diskSpace)}
            </Text>
            <Text size={300}>Available</Text>
          </div>
        </Card>

        {/* Network Access */}
        <Card className={styles.metricCard}>
          <CardHeader
            header={
              <div className={styles.metricHeader}>
                <GlobeRegular className={styles.metricIcon} />
                <Text weight="semibold">Network</Text>
                <Badge 
                  appearance="filled" 
                  color={diagnostics.networkAccess ? 'success' : 'danger'}
                  className={styles.statusBadge}
                >
                  {diagnostics.networkAccess ? 'Connected' : 'Offline'}
                </Badge>
              </div>
            }
          />
          <div>
            <Text className={styles.metricValue}>
              {diagnostics.networkAccess ? 'Online' : 'Offline'}
            </Text>
            <Text size={300}>Internet connectivity</Text>
          </div>
        </Card>
      </div>

      {/* Issues */}
      {(diagnostics.permissionIssues.length > 0 || 
        diagnostics.missingDependencies.length > 0 || 
        diagnostics.configErrors.length > 0) && (
        <div>
          <Text size={400} weight="semibold">Issues Detected</Text>
          
          {/* Permission Issues */}
          {diagnostics.permissionIssues.length > 0 && (
            <MessageBar intent="warning" style={{ marginTop: tokens.spacingVerticalM }}>
              <MessageBarTitle>Permission Issues</MessageBarTitle>
              <MessageBarBody>
                {diagnostics.permissionIssues.join(', ')}
              </MessageBarBody>
            </MessageBar>
          )}

          {/* Missing Dependencies */}
          {diagnostics.missingDependencies.length > 0 && (
            <MessageBar intent="error" style={{ marginTop: tokens.spacingVerticalM }}>
              <MessageBarTitle>Missing Dependencies</MessageBarTitle>
              <MessageBarBody>
                {diagnostics.missingDependencies.join(', ')}
              </MessageBarBody>
            </MessageBar>
          )}

          {/* Configuration Errors */}
          {diagnostics.configErrors.length > 0 && (
            <div style={{ marginTop: tokens.spacingVerticalM }}>
              <Text size={400} weight="semibold">Configuration Issues</Text>
              <DataGrid
                items={diagnostics.configErrors}
                columns={errorColumns}
                sortable
                style={{ marginTop: tokens.spacingVerticalM }}
              >
                <DataGridHeader>
                  <DataGridRow>
                    {({ renderHeaderCell }) => (
                      <DataGridHeaderCell>{renderHeaderCell()}</DataGridHeaderCell>
                    )}
                  </DataGridRow>
                </DataGridHeader>
                <DataGridBody<ConfigValidationError>>
                  {({ item, rowId }) => (
                    <DataGridRow<ConfigValidationError> key={rowId}>
                      {({ renderCell }) => (
                        <DataGridCell>{renderCell(item)}</DataGridCell>
                      )}
                    </DataGridRow>
                  )}
                </DataGridBody>
              </DataGrid>
            </div>
          )}
        </div>
      )}

      {/* Success State */}
      {diagnostics.permissionIssues.length === 0 && 
       diagnostics.missingDependencies.length === 0 && 
       diagnostics.configErrors.length === 0 && (
        <MessageBar intent="success">
          <MessageBarTitle>System Healthy</MessageBarTitle>
          <MessageBarBody>
            All system checks passed. No issues detected.
          </MessageBarBody>
        </MessageBar>
      )}
    </div>
  );
};

export default SystemDiagnostics;
