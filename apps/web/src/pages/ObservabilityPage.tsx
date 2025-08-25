/**
 * Observability Page - M5 Implementation
 * Main observability interface with live logs, debug traces, and export capabilities
 */

import React from 'react';
import {
  TabList,
  Tab,
  Text,
  Button,
  Badge,
  MessageBar,
  MessageBarTitle,
  MessageBarBody,
  mergeClasses,
  makeStyles,
  tokens,
  Card,
  CardHeader,
} from '@fluentui/react-components';
import {
  Pause24Regular,
  Play24Regular,
  ErrorCircle24Regular,
  CheckmarkCircle24Regular,
  Clock24Regular,
  Save24Regular,
  Document24Regular,
} from '@fluentui/react-icons';
import { useObservabilityStore } from '@/stores/observabilityStore';
import LiveLogStream from '@/components/observability/LiveLogStream';
import DebugTraceViewer from '@/components/observability/DebugTraceViewer';
import ExportManager from '@/components/observability/ExportManager';
import { LogEntry, DebugTrace as DebugTraceType, ExportResult } from '@/types/observability';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    padding: tokens.spacingVerticalL,
    gap: tokens.spacingVerticalL,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statusCards: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: tokens.spacingVerticalM,
    marginBottom: tokens.spacingVerticalL,
  },
  statusCard: {
    textAlign: 'center',
  },
  statusValue: {
    fontSize: tokens.fontSizeBase600,
    fontWeight: tokens.fontWeightSemibold,
    marginTop: tokens.spacingVerticalXS,
  },
  statusIcon: {
    fontSize: '24px',
    marginBottom: tokens.spacingVerticalXS,
  },
  streamingIcon: {
    color: tokens.colorPaletteGreenForeground1,
  },
  errorIcon: {
    color: tokens.colorPaletteRedForeground1,
  },
  infoIcon: {
  color: tokens.colorPaletteBlueForeground2,
  },
  tabContent: {
    flex: 1,
    overflow: 'hidden',
    paddingTop: tokens.spacingVerticalM,
  },
  connectionStatus: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalS,
  },
  connectionDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: tokens.colorPaletteGreenBackground3,
  },
  disconnectedDot: {
    backgroundColor: tokens.colorPaletteRedBackground3,
  },
  quickActions: {
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    alignItems: 'center',
  },
});

const ObservabilityPage: React.FC = () => {
  const styles = useStyles();
  const {
    // State
    logs,
    isStreaming,
    streamConnection,
    logFilter,
    traces,
    selectedTrace,
    selectedEvent,
    activeExports,
    error,
    selectedTab,
    autoScroll,
    
    // Actions
    startLogStream,
    stopLogStream,
    updateLogFilter,
    clearLogs,
    loadTraces,
    selectTrace,
    selectEvent,
    exportLogs,
    exportTrace,
    downloadExport,
    deleteExport,
    setSelectedTab,
  toggleAutoScroll,
  } = useObservabilityStore() as any;

  // Load initial data
  React.useEffect(() => {
    loadTraces();
  }, [loadTraces]);

  // Calculate status metrics
  const statusMetrics = React.useMemo(() => {
  const totalLogs = (logs as LogEntry[]).length;
  const errorLogs = (logs as LogEntry[]).filter((log: LogEntry) => log.level === 'error' || log.level === 'fatal').length;
  const completedTraces = (traces as DebugTraceType[]).filter((trace: DebugTraceType) => trace.status === 'completed').length;
  const runningTraces = (traces as DebugTraceType[]).filter((trace: DebugTraceType) => trace.status === 'running').length;
  const pendingExports = (activeExports as ExportResult[]).filter((exp: ExportResult) => exp.status === 'pending').length;
    
    return {
      totalLogs,
      errorLogs,
      completedTraces,
      runningTraces,
      pendingExports,
      errorRate: totalLogs > 0 ? (errorLogs / totalLogs * 100).toFixed(1) : '0',
    };
  }, [logs, traces, activeExports]);

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <Text size={600} weight="semibold">Observability</Text>
          <div className={styles.connectionStatus}>
            <div className={mergeClasses(
              styles.connectionDot,
              !isStreaming && styles.disconnectedDot
            )} />
            <Text size={300}>
              {isStreaming ? 'Live monitoring active' : 'Monitoring paused'}
              {streamConnection && ` • ${streamConnection.messageCount} messages`}
            </Text>
          </div>
        </div>
        
        <div className={styles.quickActions}>
          <Button
            appearance={isStreaming ? "secondary" : "primary"}
            icon={isStreaming ? <Pause24Regular /> : <Play24Regular />}
            onClick={isStreaming ? stopLogStream : startLogStream}
          >
            {isStreaming ? 'Pause' : 'Start'} Monitoring
          </Button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
  <MessageBar intent="error">
          <MessageBarTitle>Observability Error</MessageBarTitle>
          <MessageBarBody>{error}</MessageBarBody>
        </MessageBar>
      )}

      {/* Status Overview */}
      <div className={styles.statusCards}>
        <Card className={styles.statusCard}>
          <CardHeader
            header={
              <div>
                <div className={mergeClasses(styles.statusIcon, styles.infoIcon)}>
                  <Document24Regular />
                </div>
                <Text size={300}>Total Logs</Text>
                <Text className={styles.statusValue}>{statusMetrics.totalLogs.toLocaleString()}</Text>
              </div>
            }
          />
        </Card>

        <Card className={styles.statusCard}>
          <CardHeader
            header={
              <div>
                <div className={mergeClasses(styles.statusIcon, styles.errorIcon)}>
                  <ErrorCircle24Regular />
                </div>
                <Text size={300}>Error Rate</Text>
                <Text className={styles.statusValue}>{statusMetrics.errorRate}%</Text>
                <Text size={200}>{statusMetrics.errorLogs} errors</Text>
              </div>
            }
          />
        </Card>

        <Card className={styles.statusCard}>
          <CardHeader
            header={
              <div>
                <div className={mergeClasses(styles.statusIcon, styles.streamingIcon)}>
                  <CheckmarkCircle24Regular />
                </div>
                <Text size={300}>Completed Traces</Text>
                <Text className={styles.statusValue}>{statusMetrics.completedTraces}</Text>
              </div>
            }
          />
        </Card>

        <Card className={styles.statusCard}>
          <CardHeader
            header={
              <div>
                <div className={mergeClasses(styles.statusIcon, styles.infoIcon)}>
                  <Clock24Regular />
                </div>
                <Text size={300}>Running Traces</Text>
                <Text className={styles.statusValue}>{statusMetrics.runningTraces}</Text>
              </div>
            }
          />
        </Card>

        <Card className={styles.statusCard}>
          <CardHeader
            header={
              <div>
                <div className={mergeClasses(styles.statusIcon, styles.infoIcon)}>
                  <Save24Regular />
                </div>
                <Text size={300}>Active Exports</Text>
                <Text className={styles.statusValue}>{activeExports.length}</Text>
                {statusMetrics.pendingExports > 0 && (
                  <Text size={200}>{statusMetrics.pendingExports} pending</Text>
                )}
              </div>
            }
          />
        </Card>
      </div>

      {/* Main Content Tabs */}
      <TabList
        selectedValue={selectedTab}
        onTabSelect={(_, data) => setSelectedTab(data.value as any)}
      >
        <Tab value="logs">
          Live Log Stream
          {isStreaming && <Badge appearance="filled" color="success" size="small">Live</Badge>}
        </Tab>
        <Tab value="traces">
          Debug Traces
          <Badge appearance="outline" size="small">{traces.length}</Badge>
        </Tab>
        <Tab value="exports">
          Export Manager
          {statusMetrics.pendingExports > 0 && (
            <Badge appearance="filled" color="warning" size="small">
              {statusMetrics.pendingExports}
            </Badge>
          )}
        </Tab>
      </TabList>

      <div className={styles.tabContent}>
        {selectedTab === 'logs' && (
          <LiveLogStream
            logs={logs}
            isStreaming={isStreaming}
            filter={logFilter}
            autoScroll={autoScroll}
            onStartStream={startLogStream}
            onStopStream={stopLogStream}
            onUpdateFilter={updateLogFilter}
            onToggleAutoScroll={toggleAutoScroll}
            onClearLogs={clearLogs}
          />
        )}
        {selectedTab === 'traces' && (
          <DebugTraceViewer
            traces={traces}
            selectedTrace={selectedTrace}
            selectedEvent={selectedEvent}
            onSelectTrace={selectTrace}
            onSelectEvent={selectEvent}
          />
        )}
        {selectedTab === 'exports' && (
          <ExportManager
            activeExports={activeExports}
            onExportLogs={exportLogs}
            onExportTrace={exportTrace}
            onDownloadExport={downloadExport}
            onDeleteExport={deleteExport}
          />
        )}
      </div>
    </div>
  );
};

export default ObservabilityPage;
