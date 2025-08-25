/**
 * Live Log Stream Component - M5 Implementation
 * Virtualized DataGrid for high-performance log streaming with filters
 */

import React from 'react';
import {
  Text,
  Button,
  Input,
  Badge,
  Dropdown,
  Option,
  Tooltip,
  mergeClasses,
  makeStyles,
  tokens,
  DataGrid,
  DataGridHeader,
  DataGridRow,
  DataGridHeaderCell,
  DataGridCell,
  DataGridBody,
  TableColumnDefinition,
  createTableColumn,
  Dialog,
  DialogSurface,
  DialogTitle,
  DialogContent,
  DialogBody,
  DialogActions,
  Textarea,
} from '@fluentui/react-components';
import {
  Play24Regular,
  Pause24Regular,
  Delete24Regular,
  Filter24Regular,
  Eye24Regular,
  Copy24Regular,
  ArrowDown24Regular,
  Search24Regular,
} from '@fluentui/react-icons';
import { LogEntry, LogLevel, LogFilter } from '../../types/observability';

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
  },
  controls: {
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    alignItems: 'center',
  },
  filters: {
    display: 'grid',
    gridTemplateColumns: '1fr 200px 200px 1fr auto',
    gap: tokens.spacingHorizontalM,
    alignItems: 'center',
    padding: tokens.spacingVerticalM,
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusMedium,
    marginBottom: tokens.spacingVerticalM,
  },
  logGrid: {
    flex: 1,
    minHeight: '400px',
    maxHeight: '70vh',
    overflowY: 'auto',
  },
  levelBadge: {
    minWidth: '60px',
    textAlign: 'center',
  },
  sourceCell: {
    fontFamily: tokens.fontFamilyMonospace,
    fontSize: tokens.fontSizeBase200,
  },
  messageCell: {
    maxWidth: '400px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  timestampCell: {
    fontFamily: tokens.fontFamilyMonospace,
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
  contextCell: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground2,
  },
  statusIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXS,
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
  autoScrollButton: {
    position: 'relative',
  },
  autoScrollIndicator: {
    position: 'absolute',
    top: '-4px',
    right: '-4px',
    width: '12px',
    height: '12px',
    borderRadius: '50%',
    backgroundColor: tokens.colorPaletteGreenBackground3,
  },
});

interface LiveLogStreamProps {
  logs: LogEntry[];
  isStreaming: boolean;
  filter: LogFilter;
  autoScroll: boolean;
  onStartStream: () => void;
  onStopStream: () => void;
  onUpdateFilter: (filter: Partial<LogFilter>) => void;
  onToggleAutoScroll: () => void;
  onClearLogs: () => void;
  className?: string;
}

const LiveLogStream: React.FC<LiveLogStreamProps> = ({
  logs,
  isStreaming,
  filter,
  autoScroll,
  onStartStream,
  onStopStream,
  onUpdateFilter,
  onToggleAutoScroll,
  onClearLogs,
  className,
}) => {
  const styles = useStyles();
  const [selectedLog, setSelectedLog] = React.useState<LogEntry | null>(null);
  const [showDetailDialog, setShowDetailDialog] = React.useState(false);
  const gridRef = React.useRef<HTMLDivElement>(null);

  // Auto scroll to bottom when new logs arrive
  React.useEffect(() => {
    if (autoScroll && gridRef.current) {
      gridRef.current.scrollTop = gridRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  // Filter logs based on current filter
  const filteredLogs = React.useMemo(() => {
    return logs.filter(log => {
      // Level filter
      if (!filter.levels.includes(log.level)) return false;
      
      // Source filter
      if (filter.sources.length > 0 && !filter.sources.includes(log.source)) return false;
      
      // Search filter
      if (filter.searchQuery && !log.message.toLowerCase().includes(filter.searchQuery.toLowerCase())) {
        return false;
      }
      
      // Time range filter
      if (filter.timeRange.start && log.timestamp < filter.timeRange.start) return false;
      if (filter.timeRange.end && log.timestamp > filter.timeRange.end) return false;
      
      return true;
    }).slice(0, filter.maxResults);
  }, [logs, filter]);

  const getLevelColor = (level: LogLevel) => {
    const colors = {
      debug: 'subtle',
      info: 'informative',
      warn: 'warning',
      error: 'danger',
      fatal: 'severe',
    } as const;
    return colors[level];
  };

  const formatTimestamp = (date: Date): string => {
    const base = date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
    const ms = String(date.getMilliseconds()).padStart(3, '0');
    return `${base}.${ms}`;
  };

  const formatContext = (context: Record<string, any> | undefined): string => {
    if (!context) return '';
    return Object.entries(context)
      .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
      .join(' ');
  };

  // Table columns
  const columns: TableColumnDefinition<LogEntry>[] = [
    createTableColumn<LogEntry>({
      columnId: 'timestamp',
      compare: (a, b) => b.timestamp.getTime() - a.timestamp.getTime(),
      renderHeaderCell: () => 'Time',
      renderCell: (item) => (
        <Text className={styles.timestampCell}>
          {formatTimestamp(item.timestamp)}
        </Text>
      ),
    }),
    createTableColumn<LogEntry>({
      columnId: 'level',
      compare: (a, b) => a.level.localeCompare(b.level),
      renderHeaderCell: () => 'Level',
      renderCell: (item) => (
        <Badge 
          appearance="filled" 
          color={getLevelColor(item.level)}
          className={styles.levelBadge}
        >
          {item.level.toUpperCase()}
        </Badge>
      ),
    }),
    createTableColumn<LogEntry>({
      columnId: 'source',
      compare: (a, b) => a.source.localeCompare(b.source),
      renderHeaderCell: () => 'Source',
      renderCell: (item) => (
        <Text className={styles.sourceCell}>{item.source}</Text>
      ),
    }),
    createTableColumn<LogEntry>({
      columnId: 'message',
      renderHeaderCell: () => 'Message',
      renderCell: (item) => (
        <Tooltip content={item.message} relationship="label">
          <Text className={styles.messageCell}>{item.message}</Text>
        </Tooltip>
      ),
    }),
    createTableColumn<LogEntry>({
      columnId: 'context',
      renderHeaderCell: () => 'Context',
      renderCell: (item) => (
        <Text className={styles.contextCell}>
          {formatContext(item.context)}
        </Text>
      ),
    }),
    createTableColumn<LogEntry>({
      columnId: 'actions',
      renderHeaderCell: () => 'Actions',
      renderCell: (item) => (
        <div style={{ display: 'flex', gap: tokens.spacingHorizontalXS }}>
          <Button
            appearance="subtle"
            size="small"
            icon={<Eye24Regular />}
            onClick={() => {
              setSelectedLog(item);
              setShowDetailDialog(true);
            }}
          />
          <Button
            appearance="subtle"
            size="small"
            icon={<Copy24Regular />}
            onClick={() => {
              navigator.clipboard.writeText(JSON.stringify(item, null, 2));
            }}
          />
        </div>
      ),
    }),
  ];

  const logLevels: LogLevel[] = ['debug', 'info', 'warn', 'error', 'fatal'];
  const availableSources = Array.from(new Set(logs.map(log => log.source)));

  return (
    <div className={mergeClasses(styles.container, className)}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <Text size={500} weight="semibold">Live Log Stream</Text>
          <div className={styles.statusIndicator}>
            <div className={mergeClasses(
              styles.connectionDot, 
              !isStreaming && styles.disconnectedDot
            )} />
            <Text size={300}>
              {isStreaming ? 'Streaming' : 'Disconnected'} • {filteredLogs.length} logs
            </Text>
          </div>
        </div>
        
        <div className={styles.controls}>
          <Button
            appearance="subtle"
            icon={<Filter24Regular />}
            onClick={() => {/* Toggle filters visibility */}}
          >
            Filters
          </Button>
          
          <Button
            appearance="subtle"
            icon={<ArrowDown24Regular />}
            onClick={onToggleAutoScroll}
            className={styles.autoScrollButton}
          >
            {autoScroll && <div className={styles.autoScrollIndicator} />}
            Auto Scroll
          </Button>
          
          <Button
            appearance="subtle"
            icon={<Delete24Regular />}
            onClick={onClearLogs}
          >
            Clear
          </Button>
          
          <Button
            appearance={isStreaming ? "secondary" : "primary"}
            icon={isStreaming ? <Pause24Regular /> : <Play24Regular />}
            onClick={isStreaming ? onStopStream : onStartStream}
          >
            {isStreaming ? 'Stop' : 'Start'} Stream
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className={styles.filters}>
        <Input
          placeholder="Search messages..."
          value={filter.searchQuery}
          onChange={(_, data) => onUpdateFilter({ searchQuery: data.value })}
          contentBefore={<Search24Regular />}
        />
        
        <Dropdown
          placeholder="Log Levels"
          multiselect
          value={filter.levels.join(', ')}
          selectedOptions={filter.levels}
          onOptionSelect={(_, data) => {
            const newLevels = data.selectedOptions as LogLevel[];
            onUpdateFilter({ levels: newLevels });
          }}
        >
          {logLevels.map(level => (
            <Option key={level} value={level} text={level.toUpperCase()}>
              <Badge color={getLevelColor(level)}>{level.toUpperCase()}</Badge>
            </Option>
          ))}
        </Dropdown>
        
        <Dropdown
          placeholder="Sources"
          multiselect
          value={filter.sources.join(', ')}
          selectedOptions={filter.sources}
          onOptionSelect={(_, data) => {
            onUpdateFilter({ sources: data.selectedOptions as string[] });
          }}
        >
          {availableSources.map(source => (
            <Option key={source} value={source} text={source}>{source}</Option>
          ))}
        </Dropdown>
        
        <Input
          type="number"
          placeholder="Max Results"
          value={String(filter.maxResults)}
          onChange={(_, data) => onUpdateFilter({ 
            maxResults: parseInt(data.value) || 1000 
          })}
        />
        
        <Text size={300}>
          Showing {filteredLogs.length} of {logs.length} logs
        </Text>
      </div>

      {/* Log Grid */}
      <div className={styles.logGrid} ref={gridRef}>
        <DataGrid
          items={filteredLogs}
          columns={columns}
          sortable
          selectionMode="single"
          getRowId={(item) => item.id}
        >
          <DataGridHeader>
            <DataGridRow>
              {({ renderHeaderCell }) => (
                <DataGridHeaderCell>{renderHeaderCell()}</DataGridHeaderCell>
              )}
            </DataGridRow>
          </DataGridHeader>
          <DataGridBody<LogEntry>>
            {({ item, rowId }) => (
              <DataGridRow<LogEntry> key={rowId}>
                {({ renderCell }) => (
                  <DataGridCell>{renderCell(item)}</DataGridCell>
                )}
              </DataGridRow>
            )}
          </DataGridBody>
        </DataGrid>
      </div>

      {/* Log Detail Dialog */}
      <Dialog 
        open={showDetailDialog} 
        onOpenChange={(_, data) => setShowDetailDialog(data.open)}
      >
        <DialogSurface>
          <DialogTitle>Log Entry Details</DialogTitle>
          <DialogContent>
            <DialogBody>
              {selectedLog && (
                <Textarea
                  value={JSON.stringify(selectedLog, null, 2)}
                  readOnly
                  rows={20}
                  style={{ 
                    fontFamily: tokens.fontFamilyMonospace,
                    fontSize: tokens.fontSizeBase200,
                  }}
                />
              )}
            </DialogBody>
            <DialogActions>
              <Button
                appearance="secondary"
                onClick={() => setShowDetailDialog(false)}
              >
                Close
              </Button>
              <Button
                appearance="primary"
                icon={<Copy24Regular />}
                onClick={() => {
                  if (selectedLog) {
                    navigator.clipboard.writeText(JSON.stringify(selectedLog, null, 2));
                  }
                }}
              >
                Copy JSON
              </Button>
            </DialogActions>
          </DialogContent>
        </DialogSurface>
      </Dialog>
    </div>
  );
};

export default LiveLogStream;
