/**
 * Debug Trace Viewer Component - M5 Implementation
 * Timeline visualization for observe_debug traces with event details
 */

import React from 'react';
import {
  Card,
  CardHeader,
  Text,
  Button,
  Badge,
  Input,
  Dropdown,
  Option,
  mergeClasses,
  makeStyles,
  tokens,
  ProgressBar,
  Dialog,
  DialogSurface,
  DialogTitle,
  DialogContent,
  DialogBody,
  DialogActions,
} from '@fluentui/react-components';
import {
  PlayRegular,
  ClockRegular,
  ErrorCircleRegular,
  CheckmarkCircleRegular,
  DocumentRegular,
  ImageRegular,
  VideoRegular,
  SearchRegular,
  ChevronDown24Regular,
  ChevronUp24Regular,
} from '@fluentui/react-icons';
import { DebugTrace, DebugEvent, DebugArtifact } from '../../types/observability';

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
  filters: {
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    alignItems: 'center',
    padding: tokens.spacingVerticalM,
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusMedium,
  },
  splitterContainer: {
    flex: 1,
    minHeight: '600px',
  },
  splitLayout: {
    display: 'grid',
    gridTemplateColumns: 'minmax(300px, 420px) 6px 1fr',
    height: '100%',
  },
  leftPane: {
    display: 'flex',
    flexDirection: 'column',
    minHeight: 0,
    overflow: 'hidden',
  },
  rightPane: {
    minHeight: 0,
    overflow: 'hidden',
  },
  splitterHandle: {
    backgroundColor: tokens.colorNeutralStroke2,
    cursor: 'col-resize',
  },
  tracesList: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
    padding: tokens.spacingVerticalM,
    height: '100%',
    overflowY: 'auto',
  },
  traceCard: {
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    '&:hover': {
      backgroundColor: tokens.colorNeutralBackground3,
    },
  },
  selectedTrace: {
    backgroundColor: tokens.colorBrandBackground2,
  // visual accent handled by background only (borderColor removed for compatibility)
  },
  traceHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: tokens.spacingVerticalXS,
  },
  traceMetadata: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: tokens.spacingVerticalXS,
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
  statusBadge: {
    marginLeft: 'auto',
  },
  timeline: {
    padding: tokens.spacingVerticalM,
    height: '100%',
    overflowY: 'auto',
  },
  timelineItem: {
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    marginBottom: tokens.spacingVerticalM,
    position: 'relative',
  },
  timelineConnector: {
    position: 'absolute',
    left: '16px',
    top: '32px',
    bottom: '-16px',
    width: '2px',
    backgroundColor: tokens.colorNeutralStroke2,
  },
  lastTimelineItem: {
    '& $timelineConnector': {
      display: 'none',
    },
  },
  timelineIcon: {
    width: '32px',
    height: '32px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: tokens.colorNeutralBackground3,
    border: `2px solid ${tokens.colorNeutralStroke1}`,
    fontSize: '16px',
    zIndex: 1,
    position: 'relative',
  },
  actionIcon: {
    backgroundColor: tokens.colorPaletteBlueBorderActive,
    color: tokens.colorNeutralForegroundOnBrand,
  },
  thoughtIcon: {
    backgroundColor: tokens.colorPaletteYellowBorderActive,
    color: tokens.colorNeutralForeground1,
  },
  errorIcon: {
    backgroundColor: tokens.colorPaletteRedBorderActive,
    color: tokens.colorNeutralForegroundOnBrand,
  },
  successIcon: {
    backgroundColor: tokens.colorPaletteGreenBorderActive,
    color: tokens.colorNeutralForegroundOnBrand,
  },
  timelineContent: {
    flex: 1,
    minWidth: 0,
  },
  eventCard: {
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    '&:hover': {
      backgroundColor: tokens.colorNeutralBackground3,
    },
  },
  selectedEvent: {
    backgroundColor: tokens.colorBrandBackground2,
  // visual accent handled by background only (borderColor removed for compatibility)
  },
  eventHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: tokens.spacingVerticalXS,
  },
  eventTimestamp: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
    fontFamily: tokens.fontFamilyMonospace,
  },
  eventDescription: {
    marginBottom: tokens.spacingVerticalS,
  },
  artifactsList: {
    display: 'flex',
    gap: tokens.spacingHorizontalS,
    marginTop: tokens.spacingVerticalS,
  },
  artifactChip: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXS,
    padding: `${tokens.spacingVerticalXS} ${tokens.spacingHorizontalS}`,
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusMedium,
    fontSize: tokens.fontSizeBase200,
    cursor: 'pointer',
    '&:hover': {
      backgroundColor: tokens.colorNeutralBackground3,
    },
  },
  eventDetails: {
    marginTop: tokens.spacingVerticalS,
    padding: tokens.spacingVerticalS,
    backgroundColor: tokens.colorNeutralBackground1,
    borderRadius: tokens.borderRadiusMedium,
    fontFamily: tokens.fontFamilyMonospace,
    fontSize: tokens.fontSizeBase200,
    whiteSpace: 'pre-wrap',
  },
  durationBar: {
    marginTop: tokens.spacingVerticalXS,
  },
  expandButton: {
    marginLeft: tokens.spacingHorizontalS,
  },
});

interface DebugTraceViewerProps {
  traces: DebugTrace[];
  selectedTrace: DebugTrace | null;
  selectedEvent: DebugEvent | null;
  onSelectTrace: (traceId: string) => void;
  onSelectEvent: (eventId: string) => void;
  className?: string;
}

const DebugTraceViewer: React.FC<DebugTraceViewerProps> = ({
  traces,
  selectedTrace,
  selectedEvent,
  onSelectTrace,
  onSelectEvent,
  className,
}) => {
  const styles = useStyles();
  const [searchQuery, setSearchQuery] = React.useState('');
  const [statusFilter, setStatusFilter] = React.useState<DebugTrace['status'][]>([
    'running', 'completed', 'error'
  ]);
  const [expandedEvents, setExpandedEvents] = React.useState<Set<string>>(new Set());
  const [selectedArtifact, setSelectedArtifact] = React.useState<DebugArtifact | null>(null);

  // Filter traces
  const filteredTraces = React.useMemo(() => {
    return traces.filter(trace => {
      if (!statusFilter.includes(trace.status)) return false;
      if (searchQuery && !trace.metadata.url.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }
      return true;
    });
  }, [traces, statusFilter, searchQuery]);

  const getStatusColor = (status: DebugTrace['status']) => {
    const colors = {
      running: 'informative',
      completed: 'success',
      error: 'danger',
      cancelled: 'warning',
    } as const;
    return colors[status];
  };

  const getEventIcon = (event: DebugEvent) => {
    switch (event.type) {
      case 'action':
        return <PlayRegular />;
      case 'thought':
        return <ClockRegular />;
      case 'error':
        return <ErrorCircleRegular />;
      case 'screenshot':
        return <ImageRegular />;
      case 'dom_snapshot':
        return <DocumentRegular />;
      default:
        return <CheckmarkCircleRegular />;
    }
  };

  const getEventIconStyle = (event: DebugEvent) => {
    switch (event.status) {
      case 'error':
        return styles.errorIcon;
      case 'success':
        return styles.successIcon;
      case 'warning':
        return mergeClasses(styles.timelineIcon, styles.thoughtIcon);
      default:
        if (event.type === 'action') return styles.actionIcon;
        if (event.type === 'thought') return styles.thoughtIcon;
        return styles.timelineIcon;
    }
  };

  const getArtifactIcon = (artifact: DebugArtifact) => {
    switch (artifact.type) {
      case 'screenshot':
        return <ImageRegular />;
      case 'video':
        return <VideoRegular />;
      default:
        return <DocumentRegular />;
    }
  };

  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
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

  const toggleEventExpansion = (eventId: string) => {
    setExpandedEvents(prev => {
      const newSet = new Set(prev);
      if (newSet.has(eventId)) {
        newSet.delete(eventId);
      } else {
        newSet.add(eventId);
      }
      return newSet;
    });
  };

  const maxDuration = selectedTrace ? 
    Math.max(...selectedTrace.events.map(e => e.duration || 0)) : 
    0;

  return (
    <div className={mergeClasses(styles.container, className)}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <Text size={500} weight="semibold">Debug Trace Viewer</Text>
          <Text size={300}>
            Timeline visualization of agent execution traces
          </Text>
        </div>
      </div>

      {/* Filters */}
      <div className={styles.filters}>
        <Input
          placeholder="Search by URL..."
          value={searchQuery}
          onChange={(_, data) => setSearchQuery(data.value)}
          contentBefore={<SearchRegular />}
        />
        
        <Dropdown
          placeholder="Status Filter"
          multiselect
          value={statusFilter.join(', ')}
          selectedOptions={statusFilter}
          onOptionSelect={(_, data) => {
            setStatusFilter(data.selectedOptions as DebugTrace['status'][]);
          }}
        >
          <Option value="running" text="Running">
            <Badge color="informative">Running</Badge>
          </Option>
          <Option value="completed" text="Completed">
            <Badge color="success">Completed</Badge>
          </Option>
          <Option value="error" text="Error">
            <Badge color="danger">Error</Badge>
          </Option>
          <Option value="cancelled" text="Cancelled">
            <Badge color="warning">Cancelled</Badge>
          </Option>
        </Dropdown>
        
        <Text size={300}>
          {filteredTraces.length} trace(s)
        </Text>
      </div>

      {/* Main Content */}
      <div className={mergeClasses(styles.splitterContainer, styles.splitLayout)}>
        {/* Traces List */}
        <div className={styles.leftPane}>
          <div className={styles.tracesList}>
            {filteredTraces.map(trace => (
              <Card
                key={trace.id}
                className={mergeClasses(
                  styles.traceCard,
                  selectedTrace?.id === trace.id && styles.selectedTrace
                )}
                onClick={() => onSelectTrace(trace.id)}
              >
                <CardHeader
                  header={
                    <div className={styles.traceHeader}>
                      <Text weight="semibold" size={300}>
                        Session {trace.sessionId.slice(-8)}
                      </Text>
                      <Badge 
                        appearance="filled" 
                        color={getStatusColor(trace.status)}
                        className={styles.statusBadge}
                      >
                        {trace.status}
                      </Badge>
                    </div>
                  }
                />
                <div className={styles.traceMetadata}>
                  <Text size={200}>URL: {trace.metadata.url}</Text>
                  <Text size={200}>Duration: {trace.metadata.duration ? formatDuration(trace.metadata.duration) : 'Running...'}</Text>
                  <Text size={200}>Actions: {trace.metadata.totalActions}</Text>
                  <Text size={200}>Events: {trace.events.length}</Text>
                </div>
              </Card>
            ))}
          </div>
        </div>

        <div className={styles.splitterHandle} aria-hidden />

        {/* Timeline */}
        <div className={styles.rightPane}>
          <div className={styles.timeline}>
            {selectedTrace ? (
              <>
                <Text size={400} weight="semibold" style={{ marginBottom: tokens.spacingVerticalM }}>
                  Execution Timeline
                </Text>
                
                {selectedTrace.events.map((event, index) => {
                  const isExpanded = expandedEvents.has(event.id);
                  const isLast = index === selectedTrace.events.length - 1;
                  
                  return (
                    <div 
                      key={event.id} 
                      className={mergeClasses(
                        styles.timelineItem,
                        isLast && styles.lastTimelineItem
                      )}
                    >
                      {!isLast && <div className={styles.timelineConnector} />}
                      
                      <div className={mergeClasses(styles.timelineIcon, getEventIconStyle(event))}>
                        {getEventIcon(event)}
                      </div>
                      
                      <div className={styles.timelineContent}>
                        <Card
                          className={mergeClasses(
                            styles.eventCard,
                            selectedEvent?.id === event.id && styles.selectedEvent
                          )}
                          onClick={() => onSelectEvent(event.id)}
                        >
                          <div className={styles.eventHeader}>
                            <div>
                              <Text weight="semibold">{event.title}</Text>
                              <Text className={styles.eventTimestamp}>
                                {formatTimestamp(event.timestamp)}
                              </Text>
                            </div>
                            
                            <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacingHorizontalS }}>
                              <Badge color={getStatusColor(event.status as any)}>
                                {event.status}
                              </Badge>
                              
                              {event.duration && (
                                <Text size={200}>{formatDuration(event.duration)}</Text>
                              )}
                              
                              <Button
                                appearance="subtle"
                                size="small"
                                icon={isExpanded ? <ChevronUp24Regular /> : <ChevronDown24Regular />}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  toggleEventExpansion(event.id);
                                }}
                                className={styles.expandButton}
                              />
                            </div>
                          </div>
                          
                          <Text className={styles.eventDescription}>
                            {event.description}
                          </Text>
                          
                          {event.duration && (
                            <div className={styles.durationBar}>
                              <ProgressBar 
                                value={event.duration / maxDuration} 
                                color="brand"
                              />
                            </div>
                          )}
                          
                          {/* Artifacts */}
                          {event.artifacts && event.artifacts.length > 0 && (
                            <div className={styles.artifactsList}>
                              {event.artifacts.map(artifact => (
                                <div
                                  key={artifact.id}
                                  className={styles.artifactChip}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setSelectedArtifact(artifact);
                                  }}
                                >
                                  {getArtifactIcon(artifact)}
                                  <Text size={200}>{artifact.name}</Text>
                                </div>
                              ))}
                            </div>
                          )}
                          
                          {/* Expanded Details */}
                          {isExpanded && (
                            <div className={styles.eventDetails}>
                              {JSON.stringify(event.details, null, 2)}
                            </div>
                          )}
                        </Card>
                      </div>
                    </div>
                  );
                })}
              </>
            ) : (
              <Text>Select a trace to view its timeline</Text>
            )}
          </div>
        </div>
      </div>

      {/* Artifact Viewer Dialog */}
      <Dialog 
        open={!!selectedArtifact} 
        onOpenChange={(_, data) => !data.open && setSelectedArtifact(null)}
      >
        <DialogSurface>
          <DialogTitle>{selectedArtifact?.name}</DialogTitle>
          <DialogContent>
            <DialogBody>
              {selectedArtifact?.type === 'screenshot' && (
                <img 
                  src={selectedArtifact.url} 
                  alt={selectedArtifact.name}
                  style={{ maxWidth: '100%', maxHeight: '500px' }}
                />
              )}
              {selectedArtifact?.type !== 'screenshot' && (
                <Text>
                  {selectedArtifact?.type.toUpperCase()} file: {selectedArtifact?.size} bytes
                </Text>
              )}
            </DialogBody>
            <DialogActions>
              <Button
                appearance="secondary"
                onClick={() => setSelectedArtifact(null)}
              >
                Close
              </Button>
              <Button
                appearance="primary"
                onClick={() => {
                  if (selectedArtifact) {
                    window.open(selectedArtifact.url, '_blank');
                  }
                }}
              >
                Open
              </Button>
            </DialogActions>
          </DialogContent>
        </DialogSurface>
      </Dialog>
    </div>
  );
};

export default DebugTraceViewer;
