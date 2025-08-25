/**
 * Observability Types - M5 Implementation
 * Data structures for logging, tracing, and debugging
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'fatal';

export interface LogEntry {
  id: string;
  timestamp: Date;
  level: LogLevel;
  message: string;
  source: string;
  context?: Record<string, any>;
  stackTrace?: string;
  sessionId?: string;
  actionId?: string;
  userId?: string;
  tags: string[];
}

export interface LogFilter {
  levels: LogLevel[];
  sources: string[];
  timeRange: {
    start?: Date;
    end?: Date;
  };
  searchQuery: string;
  sessionId?: string;
  tags: string[];
  maxResults: number;
}

export interface DebugTrace {
  id: string;
  sessionId: string;
  startTime: Date;
  endTime?: Date;
  status: 'running' | 'completed' | 'error' | 'cancelled';
  events: DebugEvent[];
  metadata: {
    userAgent: string;
    url: string;
    viewport: { width: number; height: number };
    totalActions: number;
    totalThoughts: number;
    duration?: number;
  };
}

export interface DebugEvent {
  id: string;
  timestamp: Date;
  type: 'action' | 'thought' | 'observation' | 'error' | 'screenshot' | 'dom_snapshot';
  category: string;
  title: string;
  description: string;
  details: Record<string, any>;
  duration?: number;
  status: 'pending' | 'success' | 'error' | 'warning';
  parentId?: string;
  children?: string[];
  artifacts?: DebugArtifact[];
}

export interface DebugArtifact {
  id: string;
  type: 'screenshot' | 'html' | 'json' | 'text' | 'video';
  name: string;
  url: string;
  size: number;
  mimeType: string;
  thumbnail?: string;
}

export interface LogSource {
  name: string;
  count: number;
  lastSeen: Date;
  isActive: boolean;
}

export interface LogStats {
  totalLogs: number;
  logsByLevel: Record<LogLevel, number>;
  sources: LogSource[];
  timeRange: {
    earliest: Date;
    latest: Date;
  };
  logsPerMinute: Array<{
    timestamp: Date;
    count: number;
  }>;
}

export interface ExportOptions {
  format: 'json' | 'csv' | 'txt' | 'har';
  includeStackTraces: boolean;
  includeContext: boolean;
  timeRange?: {
    start: Date;
    end: Date;
  };
  filters?: Partial<LogFilter>;
  maxSize: number; // MB
}

export interface ExportResult {
  id: string;
  status: 'pending' | 'completed' | 'error';
  format: string;
  filename: string;
  size: number;
  downloadUrl?: string;
  error?: string;
  createdAt: Date;
  expiresAt: Date;
}

export interface StreamConnection {
  id: string;
  isConnected: boolean;
  lastHeartbeat: Date;
  messageCount: number;
  reconnectAttempts: number;
  filters: LogFilter;
}

// Form state for observability interface
export interface ObservabilityState {
  // Log streaming
  logs: LogEntry[];
  isStreaming: boolean;
  streamConnection: StreamConnection | null;
  logFilter: LogFilter;
  logStats: LogStats | null;
  
  // Debug traces
  traces: DebugTrace[];
  selectedTrace: DebugTrace | null;
  selectedEvent: DebugEvent | null;
  traceFilter: {
    status: DebugTrace['status'][];
    timeRange: { start?: Date; end?: Date };
    searchQuery: string;
  };
  
  // Exports
  activeExports: ExportResult[];
  
  // UI state
  isLoading: boolean;
  error: string | null;
  selectedTab: 'logs' | 'traces' | 'exports';
  logsPanelHeight: number;
  tracesPanelWidth: number;
  autoScroll: boolean;
  showFilters: boolean;
  showStats: boolean;
}
