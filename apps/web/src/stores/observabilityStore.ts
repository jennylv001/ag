/**
 * Observability Store - M5 Implementation
 * Zustand store for log streaming, debug traces, and exports
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { 
  ObservabilityState,
  LogEntry,
  LogLevel,
  LogFilter,
  DebugTrace,
  DebugEvent,
  ExportOptions,
  ExportResult,
  LogStats,
  StreamConnection,
  LogSource,
} from '../types/observability';

interface ObservabilityStore extends ObservabilityState {
  // Log streaming actions
  startLogStream: () => Promise<void>;
  stopLogStream: () => void;
  updateLogFilter: (filter: Partial<LogFilter>) => void;
  clearLogs: () => void;
  
  // Debug trace actions
  loadTraces: () => Promise<void>;
  selectTrace: (traceId: string) => void;
  selectEvent: (eventId: string) => void;
  updateTraceFilter: (filter: any) => void;
  
  // Export actions
  exportLogs: (options: ExportOptions) => Promise<void>;
  exportTrace: (traceId: string, options: ExportOptions) => Promise<void>;
  downloadExport: (exportId: string) => void;
  deleteExport: (exportId: string) => void;
  
  // UI actions
  setSelectedTab: (tab: 'logs' | 'traces' | 'exports') => void;
  toggleAutoScroll: () => void;
  toggleFilters: () => void;
  toggleStats: () => void;
  setError: (error: string | null) => void;
}

// Mock data generators
const generateMockLogEntry = (level: LogLevel, source: string): LogEntry => {
  const messages: Record<LogLevel, string[]> = {
    debug: [
      'Element selector: button[data-testid="submit"]',
      'DOM query executed successfully',
      'Screenshot captured: /tmp/screenshot_001.png',
    ],
    info: [
      'Action completed: click on login button',
      'Page navigation to https://example.com/dashboard',
      'Session started with ID: sess_abc123',
    ],
    warn: [
      'Element not immediately visible, waiting...',
      'Retry attempt 2/3 for action',
      'Slow network response detected',
    ],
    error: [
      'Element not found after timeout',
      'Failed to execute JavaScript injection',
      'Network request failed with status 500',
    ],
    fatal: [
      'Browser crashed unexpectedly',
      'Critical system error occurred',
      'Memory limit exceeded',
    ],
  };

  const contexts = {
    'browser-use': { url: 'https://example.com', actionType: 'click' },
    'llm-provider': { model: 'gpt-4o', tokens: 1247 },
    'agent-core': { sessionId: 'sess_' + Math.random().toString(36).substr(2, 9) },
    'stealth-engine': { feature: 'user-agent-override', enabled: true },
  };

  return {
    id: 'log_' + Math.random().toString(36).substr(2, 9),
    timestamp: new Date(Date.now() - Math.random() * 3600000),
    level,
    message: messages[level][Math.floor(Math.random() * messages[level].length)],
    source,
    context: contexts[source as keyof typeof contexts],
    stackTrace: level === 'error' || level === 'fatal' ? 
      `Error: ${messages[level][0]}\n    at action (/app/src/agent.js:45:12)\n    at execute (/app/src/executor.js:23:8)` : 
      undefined,
    sessionId: 'sess_' + Math.random().toString(36).substr(2, 9),
    actionId: 'action_' + Math.random().toString(36).substr(2, 9),
    tags: ['automated', source.split('-')[0]],
  };
};

const generateMockLogs = (count: number): LogEntry[] => {
  const levels: LogLevel[] = ['debug', 'info', 'warn', 'error'];
  const sources = ['browser-use', 'llm-provider', 'agent-core', 'stealth-engine'];
  
  return Array.from({ length: count }, () => {
    const level = levels[Math.floor(Math.random() * levels.length)];
    const source = sources[Math.floor(Math.random() * sources.length)];
    return generateMockLogEntry(level, source);
  }).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
};

const generateMockDebugTrace = (): DebugTrace => {
  const sessionId = 'sess_' + Math.random().toString(36).substr(2, 9);
  const startTime = new Date(Date.now() - Math.random() * 86400000);
  const duration = 30000 + Math.random() * 120000;
  
  const events: DebugEvent[] = [
    {
      id: 'evt_1',
      timestamp: new Date(startTime.getTime() + 1000),
      type: 'action',
      category: 'navigation',
      title: 'Navigate to URL',
      description: 'Navigating to https://example.com/login',
      details: { url: 'https://example.com/login', method: 'GET' },
      status: 'success',
      duration: 2500,
      artifacts: [{
        id: 'art_1',
        type: 'screenshot',
        name: 'page_load.png',
        url: '/artifacts/screenshots/page_load.png',
        size: 245760,
        mimeType: 'image/png',
        thumbnail: '/artifacts/thumbnails/page_load_thumb.png',
      }],
    },
    {
      id: 'evt_2',
      timestamp: new Date(startTime.getTime() + 5000),
      type: 'thought',
      category: 'planning',
      title: 'Analyze page structure',
      description: 'Identifying login form elements and required actions',
      details: { 
        elements_found: ['email_input', 'password_input', 'submit_button'],
        confidence: 0.95 
      },
      status: 'success',
      duration: 1200,
    },
    {
      id: 'evt_3',
      timestamp: new Date(startTime.getTime() + 8000),
      type: 'action',
      category: 'interaction',
      title: 'Fill email field',
      description: 'Entering email address into login form',
      details: { 
        selector: 'input[type="email"]',
        value: 'user@example.com',
        method: 'type' 
      },
      status: 'success',
      duration: 800,
    },
  ];
  
  return {
    id: 'trace_' + Math.random().toString(36).substr(2, 9),
    sessionId,
    startTime,
    endTime: new Date(startTime.getTime() + duration),
    status: Math.random() > 0.8 ? 'error' : 'completed',
    events,
    metadata: {
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      url: 'https://example.com/login',
      viewport: { width: 1920, height: 1080 },
      totalActions: events.filter(e => e.type === 'action').length,
      totalThoughts: events.filter(e => e.type === 'thought').length,
      duration,
    },
  };
};

const generateMockLogStats = (logs: LogEntry[]): LogStats => {
  const sources = logs.reduce<LogSource[]>((acc, log) => {
    const existing = acc.find((s: LogSource) => s.name === log.source);
    if (existing) {
      existing.count++;
      if (log.timestamp > existing.lastSeen) {
        existing.lastSeen = log.timestamp;
      }
    } else {
      acc.push({
        name: log.source,
        count: 1,
        lastSeen: log.timestamp,
        isActive: Math.random() > 0.3,
      });
    }
    return acc;
  }, []);

  const logsByLevel = logs.reduce((acc, log) => {
    acc[log.level] = (acc[log.level] || 0) + 1;
    return acc;
  }, {} as Record<LogLevel, number>);

  return {
    totalLogs: logs.length,
    logsByLevel,
    sources,
    timeRange: {
      earliest: new Date(Math.min(...logs.map(l => l.timestamp.getTime()))),
      latest: new Date(Math.max(...logs.map(l => l.timestamp.getTime()))),
    },
    logsPerMinute: Array.from({ length: 60 }, (_, i) => ({
      timestamp: new Date(Date.now() - (59 - i) * 60000),
      count: Math.floor(Math.random() * 20),
    })),
  };
};

export const useObservabilityStore = create<ObservabilityStore>()(
  devtools(
    (set, get) => ({
      // Initial state
      logs: [],
      isStreaming: false,
      streamConnection: null,
      logFilter: {
        levels: ['debug', 'info', 'warn', 'error', 'fatal'],
        sources: [],
        timeRange: {},
        searchQuery: '',
        tags: [],
        maxResults: 1000,
      },
      logStats: null,
      traces: [],
      selectedTrace: null,
      selectedEvent: null,
      traceFilter: {
        status: ['running', 'completed', 'error'],
        timeRange: {},
        searchQuery: '',
      },
      activeExports: [],
      isLoading: false,
      error: null,
      selectedTab: 'logs',
      logsPanelHeight: 400,
      tracesPanelWidth: 300,
      autoScroll: true,
      showFilters: true,
      showStats: false,

      // Log streaming actions
      startLogStream: async () => {
        set({ isStreaming: true, error: null });
        
        // Generate initial logs
        const initialLogs = generateMockLogs(50);
        const stats = generateMockLogStats(initialLogs);
        
        const connection: StreamConnection = {
          id: 'conn_' + Math.random().toString(36).substr(2, 9),
          isConnected: true,
          lastHeartbeat: new Date(),
          messageCount: initialLogs.length,
          reconnectAttempts: 0,
          filters: get().logFilter,
        };
        
        set({ 
          logs: initialLogs, 
          logStats: stats,
          streamConnection: connection 
        });
        
        // Simulate live streaming
        const streamInterval = setInterval(() => {
          const { isStreaming, logFilter } = get();
          if (!isStreaming) {
            clearInterval(streamInterval);
            return;
          }
          
          // Add new log every 2-5 seconds
          const newLog = generateMockLogEntry(
            ['debug', 'info', 'warn', 'error'][Math.floor(Math.random() * 4)] as LogLevel,
            ['browser-use', 'llm-provider', 'agent-core'][Math.floor(Math.random() * 3)]
          );
          
          set((state) => {
            const updatedLogs = [newLog, ...state.logs].slice(0, logFilter.maxResults);
            return {
              logs: updatedLogs,
              logStats: generateMockLogStats(updatedLogs),
              streamConnection: state.streamConnection ? {
                ...state.streamConnection,
                lastHeartbeat: new Date(),
                messageCount: state.streamConnection.messageCount + 1,
              } : null,
            };
          });
        }, 2000 + Math.random() * 3000);
      },

      stopLogStream: () => {
        set({ 
          isStreaming: false, 
          streamConnection: null 
        });
      },

      updateLogFilter: (filter) => {
        set((state) => ({ 
          logFilter: { ...state.logFilter, ...filter } 
        }));
      },

      clearLogs: () => {
        set({ logs: [], logStats: null });
      },

      // Debug trace actions
      loadTraces: async () => {
        set({ isLoading: true, error: null });
        try {
          await new Promise(resolve => setTimeout(resolve, 1000));
          const traces = Array.from({ length: 10 }, () => generateMockDebugTrace());
          set({ traces, isLoading: false });
        } catch (error) {
          set({ 
            error: 'Failed to load traces',
            isLoading: false 
          });
        }
      },

      selectTrace: (traceId) => {
        const { traces } = get();
        const trace = traces.find((t: DebugTrace) => t.id === traceId);
        set({ 
          selectedTrace: trace || null,
          selectedEvent: null 
        });
      },

      selectEvent: (eventId) => {
        const { selectedTrace } = get();
        if (selectedTrace) {
          const event = selectedTrace.events.find((e: DebugEvent) => e.id === eventId);
          set({ selectedEvent: event || null });
        }
      },

      updateTraceFilter: (filter) => {
        set((state) => ({ 
          traceFilter: { ...state.traceFilter, ...filter } 
        }));
      },

      // Export actions
      exportLogs: async (options) => {
        const exportId = 'export_' + Math.random().toString(36).substr(2, 9);
        const exportResult: ExportResult = {
          id: exportId,
          status: 'pending',
          format: options.format,
          filename: `logs_${new Date().toISOString().split('T')[0]}.${options.format}`,
          size: 0,
          createdAt: new Date(),
          expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24 hours
        };
        
        set((state) => ({
          activeExports: [...state.activeExports, exportResult]
        }));
        
        // Simulate export processing
        setTimeout(() => {
          set((state) => ({
            activeExports: state.activeExports.map((exp: ExportResult) => 
              exp.id === exportId 
                ? { 
                    ...exp, 
                    status: 'completed',
                    size: 1024 + Math.random() * 1024 * 1024,
                    downloadUrl: `/api/exports/${exportId}/download`
                  }
                : exp
            )
          }));
        }, 2000);
      },

      exportTrace: async (_traceId, options) => {
        // Similar to exportLogs but for a specific trace
        get().exportLogs(options);
      },

      downloadExport: (exportId) => {
        // In real implementation, would trigger download
        console.log('Downloading export:', exportId);
      },

      deleteExport: (exportId) => {
        set((state) => ({
          activeExports: state.activeExports.filter((exp: ExportResult) => exp.id !== exportId)
        }));
      },

      // UI actions
      setSelectedTab: (tab) => {
        set({ selectedTab: tab });
      },

      toggleAutoScroll: () => {
        set((state) => ({ autoScroll: !state.autoScroll }));
      },

      toggleFilters: () => {
        set((state) => ({ showFilters: !state.showFilters }));
      },

      toggleStats: () => {
        set((state) => ({ showStats: !state.showStats }));
      },

      setError: (error) => {
        set({ error });
      },
    }),
    { name: 'observability-store' }
  )
);
