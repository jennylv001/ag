/**
 * Session Store - M3 Implementation
 * State management for session control panel
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { 
  BrowserSession, 
  SessionFormState, 
  SessionFilters, 
  SessionStats,
  StealthFeature
} from '../types/session';

interface SessionControlStore extends SessionFormState {
  // Actions
  loadSessions: () => Promise<void>;
  createSession: (profileId: string, config: Partial<BrowserSession['launchConfig']>) => Promise<string>;
  startSession: (sessionId: string) => Promise<void>;
  stopSession: (sessionId: string) => Promise<void>;
  connectToSession: (sessionId: string) => Promise<void>;
  selectSession: (session: BrowserSession | null) => void;
  setFilters: (filters: Partial<SessionFilters>) => void;
  setActiveTab: (tab: SessionFormState['activeTab']) => void;
  refreshSession: (sessionId: string) => Promise<void>;
  
  // DevTools actions
  openDevTools: (sessionId: string) => Promise<void>;
  takeScreenshot: (sessionId: string) => Promise<void>;
  
  // Artifacts actions
  downloadArtifact: (artifactId: string) => Promise<void>;
  deleteArtifact: (artifactId: string) => Promise<void>;
}

// Mock data generators
const generateStealthFeatures = (): StealthFeature[] => [
  {
    name: 'User Agent Override',
    enabled: true,
    status: 'active',
    description: 'Custom user agent string applied',
    category: 'fingerprint'
  },
  {
    name: 'WebRTC IP Leak Protection',
    enabled: true,
    status: 'active',
    description: 'Prevents WebRTC IP address leaks',
    category: 'network'
  },
  {
    name: 'Canvas Fingerprint Blocking',
    enabled: true,
    status: 'active',
    description: 'Randomizes canvas fingerprint',
    category: 'fingerprint'
  },
  {
    name: 'Automation Detection Bypass',
    enabled: true,
    status: 'active',
    description: 'Hides automation detection markers',
    category: 'detection'
  },
  {
    name: 'Timezone Spoofing',
    enabled: false,
    status: 'inactive',
    description: 'Override system timezone',
    category: 'fingerprint'
  },
  {
    name: 'Language Override',
    enabled: true,
    status: 'active',
    description: 'Custom Accept-Language header',
    category: 'fingerprint'
  },
  {
    name: 'WebGL Vendor Override',
    enabled: false,
    status: 'inactive',
    description: 'Custom WebGL vendor/renderer',
    category: 'fingerprint'
  },
  {
    name: 'Mouse Movement Simulation',
    enabled: true,
    status: 'active',
    description: 'Human-like mouse patterns',
    category: 'behavior'
  }
];

const generateMockSession = (index: number): BrowserSession => {
  const statuses: BrowserSession['status'][] = ['running', 'stopped', 'launching', 'error'];
  const status = statuses[index % statuses.length];
  const createdAt = new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString();
  
  return {
    id: `session-${index + 1}`,
    name: `Session ${index + 1}`,
    profileId: `profile-${(index % 3) + 1}`,
    profileName: `Profile ${(index % 3) + 1}`,
    status,
    launchConfig: {
      channel: index % 2 === 0 ? 'chrome' : 'msedge',
      headless: index % 3 === 0,
      devtools: index % 2 === 1,
      timeout: 30000 + (index * 5000),
      proxy: index % 4 === 0 ? {
        server: 'http://proxy.example.com:8080',
        username: 'user',
        password: 'pass'
      } : undefined,
    },
    connectionInfo: status === 'running' ? {
      wssUrl: `ws://localhost:${9222 + index}`,
      cdpUrl: `http://localhost:${9222 + index}`,
      browserPid: 1000 + index,
      debuggerUrl: `http://localhost:${9222 + index}/json`,
    } : undefined,
    createdAt,
    startedAt: status !== 'stopped' ? createdAt : undefined,
    lastActivity: status === 'running' ? new Date().toISOString() : undefined,
    runtime: {
      memoryUsage: status === 'running' ? 100 + Math.random() * 400 : undefined,
      cpuUsage: status === 'running' ? Math.random() * 50 : undefined,
      activePages: status === 'running' ? Math.floor(Math.random() * 5) + 1 : 0,
      totalRequests: Math.floor(Math.random() * 1000),
      errors: Math.floor(Math.random() * 10),
    },
    stealthFeatures: generateStealthFeatures(),
    artifacts: [
      {
        id: `har-${index}`,
        type: 'har',
        name: `session-${index + 1}.har`,
        path: `/artifacts/session-${index + 1}.har`,
        size: Math.floor(Math.random() * 10000000),
        createdAt,
        status: status === 'running' ? 'recording' : 'completed',
      },
      {
        id: `video-${index}`,
        type: 'video',
        name: `session-${index + 1}.mp4`,
        path: `/artifacts/session-${index + 1}.mp4`,
        size: Math.floor(Math.random() * 50000000),
        createdAt,
        status: status === 'running' ? 'recording' : 'completed',
      }
    ],
  };
};

const calculateStats = (sessions: BrowserSession[]): SessionStats => ({
  total: sessions.length,
  running: sessions.filter(s => s.status === 'running').length,
  stopped: sessions.filter(s => s.status === 'stopped').length,
  errors: sessions.filter(s => s.status === 'error').length,
  totalPages: sessions.reduce((sum, s) => sum + s.runtime.activePages, 0),
  totalArtifacts: sessions.reduce((sum, s) => sum + s.artifacts.length, 0),
});

export const useSessionControlStore = create<SessionControlStore>()(
  devtools(
    (set, get) => ({
      // Initial state
      sessions: [],
      selectedSession: null,
      filters: {},
      stats: { total: 0, running: 0, stopped: 0, errors: 0, totalPages: 0, totalArtifacts: 0 },
      isLoading: false,
      error: null,
      activeTab: 'launch',
      isConnecting: false,
      isLaunching: false,
      isStopping: false,

      // Actions
      loadSessions: async () => {
        set({ isLoading: true, error: null });
        try {
          // Mock API call
          await new Promise(resolve => setTimeout(resolve, 800));
          
          const mockSessions = Array.from({ length: 8 }, (_, i) => generateMockSession(i));
          const stats = calculateStats(mockSessions);
          
          set({ 
            sessions: mockSessions,
            stats,
            isLoading: false 
          });
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to load sessions',
            isLoading: false 
          });
        }
      },

      createSession: async (profileId: string, config: Partial<BrowserSession['launchConfig']>) => {
        set({ isLaunching: true, error: null });
        try {
          // Mock API call
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          const newSession: BrowserSession = {
            id: `session-${Date.now()}`,
            name: `Session ${Date.now()}`,
            profileId,
            profileName: `Profile ${profileId}`,
            status: 'launching',
            launchConfig: {
              headless: false,
              devtools: false,
              timeout: 30000,
              ...config,
            },
            createdAt: new Date().toISOString(),
            runtime: {
              activePages: 0,
              totalRequests: 0,
              errors: 0,
            },
            stealthFeatures: generateStealthFeatures(),
            artifacts: [],
          };

          const { sessions } = get();
          const updatedSessions = [newSession, ...sessions];
          
          set({ 
            sessions: updatedSessions,
            stats: calculateStats(updatedSessions),
            selectedSession: newSession,
            isLaunching: false 
          });

          // Simulate session startup
          setTimeout(() => {
            const { sessions } = get();
            const updated = sessions.map(s => 
              s.id === newSession.id 
                ? { 
                    ...s, 
                    status: 'running' as const,
                    startedAt: new Date().toISOString(),
                    connectionInfo: {
                      wssUrl: `ws://localhost:9222`,
                      cdpUrl: `http://localhost:9222`,
                      browserPid: Math.floor(Math.random() * 10000),
                    },
                    runtime: {
                      ...s.runtime,
                      activePages: 1,
                    }
                  }
                : s
            );
            set({ 
              sessions: updated,
              stats: calculateStats(updated),
              selectedSession: updated.find(s => s.id === newSession.id) || null
            });
          }, 2000);

          return newSession.id;
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to create session',
            isLaunching: false 
          });
          throw error;
        }
      },

      startSession: async (sessionId: string) => {
        set({ error: null });
        try {
          await new Promise(resolve => setTimeout(resolve, 1500));
          
          const { sessions } = get();
          const updated = sessions.map(s => 
            s.id === sessionId 
              ? { 
                  ...s, 
                  status: 'running' as const,
                  startedAt: new Date().toISOString(),
                  connectionInfo: {
                    wssUrl: `ws://localhost:9222`,
                    cdpUrl: `http://localhost:9222`,
                    browserPid: Math.floor(Math.random() * 10000),
                  }
                }
              : s
          );
          
          set({ 
            sessions: updated,
            stats: calculateStats(updated),
            selectedSession: updated.find(s => s.id === sessionId) || null
          });
        } catch (error) {
          set({ error: error instanceof Error ? error.message : 'Failed to start session' });
        }
      },

      stopSession: async (sessionId: string) => {
        set({ isStopping: true, error: null });
        try {
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          const { sessions } = get();
          const updated = sessions.map(s => 
            s.id === sessionId 
              ? { 
                  ...s, 
                  status: 'stopped' as const,
                  stoppedAt: new Date().toISOString(),
                  connectionInfo: undefined,
                  runtime: {
                    ...s.runtime,
                    activePages: 0,
                  }
                }
              : s
          );
          
          set({ 
            sessions: updated,
            stats: calculateStats(updated),
            selectedSession: updated.find(s => s.id === sessionId) || null,
            isStopping: false
          });
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to stop session',
            isStopping: false 
          });
        }
      },

  connectToSession: async (_sessionId: string) => {
        set({ isConnecting: true, error: null });
        try {
          await new Promise(resolve => setTimeout(resolve, 500));
          // Mock connection logic
          set({ isConnecting: false });
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to connect to session',
            isConnecting: false 
          });
        }
      },

      selectSession: (session: BrowserSession | null) => {
        set({ selectedSession: session, activeTab: 'launch' });
      },

      setFilters: (filters: Partial<SessionFilters>) => {
        set(state => ({ filters: { ...state.filters, ...filters } }));
      },

      setActiveTab: (tab: SessionFormState['activeTab']) => {
        set({ activeTab: tab });
      },

      refreshSession: async (_sessionId: string) => {
        // Mock refresh - in real implementation, would fetch latest session data
        const { sessions } = get();
        const session = sessions.find(s => s.id === _sessionId);
        if (session && session.status === 'running') {
          const updated = sessions.map(s => 
            s.id === _sessionId 
              ? { 
                  ...s, 
                  lastActivity: new Date().toISOString(),
                  runtime: {
                    ...s.runtime,
                    memoryUsage: 100 + Math.random() * 400,
                    cpuUsage: Math.random() * 50,
                    totalRequests: s.runtime.totalRequests + Math.floor(Math.random() * 10),
                  }
                }
              : s
          );
          
          set({ 
            sessions: updated,
            selectedSession: updated.find(s => s.id === _sessionId) || null
          });
        }
      },

      openDevTools: async (sessionId: string) => {
        const { sessions } = get();
        const session = sessions.find(s => s.id === sessionId);
        if (session?.connectionInfo?.debuggerUrl) {
          // In real implementation, would open DevTools
          window.open(session.connectionInfo.debuggerUrl, '_blank');
        }
      },

      takeScreenshot: async (sessionId: string) => {
        // Mock screenshot functionality
        await new Promise(resolve => setTimeout(resolve, 500));
        console.log(`Taking screenshot for session ${sessionId}`);
      },

      downloadArtifact: async (artifactId: string) => {
        // Mock download functionality
        await new Promise(resolve => setTimeout(resolve, 300));
        console.log(`Downloading artifact ${artifactId}`);
      },

      deleteArtifact: async (artifactId: string) => {
        // Mock delete functionality
        await new Promise(resolve => setTimeout(resolve, 200));
        console.log(`Deleting artifact ${artifactId}`);
      },
    }),
    {
      name: 'session-control-store',
    }
  )
);
