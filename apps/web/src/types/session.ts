/**
 * Session Types - M3 Implementation
 * Data structures for session management and monitoring
 */

export interface BrowserSession {
  id: string;
  name: string;
  profileId: string;
  profileName: string;
  status: 'launching' | 'running' | 'stopping' | 'stopped' | 'error';
  
  // Launch configuration
  launchConfig: {
    channel?: string;
    executablePath?: string;
    headless: boolean;
    devtools: boolean;
    proxy?: {
      server: string;
      username?: string;
      password?: string;
    };
    timeout: number;
  };
  
  // Connection details (available when running)
  connectionInfo?: {
    wssUrl: string;
    cdpUrl: string;
    browserPid: number;
    debuggerUrl?: string;
  };
  
  // Session metadata
  createdAt: string;
  startedAt?: string;
  stoppedAt?: string;
  lastActivity?: string;
  
  // Runtime information
  runtime: {
    memoryUsage?: number;
    cpuUsage?: number;
    activePages: number;
    totalRequests: number;
    errors: number;
  };
  
  // Stealth features
  stealthFeatures: StealthFeature[];
  
  // Artifacts
  artifacts: SessionArtifact[];
}

export interface StealthFeature {
  name: string;
  enabled: boolean;
  status: 'active' | 'inactive' | 'error';
  description: string;
  category: 'detection' | 'fingerprint' | 'behavior' | 'network';
}

export interface SessionArtifact {
  id: string;
  type: 'har' | 'video' | 'trace' | 'screenshot';
  name: string;
  path: string;
  size: number;
  createdAt: string;
  status: 'recording' | 'completed' | 'error';
}

export interface ActivePage {
  id: string;
  url: string;
  title: string;
  status: 'loading' | 'complete' | 'error';
  isActive: boolean;
  favicon?: string;
  lastNavigated: string;
  requestCount: number;
  errorCount: number;
}

export interface SessionFilters {
  status?: BrowserSession['status'][];
  profileId?: string;
  search?: string;
  dateRange?: {
    from: Date;
    to: Date;
  };
  stealthEnabled?: boolean;
  hasArtifacts?: boolean;
}

export interface SessionStats {
  total: number;
  running: number;
  stopped: number;
  errors: number;
  totalPages: number;
  totalArtifacts: number;
}

// Form state for session management
export interface SessionFormState {
  sessions: BrowserSession[];
  selectedSession: BrowserSession | null;
  filters: SessionFilters;
  stats: SessionStats;
  isLoading: boolean;
  error: string | null;
  
  // Detail view state
  activeTab: 'launch' | 'connect' | 'pages' | 'artifacts';
  isConnecting: boolean;
  isLaunching: boolean;
  isStopping: boolean;
}
