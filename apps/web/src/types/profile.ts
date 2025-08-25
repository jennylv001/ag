/**
 * Profile Builder Types - M2 Implementation
 * Maps to BrowserProfile configuration from backend system
 */

export interface ViewportSettings {
  width: number;
  height: number;
  deviceScaleFactor: number;
  isMobile: boolean;
  hasTouch: boolean;
}

export interface GeolocationSettings {
  latitude: number;
  longitude: number;
  accuracy: number;
}

export interface ProxySettings {
  server: string;
  username?: string;
  password?: string;
  bypass?: string[];
}

export interface StealthSettings {
  enabled: boolean;
  advanced_stealth: boolean;
  webgl_vendor_override?: string;
  webgl_renderer_override?: string;
  timezone?: string;
  locale?: string;
  user_agent_override?: string;
  accept_language?: string;
}

export interface SecuritySettings {
  bypassCSP: boolean;
  ignoreHTTPSErrors: boolean;
  acceptDownloads: boolean;
  javaScriptEnabled: boolean;
  webSecurity: boolean;
  permissions: string[];
}

export interface NetworkSettings {
  proxy?: ProxySettings;
  extraHTTPHeaders: Record<string, string>;
  userAgent?: string;
  offline: boolean;
  downloadPath?: string;
}

export interface RecordingSettings {
  video?: {
    dir: string;
    size?: ViewportSettings;
  };
  trace?: {
    dir: string;
    screenshots: boolean;
    snapshots: boolean;
  };
  har?: {
    path: string;
  };
}

export interface PersistenceSettings {
  userDataDir?: string;
  storageState?: string;
  channel?: string;
  executablePath?: string;
}

export interface LaunchArgsSettings {
  args: string[];
  ignoreDefaultArgs: boolean | string[];
  env: Record<string, string>;
  timeout: number;
  chromiumSandbox: boolean;
}

export interface BrowserProfile {
  id: string;
  name: string;
  description?: string;
  headless: boolean;
  
  // Tab configurations
  stealth: StealthSettings;
  security: SecuritySettings;
  viewport: ViewportSettings;
  network: NetworkSettings;
  recording: RecordingSettings;
  persistence: PersistenceSettings;
  launchArgs: LaunchArgsSettings;
  
  // Metadata
  createdAt: string;
  updatedAt: string;
  isDefault: boolean;
}

export interface ProfileValidationError {
  field: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
}

export interface ProfileValidationResult {
  isValid: boolean;
  errors: ProfileValidationError[];
  computedFlags: string[];
}

export interface ChromeFlag {
  flag: string;
  value: string | boolean;
  source: 'stealth' | 'security' | 'viewport' | 'args' | 'computed';
  rationale: string;
}

// Form state types for Zustand
export interface ProfileFormState {
  currentProfile: BrowserProfile | null;
  isDirty: boolean;
  activeTab: string;
  validationResult: ProfileValidationResult | null;
  isLoading: boolean;
  isSaving: boolean;
}
