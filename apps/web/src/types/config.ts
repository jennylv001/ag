/**
 * Configuration Types - M4 Implementation
 * Data structures for configuration management with sources and precedence
 */

export type ConfigSource = 'database' | 'environment' | 'runtime' | 'default';

export interface ConfigValue<T = any> {
  value: T;
  source: ConfigSource;
  priority: number; // Higher number = higher priority
  isOverridden: boolean;
  originalValue?: T;
  description?: string;
}

export interface ConfigSection {
  [key: string]: ConfigValue | ConfigSection;
}

// Environment variable mappings
export interface EnvironmentVariables {
  // Browser configuration
  BROWSER_USE_HEADLESS: boolean;
  BROWSER_USE_ALLOWED_DOMAINS: string[];
  BROWSER_USE_STEALTH: boolean;
  BROWSER_USE_TIMEOUT: number;
  BROWSER_USE_PROXY: string;
  
  // LLM configuration
  LLM_MODEL: string;
  LLM_API_KEY: string;
  LLM_BASE_URL: string;
  LLM_TEMPERATURE: number;
  LLM_MAX_TOKENS: number;
  
  // Agent configuration
  AGENT_MAX_ACTIONS: number;
  AGENT_THINK_TIMEOUT: number;
  AGENT_ACTION_TIMEOUT: number;
  AGENT_DEBUG: boolean;
  
  // System configuration
  LOG_LEVEL: string;
  DEBUG_MODE: boolean;
  DATA_DIR: string;
  CACHE_DIR: string;
}

export interface AgentConfig {
  maxActions: ConfigValue<number>;
  thinkTimeout: ConfigValue<number>;
  actionTimeout: ConfigValue<number>;
  debugMode: ConfigValue<boolean>;
  retryAttempts: ConfigValue<number>;
  enableTelemetry: ConfigValue<boolean>;
  workingDirectory: ConfigValue<string>;
}

export interface LLMConfig {
  model: ConfigValue<string>;
  apiKey: ConfigValue<string>;
  baseUrl: ConfigValue<string>;
  temperature: ConfigValue<number>;
  maxTokens: ConfigValue<number>;
  timeout: ConfigValue<number>;
  retryAttempts: ConfigValue<number>;
  fallbackModel: ConfigValue<string>;
}

export interface BrowserConfig {
  headless: ConfigValue<boolean>;
  allowedDomains: ConfigValue<string[]>;
  stealthMode: ConfigValue<boolean>;
  timeout: ConfigValue<number>;
  proxy: ConfigValue<string>;
  userDataDir: ConfigValue<string>;
  executablePath: ConfigValue<string>;
  devtools: ConfigValue<boolean>;
}

export interface SystemConfig {
  logLevel: ConfigValue<string>;
  debugMode: ConfigValue<boolean>;
  dataDirectory: ConfigValue<string>;
  cacheDirectory: ConfigValue<string>;
  maxLogFiles: ConfigValue<number>;
  enableMetrics: ConfigValue<boolean>;
}

export interface ApplicationConfig {
  agent: AgentConfig;
  llm: LLMConfig;
  browser: BrowserConfig;
  system: SystemConfig;
}

export interface ConfigDiagnostics {
  inDocker: boolean;
  isInEvals: boolean;
  nodeVersion: string;
  platform: string;
  architecture: string;
  environmentType: 'development' | 'production' | 'test';
  availableMemory: number;
  diskSpace: number;
  networkAccess: boolean;
  permissionIssues: string[];
  missingDependencies: string[];
  configErrors: ConfigValidationError[];
}

export interface ConfigValidationError {
  path: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
  source: ConfigSource;
}

export interface ConfigPreview {
  profileId: string;
  profileName: string;
  effectiveConfig: ApplicationConfig;
  overrides: ConfigOverride[];
  warnings: string[];
  isValid: boolean;
}

export interface ConfigOverride {
  path: string;
  originalValue: any;
  newValue: any;
  source: ConfigSource;
  reason: string;
}

// Form state for configuration management
export interface ConfigFormState {
  config: ApplicationConfig;
  diagnostics: ConfigDiagnostics;
  selectedSection: keyof ApplicationConfig;
  selectedProfile?: string;
  preview?: ConfigPreview;
  isLoading: boolean;
  isDirty: boolean;
  isSaving: boolean;
  error: string | null;
  
  // Environment variable overrides
  envOverrides: Partial<EnvironmentVariables>;
  showEnvEditor: boolean;
  showDiagnostics: boolean;
  showPreview: boolean;
}
