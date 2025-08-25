/**
 * Config Manager Store - M4 Implementation
 * Zustand store for configuration management with sources and precedence
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { 
  ConfigFormState, 
  ApplicationConfig, 
  ConfigDiagnostics, 
  ConfigPreview,
  ConfigValue,
  ConfigSource,
  EnvironmentVariables
} from '../types/config';

interface ConfigManagerStore extends ConfigFormState {
  // Core actions
  loadConfig: () => Promise<void>;
  saveConfig: () => Promise<void>;
  resetConfig: () => void;
  
  // Section management
  selectSection: (section: keyof ApplicationConfig) => void;
  
  // Configuration value updates
  updateConfigValue: <T>(path: string, value: T, source?: ConfigSource) => void;
  
  // Environment overrides
  updateEnvOverride: <K extends keyof EnvironmentVariables>(
    key: K, 
    value: EnvironmentVariables[K]
  ) => void;
  removeEnvOverride: (key: keyof EnvironmentVariables) => void;
  applyEnvOverrides: () => void;
  
  // Diagnostics
  runDiagnostics: () => Promise<void>;
  
  // Profile integration
  selectProfile: (profileId: string) => void;
  generatePreview: () => Promise<void>;
  applyToProfile: () => Promise<void>;
  
  // UI state
  toggleEnvEditor: () => void;
  toggleDiagnostics: () => void;
  togglePreview: () => void;
  setError: (error: string | null) => void;
}

// Mock data generators
const createConfigValue = <T>(
  value: T, 
  source: ConfigSource = 'default',
  description?: string
): ConfigValue<T> => ({
  value,
  source,
  priority: getPriorityForSource(source),
  isOverridden: source !== 'default',
  originalValue: source !== 'default' ? undefined : value,
  description,
});

const getPriorityForSource = (source: ConfigSource): number => {
  const priorities: Record<ConfigSource, number> = { default: 1, database: 2, environment: 3, runtime: 4 };
  return priorities[source];
};

const generateMockConfig = (): ApplicationConfig => ({
  agent: {
    maxActions: createConfigValue(100, 'default', 'Maximum number of actions per session'),
    thinkTimeout: createConfigValue(30000, 'environment', 'Timeout for agent thinking phase'),
    actionTimeout: createConfigValue(10000, 'default', 'Timeout for individual actions'),
    debugMode: createConfigValue(false, 'environment', 'Enable debug logging'),
    retryAttempts: createConfigValue(3, 'default', 'Number of retry attempts'),
    enableTelemetry: createConfigValue(true, 'database', 'Enable telemetry collection'),
    workingDirectory: createConfigValue('/tmp/browser-use', 'default', 'Working directory'),
  },
  llm: {
    model: createConfigValue('gpt-4o', 'environment', 'LLM model to use'),
    apiKey: createConfigValue('sk-...', 'environment', 'API key for LLM service'),
    baseUrl: createConfigValue('https://api.openai.com/v1', 'default', 'Base URL for API'),
    temperature: createConfigValue(0.7, 'database', 'Sampling temperature'),
    maxTokens: createConfigValue(4096, 'default', 'Maximum tokens per request'),
    timeout: createConfigValue(30000, 'default', 'Request timeout'),
    retryAttempts: createConfigValue(3, 'default', 'Number of retry attempts'),
    fallbackModel: createConfigValue('gpt-3.5-turbo', 'default', 'Fallback model'),
  },
  browser: {
    headless: createConfigValue(true, 'environment', 'Run browser in headless mode'),
    allowedDomains: createConfigValue(['*'], 'environment', 'Allowed domains list'),
    stealthMode: createConfigValue(true, 'database', 'Enable stealth features'),
    timeout: createConfigValue(30000, 'default', 'Page load timeout'),
    proxy: createConfigValue('', 'default', 'Proxy server configuration'),
    userDataDir: createConfigValue('', 'default', 'User data directory'),
    executablePath: createConfigValue('', 'default', 'Chrome executable path'),
    devtools: createConfigValue(false, 'runtime', 'Open DevTools'),
  },
  system: {
    logLevel: createConfigValue('info', 'environment', 'Logging level'),
    debugMode: createConfigValue(false, 'environment', 'System debug mode'),
    dataDirectory: createConfigValue('./data', 'default', 'Data storage directory'),
    cacheDirectory: createConfigValue('./cache', 'default', 'Cache directory'),
    maxLogFiles: createConfigValue(10, 'default', 'Maximum log files to keep'),
    enableMetrics: createConfigValue(true, 'database', 'Enable metrics collection'),
  },
});

const generateMockDiagnostics = (): ConfigDiagnostics => {
  // Safe env helpers for browser + node
  const viteEnv = (typeof import.meta !== 'undefined' && (import.meta as any).env) || {};
  const nodeEnv = (typeof process !== 'undefined' && (process as any).env) || {};
  const getBool = (viteKey: string, nodeKey?: string) => {
    const v = viteEnv[viteKey] ?? nodeEnv[(nodeKey || viteKey)];
    if (v === undefined) return false;
    return String(v).toLowerCase() === 'true';
  };

  return {
    inDocker: getBool('VITE_IN_DOCKER', 'IN_DOCKER'),
    isInEvals: getBool('VITE_IS_IN_EVALS', 'IS_IN_EVALS'),
    nodeVersion:
      (typeof process !== 'undefined' && (process as any).versions?.node) || 'browser',
    platform:
      (typeof navigator !== 'undefined' && (navigator.platform || 'browser')) ||
      (typeof process !== 'undefined' && (process as any).platform) ||
      'browser',
    architecture:
      (typeof process !== 'undefined' && (process as any).arch) || 'x64',
    environmentType: viteEnv.MODE || 'development',
    availableMemory: 8192,
    diskSpace: 50000,
    networkAccess: true,
    permissionIssues: [],
    missingDependencies: [],
    configErrors: [
      {
        path: 'llm.apiKey',
        message: 'API key is using placeholder value',
        severity: 'warning',
        source: 'environment',
      },
    ],
  };
};

export const useConfigManagerStore = create<ConfigManagerStore>()(
  devtools(
    (set, get) => ({
      // Initial state
      config: generateMockConfig(),
      diagnostics: generateMockDiagnostics(),
      selectedSection: 'agent',
      selectedProfile: undefined,
      preview: undefined,
      isLoading: false,
      isDirty: false,
      isSaving: false,
      error: null,
      envOverrides: {},
      showEnvEditor: false,
      showDiagnostics: false,
      showPreview: false,

      // Core actions
      loadConfig: async () => {
        set({ isLoading: true, error: null });
        try {
          // Simulate API call
          await new Promise(resolve => setTimeout(resolve, 500));
          const config = generateMockConfig();
          set({ config, isLoading: false, isDirty: false });
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to load config',
            isLoading: false 
          });
        }
      },

      saveConfig: async () => {
        set({ isSaving: true, error: null });
        try {
          // Simulate API call
          await new Promise(resolve => setTimeout(resolve, 1000));
          set({ isSaving: false, isDirty: false });
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to save config',
            isSaving: false 
          });
        }
      },

      resetConfig: () => {
        const config = generateMockConfig();
        set({ config, isDirty: false, error: null });
      },

      // Section management
      selectSection: (section) => {
        set({ selectedSection: section });
      },

      // Configuration value updates
      updateConfigValue: (path, value, source = 'runtime') => {
        const { config } = get();
        const pathParts = path.split('.');
        
        if (pathParts.length === 2) {
          const [section, key] = pathParts;
          const sectionConfig = config[section as keyof ApplicationConfig];
          
          if (sectionConfig && typeof sectionConfig === 'object' && key in sectionConfig) {
            const currentValue = (sectionConfig as any)[key] as ConfigValue;
            const newValue = createConfigValue(
              value,
              source,
              currentValue.description
            );
            newValue.originalValue = currentValue.originalValue ?? currentValue.value;
            newValue.isOverridden = source !== 'default';
            
            const updatedConfig = {
              ...config,
              [section]: {
                ...sectionConfig,
                [key]: newValue,
              },
            };
            
            set({ config: updatedConfig, isDirty: true });
          }
        }
      },

      // Environment overrides
      updateEnvOverride: (key, value) => {
        const { envOverrides } = get();
        set({ 
          envOverrides: { ...envOverrides, [key]: value },
          isDirty: true
        });
      },

      removeEnvOverride: (key) => {
        const { envOverrides } = get();
        const updated = { ...envOverrides };
        delete updated[key];
        set({ envOverrides: updated, isDirty: true });
      },

      applyEnvOverrides: () => {
        const { envOverrides } = get();
        
        // Apply environment overrides to configuration
        Object.entries(envOverrides).forEach(([key, value]) => {
          const mapping = getConfigPathForEnvVar(key as keyof EnvironmentVariables);
          if (mapping) {
            get().updateConfigValue(mapping, value, 'environment');
          }
        });
      },

      // Diagnostics
      runDiagnostics: async () => {
        set({ isLoading: true });
        try {
          await new Promise(resolve => setTimeout(resolve, 1000));
          const diagnostics = generateMockDiagnostics();
          set({ diagnostics, isLoading: false });
        } catch (error) {
          set({ 
            error: 'Failed to run diagnostics',
            isLoading: false 
          });
        }
      },

      // Profile integration
      selectProfile: (profileId) => {
        set({ selectedProfile: profileId });
      },

      generatePreview: async () => {
        const { config, selectedProfile } = get();
        if (!selectedProfile) return;

        set({ isLoading: true });
        try {
          await new Promise(resolve => setTimeout(resolve, 500));
          
          const preview: ConfigPreview = {
            profileId: selectedProfile,
            profileName: `Profile ${selectedProfile}`,
            effectiveConfig: config,
            overrides: [
              {
                path: 'browser.headless',
                originalValue: true,
                newValue: false,
                source: 'runtime',
                reason: 'Profile override for debugging',
              },
            ],
            warnings: ['Some stealth features may be disabled'],
            isValid: true,
          };
          
          set({ preview, isLoading: false });
        } catch (error) {
          set({ 
            error: 'Failed to generate preview',
            isLoading: false 
          });
        }
      },

      applyToProfile: async () => {
        const { selectedProfile } = get();
        if (!selectedProfile) return;

        set({ isSaving: true });
        try {
          await new Promise(resolve => setTimeout(resolve, 1000));
          set({ isSaving: false });
        } catch (error) {
          set({ 
            error: 'Failed to apply config to profile',
            isSaving: false 
          });
        }
      },

      // UI state
      toggleEnvEditor: () => {
        set(state => ({ showEnvEditor: !state.showEnvEditor }));
      },

      toggleDiagnostics: () => {
        set(state => ({ showDiagnostics: !state.showDiagnostics }));
      },

      togglePreview: () => {
        set(state => ({ showPreview: !state.showPreview }));
      },

      setError: (error) => {
        set({ error });
      },
    }),
    { name: 'config-manager-store' }
  )
);

// Helper function to map environment variables to config paths
const getConfigPathForEnvVar = (envVar: keyof EnvironmentVariables): string | null => {
  const mapping: Partial<Record<keyof EnvironmentVariables, string>> = {
    BROWSER_USE_HEADLESS: 'browser.headless',
    BROWSER_USE_ALLOWED_DOMAINS: 'browser.allowedDomains',
    BROWSER_USE_STEALTH: 'browser.stealthMode',
    BROWSER_USE_TIMEOUT: 'browser.timeout',
    BROWSER_USE_PROXY: 'browser.proxy',
    LLM_MODEL: 'llm.model',
    LLM_API_KEY: 'llm.apiKey',
    LLM_BASE_URL: 'llm.baseUrl',
    LLM_TEMPERATURE: 'llm.temperature',
    LLM_MAX_TOKENS: 'llm.maxTokens',
    AGENT_MAX_ACTIONS: 'agent.maxActions',
    AGENT_THINK_TIMEOUT: 'agent.thinkTimeout',
    AGENT_ACTION_TIMEOUT: 'agent.actionTimeout',
    AGENT_DEBUG: 'agent.debugMode',
    LOG_LEVEL: 'system.logLevel',
    DEBUG_MODE: 'system.debugMode',
    DATA_DIR: 'system.dataDirectory',
    CACHE_DIR: 'system.cacheDirectory',
  };
  
  return mapping[envVar] || null;
};
