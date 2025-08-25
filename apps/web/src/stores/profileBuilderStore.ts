/**
 * Profile Builder Store - M2 Implementation
 * Manages profile editing state and validation
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { 
  BrowserProfile, 
  ProfileFormState, 
  ProfileValidationResult, 
  ProfileValidationError,
  ChromeFlag 
} from '../types/profile';

interface ProfileBuilderStore extends ProfileFormState {
  // Actions
  loadProfile: (profileId: string) => Promise<void>;
  createNewProfile: () => void;
  updateProfile: (updates: Partial<BrowserProfile>) => void;
  validateProfile: () => ProfileValidationResult;
  saveProfile: () => Promise<void>;
  setActiveTab: (tab: string) => void;
  resetForm: () => void;
  
  // Computed values
  getComputedFlags: () => ChromeFlag[];
}

const createDefaultProfile = (): BrowserProfile => ({
  id: crypto.randomUUID(),
  name: 'New Profile',
  description: '',
  headless: false,
  stealth: {
    enabled: true,
    advanced_stealth: false,
    timezone: 'America/New_York',
    locale: 'en-US',
  },
  security: {
    bypassCSP: false,
    ignoreHTTPSErrors: false,
    acceptDownloads: true,
    javaScriptEnabled: true,
    webSecurity: true,
    permissions: [],
  },
  viewport: {
    width: 1920,
    height: 1080,
    deviceScaleFactor: 1,
    isMobile: false,
    hasTouch: false,
  },
  network: {
    extraHTTPHeaders: {},
    offline: false,
  },
  recording: {},
  persistence: {},
  launchArgs: {
    args: [],
    ignoreDefaultArgs: false,
    env: {},
    timeout: 30000,
    chromiumSandbox: true,
  },
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  isDefault: false,
});

export const useProfileBuilderStore = create<ProfileBuilderStore>()(
  devtools(
    (set, get) => ({
      // Initial state
      currentProfile: null,
      isDirty: false,
      activeTab: 'stealth',
      validationResult: null,
      isLoading: false,
      isSaving: false,

      // Actions
      loadProfile: async (profileId: string) => {
        set({ isLoading: true });
        try {
          // Mock API call - replace with actual backend call
          await new Promise(resolve => setTimeout(resolve, 500));
          
          // For now, return default profile with the requested ID
          const profile = { ...createDefaultProfile(), id: profileId, name: `Profile ${profileId}` };
          
          set({ 
            currentProfile: profile, 
            isDirty: false, 
            isLoading: false,
            validationResult: null 
          });
        } catch (error) {
          console.error('Failed to load profile:', error);
          set({ isLoading: false });
        }
      },

      createNewProfile: () => {
        const newProfile = createDefaultProfile();
        set({ 
          currentProfile: newProfile, 
          isDirty: true, 
          validationResult: null 
        });
      },

      updateProfile: (updates: Partial<BrowserProfile>) => {
        const { currentProfile } = get();
        if (!currentProfile) return;

        const updatedProfile = {
          ...currentProfile,
          ...updates,
          updatedAt: new Date().toISOString(),
        };

        set({ 
          currentProfile: updatedProfile, 
          isDirty: true,
          validationResult: null 
        });
      },

      validateProfile: (): ProfileValidationResult => {
        const { currentProfile } = get();
        if (!currentProfile) {
          return { isValid: false, errors: [], computedFlags: [] };
        }

        const errors: ProfileValidationError[] = [];
        const computedFlags: string[] = [];

        // Validation Rule: advanced_stealth ⇒ stealth
        if (currentProfile.stealth.advanced_stealth && !currentProfile.stealth.enabled) {
          errors.push({
            field: 'stealth.advanced_stealth',
            message: 'Advanced stealth requires stealth to be enabled',
            severity: 'error'
          });
        }

        // Validation Rule: headless ↔ viewport constraints
        if (currentProfile.headless && currentProfile.viewport.isMobile) {
          errors.push({
            field: 'viewport.isMobile',
            message: 'Mobile viewport simulation not supported in headless mode',
            severity: 'warning'
          });
        }

        // Validation Rule: storage_state vs user_data_dir warning
        if (currentProfile.persistence.storageState && currentProfile.persistence.userDataDir) {
          errors.push({
            field: 'persistence',
            message: 'Using both storage_state and user_data_dir may cause conflicts',
            severity: 'warning'
          });
        }

        // Computed flags generation
        if (currentProfile.stealth.enabled) {
          computedFlags.push('--disable-blink-features=AutomationControlled');
          computedFlags.push('--disable-dev-shm-usage');
        }

        if (currentProfile.stealth.advanced_stealth) {
          computedFlags.push('--disable-web-security');
          computedFlags.push('--disable-features=VizDisplayCompositor');
        }

        if (currentProfile.headless) {
          computedFlags.push('--headless=new');
          computedFlags.push('--disable-gpu');
        }

        const result = {
          isValid: errors.filter(e => e.severity === 'error').length === 0,
          errors,
          computedFlags
        };

        set({ validationResult: result });
        return result;
      },

      saveProfile: async () => {
        const { currentProfile, validateProfile } = get();
        if (!currentProfile) return;

        const validation = validateProfile();
        if (!validation.isValid) {
          throw new Error('Cannot save profile with validation errors');
        }

        set({ isSaving: true });
        try {
          // Mock API call - replace with actual backend call
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          set({ 
            isDirty: false, 
            isSaving: false,
            currentProfile: {
              ...currentProfile,
              updatedAt: new Date().toISOString(),
            }
          });
        } catch (error) {
          console.error('Failed to save profile:', error);
          set({ isSaving: false });
          throw error;
        }
      },

      setActiveTab: (tab: string) => {
        set({ activeTab: tab });
      },

      resetForm: () => {
        set({ 
          currentProfile: null, 
          isDirty: false, 
          validationResult: null,
          activeTab: 'stealth'
        });
      },

      getComputedFlags: (): ChromeFlag[] => {
        const { currentProfile } = get();
        if (!currentProfile) return [];

        const flags: ChromeFlag[] = [];

        // Stealth flags
        if (currentProfile.stealth.enabled) {
          flags.push({
            flag: '--disable-blink-features',
            value: 'AutomationControlled',
            source: 'stealth',
            rationale: 'Removes automation detection markers'
          });
          flags.push({
            flag: '--disable-dev-shm-usage',
            value: true,
            source: 'stealth',
            rationale: 'Prevents shared memory issues in containerized environments'
          });
        }

        if (currentProfile.stealth.advanced_stealth) {
          flags.push({
            flag: '--disable-web-security',
            value: true,
            source: 'stealth',
            rationale: 'Bypasses CORS and security policies for advanced stealth'
          });
          flags.push({
            flag: '--disable-features',
            value: 'VizDisplayCompositor',
            source: 'stealth',
            rationale: 'Disables compositor to reduce detection fingerprints'
          });
        }

        // Security flags
        if (currentProfile.security.bypassCSP) {
          flags.push({
            flag: '--disable-web-security',
            value: true,
            source: 'security',
            rationale: 'Disables Content Security Policy enforcement'
          });
        }

        // Viewport flags
        if (currentProfile.headless) {
          flags.push({
            flag: '--headless',
            value: 'new',
            source: 'viewport',
            rationale: 'Runs browser in headless mode with new implementation'
          });
          flags.push({
            flag: '--disable-gpu',
            value: true,
            source: 'viewport',
            rationale: 'Disables GPU acceleration in headless mode'
          });
        }

        // Custom launch args
        currentProfile.launchArgs.args.forEach(arg => {
          const [flag, value] = arg.split('=');
          flags.push({
            flag,
            value: value || true,
            source: 'args',
            rationale: 'User-defined launch argument'
          });
        });

        return flags;
      },
    }),
    {
      name: 'profile-builder-store',
    }
  )
);
