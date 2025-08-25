/**
 * Profile Builder Page - M2 Implementation
 * Main interface for creating and editing browser profiles
 */

import React, { useEffect } from 'react';
import {
  Text,
  Button,
  Input,
  Textarea,
  Tab,
  TabList,
  Field,
  MessageBar,
  MessageBarBody,
  MessageBarTitle,
  Spinner,
  makeStyles,
  tokens,
} from '@fluentui/react-components';
import {
  Save24Regular,
  Play24Regular,
  Checkmark24Regular,
} from '@fluentui/react-icons';
import { useParams, useNavigate } from '@tanstack/react-router';

import { useProfileBuilderStore } from '@/stores/profileBuilderStore';
import { StealthTab } from '@/components/profile/StealthTab';
import { SecurityTab } from '@/components/profile/SecurityTab';
import { ViewportTab } from '@/components/profile/ViewportTab';
import { NetworkTab } from '@/components/profile/NetworkTab';
import { LaunchArgsTab } from '@/components/profile/LaunchArgsTab';
import { ChromeFlagInspector } from '@/components/profile/ChromeFlagInspector';
import { ProfileValidationError } from '@/types/profile';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: tokens.spacingVerticalL,
    borderBottom: `1px solid ${tokens.colorNeutralStroke2}`,
    backgroundColor: tokens.colorNeutralBackground1,
  },
  headerLeft: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
  },
  headerRight: {
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    alignItems: 'center',
  },
  basicInfo: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: tokens.spacingHorizontalL,
    padding: tokens.spacingVerticalL,
  },
  tabContainer: {
    display: 'flex',
    flexDirection: 'column',
    flex: 1,
    overflow: 'hidden',
  },
  tabContent: {
    flex: 1,
    overflow: 'auto',
  },
  validationPanel: {
    padding: tokens.spacingVerticalL,
    borderTop: `1px solid ${tokens.colorNeutralStroke2}`,
    backgroundColor: tokens.colorNeutralBackground2,
  },
  loadingOverlay: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    zIndex: 1000,
  },
});

const TabPanelContent: React.FC<React.PropsWithChildren<{ value: string; active: string }>> = ({ children, value, active }) => {
  if (value !== active) return null;
  return <>{children}</>;
};

export const ProfileBuilderPage: React.FC = () => {
  const styles = useStyles();
  const navigate = useNavigate();
  const { profileId } = useParams({ strict: false }) as { profileId?: string };
  
  const {
    currentProfile,
    activeTab,
    isDirty,
    validationResult,
    isLoading,
    isSaving,
    loadProfile,
    createNewProfile,
    updateProfile,
    validateProfile,
    saveProfile,
    setActiveTab,
    getComputedFlags,
  } = useProfileBuilderStore();

  useEffect(() => {
    if (profileId && profileId !== 'new') {
      loadProfile(profileId);
    } else if (profileId === 'new') {
      createNewProfile();
    }
  }, [profileId, loadProfile, createNewProfile]);

  useEffect(() => {
    if (currentProfile) {
      validateProfile();
    }
  }, [currentProfile, validateProfile]);

  const handleSave = async () => {
    try {
      await saveProfile();
      // Navigate to profiles list or show success message
    } catch (error) {
      console.error('Failed to save profile:', error);
    }
  };

  const handleTestRun = () => {
    if (currentProfile) {
      // TODO: Implement test run functionality
      console.log('Test run for profile:', currentProfile.name);
    }
  };

  const getValidationIntent = (severity: ProfileValidationError['severity']) => {
    switch (severity) {
      case 'error': return 'error' as const;
      case 'warning': return 'warning' as const;
      case 'info': return 'info' as const;
    }
  };

  if (isLoading) {
    return (
      <div className={styles.loadingOverlay}>
        <Spinner label="Loading profile..." />
      </div>
    );
  }

  if (!currentProfile) {
    return (
      <div style={{ padding: tokens.spacingVerticalXL, textAlign: 'center' }}>
        <Text>Profile not found or failed to load.</Text>
        <Button onClick={() => navigate({ to: '/', params: (prev) => prev, search: (prev) => prev })}>
          Back to Home
        </Button>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <Text size={600} weight="semibold">
            {profileId === 'new' ? 'Create New Profile' : 'Edit Profile'}
          </Text>
          <Text size={200}>
            Configure browser settings, stealth options, and launch parameters
          </Text>
        </div>
        <div className={styles.headerRight}>
          {isSaving && <Spinner size="small" />}
          <Button
            appearance="secondary"
            icon={<Play24Regular />}
            onClick={handleTestRun}
            disabled={!validationResult?.isValid || isSaving}
          >
            Test Run
          </Button>
          <Button
            appearance="primary"
            icon={<Save24Regular />}
            onClick={handleSave}
            disabled={!isDirty || !validationResult?.isValid || isSaving}
          >
            {isSaving ? 'Saving...' : 'Save Profile'}
          </Button>
        </div>
      </div>

      {/* Basic Info */}
      <div className={styles.basicInfo}>
        <Field label="Profile Name" required>
          <Input
            value={currentProfile.name}
            onChange={(_, data) => updateProfile({ name: data.value })}
            placeholder="Enter profile name"
          />
        </Field>
        <Field label="Description">
          <Textarea
            value={currentProfile.description || ''}
            onChange={(_, data) => updateProfile({ description: data.value })}
            placeholder="Optional description"
          />
        </Field>
      </div>

      {/* Tabs */}
      <div className={styles.tabContainer}>
        <TabList
          selectedValue={activeTab}
          onTabSelect={(_, data) => setActiveTab(data.value as string)}
        >
          <Tab value="stealth">Stealth</Tab>
          <Tab value="security">Security</Tab>
          <Tab value="viewport">Display/Viewport</Tab>
          <Tab value="network">Network/Proxy</Tab>
          <Tab value="recording">Recording/Tracing</Tab>
          <Tab value="persistence">Persistence</Tab>
          <Tab value="launchargs">Launch Args</Tab>
          <Tab value="inspector">Flag Inspector</Tab>
        </TabList>

        <div className={styles.tabContent}>
          <TabPanelContent value="stealth" active={activeTab}>
            <StealthTab
              settings={currentProfile.stealth}
              onChange={(stealth) => updateProfile({ stealth })}
            />
          </TabPanelContent>

          <TabPanelContent value="security" active={activeTab}>
            <SecurityTab
              settings={currentProfile.security}
              onChange={(security) => updateProfile({ security })}
            />
          </TabPanelContent>

          <TabPanelContent value="viewport" active={activeTab}>
            <ViewportTab
              settings={currentProfile.viewport}
              headless={currentProfile.headless}
              onChange={(viewport) => updateProfile({ viewport })}
              onHeadlessChange={(headless) => updateProfile({ headless })}
            />
          </TabPanelContent>

          <TabPanelContent value="network" active={activeTab}>
            <NetworkTab
              settings={currentProfile.network}
              onChange={(network) => updateProfile({ network })}
            />
          </TabPanelContent>

          <TabPanelContent value="recording" active={activeTab}>
            <div style={{ padding: tokens.spacingVerticalL }}>
              <Text>Recording tab - TODO: Implement RecordingTab component</Text>
            </div>
          </TabPanelContent>

          <TabPanelContent value="persistence" active={activeTab}>
            <div style={{ padding: tokens.spacingVerticalL }}>
              <Text>Persistence tab - TODO: Implement PersistenceTab component</Text>
            </div>
          </TabPanelContent>

          <TabPanelContent value="launchargs" active={activeTab}>
            <LaunchArgsTab
              settings={currentProfile.launchArgs}
              onChange={(launchArgs) => updateProfile({ launchArgs })}
            />
          </TabPanelContent>

          <TabPanelContent value="inspector" active={activeTab}>
            <div style={{ padding: tokens.spacingVerticalL }}>
              <ChromeFlagInspector flags={getComputedFlags()} />
            </div>
          </TabPanelContent>
        </div>
      </div>

      {/* Validation Panel */}
      {validationResult && validationResult.errors.length > 0 && (
        <div className={styles.validationPanel}>
          <Text size={400} weight="semibold" style={{ marginBottom: tokens.spacingVerticalS }}>
            Validation Results
          </Text>
          {validationResult.errors.map((error, index) => (
            <MessageBar
              key={index}
              intent={getValidationIntent(error.severity)}
              style={{ marginBottom: tokens.spacingVerticalS }}
            >
              <MessageBarBody>
                <MessageBarTitle>{error.field}</MessageBarTitle>
                {error.message}
              </MessageBarBody>
            </MessageBar>
          ))}
        </div>
      )}

      {validationResult?.isValid && (
        <div className={styles.validationPanel}>
          <MessageBar intent="success">
            <MessageBarBody>
              <MessageBarTitle>
                <Checkmark24Regular /> Profile Configuration Valid
              </MessageBarTitle>
              All settings have been validated successfully. Ready to save and use.
            </MessageBarBody>
          </MessageBar>
        </div>
      )}
    </div>
  );
};
