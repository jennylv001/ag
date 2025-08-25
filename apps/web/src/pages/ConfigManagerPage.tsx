/**
 * Config Manager Page - M4 Implementation
 * Main configuration management interface with tabs and comprehensive features
 */

import React from 'react';
import {
  TabList,
  Tab,
  Text,
  Button,
  MessageBar,
  MessageBarTitle,
  MessageBarBody,
  makeStyles,
  tokens,
  Spinner,
} from '@fluentui/react-components';
import {
  Settings24Regular,
  Database24Regular,
  BrainCircuit24Regular,
  Eye24Regular,
  Warning24Regular,
  Save24Regular,
  ArrowReset24Regular,
} from '@fluentui/react-icons';
import { useConfigManagerStore } from '@/stores/configManagerStore';
import SourcesPrecedenceView from '@/components/config/SourcesPrecedenceView';
import EnvironmentOverridesEditor from '@/components/config/EnvironmentOverridesEditor';
import AgentLLMEditors from '@/components/config/AgentLLMEditors';
import SystemDiagnostics from '@/components/config/SystemDiagnostics';
import ConfigPreviewApply from '@/components/config/ConfigPreviewApply';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    padding: tokens.spacingVerticalL,
    gap: tokens.spacingVerticalL,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerActions: {
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    alignItems: 'center',
  },
  dirtyIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXS,
  color: tokens.colorPaletteYellowForeground2,
  },
  tabContent: {
    flex: 1,
    overflow: 'auto',
    paddingTop: tokens.spacingVerticalM,
  },
  loadingContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '200px',
  },
});

const ConfigManagerPage: React.FC = () => {
  const styles = useStyles();
  const {
    // State
    config,
    diagnostics,
    selectedProfile,
    preview,
    isLoading,
    isDirty,
    isSaving,
    error,
    envOverrides,
    
    // Actions
    loadConfig,
    saveConfig,
    resetConfig,
    updateConfigValue,
    updateEnvOverride,
    removeEnvOverride,
    applyEnvOverrides,
    runDiagnostics,
    selectProfile,
    generatePreview,
    applyToProfile,
    setError,
  } = useConfigManagerStore() as any;

  // Mock profiles for demonstration
  const availableProfiles = [
    { id: 'profile-1', name: 'Development Profile' },
    { id: 'profile-2', name: 'Production Profile' },
    { id: 'profile-3', name: 'Testing Profile' },
  ];

  // Tab state
  const [activeTab, setActiveTab] = React.useState('sources');

  // Load initial data
  React.useEffect(() => {
    loadConfig();
    runDiagnostics();
  }, [loadConfig, runDiagnostics]);

  // Mock path selector (would use native file dialog in real implementation)
  const handleSelectPath = async (type: 'file' | 'folder'): Promise<string> => {
    // Simulate file dialog
    return new Promise((resolve) => {
      setTimeout(() => {
        const mockPath = type === 'file' 
          ? '/path/to/selected/file.txt'
          : '/path/to/selected/folder';
        resolve(mockPath);
      }, 500);
    });
  };

  const handleUpdateAgent = (key: string, value: any) => {
    updateConfigValue(`agent.${key}`, value, 'runtime');
  };

  const handleUpdateLLM = (key: string, value: any) => {
    updateConfigValue(`llm.${key}`, value, 'runtime');
  };

  const handleSave = async () => {
    try {
      await saveConfig();
      setError(null);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Save failed');
    }
  };

  const handleReset = () => {
    resetConfig();
    setError(null);
  };

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <Text size={600} weight="semibold">Configuration Manager</Text>
          <Text size={300}>
            Manage application configuration with source precedence and environment overrides
          </Text>
        </div>
        
        <div className={styles.headerActions}>
          {isDirty && (
            <div className={styles.dirtyIndicator}>
              <Warning24Regular />
              <Text size={200}>Unsaved changes</Text>
            </div>
          )}
          
          <Button
            appearance="secondary"
            icon={<ArrowReset24Regular />}
            onClick={handleReset}
            disabled={isLoading || !isDirty}
          >
            Reset
          </Button>
          
          <Button
            appearance="primary"
            icon={<Save24Regular />}
            onClick={handleSave}
            disabled={isLoading || !isDirty || isSaving}
          >
            {isSaving ? 'Saving...' : 'Save Configuration'}
          </Button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <MessageBar intent="error">
          <MessageBarTitle>Configuration Error</MessageBarTitle>
          <MessageBarBody>{error}</MessageBarBody>
        </MessageBar>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className={styles.loadingContainer}>
          <Spinner size="large" />
        </div>
      )}

      {/* Main Content */}
      {!isLoading && (
        <>
          <TabList
            selectedValue={activeTab}
            onTabSelect={(_, data) => setActiveTab(data.value as string)}
          >
            <Tab value="sources" icon={<Database24Regular />}>
              Sources & Precedence
            </Tab>
            <Tab value="environment" icon={<Settings24Regular />}>
              Environment Overrides
            </Tab>
            <Tab value="agent-llm" icon={<BrainCircuit24Regular />}>
              Agent & LLM
            </Tab>
            <Tab value="diagnostics" icon={<Warning24Regular />}>
              Diagnostics
            </Tab>
            <Tab value="preview" icon={<Eye24Regular />}>
              Preview & Apply
            </Tab>
          </TabList>

          <div className={styles.tabContent}>
            {activeTab === 'sources' && (
              <SourcesPrecedenceView config={config} />
            )}
            {activeTab === 'environment' && (
              <EnvironmentOverridesEditor
                overrides={envOverrides}
                onUpdateOverride={updateEnvOverride}
                onRemoveOverride={removeEnvOverride}
                onApplyOverrides={applyEnvOverrides}
              />
            )}
            {activeTab === 'agent-llm' && (
              <AgentLLMEditors
                agentConfig={config.agent}
                llmConfig={config.llm}
                onUpdateAgent={handleUpdateAgent}
                onUpdateLLM={handleUpdateLLM}
                onSelectPath={handleSelectPath}
              />
            )}
            {activeTab === 'diagnostics' && (
              <SystemDiagnostics
                diagnostics={diagnostics}
                onRefresh={runDiagnostics}
                isLoading={isLoading}
              />
            )}
            {activeTab === 'preview' && (
              <ConfigPreviewApply
                preview={preview}
                availableProfiles={availableProfiles}
                selectedProfile={selectedProfile}
                isGenerating={isLoading}
                isApplying={isSaving}
                onSelectProfile={selectProfile}
                onGeneratePreview={generatePreview}
                onApplyToProfile={applyToProfile}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default ConfigManagerPage;
