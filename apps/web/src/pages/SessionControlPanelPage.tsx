/**
 * Session Control Panel Page - M3 Implementation
 * Main interface for session management with split view
 */

import React, { useEffect, useState } from 'react';
import {
  makeStyles,
  tokens,
  Dialog,
  DialogSurface,
  DialogTitle,
  DialogContent,
  DialogBody,
  DialogActions,
  Button,
  Field,
  Dropdown,
  Option,
  Switch,
  Input,
  SpinButton,
  Text,
} from '@fluentui/react-components';
import { Add24Regular } from '@fluentui/react-icons';

import { useSessionControlStore } from '@/stores/sessionControlStore';
import { SessionsDataGrid } from '@/components/sessions/SessionsDataGrid';
import { SessionDetailView } from '@/components/sessions/SessionDetailView';
// import { BrowserSession } from '@/types/session';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    overflow: 'hidden',
  },
  splitterContainer: {
    flex: 1,
    overflow: 'hidden',
  },
  splitLayout: {
    display: 'grid',
    gridTemplateColumns: 'minmax(320px, 1fr) 6px minmax(300px, 1fr)',
    height: '100%',
  },
  splitterHandle: {
    backgroundColor: tokens.colorNeutralStroke2,
    cursor: 'col-resize',
  },
  leftPanel: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
  },
  rightPanel: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
  borderLeft: `1px solid ${tokens.colorNeutralStroke2}`,
  },
  emptyDetailView: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    padding: tokens.spacingVerticalXXL,
    textAlign: 'center',
    color: tokens.colorNeutralForeground3,
  },
  newSessionDialog: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
    minWidth: '400px',
  },
  configRow: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: tokens.spacingHorizontalM,
  },
});

const MOCK_PROFILES = [
  { id: 'profile-1', name: 'Default Profile' },
  { id: 'profile-2', name: 'Stealth Profile' },
  { id: 'profile-3', name: 'Mobile Profile' },
  { id: 'profile-4', name: 'Headless Profile' },
];

export const SessionControlPanelPage: React.FC = () => {
  const styles = useStyles();
  const [isNewSessionDialogOpen, setIsNewSessionDialogOpen] = useState(false);
  const [newSessionConfig, setNewSessionConfig] = useState({
    profileId: '',
    channel: 'chrome',
    headless: false,
    devtools: false,
    timeout: 30000,
    proxy: {
      enabled: false,
      server: '',
      username: '',
      password: '',
    },
  });

  const {
    sessions,
    selectedSession,
    filters,
    stats,
    isLoading,
    activeTab,
    isLaunching,
    isStopping,
    error,
    loadSessions,
    createSession,
    startSession,
    stopSession,
    selectSession,
    setFilters,
    setActiveTab,
    refreshSession,
    openDevTools,
    takeScreenshot,
    downloadArtifact,
    deleteArtifact,
  } = useSessionControlStore();

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Auto-refresh running sessions
  useEffect(() => {
    const interval = setInterval(() => {
      sessions
        .filter(s => s.status === 'running')
        .forEach(s => refreshSession(s.id));
    }, 5000);

    return () => clearInterval(interval);
  }, [sessions, refreshSession]);

  const handleCreateSession = async () => {
    if (!newSessionConfig.profileId) return;

    try {
      const launchConfig = {
        channel: newSessionConfig.channel,
        headless: newSessionConfig.headless,
        devtools: newSessionConfig.devtools,
        timeout: newSessionConfig.timeout,
        proxy: newSessionConfig.proxy.enabled ? {
          server: newSessionConfig.proxy.server,
          username: newSessionConfig.proxy.username || undefined,
          password: newSessionConfig.proxy.password || undefined,
        } : undefined,
      };

      await createSession(newSessionConfig.profileId, launchConfig);
      setIsNewSessionDialogOpen(false);
      setNewSessionConfig({
        profileId: '',
        channel: 'chrome',
        headless: false,
        devtools: false,
        timeout: 30000,
        proxy: {
          enabled: false,
          server: '',
          username: '',
          password: '',
        },
      });
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    // Mock delete - in real implementation would call API
    console.log('Delete session:', sessionId);
  };

  return (
    <div className={styles.container}>
      <div className={styles.splitterContainer}>
        <div className={styles.splitLayout}>
          <div className={styles.leftPanel}>
            <SessionsDataGrid
              sessions={sessions}
              stats={stats}
              filters={filters}
              isLoading={isLoading}
              selectedSession={selectedSession}
              onSelectSession={selectSession}
              onFiltersChange={setFilters}
              onStartSession={startSession}
              onStopSession={stopSession}
              onDeleteSession={handleDeleteSession}
              onCreateSession={() => setIsNewSessionDialogOpen(true)}
            />
          </div>
          <div className={styles.splitterHandle} aria-hidden />
          <div className={styles.rightPanel}>
            {selectedSession ? (
              <SessionDetailView
                session={selectedSession}
                activeTab={activeTab}
                onTabChange={setActiveTab}
                onStartSession={() => startSession(selectedSession.id)}
                onStopSession={() => stopSession(selectedSession.id)}
                onOpenDevTools={() => openDevTools(selectedSession.id)}
                onTakeScreenshot={() => takeScreenshot(selectedSession.id)}
                onDownloadArtifact={downloadArtifact}
                onDeleteArtifact={deleteArtifact}
                isLoading={isLaunching || isStopping}
              />
            ) : (
              <div className={styles.emptyDetailView}>
                <Add24Regular style={{ fontSize: '48px', marginBottom: tokens.spacingVerticalM }} />
                <Text size={500} weight="semibold">Select a Session</Text>
                <Text size={300}>
                  Choose a session from the list to view details and controls
                </Text>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* New Session Dialog */}
      <Dialog 
        open={isNewSessionDialogOpen} 
        onOpenChange={(_, data) => setIsNewSessionDialogOpen(data.open)}
      >
        <DialogSurface>
          <DialogTitle>Create New Session</DialogTitle>
          <DialogContent>
            <DialogBody>
              <div className={styles.newSessionDialog}>
                <Field label="Profile" required>
                  <Dropdown
                    placeholder="Select a profile"
                    value={newSessionConfig.profileId}
                    onOptionSelect={(_, data) => 
                      setNewSessionConfig(prev => ({ ...prev, profileId: data.optionValue || '' }))
                    }
                  >
                    {MOCK_PROFILES.map(profile => (
                      <Option key={profile.id} value={profile.id}>
                        {profile.name}
                      </Option>
                    ))}
                  </Dropdown>
                </Field>

                <div className={styles.configRow}>
                  <Field label="Browser Channel">
                    <Dropdown
                      value={newSessionConfig.channel}
                      onOptionSelect={(_, data) => 
                        setNewSessionConfig(prev => ({ ...prev, channel: data.optionValue || 'chrome' }))
                      }
                    >
                      <Option value="chrome">Chrome</Option>
                      <Option value="msedge">Edge</Option>
                      <Option value="chromium">Chromium</Option>
                    </Dropdown>
                  </Field>

                  <Field label="Timeout (ms)">
                    <SpinButton
                      value={newSessionConfig.timeout}
                      onChange={(_, data) => 
                        setNewSessionConfig(prev => ({ ...prev, timeout: data.value || 30000 }))
                      }
                      min={5000}
                      max={300000}
                      step={5000}
                    />
                  </Field>
                </div>

                <div className={styles.configRow}>
                  <Field>
                    <Switch
                      checked={newSessionConfig.headless}
                      onChange={(_, data) => 
                        setNewSessionConfig(prev => ({ ...prev, headless: data.checked }))
                      }
                      label="Headless Mode"
                    />
                  </Field>

                  <Field>
                    <Switch
                      checked={newSessionConfig.devtools}
                      onChange={(_, data) => 
                        setNewSessionConfig(prev => ({ ...prev, devtools: data.checked }))
                      }
                      label="Open DevTools"
                    />
                  </Field>
                </div>

                <Field>
                  <Switch
                    checked={newSessionConfig.proxy.enabled}
                    onChange={(_, data) => 
                      setNewSessionConfig(prev => ({ 
                        ...prev, 
                        proxy: { ...prev.proxy, enabled: data.checked }
                      }))
                    }
                    label="Use Proxy"
                  />
                </Field>

                {newSessionConfig.proxy.enabled && (
                  <>
                    <Field label="Proxy Server" required>
                      <Input
                        value={newSessionConfig.proxy.server}
                        onChange={(_, data) => 
                          setNewSessionConfig(prev => ({ 
                            ...prev, 
                            proxy: { ...prev.proxy, server: data.value }
                          }))
                        }
                        placeholder="http://proxy.example.com:8080"
                      />
                    </Field>

                    <div className={styles.configRow}>
                      <Field label="Username">
                        <Input
                          value={newSessionConfig.proxy.username}
                          onChange={(_, data) => 
                            setNewSessionConfig(prev => ({ 
                              ...prev, 
                              proxy: { ...prev.proxy, username: data.value }
                            }))
                          }
                          placeholder="Optional"
                        />
                      </Field>

                      <Field label="Password">
                        <Input
                          type="password"
                          value={newSessionConfig.proxy.password}
                          onChange={(_, data) => 
                            setNewSessionConfig(prev => ({ 
                              ...prev, 
                              proxy: { ...prev.proxy, password: data.value }
                            }))
                          }
                          placeholder="Optional"
                        />
                      </Field>
                    </div>
                  </>
                )}
              </div>
            </DialogBody>
            <DialogActions>
              <Button
                appearance="secondary"
                onClick={() => setIsNewSessionDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                appearance="primary"
                onClick={handleCreateSession}
                disabled={!newSessionConfig.profileId || isLaunching}
              >
                {isLaunching ? 'Creating...' : 'Create Session'}
              </Button>
            </DialogActions>
          </DialogContent>
        </DialogSurface>
      </Dialog>

      {/* Error Display */}
      {error && (
        <div style={{ 
          position: 'fixed', 
          bottom: tokens.spacingVerticalM, 
          right: tokens.spacingVerticalM,
          zIndex: 1000 
        }}>
          <Text style={{ color: tokens.colorPaletteRedForeground1 }}>
            {error}
          </Text>
        </div>
      )}
    </div>
  );
};
