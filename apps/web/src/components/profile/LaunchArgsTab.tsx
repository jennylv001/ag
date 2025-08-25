/**
 * Launch Args Tab - M2 Implementation
 * Configuration for Chrome launch arguments and environment variables
 */

import React from 'react';
import {
  Field,
  Switch,
  SpinButton,
  Text,
  makeStyles,
  tokens,
} from '@fluentui/react-components';
import { LaunchArgsSettings } from '../../types/profile';
import { KeyValueEditor, ListEditor } from '../editors/CompositeEditors';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalL,
    padding: tokens.spacingVerticalL,
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
  },
  row: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: tokens.spacingHorizontalL,
  },
});

interface LaunchArgsTabProps {
  settings: LaunchArgsSettings;
  onChange: (settings: LaunchArgsSettings) => void;
}

export const LaunchArgsTab: React.FC<LaunchArgsTabProps> = ({ settings, onChange }) => {
  const styles = useStyles();

  const handleChange = (field: keyof LaunchArgsSettings, value: any) => {
    onChange({ ...settings, [field]: value });
  };

  return (
    <div className={styles.container}>
      <div className={styles.section}>
        <Text size={500} weight="semibold">Chrome Launch Arguments</Text>
        
        <ListEditor
          data={settings.args}
          onChange={(args) => handleChange('args', args)}
          title="Custom Launch Arguments"
          placeholder="--flag-name=value or --flag-name"
          itemLabel="Argument"
        />
        
        <Text size={200}>
          Example arguments: --disable-extensions, --no-sandbox, --disable-dev-shm-usage
        </Text>
      </div>

      <div className={styles.section}>
        <Text size={500} weight="semibold">Environment Variables</Text>
        
        <KeyValueEditor
          data={settings.env}
          onChange={(env) => handleChange('env', env)}
          title="Environment Variables"
          keyPlaceholder="VARIABLE_NAME"
          valuePlaceholder="value"
        />
        
        <Text size={200}>
          Set environment variables for the browser process (e.g., DISPLAY, PATH modifications)
        </Text>
      </div>

      <div className={styles.section}>
        <Text size={500} weight="semibold">Launch Configuration</Text>
        
        <div className={styles.row}>
          <Field label="Launch Timeout (ms)">
            <SpinButton
              value={settings.timeout}
              onChange={(_, data) => handleChange('timeout', data.value || 30000)}
              min={1000}
              max={120000}
              step={1000}
            />
          </Field>

          <Field>
            <Switch
              checked={settings.chromiumSandbox}
              onChange={(_, data) => handleChange('chromiumSandbox', data.checked)}
              label="Chromium Sandbox"
            />
            <Text size={200}>
              Enable Chromium's security sandbox (disable for Docker environments)
            </Text>
          </Field>
        </div>

        <Field>
          <Switch
            checked={typeof settings.ignoreDefaultArgs === 'boolean' ? settings.ignoreDefaultArgs : false}
            onChange={(_, data) => handleChange('ignoreDefaultArgs', data.checked)}
            label="Ignore All Default Arguments"
          />
          <Text size={200}>
            Skip Playwright's default Chrome arguments (advanced users only)
          </Text>
        </Field>
      </div>
    </div>
  );
};
