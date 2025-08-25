/**
 * Network Tab - M2 Implementation
 * Configuration for network settings, proxy, and headers
 */

import React from 'react';
import {
  Field,
  Switch,
  Input,
  Text,
  makeStyles,
  tokens,
} from '@fluentui/react-components';
import { NetworkSettings } from '../../types/profile';
import { KeyValueEditor, FileFolderPicker } from '../editors/CompositeEditors';

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
  threeColumn: {
    display: 'grid',
    gridTemplateColumns: '2fr 1fr 1fr',
    gap: tokens.spacingHorizontalM,
  },
});

interface NetworkTabProps {
  settings: NetworkSettings;
  onChange: (settings: NetworkSettings) => void;
}

export const NetworkTab: React.FC<NetworkTabProps> = ({ settings, onChange }) => {
  const styles = useStyles();

  const handleChange = (field: keyof NetworkSettings, value: any) => {
    onChange({ ...settings, [field]: value });
  };

  const handleProxyChange = (field: keyof NonNullable<NetworkSettings['proxy']>, value: any) => {
    const current = settings.proxy;
    if (field === 'server') {
      const server = String(value || '').trim();
      if (!server) {
        // Remove entire proxy config if server cleared
        onChange({ ...settings, proxy: undefined });
        return;
      }
      const next = { server } as NonNullable<NetworkSettings['proxy']>;
      if (current?.username) next.username = current.username;
      if (current?.password) next.password = current.password;
      if (current?.bypass) next.bypass = current.bypass;
      onChange({ ...settings, proxy: next });
      return;
    }
    // For non-server fields, only apply if we already have a proxy server set
    if (current?.server) {
      const next: NonNullable<NetworkSettings['proxy']> = { ...current };
      (next as any)[field] = value;
      onChange({ ...settings, proxy: next });
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.section}>
        <Text size={500} weight="semibold">Proxy Configuration</Text>
        
        <div className={styles.threeColumn}>
          <Field label="Proxy Server">
            <Input
              value={settings.proxy?.server || ''}
              onChange={(_, data) => handleProxyChange('server', data.value)}
              placeholder="http://proxy.example.com:8080"
            />
          </Field>

          <Field label="Username">
            <Input
              value={settings.proxy?.username || ''}
              onChange={(_, data) => handleProxyChange('username', data.value)}
              placeholder="Optional"
              disabled={!settings.proxy?.server}
            />
          </Field>

          <Field label="Password">
            <Input
              type="password"
              value={settings.proxy?.password || ''}
              onChange={(_, data) => handleProxyChange('password', data.value)}
              placeholder="Optional"
              disabled={!settings.proxy?.server}
            />
          </Field>
        </div>

        <Field label="Bypass List (comma-separated)">
          <Input
            value={settings.proxy?.bypass?.join(', ') || ''}
            onChange={(_, data) => {
              const bypass = data.value ? data.value.split(',').map(s => s.trim()).filter(Boolean) : [];
              handleProxyChange('bypass', bypass);
            }}
            placeholder="localhost, 127.0.0.1, *.internal.com"
            disabled={!settings.proxy?.server}
          />
        </Field>
      </div>

      <div className={styles.section}>
        <Text size={500} weight="semibold">HTTP Headers</Text>
        
        <KeyValueEditor
          data={settings.extraHTTPHeaders}
          onChange={(headers) => handleChange('extraHTTPHeaders', headers)}
          title="Extra HTTP Headers"
          keyPlaceholder="Header-Name"
          valuePlaceholder="Header Value"
        />
        
        <Text size={200}>
          Custom headers sent with every request (e.g., Authorization, X-Custom-Header)
        </Text>
      </div>

      <div className={styles.section}>
        <Text size={500} weight="semibold">User Agent &amp; Downloads</Text>
        
        <Field label="Custom User Agent">
          <Input
            value={settings.userAgent || ''}
            onChange={(_, data) => handleChange('userAgent', data.value)}
            placeholder="Leave empty to use default"
          />
        </Field>

        <div className={styles.row}>
          <FileFolderPicker
            type="folder"
            label="Download Directory"
            value={settings.downloadPath}
            onChange={(path) => handleChange('downloadPath', path)}
            placeholder="Default downloads folder"
          />

          <Field>
            <Switch
              checked={settings.offline}
              onChange={(_, data) => handleChange('offline', data.checked)}
              label="Offline Mode"
            />
            <Text size={200}>
              Simulate offline network condition
            </Text>
          </Field>
        </div>
      </div>
    </div>
  );
};
