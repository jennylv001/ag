/**
 * Security Tab - M2 Implementation
 * Configuration for security settings and permissions
 */

import React from 'react';
import {
  Field,
  Switch,
  Text,
  makeStyles,
  tokens,
} from '@fluentui/react-components';
import { SecuritySettings } from '../../types/profile';
import { ListEditor } from '../editors/CompositeEditors';

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

interface SecurityTabProps {
  settings: SecuritySettings;
  onChange: (settings: SecuritySettings) => void;
}

export const SecurityTab: React.FC<SecurityTabProps> = ({ settings, onChange }) => {
  const styles = useStyles();

  const handleChange = (field: keyof SecuritySettings, value: any) => {
    onChange({ ...settings, [field]: value });
  };

  return (
    <div className={styles.container}>
      <div className={styles.section}>
        <Text size={500} weight="semibold">Security Policies</Text>
        
        <div className={styles.row}>
          <Field>
            <Switch
              checked={settings.bypassCSP}
              onChange={(_, data) => handleChange('bypassCSP', data.checked)}
              label="Bypass Content Security Policy"
            />
            <Text size={200}>
              Disable CSP enforcement for all pages
            </Text>
          </Field>

          <Field>
            <Switch
              checked={settings.ignoreHTTPSErrors}
              onChange={(_, data) => handleChange('ignoreHTTPSErrors', data.checked)}
              label="Ignore HTTPS Errors"
            />
            <Text size={200}>
              Accept self-signed and invalid certificates
            </Text>
          </Field>
        </div>

        <div className={styles.row}>
          <Field>
            <Switch
              checked={settings.webSecurity}
              onChange={(_, data) => handleChange('webSecurity', data.checked)}
              label="Web Security"
            />
            <Text size={200}>
              Enable standard web security policies
            </Text>
          </Field>

          <Field>
            <Switch
              checked={settings.javaScriptEnabled}
              onChange={(_, data) => handleChange('javaScriptEnabled', data.checked)}
              label="JavaScript Enabled"
            />
            <Text size={200}>
              Allow JavaScript execution on pages
            </Text>
          </Field>
        </div>

        <Field>
          <Switch
            checked={settings.acceptDownloads}
            onChange={(_, data) => handleChange('acceptDownloads', data.checked)}
            label="Accept Downloads"
          />
          <Text size={200}>
            Allow file downloads to be initiated
          </Text>
        </Field>
      </div>

      <div className={styles.section}>
        <Text size={500} weight="semibold">Permissions</Text>
        
        <ListEditor
          data={settings.permissions}
          onChange={(permissions) => handleChange('permissions', permissions)}
          title="Granted Permissions"
          placeholder="camera, microphone, geolocation, notifications"
          itemLabel="Permission"
        />
        
        <Text size={200}>
          Common permissions: camera, microphone, geolocation, notifications, clipboard-read, clipboard-write
        </Text>
      </div>
    </div>
  );
};
