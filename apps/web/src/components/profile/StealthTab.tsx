/**
 * Stealth Tab - M2 Implementation
 * Configuration for stealth and anti-detection settings
 */

import React from 'react';
import {
  Field,
  Switch,
  Input,
  Dropdown,
  Option,
  Text,
  InfoLabel,
  makeStyles,
  tokens,
} from '@fluentui/react-components';
import { StealthSettings } from '../../types/profile';

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
  warning: {
    padding: tokens.spacingVerticalM,
    backgroundColor: tokens.colorPaletteYellowBackground2,
    border: `1px solid ${tokens.colorPaletteYellowBorder2}`,
    borderRadius: tokens.borderRadiusMedium,
  },
});

interface StealthTabProps {
  settings: StealthSettings;
  onChange: (settings: StealthSettings) => void;
}

export const StealthTab: React.FC<StealthTabProps> = ({ settings, onChange }) => {
  const styles = useStyles();

  const handleChange = (field: keyof StealthSettings, value: any) => {
    onChange({ ...settings, [field]: value });
  };

  const timezones = [
    'America/New_York',
    'America/Los_Angeles',
    'Europe/London',
    'Europe/Paris',
    'Asia/Tokyo',
    'Australia/Sydney',
    'UTC',
  ];

  const locales = [
    'en-US',
    'en-GB',
    'fr-FR',
    'de-DE',
    'es-ES',
    'ja-JP',
    'zh-CN',
  ];

  return (
    <div className={styles.container}>
      <div className={styles.section}>
        <Text size={500} weight="semibold">Basic Stealth Settings</Text>
        
        <Field>
          <Switch
            checked={settings.enabled}
            onChange={(_, data) => handleChange('enabled', data.checked)}
            label="Enable Stealth Mode"
          />
          <Text size={200}>
            Applies basic anti-detection measures to avoid automation detection
          </Text>
        </Field>

        <Field>
          <Switch
            checked={settings.advanced_stealth}
            onChange={(_, data) => handleChange('advanced_stealth', data.checked)}
            label="Advanced Stealth"
            disabled={!settings.enabled}
          />
          <Text size={200}>
            Enables aggressive anti-detection techniques (may impact stability)
          </Text>
        </Field>

        {settings.advanced_stealth && !settings.enabled && (
          <div className={styles.warning}>
            <Text weight="semibold">⚠️ Validation Error</Text>
            <Text size={200}>Advanced stealth requires basic stealth to be enabled first.</Text>
          </div>
        )}
      </div>

      <div className={styles.section}>
        <Text size={500} weight="semibold">Fingerprint Override</Text>
        
        <div className={styles.row}>
          <Field label="Timezone">
            <Dropdown
              value={settings.timezone || ''}
              onOptionSelect={(_, data) => handleChange('timezone', data.optionValue)}
              placeholder="Select timezone"
            >
              {timezones.map(tz => (
                <Option key={tz} value={tz}>{tz}</Option>
              ))}
            </Dropdown>
          </Field>

          <Field label="Locale">
            <Dropdown
              value={settings.locale || ''}
              onOptionSelect={(_, data) => handleChange('locale', data.optionValue)}
              placeholder="Select locale"
            >
              {locales.map(locale => (
                <Option key={locale} value={locale}>{locale}</Option>
              ))}
            </Dropdown>
          </Field>
        </div>

        <Field 
          label={
            <InfoLabel
              info={{
                content: 'Custom User-Agent string to override browser identification',
              }}
            >
              User Agent Override
            </InfoLabel>
          }
        >
          <Input
            value={settings.user_agent_override || ''}
            onChange={(_, data) => handleChange('user_agent_override', data.value)}
            placeholder="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36..."
          />
        </Field>

        <Field 
          label={
            <InfoLabel
              info={{
                content: 'Comma-separated list of languages for Accept-Language header',
              }}
            >
              Accept Language
            </InfoLabel>
          }
        >
          <Input
            value={settings.accept_language || ''}
            onChange={(_, data) => handleChange('accept_language', data.value)}
            placeholder="en-US,en;q=0.9"
          />
        </Field>
      </div>

      <div className={styles.section}>
        <Text size={500} weight="semibold">WebGL Override</Text>
        
        <div className={styles.row}>
          <Field 
            label={
              <InfoLabel
                info={{
                  content: 'Override WebGL vendor to mask hardware fingerprint',
                }}
              >
                WebGL Vendor
              </InfoLabel>
            }
          >
            <Input
              value={settings.webgl_vendor_override || ''}
              onChange={(_, data) => handleChange('webgl_vendor_override', data.value)}
              placeholder="Intel Inc."
            />
          </Field>

          <Field 
            label={
              <InfoLabel
                info={{
                  content: 'Override WebGL renderer to mask hardware fingerprint',
                }}
              >
                WebGL Renderer
              </InfoLabel>
            }
          >
            <Input
              value={settings.webgl_renderer_override || ''}
              onChange={(_, data) => handleChange('webgl_renderer_override', data.value)}
              placeholder="Intel Iris OpenGL Engine"
            />
          </Field>
        </div>
      </div>
    </div>
  );
};
