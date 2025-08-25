/**
 * Viewport Tab - M2 Implementation
 * Configuration for display, viewport, and visual settings
 */

import React from 'react';
import {
  Field,
  Switch,
  Dropdown,
  Option,
  Text,
  SpinButton,
  makeStyles,
  tokens,
} from '@fluentui/react-components';
import { ViewportSettings } from '../../types/profile';

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
    gridTemplateColumns: '1fr 1fr 1fr',
    gap: tokens.spacingHorizontalM,
  },
  warning: {
    padding: tokens.spacingVerticalM,
    backgroundColor: tokens.colorPaletteYellowBackground2,
    border: `1px solid ${tokens.colorPaletteYellowBorder2}`,
    borderRadius: tokens.borderRadiusMedium,
  },
});

interface ViewportTabProps {
  settings: ViewportSettings;
  headless: boolean;
  onChange: (settings: ViewportSettings) => void;
  onHeadlessChange: (headless: boolean) => void;
}

export const ViewportTab: React.FC<ViewportTabProps> = ({ 
  settings, 
  headless,
  onChange, 
  onHeadlessChange 
}) => {
  const styles = useStyles();

  const handleChange = (field: keyof ViewportSettings, value: any) => {
    onChange({ ...settings, [field]: value });
  };

  const presetSizes = [
    { name: 'Desktop HD', width: 1920, height: 1080 },
    { name: 'Desktop FHD', width: 1366, height: 768 },
    { name: 'MacBook Pro', width: 1440, height: 900 },
    { name: 'iPad', width: 768, height: 1024, mobile: true },
    { name: 'iPhone 14', width: 390, height: 844, mobile: true, scale: 3 },
    { name: 'Custom', width: settings.width, height: settings.height },
  ];

  const handlePresetChange = (presetName: string) => {
    const preset = presetSizes.find(p => p.name === presetName);
    if (preset && preset.name !== 'Custom') {
      onChange({
        ...settings,
        width: preset.width,
        height: preset.height,
        isMobile: preset.mobile || false,
        deviceScaleFactor: preset.scale || 1,
        hasTouch: preset.mobile || false,
      });
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.section}>
        <Text size={500} weight="semibold">Display Mode</Text>
        
        <Field>
          <Switch
            checked={headless}
            onChange={(_, data) => onHeadlessChange(data.checked)}
            label="Headless Mode"
          />
          <Text size={200}>
            Run browser without a visible window (faster, but no visual debugging)
          </Text>
        </Field>
      </div>

      <div className={styles.section}>
        <Text size={500} weight="semibold">Viewport Settings</Text>
        
        <Field label="Size Preset">
          <Dropdown
            placeholder="Select a preset or custom"
            onOptionSelect={(_, data) => handlePresetChange(data.optionValue || '')}
          >
                  {presetSizes.map(preset => (
                    <Option key={preset.name} value={preset.name} text={preset.name}>
                      {preset.name} {preset.name !== 'Custom' && `(${preset.width}×${preset.height})`}
                    </Option>
                  ))}
          </Dropdown>
        </Field>

        <div className={styles.threeColumn}>
          <Field label="Width">
            <SpinButton
              value={settings.width}
              onChange={(_, data) => handleChange('width', data.value || 1920)}
              min={320}
              max={4096}
            />
          </Field>

          <Field label="Height">
            <SpinButton
              value={settings.height}
              onChange={(_, data) => handleChange('height', data.value || 1080)}
              min={240}
              max={2160}
            />
          </Field>

          <Field label="Scale Factor">
            <SpinButton
              value={settings.deviceScaleFactor}
              onChange={(_, data) => handleChange('deviceScaleFactor', data.value || 1)}
              min={0.5}
              max={3}
              step={0.1}
            />
          </Field>
        </div>
      </div>

      <div className={styles.section}>
        <Text size={500} weight="semibold">Device Simulation</Text>
        
        <div className={styles.row}>
          <Field>
            <Switch
              checked={settings.isMobile}
              onChange={(_, data) => handleChange('isMobile', data.checked)}
              label="Mobile Device"
              disabled={headless}
            />
            <Text size={200}>
              Simulate mobile browser behavior and viewport
            </Text>
          </Field>

          <Field>
            <Switch
              checked={settings.hasTouch}
              onChange={(_, data) => handleChange('hasTouch', data.checked)}
              label="Touch Support"
              disabled={headless}
            />
            <Text size={200}>
              Enable touch event simulation
            </Text>
          </Field>
        </div>

        {headless && settings.isMobile && (
          <div className={styles.warning}>
            <Text weight="semibold">⚠️ Compatibility Warning</Text>
            <Text size={200}>
              Mobile viewport simulation is not fully supported in headless mode. 
              Consider disabling headless for accurate mobile testing.
            </Text>
          </div>
        )}
      </div>
    </div>
  );
};
