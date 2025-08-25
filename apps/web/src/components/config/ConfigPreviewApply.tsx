/**
 * Config Preview and Apply Component - M4 Implementation
 * Preview effective configuration and apply to profiles
 */

import React from 'react';
import {
  Card,
  CardHeader,
  Text,
  Button,
  Badge,
  Dropdown,
  Option,
  mergeClasses,
  makeStyles,
  tokens,
  MessageBar,
  MessageBarTitle,
  MessageBarBody,
  Dialog,
  DialogSurface,
  DialogTitle,
  DialogContent,
  DialogBody,
  DialogActions,
  Textarea,
  Tooltip,
} from '@fluentui/react-components';
import {
  EyeRegular,
  SaveRegular,
  WarningRegular,
  ArrowSyncRegular,
} from '@fluentui/react-icons';
import { ConfigPreview, ApplicationConfig } from '../../types/config';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalL,
  },
  previewHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: tokens.spacingVerticalM,
  },
  profileSelector: {
    minWidth: '200px',
  },
  previewGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: tokens.spacingVerticalL,
  },
  overridesList: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
  },
  overrideItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: tokens.spacingVerticalS,
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusMedium,
  borderLeft: `4px solid ${tokens.colorPaletteGreenBorder1}`,
  },
  overrideDetails: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXS,
  },
  overridePath: {
    fontWeight: tokens.fontWeightSemibold,
    fontSize: tokens.fontSizeBase300,
  },
  overrideValues: {
    display: 'flex',
    gap: tokens.spacingHorizontalS,
    alignItems: 'center',
    fontSize: tokens.fontSizeBase200,
  },
  oldValue: {
    textDecoration: 'line-through',
    color: tokens.colorNeutralForeground3,
  },
  newValue: {
    color: tokens.colorBrandForeground1,
    fontWeight: tokens.fontWeightSemibold,
  },
  arrow: {
    color: tokens.colorNeutralForeground3,
  },
  configPreview: {
    maxHeight: '400px',
    overflowY: 'auto',
    backgroundColor: tokens.colorNeutralBackground1,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    borderRadius: tokens.borderRadiusMedium,
    padding: tokens.spacingVerticalM,
  },
  jsonContent: {
    fontFamily: tokens.fontFamilyMonospace,
    fontSize: tokens.fontSizeBase200,
    lineHeight: '1.4',
    whiteSpace: 'pre-wrap',
  },
  warningsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
    marginTop: tokens.spacingVerticalM,
  },
  warningItem: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalS,
    padding: tokens.spacingVerticalS,
    backgroundColor: tokens.colorPaletteYellowBackground1,
    borderRadius: tokens.borderRadiusMedium,
    border: `1px solid ${tokens.colorPaletteYellowBorder1}`,
  },
  actions: {
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    justifyContent: 'flex-end',
    marginTop: tokens.spacingVerticalL,
  },
});

interface ConfigPreviewApplyProps {
  preview: ConfigPreview | undefined;
  availableProfiles: Array<{ id: string; name: string }>;
  selectedProfile: string | undefined;
  isGenerating: boolean;
  isApplying: boolean;
  onSelectProfile: (profileId: string) => void;
  onGeneratePreview: () => Promise<void>;
  onApplyToProfile: () => Promise<void>;
  className?: string;
}

const ConfigPreviewApply: React.FC<ConfigPreviewApplyProps> = ({
  preview,
  availableProfiles,
  selectedProfile,
  isGenerating,
  isApplying,
  onSelectProfile,
  onGeneratePreview,
  onApplyToProfile,
  className,
}) => {
  const styles = useStyles();
  const [showJsonDialog, setShowJsonDialog] = React.useState(false);

  const formatValue = (value: any): string => {
    if (Array.isArray(value)) {
      return `[${value.join(', ')}]`;
    }
    if (typeof value === 'object' && value !== null) {
      return JSON.stringify(value, null, 2);
    }
    if (typeof value === 'string' && value.startsWith('sk-')) {
      return value.substring(0, 8) + '...';
    }
    return String(value);
  };

  const renderConfigJson = (config: ApplicationConfig): string => {
    // Convert ConfigValue objects to plain values for JSON display
    const plainConfig: any = {};
    
    Object.entries(config).forEach(([sectionKey, section]) => {
      plainConfig[sectionKey] = {};
      Object.entries(section as any).forEach(([key, configValue]) => {
        if (configValue && typeof configValue === 'object' && 'value' in configValue) {
          plainConfig[sectionKey][key] = (configValue as any).value;
        }
      });
    });

    return JSON.stringify(plainConfig, null, 2);
  };

  const getOverrideSourceColor = (source: string) => {
    switch (source) {
      case 'environment': return 'warning';
      case 'runtime': return 'success';
      case 'database': return 'informative';
      default: return 'subtle';
    }
  };

  return (
    <div className={mergeClasses(styles.container, className)}>
      {/* Header */}
      <div className={styles.previewHeader}>
        <div>
          <Text size={500} weight="semibold">Configuration Preview</Text>
          <Text size={300} style={{ marginTop: tokens.spacingVerticalXS }}>
            Preview effective configuration and apply to browser profiles
          </Text>
        </div>
        
        <div style={{ display: 'flex', gap: tokens.spacingHorizontalM, alignItems: 'center' }}>
          <Dropdown
            placeholder="Select profile..."
            value={selectedProfile}
            selectedOptions={selectedProfile ? [selectedProfile] : []}
            onOptionSelect={(_, data) => onSelectProfile(data.optionValue as string)}
            className={styles.profileSelector}
          >
            {availableProfiles.map(profile => (
              <Option key={profile.id} value={profile.id}>
                {profile.name}
              </Option>
            ))}
          </Dropdown>
          
          <Button
            appearance="secondary"
            icon={<EyeRegular />}
            onClick={onGeneratePreview}
            disabled={!selectedProfile || isGenerating}
          >
            {isGenerating ? 'Generating...' : 'Generate Preview'}
          </Button>
        </div>
      </div>

      {/* Preview Content */}
      {preview && (
        <div className={styles.previewGrid}>
          {/* Configuration Overrides */}
          <Card>
            <CardHeader 
              header={<Text weight="semibold">Configuration Changes</Text>}
              description={`${preview.overrides.length} override(s) detected`}
            />
            
            {preview.overrides.length > 0 ? (
              <div className={styles.overridesList}>
                {preview.overrides.map((override, index) => (
                  <div key={index} className={styles.overrideItem}>
                    <div className={styles.overrideDetails}>
                      <Text className={styles.overridePath}>{override.path}</Text>
                      <div className={styles.overrideValues}>
                        <Text className={styles.oldValue}>
                          {formatValue(override.originalValue)}
                        </Text>
                        <ArrowSyncRegular className={styles.arrow} />
                        <Text className={styles.newValue}>
                          {formatValue(override.newValue)}
                        </Text>
                      </div>
                      <Text size={200} style={{ color: tokens.colorNeutralForeground3 }}>
                        {override.reason}
                      </Text>
                    </div>
                    <Badge 
                      appearance="outline" 
                      color={getOverrideSourceColor(override.source)}
                    >
                      {override.source}
                    </Badge>
                  </div>
                ))}
              </div>
            ) : (
              <MessageBar intent="info">
                <MessageBarTitle>No Overrides</MessageBarTitle>
                <MessageBarBody>
                  Configuration matches the selected profile with no changes needed.
                </MessageBarBody>
              </MessageBar>
            )}
          </Card>

          {/* Effective Configuration */}
          <Card>
            <CardHeader 
              header={
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text weight="semibold">Effective Configuration</Text>
                  <Button
                    appearance="subtle"
                    size="small"
                    onClick={() => setShowJsonDialog(true)}
                  >
                    View JSON
                  </Button>
                </div>
              }
              description="Final configuration that will be applied"
            />
            
            <div className={styles.configPreview}>
              <Text className={styles.jsonContent}>
                {renderConfigJson(preview.effectiveConfig).substring(0, 500)}
                {renderConfigJson(preview.effectiveConfig).length > 500 && '...'}
              </Text>
            </div>
          </Card>
        </div>
      )}

      {/* Warnings */}
      {preview && preview.warnings.length > 0 && (
        <Card>
          <CardHeader 
            header={
              <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacingHorizontalS }}>
                <WarningRegular />
                <Text weight="semibold">Warnings</Text>
              </div>
            }
          />
          <div className={styles.warningsList}>
            {preview.warnings.map((warning, index) => (
              <div key={index} className={styles.warningItem}>
                <WarningRegular />
                <Text>{warning}</Text>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Validation Status */}
      {preview && (
        <MessageBar intent={preview.isValid ? 'success' : 'error'}>
          <MessageBarTitle>
            {preview.isValid ? 'Configuration Valid' : 'Configuration Issues'}
          </MessageBarTitle>
          <MessageBarBody>
            {preview.isValid 
              ? 'Configuration is valid and ready to apply'
              : 'Please resolve configuration issues before applying'
            }
          </MessageBarBody>
        </MessageBar>
      )}

      {/* Actions */}
      {preview && (
        <div className={styles.actions}>
          <Tooltip 
            content={preview.isValid ? 'Apply configuration to profile' : 'Fix configuration issues first'}
            relationship="label"
          >
            <Button
              appearance="primary"
              icon={<SaveRegular />}
              onClick={onApplyToProfile}
              disabled={!preview.isValid || isApplying}
            >
              {isApplying ? 'Applying...' : 'Apply to Profile'}
            </Button>
          </Tooltip>
        </div>
      )}

      {/* JSON Dialog */}
      <Dialog open={showJsonDialog} onOpenChange={(_, data) => setShowJsonDialog(data.open)}>
        <DialogSurface>
          <DialogTitle>Complete Configuration JSON</DialogTitle>
          <DialogContent>
            <DialogBody>
              {preview && (
                <Textarea
                  value={renderConfigJson(preview.effectiveConfig)}
                  readOnly
                  rows={20}
                  style={{ 
                    fontFamily: tokens.fontFamilyMonospace,
                    fontSize: tokens.fontSizeBase200,
                  }}
                />
              )}
            </DialogBody>
            <DialogActions>
              <Button
                appearance="secondary"
                onClick={() => setShowJsonDialog(false)}
              >
                Close
              </Button>
              <Button
                appearance="primary"
                onClick={() => {
                  if (preview) {
                    navigator.clipboard.writeText(renderConfigJson(preview.effectiveConfig));
                  }
                }}
              >
                Copy to Clipboard
              </Button>
            </DialogActions>
          </DialogContent>
        </DialogSurface>
      </Dialog>
    </div>
  );
};

export default ConfigPreviewApply;
