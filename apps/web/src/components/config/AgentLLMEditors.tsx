/**
 * Agent and LLM Configuration Editors - M4 Implementation
 * Comprehensive editors for agent and LLM settings with path pickers
 */

import React from 'react';
import {
  Card,
  CardHeader,
  Text,
  Button,
  Input,
  Field,
  Switch,
  Slider,
  Dropdown,
  Option,
  Textarea,
  SpinButton,
  mergeClasses,
  makeStyles,
  tokens,
  MessageBar,
  MessageBarTitle,
  MessageBarBody,
  Tooltip,
} from '@fluentui/react-components';
import {
  Folder24Regular as FolderRegular,
  Document24Regular as DocumentRegular,
  Info24Regular as InfoRegular,
  Warning24Regular as WarningRegular,
  CheckmarkCircle24Regular as CheckmarkCircleRegular,
} from '@fluentui/react-icons';
import { AgentConfig, LLMConfig, ConfigValue } from '../../types/config';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalL,
  },
  configGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: tokens.spacingVerticalL,
  },
  fieldGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr',
    gap: tokens.spacingVerticalM,
  },
  pathField: {
    display: 'flex',
    gap: tokens.spacingHorizontalS,
    alignItems: 'end',
  },
  pathInput: {
    flex: 1,
  },
  configSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
  },
  sourceIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXS,
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
  validationError: {
    color: tokens.colorPaletteRedForeground1,
    fontSize: tokens.fontSizeBase200,
  },
  modelSelect: {
    minWidth: '200px',
  },
});

interface AgentLLMEditorsProps {
  agentConfig: AgentConfig;
  llmConfig: LLMConfig;
  onUpdateAgent: (key: keyof AgentConfig, value: any) => void;
  onUpdateLLM: (key: keyof LLMConfig, value: any) => void;
  onSelectPath: (type: 'file' | 'folder', currentPath?: string) => Promise<string>;
  className?: string;
}

// Predefined LLM models
const LLM_MODELS = [
  { value: 'gpt-4o', label: 'GPT-4o', provider: 'OpenAI' },
  { value: 'gpt-4o-mini', label: 'GPT-4o Mini', provider: 'OpenAI' },
  { value: 'gpt-4-turbo', label: 'GPT-4 Turbo', provider: 'OpenAI' },
  { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo', provider: 'OpenAI' },
  { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet', provider: 'Anthropic' },
  { value: 'claude-3-haiku-20240307', label: 'Claude 3 Haiku', provider: 'Anthropic' },
  { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro', provider: 'Google' },
  { value: 'gemini-1.5-flash', label: 'Gemini 1.5 Flash', provider: 'Google' },
];

const AgentLLMEditors: React.FC<AgentLLMEditorsProps> = ({
  agentConfig,
  llmConfig,
  onUpdateAgent,
  onUpdateLLM,
  onSelectPath,
  className,
}) => {
  const styles = useStyles();

  // Field validation
  const validateTimeout = (value: number): string | null => {
    if (value < 1000) return 'Must be at least 1000ms';
    if (value > 300000) return 'Must be less than 5 minutes';
    return null;
  };

  const validateMaxActions = (value: number): string | null => {
    if (value < 1) return 'Must be at least 1';
    if (value > 1000) return 'Must be less than 1000';
    return null;
  };

  const validateTemperature = (value: number): string | null => {
    if (value < 0) return 'Must be at least 0';
    if (value > 2) return 'Must be at most 2';
    return null;
  };

  const validateTokens = (value: number): string | null => {
    if (value < 1) return 'Must be at least 1';
    if (value > 128000) return 'Must be less than 128k';
    return null;
  };

  const getSourceIcon = (source: string) => {
    switch (source) {
      case 'environment': return <WarningRegular />;
      case 'runtime': return <CheckmarkCircleRegular />;
      case 'database': return <InfoRegular />;
      default: return <DocumentRegular />;
    }
  };

  const renderConfigField = (
    label: string,
    configValue: ConfigValue<any>,
    onUpdate: (value: any) => void,
    fieldType: 'text' | 'number' | 'boolean' | 'slider' | 'path-file' | 'path-folder' | 'textarea' | 'dropdown',
    options?: { min?: number; max?: number; step?: number; dropdownOptions?: Array<{value: string; label: string}> },
    validator?: (value: any) => string | null
  ) => {
    const validationError = validator ? validator(configValue.value) : null;

    return (
      <Field
        label={
          <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacingHorizontalS }}>
            <Text>{label}</Text>
            <Tooltip content={`Source: ${configValue.source} (Priority: ${configValue.priority})`} relationship="label">
              <div className={styles.sourceIndicator}>
                {getSourceIcon(configValue.source)}
                <Text size={200}>{configValue.source}</Text>
              </div>
            </Tooltip>
          </div>
        }
        hint={configValue.description}
        validationState={validationError ? 'error' : 'none'}
        validationMessage={validationError}
      >
        {fieldType === 'text' && (
          <Input
            value={String(configValue.value)}
            onChange={(_, data) => onUpdate(data.value)}
          />
        )}
        {fieldType === 'number' && (
          <SpinButton
            value={configValue.value}
            onChange={(_, data) => onUpdate(data.value || 0)}
            min={options?.min}
            max={options?.max}
            step={options?.step || 1}
          />
        )}
        {fieldType === 'boolean' && (
          <Switch
            checked={configValue.value}
            onChange={(_, data) => onUpdate(data.checked)}
          />
        )}
        {fieldType === 'slider' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacingHorizontalM }}>
            <Slider
              value={configValue.value}
              onChange={(_, data) => onUpdate(data.value)}
              min={options?.min || 0}
              max={options?.max || 1}
              step={options?.step || 0.1}
              style={{ flex: 1 }}
            />
            <Text size={300} style={{ minWidth: '60px' }}>{configValue.value}</Text>
          </div>
        )}
        {fieldType === 'textarea' && (
          <Textarea
            value={String(configValue.value)}
            onChange={(_, data) => onUpdate(data.value)}
            rows={3}
          />
        )}
        {fieldType === 'dropdown' && (
          <Dropdown
            value={configValue.value}
            selectedOptions={[configValue.value]}
            onOptionSelect={(_, data) => onUpdate(data.optionValue)}
            className={styles.modelSelect}
          >
            {options?.dropdownOptions?.map(option => (
              <Option key={option.value} value={option.value}>
                {option.label}
              </Option>
            ))}
          </Dropdown>
        )}
        {(fieldType === 'path-file' || fieldType === 'path-folder') && (
          <div className={styles.pathField}>
            <Input
              value={String(configValue.value)}
              onChange={(_, data) => onUpdate(data.value)}
              className={styles.pathInput}
              placeholder={fieldType === 'path-file' ? 'Select file...' : 'Select folder...'}
            />
            <Button
              appearance="secondary"
              icon={fieldType === 'path-file' ? <DocumentRegular /> : <FolderRegular />}
              onClick={async () => {
                try {
                  const path = await onSelectPath(
                    fieldType === 'path-file' ? 'file' : 'folder',
                    configValue.value
                  );
                  if (path) onUpdate(path);
                } catch (error) {
                  console.error('Path selection failed:', error);
                }
              }}
            >
              Browse
            </Button>
          </div>
        )}
      </Field>
    );
  };

  return (
    <div className={mergeClasses(styles.container, className)}>
      <div className={styles.configGrid}>
        {/* Agent Configuration */}
        <Card>
          <CardHeader 
            header={<Text size={500} weight="semibold">Agent Configuration</Text>}
            description="Configure agent behavior and execution parameters"
          />
          <div className={styles.configSection}>
            {renderConfigField(
              'Max Actions',
              agentConfig.maxActions,
              (value) => onUpdateAgent('maxActions', value),
              'number',
              { min: 1, max: 1000 },
              validateMaxActions
            )}

            {renderConfigField(
              'Think Timeout (ms)',
              agentConfig.thinkTimeout,
              (value) => onUpdateAgent('thinkTimeout', value),
              'number',
              { min: 1000, max: 300000, step: 1000 },
              validateTimeout
            )}

            {renderConfigField(
              'Action Timeout (ms)',
              agentConfig.actionTimeout,
              (value) => onUpdateAgent('actionTimeout', value),
              'number',
              { min: 1000, max: 60000, step: 1000 },
              validateTimeout
            )}

            {renderConfigField(
              'Retry Attempts',
              agentConfig.retryAttempts,
              (value) => onUpdateAgent('retryAttempts', value),
              'number',
              { min: 0, max: 10 }
            )}

            {renderConfigField(
              'Debug Mode',
              agentConfig.debugMode,
              (value) => onUpdateAgent('debugMode', value),
              'boolean'
            )}

            {renderConfigField(
              'Enable Telemetry',
              agentConfig.enableTelemetry,
              (value) => onUpdateAgent('enableTelemetry', value),
              'boolean'
            )}

            {renderConfigField(
              'Working Directory',
              agentConfig.workingDirectory,
              (value) => onUpdateAgent('workingDirectory', value),
              'path-folder'
            )}
          </div>
        </Card>

        {/* LLM Configuration */}
        <Card>
          <CardHeader 
            header={<Text size={500} weight="semibold">LLM Configuration</Text>}
            description="Configure language model settings and API parameters"
          />
          <div className={styles.configSection}>
            {renderConfigField(
              'Model',
              llmConfig.model,
              (value) => onUpdateLLM('model', value),
              'dropdown',
              { 
                dropdownOptions: LLM_MODELS.map(model => ({
                  value: model.value,
                  label: `${model.label} (${model.provider})`
                }))
              }
            )}

            {renderConfigField(
              'API Key',
              llmConfig.apiKey,
              (value) => onUpdateLLM('apiKey', value),
              'text'
            )}

            {renderConfigField(
              'Base URL',
              llmConfig.baseUrl,
              (value) => onUpdateLLM('baseUrl', value),
              'text'
            )}

            {renderConfigField(
              'Temperature',
              llmConfig.temperature,
              (value) => onUpdateLLM('temperature', value),
              'slider',
              { min: 0, max: 2, step: 0.1 },
              validateTemperature
            )}

            {renderConfigField(
              'Max Tokens',
              llmConfig.maxTokens,
              (value) => onUpdateLLM('maxTokens', value),
              'number',
              { min: 1, max: 128000 },
              validateTokens
            )}

            {renderConfigField(
              'Timeout (ms)',
              llmConfig.timeout,
              (value) => onUpdateLLM('timeout', value),
              'number',
              { min: 1000, max: 300000, step: 1000 },
              validateTimeout
            )}

            {renderConfigField(
              'Retry Attempts',
              llmConfig.retryAttempts,
              (value) => onUpdateLLM('retryAttempts', value),
              'number',
              { min: 0, max: 10 }
            )}

            {renderConfigField(
              'Fallback Model',
              llmConfig.fallbackModel,
              (value) => onUpdateLLM('fallbackModel', value),
              'dropdown',
              { 
                dropdownOptions: LLM_MODELS.map(model => ({
                  value: model.value,
                  label: `${model.label} (${model.provider})`
                }))
              }
            )}
          </div>
        </Card>
      </div>

      {/* Configuration Tips */}
      <MessageBar intent="info">
        <MessageBarTitle>Configuration Tips</MessageBarTitle>
        <MessageBarBody>
          • Higher temperature values make the LLM more creative but less predictable
          • Increase timeouts if you experience frequent timeout errors
          • Enable debug mode for detailed logging during development
          • Use the Browse buttons to select valid file and folder paths
        </MessageBarBody>
      </MessageBar>
    </div>
  );
};

export default AgentLLMEditors;
