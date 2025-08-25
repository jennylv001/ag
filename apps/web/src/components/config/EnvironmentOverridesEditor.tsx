/**
 * Environment Overrides Editor Component - M4 Implementation
 * Manages environment variable overrides with validation
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
  Textarea,
  Badge,
  InfoLabel,
  mergeClasses,
  makeStyles,
  tokens,
  MessageBar,
  MessageBarTitle,
  MessageBarBody,
} from '@fluentui/react-components';
import {
  AddRegular,
  DeleteRegular,
  SaveRegular,
  DismissRegular,
  InfoRegular,
} from '@fluentui/react-icons';
import { EnvironmentVariables } from '../../types/config';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
  },
  envGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 2fr auto',
    gap: tokens.spacingHorizontalM,
    alignItems: 'end',
    marginBottom: tokens.spacingVerticalS,
  },
  envRow: {
    display: 'contents',
  },
  actions: {
    display: 'flex',
    gap: tokens.spacingHorizontalS,
    marginTop: tokens.spacingVerticalM,
  },
  helpText: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
  envSection: {
    marginBottom: tokens.spacingVerticalL,
  },
  sectionHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalS,
    marginBottom: tokens.spacingVerticalM,
  },
  badgeGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: tokens.spacingVerticalS,
    marginBottom: tokens.spacingVerticalM,
  },
  envBadge: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: tokens.spacingVerticalS,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    borderRadius: tokens.borderRadiusMedium,
  },
});

interface EnvironmentOverridesEditorProps {
  overrides: Partial<EnvironmentVariables>;
  onUpdateOverride: <K extends keyof EnvironmentVariables>(
    key: K,
    value: EnvironmentVariables[K]
  ) => void;
  onRemoveOverride: (key: keyof EnvironmentVariables) => void;
  onApplyOverrides: () => void;
  className?: string;
}

// Environment variable definitions with validation
const ENV_DEFINITIONS: Array<{
  key: keyof EnvironmentVariables;
  type: 'string' | 'boolean' | 'number' | 'array';
  section: string;
  description: string;
  example?: string;
  validation?: (value: any) => string | null;
}> = [
  // Browser configuration
  {
    key: 'BROWSER_USE_HEADLESS',
    type: 'boolean',
    section: 'Browser',
    description: 'Run browser in headless mode',
    example: 'true',
  },
  {
    key: 'BROWSER_USE_ALLOWED_DOMAINS',
    type: 'array',
    section: 'Browser',
    description: 'Allowed domains for navigation',
    example: '["example.com", "*.google.com"]',
    validation: (value) => {
      if (!Array.isArray(value)) return 'Must be an array';
      return null;
    },
  },
  {
    key: 'BROWSER_USE_STEALTH',
    type: 'boolean',
    section: 'Browser',
    description: 'Enable stealth features',
    example: 'true',
  },
  {
    key: 'BROWSER_USE_TIMEOUT',
    type: 'number',
    section: 'Browser',
    description: 'Page load timeout in milliseconds',
    example: '30000',
    validation: (value) => {
      if (value < 1000) return 'Must be at least 1000ms';
      return null;
    },
  },
  {
    key: 'BROWSER_USE_PROXY',
    type: 'string',
    section: 'Browser',
    description: 'Proxy server configuration',
    example: 'http://proxy:8080',
  },
  
  // LLM configuration
  {
    key: 'LLM_MODEL',
    type: 'string',
    section: 'LLM',
    description: 'Language model to use',
    example: 'gpt-4o',
  },
  {
    key: 'LLM_API_KEY',
    type: 'string',
    section: 'LLM',
    description: 'API key for LLM service',
    example: 'sk-...',
  },
  {
    key: 'LLM_BASE_URL',
    type: 'string',
    section: 'LLM',
    description: 'Base URL for LLM API',
    example: 'https://api.openai.com/v1',
  },
  {
    key: 'LLM_TEMPERATURE',
    type: 'number',
    section: 'LLM',
    description: 'Sampling temperature',
    example: '0.7',
    validation: (value) => {
      if (value < 0 || value > 2) return 'Must be between 0 and 2';
      return null;
    },
  },
  {
    key: 'LLM_MAX_TOKENS',
    type: 'number',
    section: 'LLM',
    description: 'Maximum tokens per request',
    example: '4096',
    validation: (value) => {
      if (value < 1) return 'Must be positive';
      return null;
    },
  },
  
  // Agent configuration
  {
    key: 'AGENT_MAX_ACTIONS',
    type: 'number',
    section: 'Agent',
    description: 'Maximum actions per session',
    example: '100',
    validation: (value) => {
      if (value < 1) return 'Must be positive';
      return null;
    },
  },
  {
    key: 'AGENT_THINK_TIMEOUT',
    type: 'number',
    section: 'Agent',
    description: 'Thinking phase timeout',
    example: '30000',
  },
  {
    key: 'AGENT_ACTION_TIMEOUT',
    type: 'number',
    section: 'Agent',
    description: 'Action execution timeout',
    example: '10000',
  },
  {
    key: 'AGENT_DEBUG',
    type: 'boolean',
    section: 'Agent',
    description: 'Enable debug logging',
    example: 'false',
  },
  
  // System configuration
  {
    key: 'LOG_LEVEL',
    type: 'string',
    section: 'System',
    description: 'Logging level',
    example: 'info',
    validation: (value) => {
      const levels = ['debug', 'info', 'warn', 'error'];
      if (!levels.includes(value)) return `Must be one of: ${levels.join(', ')}`;
      return null;
    },
  },
  {
    key: 'DEBUG_MODE',
    type: 'boolean',
    section: 'System',
    description: 'System debug mode',
    example: 'false',
  },
  {
    key: 'DATA_DIR',
    type: 'string',
    section: 'System',
    description: 'Data storage directory',
    example: './data',
  },
  {
    key: 'CACHE_DIR',
    type: 'string',
    section: 'System',
    description: 'Cache directory',
    example: './cache',
  },
];

const EnvironmentOverridesEditor: React.FC<EnvironmentOverridesEditorProps> = ({
  overrides,
  onUpdateOverride,
  onRemoveOverride,
  onApplyOverrides,
  className,
}) => {
  const styles = useStyles();
  const [newKey, setNewKey] = React.useState('');
  const [newValue, setNewValue] = React.useState('');

  // Group definitions by section
  const sectionGroups = React.useMemo(() => {
    const groups: Record<string, typeof ENV_DEFINITIONS> = {};
    ENV_DEFINITIONS.forEach(def => {
      if (!groups[def.section]) {
        groups[def.section] = [];
      }
      groups[def.section].push(def);
    });
    return groups;
  }, []);

  const handleValueChange = (key: keyof EnvironmentVariables, rawValue: string) => {
    const definition = ENV_DEFINITIONS.find(def => def.key === key);
    if (!definition) return;

    let parsedValue: any = rawValue;

    // Parse value based on type
    try {
      switch (definition.type) {
        case 'boolean':
          parsedValue = rawValue === 'true';
          break;
        case 'number':
          parsedValue = Number(rawValue);
          if (isNaN(parsedValue)) return;
          break;
        case 'array':
          parsedValue = JSON.parse(rawValue);
          break;
        case 'string':
        default:
          parsedValue = rawValue;
          break;
      }
    } catch (error) {
      return; // Invalid JSON for array type
    }

    // Validate if validator exists
    if (definition.validation) {
      const error = definition.validation(parsedValue);
      if (error) return;
    }

    onUpdateOverride(key, parsedValue);
  };

  const formatValue = (value: any, type: string): string => {
    if (type === 'array') {
      return JSON.stringify(value);
    }
    return String(value);
  };

  const addCustomOverride = () => {
    if (!newKey || !newValue) return;
    
    // Try to parse as JSON first, fallback to string
    let parsedValue: any = newValue;
    try {
      parsedValue = JSON.parse(newValue);
    } catch {
      // Keep as string
    }

    onUpdateOverride(newKey as keyof EnvironmentVariables, parsedValue);
    setNewKey('');
    setNewValue('');
  };

  return (
    <div className={mergeClasses(styles.container, className)}>
      {/* Header */}
      <div>
        <Text size={500} weight="semibold">Environment Variable Overrides</Text>
        <Text className={styles.helpText}>
          Override configuration values using environment variables. 
          These take precedence over database and default values.
        </Text>
      </div>

      {/* Current Overrides */}
      {Object.keys(overrides).length > 0 && (
        <Card>
          <CardHeader header={<Text weight="semibold">Active Overrides</Text>} />
          <div className={styles.badgeGrid}>
            {Object.entries(overrides).map(([key, value]) => (
              <div key={key} className={styles.envBadge}>
                <div>
                  <Text size={300} weight="semibold">{key}</Text>
                  <Text size={200}>{formatValue(value, 'string')}</Text>
                </div>
                <Button
                  appearance="subtle"
                  icon={<DismissRegular />}
                  size="small"
                  onClick={() => onRemoveOverride(key as keyof EnvironmentVariables)}
                />
              </div>
            ))}
          </div>
          <Button
            appearance="primary"
            icon={<SaveRegular />}
            onClick={onApplyOverrides}
          >
            Apply Overrides
          </Button>
        </Card>
      )}

      {/* Environment Variables by Section */}
      {Object.entries(sectionGroups).map(([section, definitions]) => (
        <div key={section} className={styles.envSection}>
          <div className={styles.sectionHeader}>
            <Text size={400} weight="semibold">{section} Variables</Text>
            <InfoLabel
              info={`Configure ${section.toLowerCase()} settings via environment variables`}
            >
              <InfoRegular />
            </InfoLabel>
          </div>

          <Card>
            <div className={styles.envGrid}>
              <Text size={300} weight="semibold">Variable</Text>
              <Text size={300} weight="semibold">Value</Text>
              <Text size={300} weight="semibold">Actions</Text>

              {definitions.map(definition => {
                const currentValue = overrides[definition.key];
                const hasOverride = currentValue !== undefined;

                return (
                  <div key={definition.key} className={styles.envRow}>
                    <Field
                      label={definition.key}
                      hint={definition.description}
                    >
                      <Text size={300}>{definition.key}</Text>
                    </Field>

                    <Field
                      hint={`Type: ${definition.type}${definition.example ? ` | Example: ${definition.example}` : ''}`}
                    >
                      {definition.type === 'boolean' ? (
                        <Switch
                          checked={currentValue === true}
                          onChange={(_, data) => 
                            onUpdateOverride(definition.key, data.checked)
                          }
                        />
                      ) : definition.type === 'array' ? (
                        <Textarea
                          value={hasOverride ? JSON.stringify(currentValue, null, 2) : ''}
                          onChange={(_, data) => 
                            handleValueChange(definition.key, data.value)
                          }
                          placeholder={definition.example}
                          rows={3}
                        />
                      ) : (
                        <Input
                          value={hasOverride ? String(currentValue) : ''}
                          onChange={(_, data) => 
                            handleValueChange(definition.key, data.value)
                          }
                          placeholder={definition.example}
                        />
                      )}
                    </Field>

                    <div>
                      {hasOverride ? (
                        <Button
                          appearance="subtle"
                          icon={<DeleteRegular />}
                          size="small"
                          onClick={() => onRemoveOverride(definition.key)}
                        />
                      ) : (
                        <Badge appearance="ghost">Not set</Badge>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      ))}

      {/* Custom Override */}
      <Card>
        <CardHeader 
          header={<Text weight="semibold">Add Custom Override</Text>}
          description="Add environment variables not in the predefined list"
        />
        <div className={styles.envGrid}>
          <Field label="Variable Name">
            <Input
              value={newKey}
              onChange={(_, data) => setNewKey(data.value)}
              placeholder="CUSTOM_VAR"
            />
          </Field>
          <Field label="Value">
            <Input
              value={newValue}
              onChange={(_, data) => setNewValue(data.value)}
              placeholder="value"
            />
          </Field>
          <div>
            <Button
              appearance="secondary"
              icon={<AddRegular />}
              onClick={addCustomOverride}
              disabled={!newKey || !newValue}
            >
              Add
            </Button>
          </div>
        </div>
      </Card>

      {/* Info Message */}
      <MessageBar intent="info">
        <MessageBarTitle>Environment Variables</MessageBarTitle>
        <MessageBarBody>
          Environment variables override configuration values with the highest precedence. 
          Changes here will be applied when you click "Apply Overrides".
        </MessageBarBody>
      </MessageBar>
    </div>
  );
};

export default EnvironmentOverridesEditor;
