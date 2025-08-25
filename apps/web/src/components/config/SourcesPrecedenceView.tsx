/**
 * Sources and Precedence View Component - M4 Implementation
 * Shows configuration sources with precedence hierarchy
 */

import React from 'react';
import {
  Card,
  CardHeader,
  CardPreview,
  Text,
  Badge,
  Tooltip,
  mergeClasses,
  makeStyles,
  tokens,
  DataGrid,
  DataGridHeader,
  DataGridRow,
  DataGridHeaderCell,
  DataGridCell,
  DataGridBody,
  TableColumnDefinition,
  createTableColumn,
} from '@fluentui/react-components';
import { 
  DatabaseRegular,
  SettingsRegular,
  PlayRegular,
  DocumentRegular,
  InfoRegular,
} from '@fluentui/react-icons';
import { ConfigValue, ConfigSource } from '../../types/config';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
  },
  sourcesGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: tokens.spacingVerticalM,
  },
  sourceCard: {
    height: '100%',
  },
  sourceHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalS,
  },
  sourceIcon: {
    fontSize: '20px',
  },
  priorityBadge: {
    marginLeft: 'auto',
  },
  configTable: {
    maxHeight: '400px',
    overflowY: 'auto',
  },
  sourceBadge: {
    fontSize: '12px',
  },
  overriddenValue: {
    textDecoration: 'line-through',
    opacity: 0.6,
  },
  currentValue: {
    fontWeight: tokens.fontWeightSemibold,
  },
});

interface SourcesPrecedenceViewProps {
  config: Record<string, any>;
  className?: string;
}

interface ConfigRow {
  path: string;
  description: string;
  currentValue: any;
  source: ConfigSource;
  priority: number;
  isOverridden: boolean;
  originalValue?: any;
}

const SourcesPrecedenceView: React.FC<SourcesPrecedenceViewProps> = ({
  config,
  className,
}) => {
  const styles = useStyles();

  // Get source information
  const sources = [
    {
      type: 'default' as ConfigSource,
      name: 'Default',
      description: 'Built-in defaults',
      icon: <DocumentRegular className={styles.sourceIcon} />,
      priority: 1,
      color: 'subtle' as const,
    },
    {
      type: 'database' as ConfigSource,
      name: 'Database',
      description: 'Saved configuration',
      icon: <DatabaseRegular className={styles.sourceIcon} />,
      priority: 2,
      color: 'informative' as const,
    },
    {
      type: 'environment' as ConfigSource,
      name: 'Environment',
      description: 'Environment variables',
      icon: <SettingsRegular className={styles.sourceIcon} />,
      priority: 3,
      color: 'warning' as const,
    },
    {
      type: 'runtime' as ConfigSource,
      name: 'Runtime',
      description: 'Runtime overrides',
      icon: <PlayRegular className={styles.sourceIcon} />,
      priority: 4,
      color: 'success' as const,
    },
  ];

  // Flatten config to rows
  const configRows: ConfigRow[] = React.useMemo(() => {
    const rows: ConfigRow[] = [];
    
    const processSection = (section: any, sectionName: string) => {
      Object.entries(section).forEach(([key, value]) => {
        if (value && typeof value === 'object' && 'value' in value) {
          const configValue = value as ConfigValue;
          rows.push({
            path: `${sectionName}.${key}`,
            description: configValue.description || '',
            currentValue: configValue.value,
            source: configValue.source,
            priority: configValue.priority,
            isOverridden: configValue.isOverridden,
            originalValue: configValue.originalValue,
          });
        }
      });
    };

    Object.entries(config).forEach(([sectionName, section]) => {
      if (section && typeof section === 'object') {
        processSection(section, sectionName);
      }
    });

    return rows.sort((a, b) => a.path.localeCompare(b.path));
  }, [config]);

  // Count values by source
  const sourceCounts = React.useMemo(() => {
    const counts: Record<ConfigSource, number> = {
      default: 0,
      database: 0,
      environment: 0,
      runtime: 0,
    };
    
    configRows.forEach(row => {
      counts[row.source]++;
    });
    
    return counts;
  }, [configRows]);

  // Table columns
  const columns: TableColumnDefinition<ConfigRow>[] = [
    createTableColumn<ConfigRow>({
      columnId: 'path',
      compare: (a, b) => a.path.localeCompare(b.path),
      renderHeaderCell: () => 'Configuration Path',
      renderCell: (item) => (
        <Text weight="semibold">{item.path}</Text>
      ),
    }),
    createTableColumn<ConfigRow>({
      columnId: 'description',
      compare: (a, b) => a.description.localeCompare(b.description),
      renderHeaderCell: () => 'Description',
      renderCell: (item) => (
        <Text>{item.description}</Text>
      ),
    }),
    createTableColumn<ConfigRow>({
      columnId: 'value',
      renderHeaderCell: () => 'Value',
      renderCell: (item) => (
        <div>
          <Text className={styles.currentValue}>
            {formatValue(item.currentValue)}
          </Text>
          {item.isOverridden && item.originalValue !== undefined && (
            <Text className={styles.overriddenValue}>
              {formatValue(item.originalValue)}
            </Text>
          )}
        </div>
      ),
    }),
    createTableColumn<ConfigRow>({
      columnId: 'source',
      compare: (a, b) => a.priority - b.priority,
      renderHeaderCell: () => 'Source',
      renderCell: (item) => {
        const source = sources.find(s => s.type === item.source);
        return (
          <Tooltip content={source?.description ?? ''} relationship="label">
            <Badge 
              appearance="outline" 
              color={source?.color}
              className={styles.sourceBadge}
            >
              {source?.name}
            </Badge>
          </Tooltip>
        );
      },
    }),
    createTableColumn<ConfigRow>({
      columnId: 'priority',
      compare: (a, b) => a.priority - b.priority,
      renderHeaderCell: () => 'Priority',
      renderCell: (item) => (
        <Text>{item.priority}</Text>
      ),
    }),
  ];

  const formatValue = (value: any): string => {
    if (Array.isArray(value)) {
      return `[${value.join(', ')}]`;
    }
    if (typeof value === 'object') {
      return JSON.stringify(value);
    }
    if (typeof value === 'string' && value.startsWith('sk-')) {
      return value.substring(0, 8) + '...';
    }
    return String(value);
  };

  return (
    <div className={mergeClasses(styles.container, className)}>
      {/* Sources Overview */}
      <div>
        <Text size={500} weight="semibold">Configuration Sources</Text>
        <Text size={300} style={{ marginTop: tokens.spacingVerticalXS }}>
          Configuration values are loaded from multiple sources with precedence hierarchy
        </Text>
      </div>

      <div className={styles.sourcesGrid}>
        {sources.map((source) => (
          <Card key={source.type} className={styles.sourceCard}>
            <CardHeader
              header={
                <div className={styles.sourceHeader}>
                  {source.icon}
                  <Text weight="semibold">{source.name}</Text>
                  <Badge 
                    appearance="filled" 
                    color={source.color}
                    className={styles.priorityBadge}
                  >
                    Priority {source.priority}
                  </Badge>
                </div>
              }
              description={source.description}
            />
            <CardPreview>
              <Text size={600} weight="semibold">
                {sourceCounts[source.type]}
              </Text>
              <Text size={300}>active values</Text>
            </CardPreview>
          </Card>
        ))}
      </div>

      {/* Configuration Table */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacingHorizontalS, marginBottom: tokens.spacingVerticalM }}>
          <Text size={500} weight="semibold">All Configuration Values</Text>
          <Tooltip content="Shows all configuration values with their sources and precedence" relationship="label">
            <InfoRegular />
          </Tooltip>
        </div>

        <DataGrid
          items={configRows}
          columns={columns}
          sortable
          className={styles.configTable}
        >
          <DataGridHeader>
            <DataGridRow>
              {({ renderHeaderCell }) => (
                <DataGridHeaderCell>{renderHeaderCell()}</DataGridHeaderCell>
              )}
            </DataGridRow>
          </DataGridHeader>
          <DataGridBody<ConfigRow>>
            {({ item, rowId }) => (
              <DataGridRow<ConfigRow> key={rowId}>
                {({ renderCell }) => (
                  <DataGridCell>{renderCell(item)}</DataGridCell>
                )}
              </DataGridRow>
            )}
          </DataGridBody>
        </DataGrid>
      </div>
    </div>
  );
};

export default SourcesPrecedenceView;
