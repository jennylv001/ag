/**
 * Export Manager Component - M5 Implementation
 * Export logs and traces in multiple formats with download management
 */

import React from 'react';
import {
  Card,
  CardHeader,
  Text,
  Button,
  Dropdown,
  Option,
  Field,
  Switch,
  Input,
  Badge,
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
  MessageBar,
  MessageBarTitle,
  MessageBarBody,
} from '@fluentui/react-components';
import {
  Save24Regular,
  ArrowDownload24Regular,
  Delete24Regular,
  Document24Regular,
  CheckmarkCircle24Regular,
  ErrorCircle24Regular,
  Clock24Regular,
} from '@fluentui/react-icons';
import { ExportOptions, ExportResult } from '../../types/observability';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    gap: tokens.spacingVerticalL,
  },
  exportSection: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: tokens.spacingVerticalL,
  },
  formGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: tokens.spacingVerticalM,
  },
  fullWidth: {
    gridColumn: '1 / -1',
  },
  exportsList: {
    marginTop: tokens.spacingVerticalL,
  },
  statusIcon: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXS,
  },
  exportActions: {
    display: 'flex',
    gap: tokens.spacingHorizontalXS,
  },
  sizeCell: {
    fontFamily: tokens.fontFamilyMonospace,
    fontSize: tokens.fontSizeBase200,
  },
  dateCell: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
  },
  progressCell: {
    minWidth: '100px',
  },
  formatBadge: {
    textTransform: 'uppercase',
  },
  previewSection: {
    marginTop: tokens.spacingVerticalM,
    padding: tokens.spacingVerticalM,
    backgroundColor: tokens.colorNeutralBackground1,
    borderRadius: tokens.borderRadiusMedium,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
  },
  previewContent: {
    fontFamily: tokens.fontFamilyMonospace,
    fontSize: tokens.fontSizeBase200,
    whiteSpace: 'pre-wrap',
    maxHeight: '200px',
    overflowY: 'auto',
    marginTop: tokens.spacingVerticalS,
  },
});

interface ExportManagerProps {
  activeExports: ExportResult[];
  onExportLogs: (options: ExportOptions) => Promise<void>;
  onExportTrace: (traceId: string, options: ExportOptions) => Promise<void>;
  onDownloadExport: (exportId: string) => void;
  onDeleteExport: (exportId: string) => void;
  className?: string;
}

const ExportManager: React.FC<ExportManagerProps> = ({
  activeExports,
  onExportLogs,
  onExportTrace,
  onDownloadExport,
  onDeleteExport,
  className,
}) => {
  const styles = useStyles();
  const [exportType, setExportType] = React.useState<'logs' | 'trace'>('logs');
  const [selectedTraceId, setSelectedTraceId] = React.useState<string>('');
  const [exportOptions, setExportOptions] = React.useState<ExportOptions>({
    format: 'json',
    includeStackTraces: true,
    includeContext: true,
    maxSize: 100, // MB
  });
  const [showPreview, setShowPreview] = React.useState(false);
  const [isExporting, setIsExporting] = React.useState(false);

  // Mock trace IDs for demonstration
  const availableTraces = [
    { id: 'trace_001', name: 'Login Flow - Session sess_abc123' },
    { id: 'trace_002', name: 'Checkout Process - Session sess_def456' },
    { id: 'trace_003', name: 'Form Submission - Session sess_ghi789' },
  ];

  const formatOptions = [
    { value: 'json', label: 'JSON', description: 'Structured JSON format' },
    { value: 'csv', label: 'CSV', description: 'Comma-separated values' },
    { value: 'txt', label: 'TXT', description: 'Plain text format' },
    { value: 'har', label: 'HAR', description: 'HTTP Archive format' },
  ];

  const handleExport = async () => {
    setIsExporting(true);
    try {
      if (exportType === 'logs') {
        await onExportLogs(exportOptions);
      } else {
        await onExportTrace(selectedTraceId, exportOptions);
      }
    } finally {
      setIsExporting(false);
    }
  };

  const getStatusIcon = (status: ExportResult['status']) => {
    switch (status) {
      case 'completed':
  return <CheckmarkCircle24Regular style={{ color: tokens.colorPaletteGreenForeground1 }} />;
      case 'error':
  return <ErrorCircle24Regular style={{ color: tokens.colorPaletteRedForeground1 }} />;
      default:
  return <Clock24Regular style={{ color: tokens.colorPaletteYellowForeground1 }} />;
    }
  };

  const getStatusColor = (status: ExportResult['status']) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'error':
        return 'danger';
      default:
        return 'warning';
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (date: Date): string => {
    return date.toLocaleString();
  };

  // Mock preview data
  const generatePreview = (): string => {
    switch (exportOptions.format) {
      case 'json':
        return JSON.stringify({
          exportInfo: {
            timestamp: new Date().toISOString(),
            format: 'json',
            includeStackTraces: exportOptions.includeStackTraces,
            includeContext: exportOptions.includeContext,
          },
          logs: [
            {
              id: 'log_001',
              timestamp: '2024-01-15T10:30:00.123Z',
              level: 'info',
              message: 'Action completed: click on login button',
              source: 'browser-use',
              context: exportOptions.includeContext ? { url: 'https://example.com' } : undefined,
            },
            // ... more logs
          ]
        }, null, 2);
      case 'csv':
        return `timestamp,level,source,message,context\n2024-01-15T10:30:00.123Z,info,browser-use,"Action completed: click on login button",${exportOptions.includeContext ? '"{""url"":""https://example.com""}"' : '""'}\n2024-01-15T10:30:01.456Z,debug,llm-provider,"Token count: 1247",${exportOptions.includeContext ? '"{""model"":""gpt-4o""}"' : '""'}`;
      case 'txt':
        return `[2024-01-15T10:30:00.123Z] INFO  browser-use: Action completed: click on login button\n[2024-01-15T10:30:01.456Z] DEBUG llm-provider: Token count: 1247`;
      case 'har':
        return JSON.stringify({
          log: {
            version: '1.2',
            creator: { name: 'Browser Use', version: '1.0' },
            entries: [
              {
                startedDateTime: '2024-01-15T10:30:00.123Z',
                request: { method: 'GET', url: 'https://example.com' },
                response: { status: 200 },
              }
            ]
          }
        }, null, 2);
      default:
        return 'Preview not available for this format';
    }
  };

  // Table columns
  const columns: TableColumnDefinition<ExportResult>[] = [
    createTableColumn<ExportResult>({
      columnId: 'status',
      compare: (a, b) => a.status.localeCompare(b.status),
      renderHeaderCell: () => 'Status',
      renderCell: (item) => (
        <div className={styles.statusIcon}>
          {getStatusIcon(item.status)}
          <Badge color={getStatusColor(item.status)}>
            {item.status}
          </Badge>
        </div>
      ),
    }),
    createTableColumn<ExportResult>({
      columnId: 'filename',
      compare: (a, b) => a.filename.localeCompare(b.filename),
      renderHeaderCell: () => 'Filename',
      renderCell: (item) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacingHorizontalS }}>
          <Document24Regular />
          <Text>{item.filename}</Text>
        </div>
      ),
    }),
    createTableColumn<ExportResult>({
      columnId: 'format',
      compare: (a, b) => a.format.localeCompare(b.format),
      renderHeaderCell: () => 'Format',
      renderCell: (item) => (
        <Badge appearance="outline" className={styles.formatBadge}>
          {item.format}
        </Badge>
      ),
    }),
    createTableColumn<ExportResult>({
      columnId: 'size',
      compare: (a, b) => a.size - b.size,
      renderHeaderCell: () => 'Size',
      renderCell: (item) => (
        <Text className={styles.sizeCell}>
          {formatFileSize(item.size)}
        </Text>
      ),
    }),
    createTableColumn<ExportResult>({
      columnId: 'created',
      compare: (a, b) => a.createdAt.getTime() - b.createdAt.getTime(),
      renderHeaderCell: () => 'Created',
      renderCell: (item) => (
        <Text className={styles.dateCell}>
          {formatDate(item.createdAt)}
        </Text>
      ),
    }),
    createTableColumn<ExportResult>({
      columnId: 'expires',
      compare: (a, b) => a.expiresAt.getTime() - b.expiresAt.getTime(),
      renderHeaderCell: () => 'Expires',
      renderCell: (item) => (
        <Text className={styles.dateCell}>
          {formatDate(item.expiresAt)}
        </Text>
      ),
    }),
    createTableColumn<ExportResult>({
      columnId: 'actions',
      renderHeaderCell: () => 'Actions',
      renderCell: (item) => (
        <div className={styles.exportActions}>
          <Button
            appearance="subtle"
            size="small"
            icon={<ArrowDownload24Regular />}
            onClick={() => onDownloadExport(item.id)}
            disabled={item.status !== 'completed'}
          />
          <Button
            appearance="subtle"
            size="small"
            icon={<Delete24Regular />}
            onClick={() => onDeleteExport(item.id)}
          />
        </div>
      ),
    }),
  ];

  return (
    <div className={mergeClasses(styles.container, className)}>
      {/* Header */}
      <div>
        <Text size={500} weight="semibold">Export Manager</Text>
        <Text size={300}>
          Export logs and traces in multiple formats for analysis and archival
        </Text>
      </div>

      <div className={styles.exportSection}>
        {/* Export Configuration */}
        <Card>
          <CardHeader 
            header={<Text weight="semibold">Create New Export</Text>}
            description="Configure export options and generate downloadable files"
          />
          
          <div className={styles.formGrid}>
            <Field label="Export Type">
              <Dropdown
                value={exportType}
                selectedOptions={[exportType]}
                onOptionSelect={(_, data) => setExportType(data.optionValue as 'logs' | 'trace')}
              >
                <Option value="logs" text="All Logs">All Logs</Option>
                <Option value="trace" text="Specific Trace">Specific Trace</Option>
              </Dropdown>
            </Field>

            <Field label="Format">
              <Dropdown
                value={exportOptions.format}
                selectedOptions={[exportOptions.format]}
                onOptionSelect={(_, data) => setExportOptions({
                  ...exportOptions,
                  format: data.optionValue as any
                })}
              >
                {formatOptions.map(format => (
                  <Option key={format.value} value={format.value} text={format.label}>
                    <div>
                      <Text weight="semibold">{format.label}</Text>
                      <Text size={200}>{format.description}</Text>
                    </div>
                  </Option>
                ))}
              </Dropdown>
            </Field>

            {exportType === 'trace' && (
              <Field label="Select Trace" className={styles.fullWidth}>
                <Dropdown
                  placeholder="Choose a trace to export..."
                  value={selectedTraceId}
                  selectedOptions={selectedTraceId ? [selectedTraceId] : []}
                  onOptionSelect={(_, data) => setSelectedTraceId(data.optionValue as string)}
                >
                  {availableTraces.map(trace => (
                    <Option key={trace.id} value={trace.id} text={trace.name}>
                      {trace.name}
                    </Option>
                  ))}
                </Dropdown>
              </Field>
            )}

            <Field label="Include Stack Traces">
              <Switch
                checked={exportOptions.includeStackTraces}
                onChange={(_, data) => setExportOptions({
                  ...exportOptions,
                  includeStackTraces: data.checked
                })}
              />
            </Field>

            <Field label="Include Context Data">
              <Switch
                checked={exportOptions.includeContext}
                onChange={(_, data) => setExportOptions({
                  ...exportOptions,
                  includeContext: data.checked
                })}
              />
            </Field>

            <Field label="Max File Size (MB)">
              <Input
                type="number"
                value={String(exportOptions.maxSize)}
                onChange={(_, data) => setExportOptions({
                  ...exportOptions,
                  maxSize: parseInt(data.value) || 100
                })}
              />
            </Field>
          </div>

          <div style={{ marginTop: tokens.spacingVerticalL, display: 'flex', gap: tokens.spacingHorizontalM }}>
            <Button
              appearance="primary"
              icon={<Save24Regular />}
              onClick={handleExport}
              disabled={isExporting || (exportType === 'trace' && !selectedTraceId)}
            >
              {isExporting ? 'Exporting...' : 'Create Export'}
            </Button>
            
            <Button
              appearance="secondary"
              onClick={() => setShowPreview(!showPreview)}
            >
              {showPreview ? 'Hide' : 'Show'} Preview
            </Button>
          </div>

          {/* Preview Section */}
          {showPreview && (
            <div className={styles.previewSection}>
              <Text weight="semibold">Export Preview</Text>
              <div className={styles.previewContent}>
                {generatePreview()}
              </div>
            </div>
          )}
        </Card>

        {/* Export Statistics */}
        <Card>
          <CardHeader 
            header={<Text weight="semibold">Export Statistics</Text>}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: tokens.spacingVerticalM }}>
            <div>
              <Text size={300}>Total Exports</Text>
              <Text size={600} weight="semibold">{activeExports.length}</Text>
            </div>
            
            <div>
              <Text size={300}>Completed</Text>
              <Text size={400}>
                {activeExports.filter(e => e.status === 'completed').length}
              </Text>
            </div>
            
            <div>
              <Text size={300}>In Progress</Text>
              <Text size={400}>
                {activeExports.filter(e => e.status === 'pending').length}
              </Text>
            </div>
            
            <div>
              <Text size={300}>Total Size</Text>
              <Text size={400}>
                {formatFileSize(activeExports.reduce((sum, e) => sum + e.size, 0))}
              </Text>
            </div>
          </div>
        </Card>
      </div>

      {/* Active Exports */}
      <div className={styles.exportsList}>
        <Text size={400} weight="semibold" style={{ marginBottom: tokens.spacingVerticalM }}>
          Export History
        </Text>
        
        {activeExports.length > 0 ? (
          <DataGrid
            items={activeExports}
            columns={columns}
            sortable
            getRowId={(item) => item.id}
          >
            <DataGridHeader>
              <DataGridRow>
                {({ renderHeaderCell }) => (
                  <DataGridHeaderCell>{renderHeaderCell()}</DataGridHeaderCell>
                )}
              </DataGridRow>
            </DataGridHeader>
            <DataGridBody<ExportResult>>
              {({ item, rowId }) => (
                <DataGridRow<ExportResult> key={rowId}>
                  {({ renderCell }) => (
                    <DataGridCell>{renderCell(item)}</DataGridCell>
                  )}
                </DataGridRow>
              )}
            </DataGridBody>
          </DataGrid>
        ) : (
          <MessageBar intent="info">
            <MessageBarTitle>No Exports</MessageBarTitle>
            <MessageBarBody>
              Create your first export using the form above.
            </MessageBarBody>
          </MessageBar>
        )}
      </div>
    </div>
  );
};

export default ExportManager;
