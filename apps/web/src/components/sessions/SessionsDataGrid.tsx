/**
 * Sessions DataGrid - M3 Implementation
 * Main sessions list with filtering and status management
 */

import React, { useEffect, useState } from 'react';
import {
  DataGrid,
  DataGridHeader,
  DataGridHeaderCell,
  DataGridBody,
  DataGridRow,
  DataGridCell,
  TableColumnDefinition,
  TableCellLayout,
  createTableColumn,
  Badge,
  Button,
  Input,
  Dropdown,
  Option,
  Card,
  CardHeader,
  Text,
  makeStyles,
  tokens,
  Spinner,
  CounterBadge,
} from '@fluentui/react-components';
import {
  Play24Regular,
  Stop24Regular,
  Delete24Regular,
  Filter24Regular,
  Search24Regular,
  Add24Regular,
} from '@fluentui/react-icons';
import { BrowserSession, SessionFilters, SessionStats } from '../../types/session';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
    height: '100%',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: tokens.spacingVerticalM,
    borderBottom: `1px solid ${tokens.colorNeutralStroke2}`,
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalM,
  },
  filters: {
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    padding: tokens.spacingVerticalM,
    backgroundColor: tokens.colorNeutralBackground2,
    borderRadius: tokens.borderRadiusMedium,
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  statsRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: tokens.spacingHorizontalM,
    marginBottom: tokens.spacingVerticalM,
  },
  statCard: {
    padding: tokens.spacingVerticalM,
    textAlign: 'center',
  },
  dataGridContainer: {
    flex: 1,
    overflow: 'hidden',
    border: `1px solid ${tokens.colorNeutralStroke2}`,
    borderRadius: tokens.borderRadiusMedium,
  },
  statusCell: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalS,
  },
  actionsCell: {
    display: 'flex',
    gap: tokens.spacingHorizontalXS,
  },
  profileCell: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXS,
  },
  clickableRow: {
    cursor: 'pointer',
    '&:hover': {
      backgroundColor: tokens.colorNeutralBackground1Hover,
    },
  },
  loadingOverlay: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '300px',
  },
});

interface SessionsDataGridProps {
  sessions: BrowserSession[];
  stats: SessionStats;
  filters: SessionFilters;
  isLoading: boolean;
  selectedSession: BrowserSession | null;
  onSelectSession: (session: BrowserSession) => void;
  onFiltersChange: (filters: Partial<SessionFilters>) => void;
  onStartSession: (sessionId: string) => void;
  onStopSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  onCreateSession: () => void;
}

export const SessionsDataGrid: React.FC<SessionsDataGridProps> = ({
  sessions,
  stats,
  filters,
  isLoading,
  selectedSession,
  onSelectSession,
  onFiltersChange,
  onStartSession,
  onStopSession,
  onDeleteSession,
  onCreateSession,
}) => {
  const styles = useStyles();
  const [searchTerm, setSearchTerm] = useState(filters.search || '');

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      onFiltersChange({ search: searchTerm });
    }, 300);
    return () => clearTimeout(timer);
  }, [searchTerm, onFiltersChange]);

  const getStatusBadge = (status: BrowserSession['status']) => {
    const config = {
      running: { appearance: 'filled' as const, color: 'success' as const },
      stopped: { appearance: 'outline' as const, color: 'subtle' as const },
      launching: { appearance: 'filled' as const, color: 'warning' as const },
      stopping: { appearance: 'filled' as const, color: 'warning' as const },
      error: { appearance: 'filled' as const, color: 'danger' as const },
    };

    return (
      <Badge
        appearance={config[status]?.appearance || 'outline'}
        color={config[status]?.color || 'subtle'}
      >
        {status.toUpperCase()}
      </Badge>
    );
  };

  const getSessionActions = (session: BrowserSession) => {
    return (
      <div className={styles.actionsCell}>
        {session.status === 'stopped' && (
          <Button
            appearance="subtle"
            icon={<Play24Regular />}
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              onStartSession(session.id);
            }}
            title="Start Session"
          />
        )}
        {session.status === 'running' && (
          <Button
            appearance="subtle"
            icon={<Stop24Regular />}
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              onStopSession(session.id);
            }}
            title="Stop Session"
          />
        )}
        <Button
          appearance="subtle"
          icon={<Delete24Regular />}
          size="small"
          onClick={(e) => {
            e.stopPropagation();
            onDeleteSession(session.id);
          }}
          title="Delete Session"
          disabled={session.status === 'running'}
        />
      </div>
    );
  };

  const columns: TableColumnDefinition<BrowserSession>[] = [
    createTableColumn<BrowserSession>({
      columnId: 'name',
      compare: (a, b) => a.name.localeCompare(b.name),
      renderHeaderCell: () => 'Session Name',
      renderCell: (session) => (
        <TableCellLayout>
          <div>
            <Text weight="semibold">{session.name}</Text>
            <Text size={200} style={{ color: tokens.colorNeutralForeground3 }}>
              ID: {session.id}
            </Text>
          </div>
        </TableCellLayout>
      ),
    }),
    createTableColumn<BrowserSession>({
      columnId: 'profile',
      compare: (a, b) => a.profileName.localeCompare(b.profileName),
      renderHeaderCell: () => 'Profile',
      renderCell: (session) => (
        <TableCellLayout>
          <div className={styles.profileCell}>
            <Text>{session.profileName}</Text>
            <Text size={200} style={{ color: tokens.colorNeutralForeground3 }}>
              {session.profileId}
            </Text>
          </div>
        </TableCellLayout>
      ),
    }),
    createTableColumn<BrowserSession>({
      columnId: 'status',
      compare: (a, b) => a.status.localeCompare(b.status),
      renderHeaderCell: () => 'Status',
      renderCell: (session) => (
        <TableCellLayout>
          <div className={styles.statusCell}>
            {getStatusBadge(session.status)}
            {session.status === 'running' && (
              <CounterBadge count={session.runtime.activePages} color="informative" />
            )}
          </div>
        </TableCellLayout>
      ),
    }),
    createTableColumn<BrowserSession>({
      columnId: 'runtime',
      renderHeaderCell: () => 'Runtime',
      renderCell: (session) => (
        <TableCellLayout>
          {session.status === 'running' ? (
            <div>
              <Text size={200}>
                Pages: {session.runtime.activePages} | Requests: {session.runtime.totalRequests}
              </Text>
              {session.runtime.memoryUsage && (
                <Text size={200} style={{ color: tokens.colorNeutralForeground3 }}>
                  Memory: {Math.round(session.runtime.memoryUsage)}MB
                </Text>
              )}
            </div>
          ) : (
            <Text size={200} style={{ color: tokens.colorNeutralForeground3 }}>
              {session.stoppedAt ? `Stopped ${new Date(session.stoppedAt).toLocaleTimeString()}` : 'Not started'}
            </Text>
          )}
        </TableCellLayout>
      ),
    }),
    createTableColumn<BrowserSession>({
      columnId: 'created',
      compare: (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
      renderHeaderCell: () => 'Created',
      renderCell: (session) => (
        <TableCellLayout>
          <Text size={200}>
            {new Date(session.createdAt).toLocaleDateString()} {new Date(session.createdAt).toLocaleTimeString()}
          </Text>
        </TableCellLayout>
      ),
    }),
    createTableColumn<BrowserSession>({
      columnId: 'actions',
      renderHeaderCell: () => 'Actions',
      renderCell: (session) => (
        <TableCellLayout>
          {getSessionActions(session)}
        </TableCellLayout>
      ),
    }),
  ];

  const filteredSessions = sessions.filter(session => {
    if (filters.search && !session.name.toLowerCase().includes(filters.search.toLowerCase()) &&
        !session.profileName.toLowerCase().includes(filters.search.toLowerCase())) {
      return false;
    }
    if (filters.status && filters.status.length > 0 && !filters.status.includes(session.status)) {
      return false;
    }
    if (filters.profileId && session.profileId !== filters.profileId) {
      return false;
    }
    return true;
  });

  if (isLoading) {
    return (
      <div className={styles.loadingOverlay}>
        <Spinner label="Loading sessions..." />
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <Text size={500} weight="semibold">Browser Sessions</Text>
          <CounterBadge count={filteredSessions.length} />
        </div>
        <Button
          appearance="primary"
          icon={<Add24Regular />}
          onClick={onCreateSession}
        >
          New Session
        </Button>
      </div>

      {/* Stats Cards */}
      <div className={styles.statsRow}>
        <Card className={styles.statCard}>
          <CardHeader
            header={<Text size={400} weight="semibold">Total Sessions</Text>}
            description={<Text size={600} weight="bold">{stats.total}</Text>}
          />
        </Card>
        <Card className={styles.statCard}>
          <CardHeader
            header={<Text size={400} weight="semibold">Running</Text>}
            description={<Text size={600} weight="bold" style={{ color: tokens.colorPaletteGreenForeground1 }}>{stats.running}</Text>}
          />
        </Card>
        <Card className={styles.statCard}>
          <CardHeader
            header={<Text size={400} weight="semibold">Active Pages</Text>}
            description={<Text size={600} weight="bold">{stats.totalPages}</Text>}
          />
        </Card>
        <Card className={styles.statCard}>
          <CardHeader
            header={<Text size={400} weight="semibold">Artifacts</Text>}
            description={<Text size={600} weight="bold">{stats.totalArtifacts}</Text>}
          />
        </Card>
      </div>

      {/* Filters */}
      <div className={styles.filters}>
        <Input
          contentBefore={<Search24Regular />}
          placeholder="Search sessions..."
          value={searchTerm}
          onChange={(_, data) => setSearchTerm(data.value)}
        />
        
        <Dropdown
          placeholder="Filter by status"
          value={filters.status?.join(', ') || ''}
          multiselect
          onOptionSelect={(_, data) => {
            const selectedOptions = data.selectedOptions;
            onFiltersChange({ 
              status: selectedOptions.length > 0 ? selectedOptions as BrowserSession['status'][] : undefined 
            });
          }}
        >
          <Option value="running">Running</Option>
          <Option value="stopped">Stopped</Option>
          <Option value="launching">Launching</Option>
          <Option value="error">Error</Option>
        </Dropdown>

        <Button
          appearance="subtle"
          icon={<Filter24Regular />}
          onClick={() => onFiltersChange({})}
        >
          Clear Filters
        </Button>
      </div>

      {/* Data Grid */}
      <div className={styles.dataGridContainer}>
        <DataGrid
          items={filteredSessions}
          columns={columns}
          sortable
          selectionMode="single"
          getRowId={(item) => item.id}
        >
          <DataGridHeader>
            <DataGridRow>
              {({ renderHeaderCell }) => (
                <DataGridHeaderCell>{renderHeaderCell()}</DataGridHeaderCell>
              )}
            </DataGridRow>
          </DataGridHeader>
          <DataGridBody<BrowserSession>>
            {({ item, rowId }) => (
              <DataGridRow<BrowserSession>
                key={rowId}
                className={styles.clickableRow}
                onClick={() => onSelectSession(item)}
                style={{
                  backgroundColor: selectedSession?.id === item.id 
                    ? tokens.colorNeutralBackground1Selected 
                    : undefined
                }}
              >
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
