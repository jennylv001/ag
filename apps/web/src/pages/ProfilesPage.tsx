/**
 * Profiles Page - M2 Implementation
 * Browser profile management and selection interface
 */

import React from 'react';
import {
  Text,
  Button,
  Card,
  CardHeader,
  Badge,
  DataGrid,
  DataGridHeader,
  DataGridRow,
  DataGridHeaderCell,
  DataGridCell,
  DataGridBody,
  TableColumnDefinition,
  createTableColumn,
  makeStyles,
  tokens,
} from '@fluentui/react-components';
import {
  Add24Regular,
  Edit24Regular,
  Play24Regular,
  Delete24Regular,
} from '@fluentui/react-icons';
import { useNavigate } from '@tanstack/react-router';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    padding: tokens.spacingVerticalL,
    gap: tokens.spacingVerticalM,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  profilesGrid: {
    flex: 1,
    minHeight: '400px',
  },
});

interface Profile {
  id: string;
  name: string;
  description?: string;
  status: 'active' | 'inactive' | 'draft';
  lastUsed?: Date;
  createdAt: Date;
}

// Mock data for demonstration
const mockProfiles: Profile[] = [
  {
    id: '1',
    name: 'Default Profile',
    description: 'Standard browser configuration',
    status: 'active',
    lastUsed: new Date('2025-08-24'),
    createdAt: new Date('2025-08-01'),
  },
  {
    id: '2',
    name: 'Stealth Mode',
    description: 'High anonymity settings',
    status: 'active',
    lastUsed: new Date('2025-08-23'),
    createdAt: new Date('2025-08-05'),
  },
  {
    id: '3',
    name: 'Testing Profile',
    description: 'For automated testing',
    status: 'draft',
    createdAt: new Date('2025-08-20'),
  },
];

export const ProfilesPage: React.FC = () => {
  const styles = useStyles();
  const navigate = useNavigate();

  const handleNewProfile = () => {
    navigate({ to: '/profiles/$profileId', params: (prev) => ({ ...(prev as any), profileId: 'new' }) });
  };

  const handleEditProfile = (profileId: string) => {
    navigate({ to: '/profiles/$profileId', params: (prev) => ({ ...(prev as any), profileId }) });
  };

  const handleRunProfile = (profileId: string) => {
    // TODO: Implement profile run functionality
    console.log('Running profile:', profileId);
  };

  const handleDeleteProfile = (profileId: string) => {
    // TODO: Implement profile deletion
    console.log('Deleting profile:', profileId);
  };

  const getStatusColor = (status: Profile['status']) => {
    switch (status) {
      case 'active': return 'success' as const;
      case 'inactive': return 'subtle' as const;
      case 'draft': return 'warning' as const;
      default: return 'subtle' as const;
    }
  };

  const columns: TableColumnDefinition<Profile>[] = [
    createTableColumn<Profile>({
      columnId: 'name',
      compare: (a, b) => a.name.localeCompare(b.name),
      renderHeaderCell: () => 'Name',
      renderCell: (item) => (
        <div>
          <Text weight="semibold">{item.name}</Text>
          {item.description && (
            <Text size={200} style={{ display: 'block', color: tokens.colorNeutralForeground3 }}>
              {item.description}
            </Text>
          )}
        </div>
      ),
    }),
    createTableColumn<Profile>({
      columnId: 'status',
      compare: (a, b) => a.status.localeCompare(b.status),
      renderHeaderCell: () => 'Status',
      renderCell: (item) => (
        <Badge color={getStatusColor(item.status)} appearance="filled">
          {item.status}
        </Badge>
      ),
    }),
    createTableColumn<Profile>({
      columnId: 'lastUsed',
      compare: (a, b) => (a.lastUsed?.getTime() || 0) - (b.lastUsed?.getTime() || 0),
      renderHeaderCell: () => 'Last Used',
      renderCell: (item) => (
        <Text size={200}>
          {item.lastUsed ? item.lastUsed.toLocaleDateString() : 'Never'}
        </Text>
      ),
    }),
    createTableColumn<Profile>({
      columnId: 'actions',
      renderHeaderCell: () => 'Actions',
      renderCell: (item) => (
        <div style={{ display: 'flex', gap: tokens.spacingHorizontalS }}>
          <Button
            appearance="subtle"
            size="small"
            icon={<Play24Regular />}
            onClick={() => handleRunProfile(item.id)}
            disabled={item.status === 'draft'}
          >
            Run
          </Button>
          <Button
            appearance="subtle"
            size="small"
            icon={<Edit24Regular />}
            onClick={() => handleEditProfile(item.id)}
          >
            Edit
          </Button>
          <Button
            appearance="subtle"
            size="small"
            icon={<Delete24Regular />}
            onClick={() => handleDeleteProfile(item.id)}
          >
            Delete
          </Button>
        </div>
      ),
    }),
  ];

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <Text size={600} weight="semibold">Browser Profiles</Text>
          <Text size={300} style={{ display: 'block', marginTop: tokens.spacingVerticalXS }}>
            Manage browser configurations, stealth settings, and automation profiles
          </Text>
        </div>
        <Button
          appearance="primary"
          icon={<Add24Regular />}
          onClick={handleNewProfile}
        >
          New Profile
        </Button>
      </div>

      {/* Profiles Grid */}
      <Card className={styles.profilesGrid}>
        <CardHeader
          header={
            <Text weight="semibold">
              {mockProfiles.length} Profile(s)
            </Text>
          }
        />
        <DataGrid
          items={mockProfiles}
          columns={columns}
          sortable
          style={{ height: '100%' }}
        >
          <DataGridHeader>
            <DataGridRow>
              {({ renderHeaderCell }) => (
                <DataGridHeaderCell>{renderHeaderCell()}</DataGridHeaderCell>
              )}
            </DataGridRow>
          </DataGridHeader>
          <DataGridBody<Profile>>
            {({ item, rowId }) => (
              <DataGridRow<Profile> key={rowId}>
                {({ renderCell }) => (
                  <DataGridCell>{renderCell(item)}</DataGridCell>
                )}
              </DataGridRow>
            )}
          </DataGridBody>
        </DataGrid>
      </Card>
    </div>
  );
};

export default ProfilesPage;
