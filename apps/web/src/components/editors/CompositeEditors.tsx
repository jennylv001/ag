/**
 * Composite Editors - M2 Implementation
 * Reusable form components for complex data types
 */

import React, { useState } from 'react';
import {
  Button,
  Input,
  Table,
  TableHeader,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  makeStyles,
  tokens,
  Text,
  Dialog,
  DialogTrigger,
  DialogSurface,
  DialogTitle,
  DialogContent,
  DialogBody,
  DialogActions,
  Field,
} from '@fluentui/react-components';
import {
  Add24Regular,
  Delete24Regular,
  Edit24Regular,
  Folder24Regular,
  Document24Regular,
} from '@fluentui/react-icons';

const useStyles = makeStyles({
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
  },
  table: {
    border: `1px solid ${tokens.colorNeutralStroke2}`,
    borderRadius: tokens.borderRadiusMedium,
  },
  toolbar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: tokens.spacingVerticalS,
  },
  addButton: {
    marginLeft: 'auto',
  },
  dialogContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
    minWidth: '400px',
  },
  listItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: tokens.spacingVerticalS,
    border: `1px solid ${tokens.colorNeutralStroke2}`,
    borderRadius: tokens.borderRadiusSmall,
    marginBottom: tokens.spacingVerticalS,
  },
  pathContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalS,
  },
});

// Key-Value Editor for headers, environment variables
interface KeyValueEditorProps {
  data: Record<string, string>;
  onChange: (data: Record<string, string>) => void;
  title?: string;
  keyPlaceholder?: string;
  valuePlaceholder?: string;
}

export const KeyValueEditor: React.FC<KeyValueEditorProps> = ({
  data,
  onChange,
  title = 'Key-Value Pairs',
  keyPlaceholder = 'Key',
  valuePlaceholder = 'Value',
}) => {
  const styles = useStyles();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [newKey, setNewKey] = useState('');
  const [newValue, setNewValue] = useState('');

  const entries = Object.entries(data);

  const handleAdd = () => {
    if (newKey && !data[newKey]) {
      onChange({ ...data, [newKey]: newValue });
      setNewKey('');
      setNewValue('');
      setIsDialogOpen(false);
    }
  };

  const handleEdit = (key: string) => {
    setEditingKey(key);
    setNewKey(key);
    setNewValue(data[key]);
    setIsDialogOpen(true);
  };

  const handleUpdate = () => {
    if (editingKey && newKey) {
      const newData = { ...data };
      if (editingKey !== newKey) {
        delete newData[editingKey];
      }
      newData[newKey] = newValue;
      onChange(newData);
      setEditingKey(null);
      setNewKey('');
      setNewValue('');
      setIsDialogOpen(false);
    }
  };

  const handleDelete = (key: string) => {
    const newData = { ...data };
    delete newData[key];
    onChange(newData);
  };

  return (
    <div className={styles.container}>
      <div className={styles.toolbar}>
        <Text weight="semibold">{title}</Text>
        <Dialog open={isDialogOpen} onOpenChange={(_, data) => setIsDialogOpen(data.open)}>
          <DialogTrigger>
            <Button
              appearance="primary"
              icon={<Add24Regular />}
              className={styles.addButton}
              onClick={() => {
                setEditingKey(null);
                setNewKey('');
                setNewValue('');
              }}
            >
              Add
            </Button>
          </DialogTrigger>
          <DialogSurface>
            <DialogTitle>{editingKey ? 'Edit Entry' : 'Add Entry'}</DialogTitle>
            <DialogContent>
              <DialogBody>
                <div className={styles.dialogContent}>
                  <Field label="Key">
                    <Input
                      value={newKey}
                      onChange={(_, data) => setNewKey(data.value)}
                      placeholder={keyPlaceholder}
                    />
                  </Field>
                  <Field label="Value">
                    <Input
                      value={newValue}
                      onChange={(_, data) => setNewValue(data.value)}
                      placeholder={valuePlaceholder}
                    />
                  </Field>
                </div>
              </DialogBody>
              <DialogActions>
                <Button
                  appearance="secondary"
                  onClick={() => setIsDialogOpen(false)}
                >
                  Cancel
                </Button>
                <Button
                  appearance="primary"
                  onClick={editingKey ? handleUpdate : handleAdd}
                  disabled={!newKey}
                >
                  {editingKey ? 'Update' : 'Add'}
                </Button>
              </DialogActions>
            </DialogContent>
          </DialogSurface>
        </Dialog>
      </div>

      {entries.length > 0 ? (
        <Table className={styles.table}>
          <TableHeader>
            <TableRow>
              <TableHeaderCell>Key</TableHeaderCell>
              <TableHeaderCell>Value</TableHeaderCell>
              <TableHeaderCell>Actions</TableHeaderCell>
            </TableRow>
          </TableHeader>
          <TableBody>
            {entries.map(([key, value]) => (
              <TableRow key={key}>
                <TableCell>{key}</TableCell>
                <TableCell>{value}</TableCell>
                <TableCell>
                  <Button
                    appearance="subtle"
                    icon={<Edit24Regular />}
                    onClick={() => handleEdit(key)}
                    size="small"
                  />
                  <Button
                    appearance="subtle"
                    icon={<Delete24Regular />}
                    onClick={() => handleDelete(key)}
                    size="small"
                  />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      ) : (
        <Text>No entries added yet. Click "Add" to create your first entry.</Text>
      )}
    </div>
  );
};

// List Editor for launch arguments, permissions
interface ListEditorProps {
  data: string[];
  onChange: (data: string[]) => void;
  title?: string;
  placeholder?: string;
  itemLabel?: string;
}

export const ListEditor: React.FC<ListEditorProps> = ({
  data,
  onChange,
  title = 'List Items',
  placeholder = 'Enter value',
  itemLabel = 'Item',
}) => {
  const styles = useStyles();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [newValue, setNewValue] = useState('');

  const handleAdd = () => {
    if (newValue.trim()) {
      onChange([...data, newValue.trim()]);
      setNewValue('');
      setIsDialogOpen(false);
    }
  };

  const handleEdit = (index: number) => {
    setEditingIndex(index);
    setNewValue(data[index]);
    setIsDialogOpen(true);
  };

  const handleUpdate = () => {
    if (editingIndex !== null && newValue.trim()) {
      const newData = [...data];
      newData[editingIndex] = newValue.trim();
      onChange(newData);
      setEditingIndex(null);
      setNewValue('');
      setIsDialogOpen(false);
    }
  };

  const handleDelete = (index: number) => {
    const newData = data.filter((_, i) => i !== index);
    onChange(newData);
  };

  return (
    <div className={styles.container}>
      <div className={styles.toolbar}>
        <Text weight="semibold">{title}</Text>
        <Dialog open={isDialogOpen} onOpenChange={(_, data) => setIsDialogOpen(data.open)}>
          <DialogTrigger>
            <Button
              appearance="primary"
              icon={<Add24Regular />}
              className={styles.addButton}
              onClick={() => {
                setEditingIndex(null);
                setNewValue('');
              }}
            >
              Add
            </Button>
          </DialogTrigger>
          <DialogSurface>
            <DialogTitle>{editingIndex !== null ? `Edit ${itemLabel}` : `Add ${itemLabel}`}</DialogTitle>
            <DialogContent>
              <DialogBody>
                <div className={styles.dialogContent}>
                  <Field label={itemLabel}>
                    <Input
                      value={newValue}
                      onChange={(_, data) => setNewValue(data.value)}
                      placeholder={placeholder}
                    />
                  </Field>
                </div>
              </DialogBody>
              <DialogActions>
                <Button
                  appearance="secondary"
                  onClick={() => setIsDialogOpen(false)}
                >
                  Cancel
                </Button>
                <Button
                  appearance="primary"
                  onClick={editingIndex !== null ? handleUpdate : handleAdd}
                  disabled={!newValue.trim()}
                >
                  {editingIndex !== null ? 'Update' : 'Add'}
                </Button>
              </DialogActions>
            </DialogContent>
          </DialogSurface>
        </Dialog>
      </div>

      {data.length > 0 ? (
        <div>
          {data.map((item, index) => (
            <div key={index} className={styles.listItem}>
              <Text>{item}</Text>
              <div>
                <Button
                  appearance="subtle"
                  icon={<Edit24Regular />}
                  onClick={() => handleEdit(index)}
                  size="small"
                />
                <Button
                  appearance="subtle"
                  icon={<Delete24Regular />}
                  onClick={() => handleDelete(index)}
                  size="small"
                />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <Text>No items added yet. Click "Add" to create your first item.</Text>
      )}
    </div>
  );
};

// File/Folder Picker with validation
interface FilePickerProps {
  value?: string;
  onChange: (value: string) => void;
  type: 'file' | 'folder';
  label: string;
  placeholder?: string;
}

export const FileFolderPicker: React.FC<FilePickerProps> = ({
  value,
  onChange,
  type,
  label,
  placeholder,
}) => {
  const styles = useStyles();

  const handleBrowse = () => {
    // In a real implementation, this would open a native file/folder picker
    // For now, we'll simulate it with a prompt
    const path = prompt(`Select ${type}:`);
    if (path) {
      onChange(path);
    }
  };

  return (
    <Field label={label}>
      <div className={styles.pathContainer}>
        {type === 'folder' ? <Folder24Regular /> : <Document24Regular />}
        <Input
          value={value || ''}
          onChange={(_, data) => onChange(data.value)}
          placeholder={placeholder || `Enter ${type} path`}
          style={{ flex: 1 }}
        />
        <Button appearance="secondary" onClick={handleBrowse}>
          Browse
        </Button>
      </div>
    </Field>
  );
};
