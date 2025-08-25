/**
 * Chrome Flag Inspector - M2 Implementation
 * Shows computed Chrome flags with source attribution and rationale tooltips
 */

import React from 'react';
import {
  Table,
  TableHeader,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  Text,
  Badge,
  Tooltip,
  makeStyles,
  tokens,
} from '@fluentui/react-components';
import { Info16Regular } from '@fluentui/react-icons';
import { ChromeFlag } from '../../types/profile';

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
  flagCell: {
    fontFamily: 'monospace',
    fontSize: tokens.fontSizeBase200,
  },
  valueCell: {
    fontFamily: 'monospace',
    fontSize: tokens.fontSizeBase200,
    maxWidth: '200px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  sourceCell: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXS,
  },
  rationaleIcon: {
    cursor: 'pointer',
    color: tokens.colorBrandForeground1,
  },
  emptyState: {
    textAlign: 'center',
    padding: tokens.spacingVerticalXL,
    color: tokens.colorNeutralForeground3,
  },
});

interface ChromeFlagInspectorProps {
  flags: ChromeFlag[];
}

export const ChromeFlagInspector: React.FC<ChromeFlagInspectorProps> = ({ flags }) => {
  const styles = useStyles();

  const getSourceBadgeAppearance = (source: ChromeFlag['source']) => {
    switch (source) {
  case 'stealth': return 'outline' as const;
  case 'security': return 'outline' as const;
  case 'viewport': return 'tint' as const;
  case 'args': return 'ghost' as const;
  case 'computed': return 'outline' as const;
  default: return 'ghost' as const;
    }
  };

  const getSourceColor = (source: ChromeFlag['source']) => {
    switch (source) {
  case 'stealth': return 'danger';
  case 'security': return 'warning';
  case 'viewport': return 'brand';
  case 'args': return 'success';
  case 'computed': return 'subtle';
  default: return 'subtle';
    }
  };

  if (flags.length === 0) {
    return (
      <div className={styles.container}>
        <Text size={500} weight="semibold">Chrome Flags Inspector</Text>
        <div className={styles.emptyState}>
          <Text>No Chrome flags will be applied with the current configuration.</Text>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Text size={500} weight="semibold">Chrome Flags Inspector</Text>
        <Text size={200}>
          {flags.length} flag{flags.length !== 1 ? 's' : ''} will be applied
        </Text>
      </div>

      <Table className={styles.table}>
        <TableHeader>
          <TableRow>
            <TableHeaderCell style={{ width: '30%' }}>Flag</TableHeaderCell>
            <TableHeaderCell style={{ width: '25%' }}>Value</TableHeaderCell>
            <TableHeaderCell style={{ width: '20%' }}>Source</TableHeaderCell>
            <TableHeaderCell style={{ width: '25%' }}>Rationale</TableHeaderCell>
          </TableRow>
        </TableHeader>
        <TableBody>
          {flags.map((flag, index) => (
            <TableRow key={`${flag.flag}-${index}`}>
              <TableCell className={styles.flagCell}>
                <Text>{flag.flag}</Text>
              </TableCell>
              <TableCell className={styles.valueCell}>
                <Text>
                  {typeof flag.value === 'boolean' 
                    ? (flag.value ? 'true' : 'false')
                    : flag.value
                  }
                </Text>
              </TableCell>
              <TableCell>
                <div className={styles.sourceCell}>
                  <Badge 
                    appearance={getSourceBadgeAppearance(flag.source)}
                    color={getSourceColor(flag.source)}
                  >
                    {flag.source}
                  </Badge>
                </div>
              </TableCell>
              <TableCell>
                <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacingHorizontalXS }}>
                  <Text size={200} style={{ flex: 1 }}>
                    {flag.rationale.length > 50 
                      ? `${flag.rationale.substring(0, 50)}...`
                      : flag.rationale
                    }
                  </Text>
                  {flag.rationale.length > 50 && (
                    <Tooltip content={flag.rationale} relationship="description">
                      <Info16Regular className={styles.rationaleIcon} />
                    </Tooltip>
                  )}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <Text size={200} style={{ color: tokens.colorNeutralForeground3 }}>
        💡 These flags are computed based on your profile configuration. 
        Some flags may be automatically added or modified by the browser runtime.
      </Text>
    </div>
  );
};
