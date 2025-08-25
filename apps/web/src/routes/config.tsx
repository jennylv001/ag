/**
 * Config Manager Routes - M4 Implementation
 * Routing configuration for the config manager page
 */

import { createFileRoute } from '@tanstack/react-router';
import ConfigManagerPage from '../pages/ConfigManagerPage';

export const Route = createFileRoute('/config')({
  component: ConfigManagerPage,
});
