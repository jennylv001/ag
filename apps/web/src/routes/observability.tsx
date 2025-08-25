/**
 * Observability Routes - M5 Implementation
 * Routing configuration for the observability page
 */

import { createFileRoute } from '@tanstack/react-router';
import ObservabilityPage from '../pages/ObservabilityPage';

export const Route = createFileRoute('/observability')({
  component: ObservabilityPage,
});
