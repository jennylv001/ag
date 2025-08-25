/**
 * Profiles Layout Route - M2 Implementation
 * Layout route for profiles section
 */

import { createFileRoute, Outlet } from '@tanstack/react-router';

export const Route = createFileRoute('/profiles')({
  component: ProfilesLayout,
});

function ProfilesLayout() {
  return <Outlet />;
}
