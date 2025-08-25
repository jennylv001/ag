import { createRootRoute, Outlet } from '@tanstack/react-router'
import { AppLayout } from '../components/layout/AppLayout'

declare global {
  // eslint-disable-next-line no-var
  var __APP_ROOT_ROUTE__: ReturnType<typeof createRootRoute> | undefined
}

export const Route =
  (globalThis as any).__APP_ROOT_ROUTE__ ||
  ((globalThis as any).__APP_ROOT_ROUTE__ = createRootRoute({
    component: () => (
      <AppLayout>
        <Outlet />
      </AppLayout>
    ),
  }))
