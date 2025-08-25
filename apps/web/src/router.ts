import { createRouter } from '@tanstack/react-router'
import { routeTree } from './routeTree.gen'

// Global singleton to survive HMR/StrictMode
declare global {
  // eslint-disable-next-line no-var
  var __APP_ROUTER__: ReturnType<typeof createRouter> | undefined
}

export const router =
  (globalThis as any).__APP_ROUTER__ ||
  ((globalThis as any).__APP_ROUTER__ = createRouter({ routeTree }))

// HMR: accept routeTree updates and dispose cleanly
if (import.meta && (import.meta as any).hot) {
  // Accept updates to the generated route tree and update the router in place
  ;(import.meta as any).hot.accept('./routeTree.gen', (mod: any) => {
    try {
      if (mod?.routeTree) {
        router.update({ routeTree: mod.routeTree })
      }
    } catch (err) {
      console.error('Router update failed, reloading page...', err)
      // Fall back to full reload to guarantee a clean tree
      window.location.reload()
    }
  })

  ;(import.meta as any).hot.dispose(() => {
    try {
      router.dispose?.()
    } finally {
      ;(globalThis as any).__APP_ROUTER__ = undefined
  ;(globalThis as any).__APP_ROUTE_TREE__ = undefined
  ;(globalThis as any).__APP_ROOT_ROUTE__ = undefined
    }
  })
}
