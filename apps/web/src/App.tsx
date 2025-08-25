import { RouterProvider } from '@tanstack/react-router'
// Use a singleton router that survives HMR to avoid duplicate __root__
import { router } from './router'

// Register the router instance for type safety
declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}

function App() {
  return <RouterProvider router={router} />
}

export default App

