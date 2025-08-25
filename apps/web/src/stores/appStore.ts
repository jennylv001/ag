import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

interface AppState {
  // UI State
  activeRoute: string
  sidebarCollapsed: boolean
  
  // Application State
  isConnected: boolean
  currentProfile: string | null
  
  // Actions
  setActiveRoute: (route: string) => void
  toggleSidebar: () => void
  setConnected: (connected: boolean) => void
  setCurrentProfile: (profileId: string | null) => void
}

export const useAppStore = create<AppState>()(
  devtools(
    (set) => ({
      // Initial state
      activeRoute: 'dashboard',
      sidebarCollapsed: false,
      isConnected: false,
      currentProfile: null,
      
      // Actions
      setActiveRoute: (route) => set({ activeRoute: route }),
      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setConnected: (connected) => set({ isConnected: connected }),
      setCurrentProfile: (profileId) => set({ currentProfile: profileId }),
    }),
    {
      name: 'app-store',
    }
  )
)
