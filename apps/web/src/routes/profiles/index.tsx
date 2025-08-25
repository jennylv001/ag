/**
 * Profiles Index Route - M2 Implementation
 */
import { createFileRoute } from '@tanstack/react-router'
import ProfilesPage from '../../pages/ProfilesPage'

export const Route = createFileRoute('/profiles/')({
  component: ProfilesPage,
})
