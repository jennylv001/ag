import { createFileRoute } from '@tanstack/react-router'
import { ProfileBuilderPage } from '../../pages/ProfileBuilderPage'

export const Route = createFileRoute('/profiles/$profileId')({
  component: ProfileBuilderPage,
})
