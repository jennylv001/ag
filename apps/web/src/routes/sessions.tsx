import { createFileRoute } from '@tanstack/react-router'
import { SessionControlPanelPage } from '../pages/SessionControlPanelPage'

export const Route = createFileRoute('/sessions')({
  component: SessionControlPanelPage,
})
