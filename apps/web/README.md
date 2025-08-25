# Browser Use Control Center

Enterprise-grade frontend interface for browser automation control and monitoring.

## Architecture

- **Framework**: React 18 + TypeScript + Vite
- **Design System**: Fluent UI v9 (Fluent 2)
- **State Management**: Zustand
- **Data Fetching**: TanStack Query
- **Routing**: TanStack Router
- **Testing**: Vitest + React Testing Library
- **Storybook**: Component development environment

## Development Setup

### Prerequisites

- Node.js 18+ 
- npm, yarn, or pnpm

### Installation

```bash
cd apps/web
npm install
```

### Development Scripts

```bash
# Start development server (http://localhost:3000)
npm run dev

# Type checking
npm run typecheck

# Linting
npm run lint
npm run lint:fix

# Testing
npm run test
npm run test:ui
npm run test:coverage

# Storybook (http://localhost:6006)
npm run storybook

# Production build
npm run build
npm run preview
```

### Project Structure

```
src/
├── components/          # Reusable UI components
│   └── layout/         # Layout components (Header, Sidebar, etc.)
├── pages/              # Page components
├── stores/             # Zustand state stores
├── hooks/              # Custom React hooks
├── types/              # TypeScript type definitions
├── test/               # Test utilities and setup
├── App.tsx             # Root App component
├── main.tsx            # Application entry point
└── index.css           # Global styles
```

## Design System

This application uses **Fluent UI v9** (Fluent 2) for consistent design patterns:

- **Layout**: AppLayout, Header, Sidebar, Content, Footer
- **Navigation**: Tree, TreeItem for sidebar navigation
- **Data Display**: DataGrid, Card, Badge, Tag
- **Forms**: Input, Dropdown, Switch, Button
- **Feedback**: Toast notifications via Sonner

## State Management

- **Zustand**: Lightweight state management for UI and application state
- **TanStack Query**: Server state management with caching, retries, and real-time updates

## Milestones

- **M1**: ✅ Foundation & UI shell (Current)
- **M2**: Profile Builder with validation
- **M3**: Session Control Panel
- **M4**: Config Manager
- **M5**: Observability & Logging
- **M6**: Tools (MCP) Panel
- **M7**: Dashboard widgets
- **M8**: Backend integration
- **M9**: QA & accessibility
- **M10**: Production deployment

## Contributing

1. Follow TypeScript strict mode
2. Use Prettier for formatting
3. Write tests for components
4. Document components in Storybook
5. Follow accessibility guidelines

## API Integration

The frontend is designed to integrate with the browser-use Python backend:

- **Profiles**: CRUD operations for browser profiles
- **Sessions**: Session lifecycle management
- **Config**: Environment and runtime configuration  
- **Observability**: Real-time logs and traces
- **MCP Tools**: Browser action controls
