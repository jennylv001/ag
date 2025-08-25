import '@testing-library/jest-dom'
import { beforeAll, vi } from 'vitest'

// Mock IntersectionObserver with full interface shape
class MockIntersectionObserver implements IntersectionObserver {
  readonly root: Document | Element | null = null
  readonly rootMargin: string = '0px'
  readonly thresholds: ReadonlyArray<number> = [0]
  constructor(_callback: IntersectionObserverCallback, _options?: IntersectionObserverInit) {}
  disconnect(): void {}
  observe(_target: Element): void {}
  takeRecords(): IntersectionObserverEntry[] { return [] }
  unobserve(_target: Element): void {}
}
;(global as any).IntersectionObserver = MockIntersectionObserver as any

// Mock ResizeObserver
global.ResizeObserver = vi.fn(() => ({
  observe: vi.fn(),
  disconnect: vi.fn(),
  unobserve: vi.fn(),
}))

// Setup global test environment
beforeAll(() => {
  // Any global setup
})
