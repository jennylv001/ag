import type { Preview } from '@storybook/react'
import { FluentProvider, webLightTheme } from '@fluentui/react-components'
import '../src/index.css'

const preview: Preview = {
  parameters: {
    actions: { argTypesRegex: '^on[A-Z].*' },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/,
      },
    },
  },
  decorators: [
    (Story) => (
      <FluentProvider theme={webLightTheme}>
        <Story />
      </FluentProvider>
    ),
  ],
}

export default preview
