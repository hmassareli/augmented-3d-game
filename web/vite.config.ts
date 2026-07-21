import { defineConfig } from 'vite'
import { resolve } from 'node:path'

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        game: resolve(__dirname, 'index.html'),
        comparison: resolve(__dirname, 'compare.html'),
      },
    },
  },
})