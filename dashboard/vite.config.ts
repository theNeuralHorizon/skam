import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3001,
    proxy: {
      '/api/chaos': {
        target: 'http://localhost:30090',
        rewrite: (path) => path.replace(/^\/api\/chaos/, ''),
      },
      '/api/anomaly': {
        target: 'http://localhost:30091',
        rewrite: (path) => path.replace(/^\/api\/anomaly/, ''),
      },
      '/api/healing': {
        target: 'http://localhost:30092',
        rewrite: (path) => path.replace(/^\/api\/healing/, ''),
      },
    },
  },
})
