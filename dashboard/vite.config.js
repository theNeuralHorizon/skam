import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/anomaly': {
        target: process.env.MOCK_TARGET || 'http://localhost:9000',
        changeOrigin: true,
      },
      '/decision/ws': {
        target: process.env.MOCK_TARGET || 'http://localhost:9000',
        changeOrigin: true,
        ws: true,
      },
      '/decision/api': {
        target: process.env.MOCK_TARGET || 'http://localhost:9000',
        changeOrigin: true,
      },
      '/decision': {
        target: process.env.MOCK_TARGET || 'http://localhost:9000',
        changeOrigin: true,
        ws: true,
      },
      '/chaos/api': {
        target: process.env.MOCK_TARGET || 'http://localhost:9000',
        changeOrigin: true,
      },
      '/chaos': {
        target: process.env.MOCK_TARGET || 'http://localhost:9000',
        changeOrigin: true,
      },
    },
  },
})
