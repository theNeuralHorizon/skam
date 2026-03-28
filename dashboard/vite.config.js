import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/anomaly': {
        target: process.env.ANOMALY_TARGET || 'http://localhost:30091',
        changeOrigin: true,
      },
      '/decision': {
        target: process.env.DECISION_TARGET || 'http://localhost:30092',
        changeOrigin: true,
        ws: true,
      },
      '/chaos': {
        target: process.env.CHAOS_TARGET || 'http://localhost:30090',
        changeOrigin: true,
      },
    },
  },
})
