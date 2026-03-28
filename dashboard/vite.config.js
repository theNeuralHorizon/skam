import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/anomaly': {
        target: 'http://localhost:9000',
        changeOrigin: true,
      },
      '/decision': {
        target: 'http://localhost:9000',
        changeOrigin: true,
        ws: true,
      },
      '/chaos': {
        target: 'http://localhost:9000',
        changeOrigin: true,
      },
    },
  },
})
