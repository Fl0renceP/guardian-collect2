import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy to Flask so the browser sees one origin — no CORS setup, and no API
    // base URL to configure per environment.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      // The biometric demo pages are plain HTML served by Flask, not React
      // routes. Without these the "Face scan" link 404s in dev, because Vite
      // would try to resolve them against the SPA.
      '^/(demos|test-scan|test-plate|test-azure-plate)$': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
