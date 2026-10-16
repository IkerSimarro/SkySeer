import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The dashboard is built into the Python package, so `skyseer serve` (and the Pi) serve it
// directly. During development (`npm run dev`) API calls go to a running `skyseer serve`.
const api = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../src/skyseer/api/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': api,
      '/media': api,
      '/ws': { target: api.replace('http', 'ws'), ws: true },
    },
  },
})
