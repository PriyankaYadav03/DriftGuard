import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// In dev, /api/* is proxied to the FastAPI server so the browser never needs CORS or an API key.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { '/api': { target: process.env.API_TARGET || 'http://localhost:8000', rewrite: p => p.replace(/^\/api/, '') } },
  },
})
