import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    proxy: {
      '/backtest': 'http://localhost:8000',
      '/evaluation': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/vol-eval': 'http://localhost:8000',
      '/price-forecast': 'http://localhost:8000',
      '/co-mov': 'http://localhost:8000',
    },
  },
})
