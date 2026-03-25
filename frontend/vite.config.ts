import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const backendTarget = process.env.AUGMENTOR_API_TARGET ?? 'http://localhost:8001'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    open: '/',
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/storage': {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
})
