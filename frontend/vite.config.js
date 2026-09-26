import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The frontend talks to the API through a same-origin dev proxy rather than a
// hardcoded http://127.0.0.1:8000. That removes cross-origin requests (and the
// CORS preflight), and stops the browser resolving 127.0.0.1 to the wrong
// machine when the dev server is reached from another device. The backend stays
// bound to loopback; only Vite needs to be reachable.
const API_TARGET = process.env.ETHS_API_TARGET || 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
});
