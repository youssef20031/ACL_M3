import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// The dev server proxies the API so the app is same-origin, like production.
// API_PROXY_TARGET picks the backend: a local uvicorn by default, or the deployed site
// with `npm run dev:live` (mode "live" reads .env.live), so frontend work needs no
// Python, Neo4j or API keys.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.API_PROXY_TARGET || 'http://localhost:8000';
  return {
    plugins: [react()],
    server: {
      port: 3000,
      proxy: {
        '/api': { target, changeOrigin: true },
        '/health': { target, changeOrigin: true },
        '/ws': { target: target.replace(/^http/, 'ws'), ws: true, changeOrigin: true },
      },
    },
  };
});
