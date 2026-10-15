import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// WHY a dev proxy instead of a hardcoded API URL in components: the browser
// page and the Flask API live on different ports during development, and a
// same-origin `/api/*` call forwarded by Vite avoids CORS entirely while
// keeping every component free of host/port knowledge. In production the
// same relative paths work behind any reverse proxy that serves both.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // WHY port 5001 as the default target: it is the API port used by
      // packages/api/app.py in this monorepo. Override with API_PORT only
      // when running the API elsewhere.
      "/api": {
        target: process.env.API_PORT
          ? `http://127.0.0.1:${process.env.API_PORT}`
          : "http://127.0.0.1:5001",
        changeOrigin: true,
      },
    },
  },
});
