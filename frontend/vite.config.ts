import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// WHY a dev proxy instead of a hardcoded API URL in components: the browser
// page and the Flask API live on different ports during development, and a
// same-origin `/api/*` call forwarded by Vite avoids CORS entirely while
// keeping every component free of host/port knowledge. In production the
// same relative paths work behind any reverse proxy that serves both.
// WHY API_PROXY_TARGET wins over API_PORT: local dev only needs to move the
// port (API_PORT=5002), while a container needs a different host entirely
// (API_PROXY_TARGET=http://backend:5001). A full target URL covers both cases,
// so API_PORT stays as a convenience shorthand and keeps the old behaviour.
const apiProxyTarget =
  process.env.API_PROXY_TARGET ??
  `http://127.0.0.1:${process.env.API_PORT ?? "5001"}`;

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // WHY host: true - inside a container the dev server must listen on all
    // interfaces or the published port never reaches it; on the host it is a
    // no-op because Vite still prints the loopback URL.
    host: true,
    proxy: {
      "/api": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
});
