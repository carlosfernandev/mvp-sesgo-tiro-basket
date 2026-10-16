# WHY node:20-alpine: the Vite dev server is pure JS with no native modules,
# so a small musl image is fine here (unlike the Python backend). Pinned to
# Node 20 to match the project's toolchain.
FROM node:20-alpine

WORKDIR /app

# WHY package files copied first: the dependency layer is cached and only
# rebuilt when the lockfile changes, not on every source edit. `npm ci` is
# used instead of `npm install` so the image gets the exact locked versions
# and the build stays reproducible.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

# WHY --host 0.0.0.0: vite.config.ts already sets `host: true`, but passing it
# explicitly keeps the container reachable on the published port even if the
# config changes.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
