# Builds the React/Vite frontend and serves it with Caddy, which also
# terminates HTTPS and reverse-proxies the API to the backend container.
# Build context is the repo root; see web.Dockerfile.dockerignore.

FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY index.html tsconfig.json tsconfig.node.json vite.config.ts tailwind.config.js postcss.config.js ./
COPY src ./src
COPY public ./public
# Public origin used for the canonical/Open Graph URLs, robots.txt and sitemap.xml
# (e.g. --build-arg SITE_URL=https://fpl.example.com). Empty means the default in vite.config.ts.
ARG SITE_URL
ENV SITE_URL=$SITE_URL
RUN npm run build

FROM caddy:2-alpine
COPY deploy/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /app/dist /srv
