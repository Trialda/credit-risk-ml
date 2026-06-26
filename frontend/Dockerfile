# ---- Base: shared dependency install ----
FROM node:24-slim AS base
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---- Dev target: local Docker Compose, hot-reload ----
FROM base AS dev
ARG VITE_API_KEY=""
ENV VITE_API_KEY=$VITE_API_KEY
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# ---- Build stage: compiles static assets for production ----
FROM base AS build
ARG VITE_API_KEY=""
ENV VITE_API_KEY=$VITE_API_KEY
COPY . .
RUN npm run build

# ---- Production target: static files served by Nginx ----
FROM nginx:1.25-alpine AS production
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]