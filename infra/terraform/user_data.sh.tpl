#!/bin/bash
set -euo pipefail

# ---- Install Docker ----
dnf install -y docker
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# Docker Compose plugin (not bundled with Amazon Linux 2023)
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# ---- Authenticate to ECR ----
aws ecr get-login-password --region ${aws_region} | \
  docker login --username AWS --password-stdin \
  "$(echo ${ecr_backend_url} | cut -d'/' -f1)"

# ---- Application directory ----
mkdir -p /opt/credit-risk-ml
cd /opt/credit-risk-ml

# ---- docker-compose.prod.yml, written directly (no git clone needed) ----
cat > docker-compose.yml <<'COMPOSE_EOF'
services:
  nginx:
    image: nginx:1.25-alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - backend
      - frontend

  backend:
    image: ${ecr_backend_url}:latest
    environment:
      DATABASE_URL: postgresql://creditrisk:creditrisk@db:5432/creditrisk
      MODEL_PATH: /app/model/model.pkl
      API_KEY: ""
    depends_on:
      db:
        condition: service_healthy

  frontend:
    image: ${ecr_frontend_url}:latest
    depends_on:
      - backend

  db:
    image: postgres:16
    environment:
      POSTGRES_USER: creditrisk
      POSTGRES_PASSWORD: creditrisk
      POSTGRES_DB: creditrisk
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U creditrisk"]
      interval: 5s
      timeout: 5s
      retries: 5
COMPOSE_EOF

# Minimal nginx.conf for demo deployment, routing only, no rate limiting 
# (full version with rate limiting lives in the repo's  nginx/nginx.conf)
cat > nginx.conf <<'NGINX_EOF'
events {}
http {
  server {
    listen 80;

    location /api/ {
      rewrite ^/api/(.*) /$1 break;
      proxy_pass http://backend:8000;
    }

    location / {
      proxy_pass http://frontend:80;
    }
  }
}
NGINX_EOF

# ---- Start the stack ----
docker compose up -d