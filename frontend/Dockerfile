FROM node:24-slim

WORKDIR /app

ARG VITE_API_KEY=""
ENV VITE_API_KEY=$VITE_API_KEY

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]