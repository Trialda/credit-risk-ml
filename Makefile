.PHONY: up down rebuild logs ps shell-backend shell-db

up:
	docker compose up -d

down:
	docker compose down

rebuild:
	docker compose down -v && docker compose up --build -d

logs:
	docker compose logs -f

ps:
	docker compose ps

shell-backend:
	docker compose exec backend bash

shell-db:
	docker compose exec db psql -U creditrisk -d creditrisk