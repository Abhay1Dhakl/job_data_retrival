.PHONY: setup build-index api docker-up docker-down docker-build-index

setup:
	uv venv
	. .venv/bin/activate && uv pip install -e backend

build-index:
	PYTHONPATH=backend python backend/scripts/build_index.py

api:
	PYTHONPATH=backend uvicorn app.main:app --reload

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-build-index:
	docker compose run --rm api python backend/scripts/build_index.py
