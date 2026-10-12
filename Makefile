.PHONY: install test lint eval eval-live serve seed docker up down

install:
	pip install -e ".[dev,local-models]"

test:
	pytest --cov=finsight --cov-report=term-missing

lint:
	ruff check src tests

eval:
	python -m finsight.eval.runner --dataset fixture --modes hybrid,dense,sparse --prompts rag_answer@v1,rag_answer@v2

eval-live:
	python -m finsight.eval.runner --dataset live --modes hybrid,dense,sparse

seed:
	finsight ingest AAPL MSFT NVDA

serve:
	finsight serve --reload

docker:
	docker build -t catsfi-finsight .

up:
	docker compose up --build -d

down:
	docker compose down
