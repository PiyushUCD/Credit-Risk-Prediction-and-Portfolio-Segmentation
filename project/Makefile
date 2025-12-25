.PHONY: install install-dev format lint test run app sync-assets

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	python -m pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt

format:
	black .
	@echo "Formatted with black."

lint:
	ruff check .
	black --check .

test:
	pytest -q

run:
	python main.py

app:
	pip install -r requirements.txt -r requirements-app.txt
	streamlit run app.py

sync-assets:
	python scripts/sync_assets.py
