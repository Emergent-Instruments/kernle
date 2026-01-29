.PHONY: help install dev test lint format clean build publish publish-test docs docs-deploy

# Default target
help:
	@echo "Kernle Development Commands"
	@echo "============================"
	@echo ""
	@echo "Development:"
	@echo "  make install      Install package in editable mode"
	@echo "  make dev          Install with dev dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linter (ruff)"
	@echo "  make format       Format code (black + ruff)"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Publishing:"
	@echo "  make build        Build distribution packages"
	@echo "  make publish-test Upload to TestPyPI"
	@echo "  make publish      Upload to PyPI (production)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         Run docs site locally"
	@echo "  make docs-deploy  Deploy docs to Mintlify"
	@echo ""

# Development
install:
	pip install -e .

dev:
	pip install -e ".[dev,mcp,local,cloud]"

test:
	pytest tests/ -v --tb=short --ignore=tests/test_postgres_storage.py

test-all:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=kernle --cov-report=html --ignore=tests/test_postgres_storage.py

lint:
	ruff check .

format:
	black .
	ruff check --fix .

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Publishing
build: clean
	python -m build

publish-test: build
	twine check dist/*
	twine upload --repository testpypi dist/*

publish: build
	twine check dist/*
	twine upload dist/*

# Documentation
docs:
	cd docs-site && mintlify dev

docs-deploy:
	cd docs-site && mintlify deploy

# Backend (Railway)
backend-dev:
	cd backend && uvicorn api.main:app --reload --port 8000

backend-docker:
	cd backend && docker build -t kernle-backend .
	docker run -p 8000:8000 --env-file backend/.env kernle-backend

# Web (Next.js)
web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build
