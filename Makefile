# Makefile for Financial RAG development tasks

.PHONY: help install install-dev lint format test test-cov security docker-build docker-run clean

# Default target
help:
	@echo "Financial RAG - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  pre-commit     Install pre-commit hooks"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run linter (ruff)"
	@echo "  format         Format code (ruff)"
	@echo "  type-check     Run type checker (mypy)"
	@echo "  security       Run security scan (bandit)"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run tests"
	@echo "  test-cov       Run tests with coverage"
	@echo "  test-unit      Run unit tests only"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container"
	@echo "  docker-stop    Stop Docker container"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          Remove cache files"

# Setup targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

pre-commit:
	pre-commit install

# Code quality targets
lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

type-check:
	mypy . --ignore-missing-imports

security:
	bandit -r . -x ./tests,./venv,./.venv -ll

# Testing targets
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/ -v -m "not integration"

# Docker targets
docker-build:
	docker build -t financial-rag:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "coverage.xml" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
