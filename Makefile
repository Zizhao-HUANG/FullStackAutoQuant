.PHONY: install install-dev lint format format-check test coverage inference webui validate-config clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install core dependencies
	pip install -e .

install-dev: ## Install all dependencies including dev tools
	pip install -e ".[all]"

lint: ## Run linting (ruff + mypy)
	ruff check fullstackautoquant/ tests/
	mypy fullstackautoquant/ --ignore-missing-imports

format: ## Auto-format code
	ruff check --fix fullstackautoquant/ tests/
	black fullstackautoquant/ tests/
	isort fullstackautoquant/ tests/

format-check: ## Check formatting without modifying files
	black --check --diff --line-length=100 --target-version=py310 fullstackautoquant/ tests/
	ruff check --select I --diff fullstackautoquant/ tests/

test: ## Run test suite
	pytest tests/ -v

coverage: ## Run tests with coverage report
	pytest tests/ -v --cov=fullstackautoquant --cov-report=term-missing --cov-config=pyproject.toml

inference: ## Run model inference (date=auto)
	python -m fullstackautoquant.model.inference --date auto

webui: ## Start Streamlit WebUI
	streamlit run fullstackautoquant/webui/app/streamlit_app.py

validate-config: ## Validate config files and JSON schemas
	@echo "Validating JSON schemas (meta-validation)..."
	@for schema in configs/schema/*.schema.json; do \
		echo "  ✓ $$schema"; \
		python -c "import json; json.load(open('$$schema'))"; \
	done
	@echo "Validating YAML configs..."
	@for cfg in configs/*.yaml configs/*.yaml.example; do \
		if [ -f "$$cfg" ]; then \
			python -c "import yaml; yaml.safe_load(open('$$cfg'))" && echo "  ✓ $$cfg"; \
		fi; \
	done
	@echo "✅ All config files valid"

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
