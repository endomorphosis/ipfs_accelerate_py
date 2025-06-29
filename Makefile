# Mojo Integration Makefile
# Provides convenient commands for testing and development

.PHONY: help test test-unit test-integration test-e2e test-performance test-all
.PHONY: test-mock test-real test-coverage test-quick test-ci
.PHONY: lint format type-check security-check quality-check
.PHONY: setup setup-dev clean clean-all
.PHONY: docker-build docker-test docker-run
.PHONY: server-start server-stop server-restart server-status
.PHONY: docs-build docs-serve reports

# Default target
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
PIP := pip
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
MYPY := mypy
BANDIT := bandit

# Directories
SRC_DIR := src
TEST_DIR := tests
SCRIPTS_DIR := scripts
DOCS_DIR := docs
RESULTS_DIR := test-results
LOGS_DIR := logs

# Test configuration
TEST_TIMEOUT := 300
TEST_WORKERS := auto
MCP_SERVER_PORT := 8004

# Colors
BOLD := \033[1m
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BOLD)Mojo Integration Test & Development Commands$(RESET)"
	@echo ""
	@echo "$(BOLD)Testing:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(test|check)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Development:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -vE "(test|check)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Examples:$(RESET)"
	@echo "  make test                    # Run all tests"
	@echo "  make test-unit               # Run only unit tests"
	@echo "  make test-e2e TEST_TIMEOUT=600  # Run E2E tests with 10min timeout"
	@echo "  make test-coverage           # Run tests with coverage"
	@echo "  make quality-check           # Run all quality checks"
	@echo "  make docker-test             # Run tests in Docker"

# Test targets
test: ## Run all tests (unit + integration + e2e)
	@echo "$(BOLD)$(BLUE)Running all Mojo tests...$(RESET)"
	$(SCRIPTS_DIR)/run_mojo_tests.sh --level all --timeout $(TEST_TIMEOUT)

test-unit: ## Run unit tests only
	@echo "$(BOLD)$(GREEN)Running unit tests...$(RESET)"
	$(PYTEST) $(TEST_DIR)/test_modular_integration.py -v -m "unit" --timeout=$(TEST_TIMEOUT)

test-integration: ## Run integration tests only
	@echo "$(BOLD)$(GREEN)Running integration tests...$(RESET)"
	$(PYTEST) $(TEST_DIR)/test_modular_integration.py -v -m "integration" --timeout=$(TEST_TIMEOUT)

test-e2e: ## Run end-to-end tests only
	@echo "$(BOLD)$(GREEN)Running E2E tests...$(RESET)"
	$(PYTEST) $(TEST_DIR)/e2e/test_mojo_e2e.py -v -s --timeout=$(TEST_TIMEOUT)

test-performance: ## Run performance tests only
	@echo "$(BOLD)$(YELLOW)Running performance tests...$(RESET)"
	$(SCRIPTS_DIR)/run_mojo_tests.sh --level performance --timeout=900

test-all: ## Run comprehensive test suite
	@echo "$(BOLD)$(BLUE)Running comprehensive test suite...$(RESET)"
	$(SCRIPTS_DIR)/run_mojo_tests.sh --level all --coverage --timeout $(TEST_TIMEOUT)

test-mock: ## Run tests in mock mode (no real Mojo required)
	@echo "$(BOLD)$(YELLOW)Running tests in mock mode...$(RESET)"
	MOJO_MOCK_MODE=true $(SCRIPTS_DIR)/run_mojo_tests.sh --level all --mock

test-real: ## Run tests with real Mojo (requires Mojo installation)
	@echo "$(BOLD)$(GREEN)Running tests with real Mojo...$(RESET)"
	$(SCRIPTS_DIR)/run_mojo_tests.sh --level all --real

test-coverage: ## Run tests with coverage reporting
	@echo "$(BOLD)$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -v --cov=$(SRC_DIR) --cov-report=html:$(RESULTS_DIR)/coverage_html --cov-report=term-missing

test-quick: ## Run quick tests (unit + basic integration)
	@echo "$(BOLD)$(GREEN)Running quick tests...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -v -m "not slow and not performance" --timeout=60 -x

test-ci: ## Run tests in CI/CD mode
	@echo "$(BOLD)$(BLUE)Running CI/CD tests...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -v --tb=short --maxfail=5 --timeout=$(TEST_TIMEOUT) --cov=$(SRC_DIR) --cov-report=xml

# Quality checks
lint: ## Run linting checks
	@echo "$(BOLD)$(YELLOW)Running linting...$(RESET)"
	$(FLAKE8) $(SRC_DIR)/ final_mcp_server.py --max-line-length=100

format: ## Format code with Black
	@echo "$(BOLD)$(BLUE)Formatting code...$(RESET)"
	$(BLACK) $(SRC_DIR)/ final_mcp_server.py $(TEST_DIR)/

format-check: ## Check code formatting without changes
	@echo "$(BOLD)$(YELLOW)Checking code formatting...$(RESET)"
	$(BLACK) --check --diff $(SRC_DIR)/ final_mcp_server.py $(TEST_DIR)/

type-check: ## Run type checking
	@echo "$(BOLD)$(BLUE)Running type checks...$(RESET)"
	$(MYPY) $(SRC_DIR)/ final_mcp_server.py --ignore-missing-imports

security-check: ## Run security checks
	@echo "$(BOLD)$(RED)Running security checks...$(RESET)"
	$(BANDIT) -r $(SRC_DIR)/ final_mcp_server.py
	safety check

quality-check: format-check lint type-check security-check ## Run all quality checks
	@echo "$(BOLD)$(GREEN)All quality checks completed!$(RESET)"

# Setup and environment
setup: ## Set up development environment
	@echo "$(BOLD)$(BLUE)Setting up development environment...$(RESET)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-cov pytest-xdist aiohttp numpy
	$(PIP) install black flake8 mypy bandit safety
	mkdir -p $(RESULTS_DIR) $(LOGS_DIR)

setup-dev: setup ## Set up development environment with additional tools
	@echo "$(BOLD)$(BLUE)Setting up enhanced development environment...$(RESET)"
	$(PIP) install pre-commit jupyter notebook
	pre-commit install

# Cleanup
clean: ## Clean temporary files and test artifacts
	@echo "$(BOLD)$(YELLOW)Cleaning temporary files...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf $(RESULTS_DIR)/.coverage*

clean-all: clean ## Clean all artifacts including test results
	@echo "$(BOLD)$(RED)Cleaning all artifacts...$(RESET)"
	rm -rf $(RESULTS_DIR)/
	rm -rf $(LOGS_DIR)/
	rm -rf .tox/
	rm -rf dist/
	rm -rf build/

# Docker operations
docker-build: ## Build Docker image for testing
	@echo "$(BOLD)$(BLUE)Building Docker image...$(RESET)"
	docker build -f Dockerfile.mojo --target testing -t ipfs-accelerate-mojo-test .

docker-test: docker-build ## Run tests in Docker container
	@echo "$(BOLD)$(GREEN)Running tests in Docker...$(RESET)"
	docker run --rm -v $(PWD)/test-results:/app/test-results ipfs-accelerate-mojo-test

docker-run: ## Run MCP server in Docker
	@echo "$(BOLD)$(BLUE)Running MCP server in Docker...$(RESET)"
	docker build -f Dockerfile.mojo --target production -t ipfs-accelerate-mojo .
	docker run -p $(MCP_SERVER_PORT):$(MCP_SERVER_PORT) ipfs-accelerate-mojo

# Server management
server-start: ## Start MCP server for testing
	@echo "$(BOLD)$(GREEN)Starting MCP server...$(RESET)"
	$(PYTHON) final_mcp_server.py --host 127.0.0.1 --port $(MCP_SERVER_PORT) --timeout 600 > $(LOGS_DIR)/mcp_server.log 2>&1 &
	echo $$! > $(LOGS_DIR)/mcp_server.pid
	@echo "Server starting... Check $(LOGS_DIR)/mcp_server.log for output"

server-stop: ## Stop MCP server
	@echo "$(BOLD)$(YELLOW)Stopping MCP server...$(RESET)"
	@if [ -f $(LOGS_DIR)/mcp_server.pid ]; then \
		kill `cat $(LOGS_DIR)/mcp_server.pid` 2>/dev/null || true; \
		rm -f $(LOGS_DIR)/mcp_server.pid; \
		echo "Server stopped"; \
	else \
		echo "No server PID file found"; \
	fi

server-restart: server-stop server-start ## Restart MCP server

server-status: ## Check MCP server status
	@echo "$(BOLD)$(BLUE)Checking MCP server status...$(RESET)"
	@if curl -s http://localhost:$(MCP_SERVER_PORT)/health > /dev/null 2>&1; then \
		echo "$(GREEN)✅ Server is running$(RESET)"; \
	else \
		echo "$(RED)❌ Server is not responding$(RESET)"; \
	fi

# Documentation and reporting
docs-build: ## Build documentation
	@echo "$(BOLD)$(BLUE)Building documentation...$(RESET)"
	@if [ -d "$(DOCS_DIR)" ]; then \
		echo "Documentation directory found"; \
	else \
		echo "No documentation directory found"; \
	fi

docs-serve: ## Serve documentation locally
	@echo "$(BOLD)$(BLUE)Serving documentation...$(RESET)"
	@echo "Documentation serving not yet implemented"

reports: ## Generate test reports
	@echo "$(BOLD)$(BLUE)Generating test reports...$(RESET)"
	@if [ -d "$(RESULTS_DIR)" ]; then \
		echo "Test results available in $(RESULTS_DIR)/"; \
		ls -la $(RESULTS_DIR)/; \
	else \
		echo "No test results found. Run tests first."; \
	fi

# Performance monitoring
benchmark: ## Run performance benchmarks
	@echo "$(BOLD)$(YELLOW)Running performance benchmarks...$(RESET)"
	$(PYTEST) $(TEST_DIR)/e2e/test_mojo_e2e.py::TestMojoPerformance -v \
		--benchmark-json=$(RESULTS_DIR)/benchmark.json

profile: ## Profile test execution
	@echo "$(BOLD)$(YELLOW)Profiling test execution...$(RESET)"
	$(PYTHON) -m cProfile -o $(RESULTS_DIR)/profile.stats -m pytest $(TEST_DIR)/ -v -x

# Development helpers
watch: ## Watch files and run tests automatically
	@echo "$(BOLD)$(BLUE)Watching files for changes...$(RESET)"
	@echo "Install 'entr' for file watching: apt-get install entr"
	find $(SRC_DIR)/ $(TEST_DIR)/ -name "*.py" | entr -c make test-quick

install-hooks: ## Install git hooks
	@echo "$(BOLD)$(BLUE)Installing git hooks...$(RESET)"
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		echo "Git hooks installed"; \
	else \
		echo "pre-commit not found. Install with: pip install pre-commit"; \
	fi

# Environment checks
check-env: ## Check development environment
	@echo "$(BOLD)$(BLUE)Checking development environment...$(RESET)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Pytest: $(shell $(PYTEST) --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Python path: $(shell $(PYTHON) -c 'import sys; print(\":\"'."".join(sys.path))')"
	@if command -v mojo >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Mojo: $(shell mojo --version)$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  Mojo: Not found (mock mode will be used)$(RESET)"; \
	fi

# Continuous Integration helpers
ci-install: ## Install CI dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-cov pytest-xdist aiohttp

ci-test: ## Run tests for CI
	$(PYTEST) --tb=short --maxfail=10 --cov=$(SRC_DIR) --cov-report=xml --timeout=$(TEST_TIMEOUT)

ci-quality: ## Run quality checks for CI
	$(BLACK) --check $(SRC_DIR)/ final_mcp_server.py
	$(FLAKE8) $(SRC_DIR)/ final_mcp_server.py --max-line-length=100
	$(MYPY) $(SRC_DIR)/ final_mcp_server.py --ignore-missing-imports

# Variables can be overridden
TEST_TIMEOUT ?= 300
TEST_WORKERS ?= auto
