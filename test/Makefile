# Makefile for IPFS Accelerate Python Testing Framework

.PHONY: all test clean dashboard viz dashboard-deps test-dashboard

# Default target
all: test

# Install dependencies
deps:
	pip install -r requirements.txt

# Install dashboard dependencies
dashboard-deps:
	pip install -r requirements_dashboard.txt

# Run tests
test:
	pytest -xvs

# Run specific tests
test-api:
	pytest -xvs api/test_api_endpoints.py

# Run dashboard tests
test-dashboard:
	pytest -xvs duckdb_api/distributed_testing/tests/test_drm_real_time_dashboard.py

# Run the DRM dashboard with default settings
dashboard:
	python run_drm_real_time_dashboard.py --browser

# Run the DRM dashboard with debug mode
dashboard-debug:
	python run_drm_real_time_dashboard.py --debug --browser

# Run DRM dashboard with simulation mode
dashboard-simulate:
	python run_drm_real_time_dashboard.py --browser --update-interval 2

# Run visualization tools
viz:
	python run_enhanced_visualization_dashboard.py --browser

# Clean up
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	find . -name "*.log" -delete
	find . -name "*.png" -delete
	find . -name "*.html" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

# Help
help:
	@echo "Available targets:"
	@echo "  all            - Run tests (default)"
	@echo "  deps           - Install dependencies"
	@echo "  dashboard-deps - Install dashboard dependencies"
	@echo "  test           - Run all tests"
	@echo "  test-api       - Run API tests"
	@echo "  test-dashboard - Run dashboard tests"
	@echo "  dashboard      - Run the DRM real-time dashboard"
	@echo "  dashboard-debug - Run the DRM dashboard in debug mode"
	@echo "  dashboard-simulate - Run the DRM dashboard with simulation mode"
	@echo "  viz            - Run enhanced visualization dashboard"
	@echo "  clean          - Clean up temporary files"
	@echo "  help           - Show this help message"