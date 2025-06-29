# Makefile for HuggingFace Model Generator

# Default paths
ROADMAP_PATH ?= /home/barberb/ipfs_accelerate_py/test/skills/HF_MODEL_COVERAGE_ROADMAP.md
TESTS_DIR ?= /home/barberb/ipfs_accelerate_py/test/skills/fixed_tests
OUTPUT_DIR ?= generated_tests
REPORTS_DIR ?= reports
BATCH_SIZE ?= 10
START_INDEX ?= 0
PRIORITY ?= high
ARCH ?=
QUERY ?=
LIMIT ?= 5
MODEL_TYPE ?=
FORMATS ?= markdown html

.PHONY: all report generate-high generate-medium generate-arch batch advanced validate \
        export-matrix advanced-batch huggingface clean help test test-unit test-integration test-all

all: report generate-high

# Generate coverage report
report:
	@echo "Generating coverage report..."
	@mkdir -p $(REPORTS_DIR)
	python scripts/coverage_report.py --roadmap $(ROADMAP_PATH) --tests-dir $(TESTS_DIR) --output-dir $(REPORTS_DIR)
	@echo "Report generated in $(REPORTS_DIR)"
	@ls -la $(REPORTS_DIR) | grep coverage_report | tail -1

# Generate high priority models
generate-high: report
	@echo "Generating high priority missing models..."
	@mkdir -p $(OUTPUT_DIR)
	python scripts/generate_missing_models.py --output-dir $(OUTPUT_DIR)
	@echo "High priority models generated in $(OUTPUT_DIR)"
	@ls -la $(OUTPUT_DIR) | tail -10

# Generate medium priority models
generate-medium: report
	@echo "Generating medium priority missing models..."
	@mkdir -p $(OUTPUT_DIR)
	python scripts/batch_generate.py \
		--report $(shell ls -t $(REPORTS_DIR)/coverage_report_*.json | head -1) \
		--output-dir $(OUTPUT_DIR) \
		--priority medium \
		--batch-size $(BATCH_SIZE) \
		--start-index $(START_INDEX)
	@echo "Medium priority models generated in $(OUTPUT_DIR)"
	@ls -la $(OUTPUT_DIR) | tail -10

# Generate specific architecture models
generate-arch:
	@if [ -z "$(ARCH)" ]; then \
		echo "Error: Architecture not specified. Use ARCH=<architecture>"; \
		exit 1; \
	fi
	@echo "Generating models for architecture: $(ARCH)..."
	@mkdir -p $(OUTPUT_DIR)
	python scripts/batch_generate.py \
		--report $(shell ls -t $(REPORTS_DIR)/coverage_report_*.json | head -1) \
		--output-dir $(OUTPUT_DIR) \
		--architecture $(ARCH) \
		--batch-size $(BATCH_SIZE) \
		--start-index $(START_INDEX)
	@echo "Architecture models generated in $(OUTPUT_DIR)"
	@ls -la $(OUTPUT_DIR) | tail -10

# Generate models in batches
batch:
	@if [ -z "$(REPORT)" ]; then \
		REPORT=$$(ls -t $(REPORTS_DIR)/coverage_report_*.json | head -1); \
		echo "Using most recent report: $$REPORT"; \
	else \
		REPORT=$(REPORT); \
	fi; \
	echo "Generating batch of models..."; \
	mkdir -p $(OUTPUT_DIR); \
	python scripts/batch_generate.py \
		--report $$REPORT \
		--output-dir $(OUTPUT_DIR) \
		--priority $(PRIORITY) \
		--batch-size $(BATCH_SIZE) \
		--start-index $(START_INDEX); \
	echo "Batch generated in $(OUTPUT_DIR)"

# Use advanced generator
advanced:
	@mkdir -p $(OUTPUT_DIR)
	python scripts/advanced_generator.py \
		--report $(shell ls -t $(REPORTS_DIR)/coverage_report_*.json | head -1) \
		--output-dir $(OUTPUT_DIR) \
		--priority $(PRIORITY) \
		--batch-size $(BATCH_SIZE) \
		--start-index $(START_INDEX) \
		$(if $(ARCH),--architecture $(ARCH),)
	@echo "Models generated with advanced generator in $(OUTPUT_DIR)"
	@ls -la $(OUTPUT_DIR) | tail -10

# Generate from HuggingFace API
huggingface:
	@if [ -z "$(MODEL_TYPE)" ]; then \
		echo "Error: Model type not specified. Use MODEL_TYPE=<model_type>"; \
		exit 1; \
	fi
	@echo "Generating models for $(MODEL_TYPE) from HuggingFace API..."
	@mkdir -p $(OUTPUT_DIR)
	python scripts/advanced_generator.py \
		--huggingface $(MODEL_TYPE) \
		--output-dir $(OUTPUT_DIR) \
		$(if $(QUERY),--query $(QUERY),) \
		--limit $(LIMIT)
	@echo "HuggingFace models generated in $(OUTPUT_DIR)"
	@ls -la $(OUTPUT_DIR) | tail -10

# Export coverage matrix
export-matrix:
	@echo "Exporting coverage matrix..."
	@mkdir -p $(REPORTS_DIR)
	python scripts/export_coverage_matrix.py \
		--report $(shell ls -t $(REPORTS_DIR)/coverage_report_*.json | head -1) \
		--output-dir $(REPORTS_DIR) \
		--formats $(FORMATS)
	@echo "Coverage matrix exported to $(REPORTS_DIR)"
	@ls -la $(REPORTS_DIR) | grep coverage_matrix | tail -3

# Validate generated models
validate:
	@echo "Validating generated models..."
	@if [ -z "$(FILE)" ]; then \
		python scripts/validate_models.py --dir $(OUTPUT_DIR) $(if $(REPORT),--report $(REPORT),); \
	else \
		python scripts/validate_models.py --file $(FILE) $(if $(REPORT),--report $(REPORT),); \
	fi

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -rf $(OUTPUT_DIR)/*
	rm -rf $(REPORTS_DIR)/*

# Run unit tests
test-unit:
	@echo "Running unit tests..."
	python tests/run_all_tests.py --unit $(ARGS)

# Run integration tests
test-integration:
	@echo "Running integration tests..."
	python tests/run_all_tests.py --integration $(ARGS)

# Run all tests
test-all:
	@echo "Running all tests..."
	python tests/run_all_tests.py $(ARGS)

# Alias for running all tests
test: test-all

# Help
help:
	@echo "HuggingFace Model Generator Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make report                        Generate coverage report"
	@echo "  make generate-high                 Generate high priority missing models"
	@echo "  make generate-medium               Generate medium priority missing models"
	@echo "  make generate-arch ARCH=<arch>     Generate models for a specific architecture"
	@echo "  make batch                         Generate a batch of models from the latest report"
	@echo "  make advanced                      Use advanced generator with auto-fixing"
	@echo "  make huggingface MODEL_TYPE=<type> Generate models from HuggingFace API"
	@echo "  make export-matrix                 Export coverage matrix in various formats"
	@echo "  make validate                      Validate generated models"
	@echo "  make clean                         Clean up generated files"
	@echo ""
	@echo "Testing:"
	@echo "  make test                          Run all tests"
	@echo "  make test-unit                     Run unit tests only"
	@echo "  make test-integration              Run integration tests only"
	@echo "  make test-all                      Run all tests (same as 'make test')"
	@echo "  make test ARGS=\"-v --pattern=hardware\"  Run specific tests with arguments"
	@echo ""
	@echo "Environment Variables:"
	@echo "  ROADMAP_PATH     Path to HF_MODEL_COVERAGE_ROADMAP.md"
	@echo "  TESTS_DIR        Directory containing test files"
	@echo "  OUTPUT_DIR       Directory to output generated files"
	@echo "  REPORTS_DIR      Directory to save reports"
	@echo "  BATCH_SIZE       Number of models to generate in a batch"
	@echo "  START_INDEX      Index to start from"
	@echo "  PRIORITY         Priority level to generate (high, medium)"
	@echo "  REPORT           Path to coverage report JSON file"
	@echo "  ARCH             Architecture to generate models for"
	@echo "  MODEL_TYPE       Model type for HuggingFace search"
	@echo "  QUERY            Search query for HuggingFace"
	@echo "  LIMIT            Maximum results from HuggingFace"
	@echo "  FORMATS          Output formats for coverage matrix (csv, markdown, html, json)"
	@echo "  FILE             Specific file to validate"
	@echo "  ARGS             Additional arguments for test commands"