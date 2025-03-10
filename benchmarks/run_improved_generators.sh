#!/bin/bash
# Phase 16 Improvements - Setup and Run Script
# This script sets up the improved generators and runs them to generate
# tests and skillsets with proper hardware detection and database integration.

set -e  # Exit immediately if a command exits with a non-zero status

# Create improvements directory if it doesn't exist
mkdir -p improvements

# Set environment variables
export BENCHMARK_DB_PATH="./benchmark_db.duckdb"
export DEPRECATE_JSON_OUTPUT=1

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Phase 16 improvements...${NC}"

# Create directories
mkdir -p generated_tests
mkdir -p generated_skillsets

# First, check if the improved versions already exist
if [ ! -f "improvements/improved_hardware_detection.py" ]; then
    echo -e "${YELLOW}Creating improved hardware detection module...${NC}"
    cp -f improvements/improved_hardware_detection.py.tmp improvements/improved_hardware_detection.py 2>/dev/null || \
    echo "File not found, please create it first"
fi

if [ ! -f "improvements/database_integration.py" ]; then
    echo -e "${YELLOW}Creating database integration module...${NC}"
    cp -f improvements/database_integration.py.tmp improvements/database_integration.py 2>/dev/null || \
    echo "File not found, please create it first"
fi

if [ ! -f "improvements/improved_merged_test_generator.py" ]; then
    echo -e "${YELLOW}Creating improved test generator...${NC}"
    cp -f improvements/improved_merged_test_generator.py.tmp improvements/improved_merged_test_generator.py 2>/dev/null || \
    echo "File not found, please create it first"
fi

if [ ! -f "improvements/improved_skillset_generator.py" ]; then
    echo -e "${YELLOW}Creating improved skillset generator...${NC}"
    cp -f improvements/improved_skillset_generator.py.tmp improvements/improved_skillset_generator.py 2>/dev/null || \
    echo "File not found, please create it first"
fi

# Create __init__.py in improvements directory
if [ ! -f "improvements/__init__.py" ]; then
    echo -e "${YELLOW}Creating __init__.py in improvements directory...${NC}"
    echo '"""Improved modules for Phase 16."""' > improvements/__init__.py
fi

# Run the improved test generator for key models
echo -e "${GREEN}Running improved test generator for key models...${NC}"
python improvements/improved_merged_test_generator.py --batch-generate bert,t5,clip,vit,whisper,wav2vec2,llava --cross-platform --output-dir ./generated_tests

# Run the improved skillset generator for key models
echo -e "${GREEN}Running improved skillset generator for key models...${NC}"
python improvements/improved_skillset_generator.py --batch-generate bert,t5,clip,vit,whisper,wav2vec2,llava --cross-platform --output-dir ./generated_skillsets

# Run a sample test to verify it works
echo -e "${GREEN}Testing a generated test file...${NC}"
if [ -f "generated_tests/test_hf_bert.py" ]; then
    python generated_tests/test_hf_bert.py -v
else
    echo "Test file not found. Generation may have failed."
fi

# Run database queries to verify data was stored properly
echo -e "${GREEN}Checking database content...${NC}"
if command -v duckdb &> /dev/null && [ -f "./benchmark_db.duckdb" ]; then
    echo "Database records count:"
    echo -e "SELECT COUNT(*) FROM test_runs;" | duckdb ./benchmark_db.duckdb
    echo -e "SELECT COUNT(*) FROM models;" | duckdb ./benchmark_db.duckdb
    echo -e "SELECT COUNT(*) FROM model_implementations;" | duckdb ./benchmark_db.duckdb
else
    echo "DuckDB or database file not found. Cannot verify database content."
fi

echo -e "${GREEN}All done!${NC}"
echo "Generated files are in 'generated_tests' and 'generated_skillsets' directories."
echo "If you encounter any issues, check the error logs and fix the underlying generator code."