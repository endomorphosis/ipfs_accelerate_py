#\!/bin/bash
# Test script for enhanced error recovery with DuckDB integration

echo "Running Enhanced Error Recovery Tests with DuckDB Integration"
echo "==========================================================="

# Run full database integration tests
echo "Running database integration tests..."
python test_error_recovery_db_integration.py

echo ""
echo "Running enhanced error recovery tests..."
python run_test_enhanced_error_recovery.py

echo ""
echo "Tests completed"

