#!/bin/bash
# Run tests for the Integrated Analysis System

set -e

echo "===== Running Integrated Analysis System Tests ====="
echo

# Create test directories if they don't exist
mkdir -p test_reports
mkdir -p test_visualizations

# Run the unit tests
python3 -m unittest tests/test_integrated_analysis_system.py

# If the tests pass, run the example
if [ $? -eq 0 ]; then
    echo
    echo "===== Running Integrated Analysis System Example ====="
    echo
    
    # Create example directories if they don't exist
    mkdir -p reports
    mkdir -p visualizations
    
    # Run the example with different options
    echo "Running basic example..."
    python3 examples/result_aggregator_example.py --cleanup
    
    echo
    echo "Running example without visualization..."
    python3 examples/result_aggregator_example.py --no-visualization --cleanup
    
    echo
    echo "Running example without ML..."
    python3 examples/result_aggregator_example.py --no-ml --cleanup
    
    echo
    echo "All tests and examples completed successfully!"
else
    echo
    echo "Tests failed, skipping example."
    exit 1
fi

# Clean up test artifacts
rm -rf test_reports
rm -rf test_visualizations
rm -rf reports
rm -rf visualizations

echo
echo "===== Cleaning up test artifacts ====="
echo "Done!"