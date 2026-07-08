#!/usr/bin/env python
"""
Test script for the fixed benchmark database query tool.

This script runs basic tests on the fixed_benchmark_db_query.py script to verify
that it works correctly with the benchmark database.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_db_query')

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

def run_test(test_name, command, expected_text=None):
    """Run a test command and check if the expected text is in the output."""
    logger.info(f"Running test: {test_name}")
    logger.info(f"Command: {command}")
    
    try:
        import subprocess
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        output = result.stdout
        
        logger.info(f"Test completed successfully")
        
        if expected_text and expected_text not in output:
            logger.warning(f"Expected text not found in output: {expected_text}")
            logger.warning(f"Output: {output[:200]}...")
            return False
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Test failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error running test: {e}")
        return False

def main():
    """Run tests on the fixed benchmark database query tool."""
    parser = argparse.ArgumentParser(description='Test benchmark database query tool')
    parser.add_argument('--db', type=str, default='./benchmark_db.duckdb',
                       help='Path to the benchmark database')
    parser.add_argument('--run-all', action='store_true',
                       help='Run all tests')
    parser.add_argument('--run-sql', action='store_true',
                       help='Test SQL query functionality')
    parser.add_argument('--run-model', action='store_true',
                       help='Test model-specific query functionality')
    parser.add_argument('--run-report', action='store_true',
                       help='Test report generation functionality')
    args = parser.parse_args()
    
    # If no specific tests are requested, run all tests
    run_all = args.run_all or not (args.run_sql or args.run_model or args.run_report)
    
    # Database path
    db_path = args.db
    
    # Check if the database exists
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return False
    
    # Check if the fixed_benchmark_db_query.py script exists
    script_path = Path(__file__).parent / 'fixed_benchmark_db_query.py'
    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        return False
    
    # List of test commands
    test_commands = []
    
    # SQL query tests
    if run_all or args.run_sql:
        test_commands.append({
            'name': 'Basic SQL Query',
            'command': f'python {script_path} --db {db_path} --sql "SELECT COUNT(*) FROM models"',
            'expected': None
        })
        
        test_commands.append({
            'name': 'SQL Query with Join',
            'command': f'python {script_path} --db {db_path} --sql "SELECT m.model_name, COUNT(*) as test_count FROM performance_results pr JOIN models m ON pr.model_id = m.model_id GROUP BY m.model_name"',
            'expected': None
        })
    
    # Model-specific tests
    if run_all or args.run_model:
        test_commands.append({
            'name': 'Model Query',
            'command': f'python {script_path} --db {db_path} --model bert',
            'expected': 'bert'
        })
        
        test_commands.append({
            'name': 'Model Query with Metric',
            'command': f'python {script_path} --db {db_path} --model bert --metric throughput --compare-hardware',
            'expected': None
        })
        
        test_commands.append({
            'name': 'Hardware Query',
            'command': f'python {script_path} --db {db_path} --hardware cuda',
            'expected': 'cuda'
        })
    
    # Report tests
    if run_all or args.run_report:
        test_commands.append({
            'name': 'Performance Report',
            'command': f'python {script_path} --db {db_path} --report performance',
            'expected': None
        })
        
        test_commands.append({
            'name': 'Summary Report',
            'command': f'python {script_path} --db {db_path} --report summary',
            'expected': None
        })
        
        # Create a temporary output file for the chart test
        chart_output = '/tmp/benchmark_chart.png'
        test_commands.append({
            'name': 'Chart Output',
            'command': f'python {script_path} --db {db_path} --model bert --metric throughput --compare-hardware --format chart --output {chart_output}',
            'expected': None
        })
    
    # Run tests
    success_count = 0
    fail_count = 0
    
    for test in test_commands:
        if run_test(test['name'], test['command'], test['expected']):
            success_count += 1
        else:
            fail_count += 1
    
    # Report results
    logger.info(f"Tests completed: {success_count} passed, {fail_count} failed")
    
    return fail_count == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)