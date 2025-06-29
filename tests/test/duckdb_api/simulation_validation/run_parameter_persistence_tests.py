#!/usr/bin/env python3
"""
Run tests for the Parameter Persistence functionality in the Database Predictive Analytics module.
"""

import os
import sys
import unittest
import argparse
import datetime
import json

# Add parent directories to path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def run_tests(args):
    """Run the parameter persistence tests."""
    import unittest
    from test.test_database_predictive_analytics import TestDatabasePredictiveAnalytics
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add standalone tests for parameter persistence
    if args.standalone:
        # Add all tests from the TestDatabasePredictiveAnalytics class
        tests = unittest.defaultTestLoader.loadTestsFromTestCase(TestDatabasePredictiveAnalytics)
        suite.addTest(tests)
        
    # Add the parameter persistence test from the comprehensive E2E test suite
    if args.integrated:
        from test.test_comprehensive_e2e import TestComprehensiveEndToEnd
        suite.addTest(TestComprehensiveEndToEnd("test_25_database_predictive_analytics_parameter_persistence"))
    
    # Run tests with the specified runner
    if args.html_report and args.output_dir:
        try:
            import HtmlTestRunner
            runner = HtmlTestRunner.HTMLTestRunner(
                output=args.output_dir,
                report_name=f"parameter_persistence_test_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                combine_reports=True,
                report_title="Parameter Persistence Test Report"
            )
        except ImportError:
            print("HtmlTestRunner not installed. Using default test runner.")
            runner = unittest.TextTestRunner(verbosity=2)
    else:
        runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    result = runner.run(suite)
    
    # Generate performance report if requested
    if args.performance_report and args.output_dir:
        from test.test_database_predictive_analytics import TestDatabasePredictiveAnalytics
        from test.test_comprehensive_e2e import TestComprehensiveEndToEnd
        
        # Create a performance test
        performance_test = TestDatabasePredictiveAnalytics()
        performance_test.setUp()
        
        # Run performance tests
        model_types = ['arima', 'exponential_smoothing', 'linear_regression']
        performance_data = {}
        
        for model_type in model_types:
            # Run with parameter persistence enabled
            config_enabled = {
                'parameter_persistence': {
                    'enabled': True,
                    'storage_path': performance_test.temp_dir,
                    'format': 'json',
                    'max_age_days': 30,
                    'revalidate_after_days': 7,
                    'cache_in_memory': True
                }
            }
            performance_test.analyzer.config = config_enabled
            
            # First run - should tune parameters
            import time
            start_time = time.time()
            result1 = performance_test.analyzer.forecast_time_series(
                performance_test.test_df,
                metric_name='test_metric',
                model_type=model_type,
                forecast_days=7
            )
            first_run_time = time.time() - start_time
            
            # Second run - should use saved parameters
            start_time = time.time()
            result2 = performance_test.analyzer.forecast_time_series(
                performance_test.test_df,
                metric_name='test_metric',
                model_type=model_type,
                forecast_days=7
            )
            second_run_time = time.time() - start_time
            
            # Run with parameter persistence disabled
            config_disabled = {
                'parameter_persistence': {
                    'enabled': False
                }
            }
            performance_test.analyzer.config = config_disabled
            
            # Run without parameter persistence
            start_time = time.time()
            result3 = performance_test.analyzer.forecast_time_series(
                performance_test.test_df,
                metric_name='test_metric',
                model_type=model_type,
                forecast_days=7
            )
            disabled_time = time.time() - start_time
            
            # Record performance data
            performance_data[model_type] = {
                'first_run_time': first_run_time,
                'second_run_time': second_run_time,
                'disabled_time': disabled_time,
                'speedup': first_run_time / second_run_time if second_run_time > 0 else 0,
                'overhead_time': first_run_time - disabled_time
            }
        
        # Clean up
        performance_test.tearDown()
        
        # Create a detailed performance report
        performance_report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'model_types': model_types,
            'performance_data': performance_data,
            'summary': {
                'average_speedup': sum([performance_data[model_type]['speedup'] for model_type in model_types]) / len(model_types),
                'max_speedup': max([performance_data[model_type]['speedup'] for model_type in model_types]),
                'min_speedup': min([performance_data[model_type]['speedup'] for model_type in model_types]),
                'average_overhead': sum([performance_data[model_type]['overhead_time'] for model_type in model_types]) / len(model_types)
            }
        }
        
        # Save report to file
        report_path = os.path.join(args.output_dir, f"parameter_persistence_performance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        # Also generate a Markdown report
        md_report_path = os.path.join(args.output_dir, f"parameter_persistence_performance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(md_report_path, 'w') as f:
            f.write(f"# Parameter Persistence Performance Report\n\n")
            f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Average Speedup: {performance_report['summary']['average_speedup']:.2f}x\n")
            f.write(f"- Maximum Speedup: {performance_report['summary']['max_speedup']:.2f}x\n")
            f.write(f"- Minimum Speedup: {performance_report['summary']['min_speedup']:.2f}x\n")
            f.write(f"- Average Overhead (first run): {performance_report['summary']['average_overhead']:.2f} seconds\n\n")
            
            f.write("## Detailed Performance\n\n")
            f.write("| Model Type | First Run (s) | Second Run (s) | Without Persistence (s) | Speedup | Overhead (s) |\n")
            f.write("|------------|--------------|---------------|-------------------------|---------|-------------|\n")
            
            for model_type in model_types:
                data = performance_data[model_type]
                f.write(f"| {model_type} | {data['first_run_time']:.2f} | {data['second_run_time']:.2f} | ")
                f.write(f"{data['disabled_time']:.2f} | {data['speedup']:.2f}x | {data['overhead_time']:.2f} |\n")
        
        print(f"\nPerformance report generated: {report_path}")
        print(f"Markdown report generated: {md_report_path}")
    
    return result.wasSuccessful()

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run tests for the Parameter Persistence functionality.")
    parser.add_argument("--standalone", action="store_true", help="Run standalone parameter persistence tests")
    parser.add_argument("--integrated", action="store_true", help="Run the parameter persistence test from the comprehensive E2E test suite")
    parser.add_argument("--performance-report", action="store_true", help="Generate detailed performance report")
    parser.add_argument("--html-report", action="store_true", help="Generate HTML test report")
    parser.add_argument("--output-dir", type=str, default="test_output", help="Directory to save test reports")
    
    args = parser.parse_args()
    
    # Default to running both test types if none specified
    if not (args.standalone or args.integrated):
        args.standalone = True
        args.integrated = True
    
    print(f"\nRunning Parameter Persistence Tests:")
    print(f"- Standalone Tests: {'Yes' if args.standalone else 'No'}")
    print(f"- Integrated Tests: {'Yes' if args.integrated else 'No'}")
    print(f"- HTML Report: {'Yes' if args.html_report else 'No'}")
    print(f"- Performance Report: {'Yes' if args.performance_report else 'No'}")
    print(f"- Output Directory: {args.output_dir}")
    print()
    
    success = run_tests(args)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()