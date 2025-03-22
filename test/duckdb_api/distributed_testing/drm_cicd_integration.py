#!/usr/bin/env python3
"""
CI/CD Integration for Dynamic Resource Management (DRM)

This module extends the CI/CD integration for the Distributed Testing Framework to specifically
support Dynamic Resource Management testing. It includes:

1. Enhanced test discovery for DRM-specific tests
2. Resource requirement analysis for DRM tests
3. Scaling scenario simulation and reporting
4. Compatibility with GitHub Actions, GitLab CI, and Jenkins

Usage examples:
    # Run DRM tests from GitHub Actions
    python -m duckdb_api.distributed_testing.drm_cicd_integration --provider github \
        --output-dir ./reports
        
    # Run specific DRM component tests
    python -m duckdb_api.distributed_testing.drm_cicd_integration --component resource_optimizer
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Local imports
try:
    from duckdb_api.distributed_testing.cicd_integration import CICDIntegration
except ImportError:
    # Handle case when running directly
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from duckdb_api.distributed_testing.cicd_integration import CICDIntegration


class DRMCICDIntegration(CICDIntegration):
    """
    Extended CI/CD integration for Dynamic Resource Management testing.
    Adds DRM-specific functionality for test discovery and reporting.
    """
    
    DRM_COMPONENTS = [
        'dynamic_resource_manager',
        'resource_performance_predictor',
        'cloud_provider_manager',
        'resource_optimizer',
        'drm_integration'
    ]
    
    def __init__(
        self, 
        coordinator_url: str,
        api_key: str,
        provider: str = 'generic',
        timeout: int = 3600,
        poll_interval: int = 15,
        verbose: bool = False,
        simulate_scaling: bool = False
    ):
        """
        Initialize the DRM CI/CD integration.
        
        Args:
            coordinator_url: URL of the coordinator server
            api_key: API key for authentication
            provider: CI/CD provider (github, gitlab, jenkins, generic)
            timeout: Maximum time to wait for test completion (seconds)
            poll_interval: How often to poll for results (seconds)
            verbose: Enable verbose logging
            simulate_scaling: Whether to run scaling scenario simulation
        """
        super().__init__(coordinator_url, api_key, provider, timeout, poll_interval, verbose)
        self.simulate_scaling = simulate_scaling
    
    def discover_drm_tests(
        self, 
        component: Optional[str] = None,
        include_e2e: bool = True
    ) -> List[str]:
        """
        Discover DRM-specific tests based on component.
        
        Args:
            component: Specific DRM component to test (or None for all)
            include_e2e: Whether to include end-to-end tests
            
        Returns:
            List of test file paths
        """
        # Get the base directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tests_dir = os.path.join(script_dir, "tests")
        
        # Initialize list of test files
        discovered_tests = []
        
        # Discover component-specific tests
        if component is None or component == 'all':
            # Add all component tests
            for comp in self.DRM_COMPONENTS:
                pattern = f"test_{comp}.py"
                test_file = os.path.join(tests_dir, pattern)
                if os.path.exists(test_file):
                    discovered_tests.append(test_file)
        else:
            # Add only the specified component
            if component in self.DRM_COMPONENTS:
                pattern = f"test_{component}.py"
                test_file = os.path.join(tests_dir, pattern)
                if os.path.exists(test_file):
                    discovered_tests.append(test_file)
            else:
                raise ValueError(f"Invalid DRM component: {component}. Valid options: {', '.join(self.DRM_COMPONENTS)}")
        
        # Add end-to-end tests if requested
        if include_e2e:
            e2e_test = os.path.join(tests_dir, "run_e2e_drm_test.py")
            if os.path.exists(e2e_test):
                discovered_tests.append(e2e_test)
        
        # Sort for deterministic ordering
        discovered_tests.sort()
        
        if self.verbose:
            print(f"Discovered {len(discovered_tests)} DRM test files")
            for test in discovered_tests:
                print(f"  - {test}")
        
        return discovered_tests
    
    def analyze_drm_test_requirements(self, test_file: str) -> Dict[str, Union[str, List[str], int]]:
        """
        Analyze a DRM test file to determine hardware and resource requirements.
        Enhanced for DRM-specific requirements.
        
        Args:
            test_file: Path to the test file
            
        Returns:
            Dictionary of test requirements
        """
        # Get base requirements from parent class
        requirements = super().analyze_test_requirements(test_file)
        
        # Set default priority higher for DRM tests
        requirements['priority'] = 7
        
        # Add DRM-specific requirements
        requirements['drm_component'] = None
        requirements['simulate_scaling'] = False
        requirements['is_e2e_test'] = False
        requirements['resource_intensive'] = False
        
        # Determine component from filename
        filename = os.path.basename(test_file)
        for component in self.DRM_COMPONENTS:
            if f"test_{component}" in filename:
                requirements['drm_component'] = component
                break
        
        # Determine if end-to-end test
        if "e2e" in filename:
            requirements['is_e2e_test'] = True
            requirements['priority'] = 8  # Higher priority for E2E tests
        
        # Read file content to determine more specific requirements
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                content = f.read()
                
                # Check for resource-intensive test
                if "TestResourceOptimizerPerformance" in content or "simulate_scaling_scenario" in content:
                    requirements['resource_intensive'] = True
                    requirements['min_memory_mb'] = 4096  # Require more memory
                
                # Check for scaling simulation
                if "simulate_scaling_scenario" in content:
                    requirements['simulate_scaling'] = True
                
                # Increase priority for resource optimizer tests
                if "ResourceOptimizer" in content:
                    requirements['priority'] = max(requirements['priority'], 8)
        
        return requirements
    
    def generate_drm_specific_report(
        self, 
        results: Dict[str, Dict], 
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate DRM-specific report with performance metrics and scaling decisions.
        
        Args:
            results: Dictionary of task results
            output_dir: Directory to write reports
            
        Returns:
            Dictionary mapping report types to file paths
        """
        if not output_dir:
            output_dir = os.getcwd()
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_files = {}
        
        # Extract DRM-specific metrics from results
        drm_metrics = {
            'resource_optimization': {
                'average_allocation_time_ms': 0,
                'allocation_success_rate': 0,
                'samples': 0
            },
            'scaling_decisions': [],
            'worker_utilization': []
        }
        
        # Extract metrics from task results
        task_count = 0
        allocation_total_time = 0
        allocation_success_count = 0
        
        for task_id, result in results.items():
            task_count += 1
            metrics = result.get('metrics', {})
            
            # Extract resource allocation metrics
            if 'allocation_time_ms' in metrics:
                allocation_total_time += metrics['allocation_time_ms']
                allocation_success_count += 1 if metrics.get('allocation_success', False) else 0
            
            # Extract scaling decisions
            if 'scaling_decisions' in metrics:
                drm_metrics['scaling_decisions'].extend(metrics['scaling_decisions'])
            
            # Extract worker utilization
            if 'worker_utilization' in metrics:
                drm_metrics['worker_utilization'].extend(metrics['worker_utilization'])
        
        # Calculate averages if data is available
        if task_count > 0:
            if allocation_total_time > 0:
                drm_metrics['resource_optimization']['average_allocation_time_ms'] = allocation_total_time / task_count
            
            if allocation_success_count > 0:
                drm_metrics['resource_optimization']['allocation_success_rate'] = allocation_success_count / task_count
            
            drm_metrics['resource_optimization']['samples'] = task_count
        
        # Sort scaling decisions by timestamp
        if drm_metrics['scaling_decisions']:
            drm_metrics['scaling_decisions'].sort(key=lambda x: x.get('timestamp', ''))
        
        # Generate DRM performance report (JSON)
        drm_report_file = os.path.join(output_dir, f"drm_performance_{timestamp}.json")
        with open(drm_report_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'provider': self.provider,
                'build_id': self.build_id,
                'repo': self.repo_name,
                'branch': self.branch,
                'commit': self.commit_sha,
                'metrics': drm_metrics,
                'tasks': task_count
            }, f, indent=2)
        
        report_files['drm_performance'] = drm_report_file
        
        # Generate DRM HTML visualization report
        if drm_metrics['scaling_decisions'] or drm_metrics['worker_utilization']:
            visualization_file = os.path.join(output_dir, f"drm_visualization_{timestamp}.html")
            self._generate_visualization_report(drm_metrics, visualization_file)
            report_files['visualization'] = visualization_file
        
        if self.verbose:
            print(f"Generated DRM-specific reports:")
            for report_type, file_path in report_files.items():
                print(f"  - {report_type}: {file_path}")
        
        return report_files
    
    def _generate_visualization_report(self, metrics: Dict, output_file: str) -> None:
        """
        Generate an HTML visualization of DRM metrics.
        
        Args:
            metrics: Dictionary of DRM metrics
            output_file: Path to output HTML file
        """
        # Generate basic HTML with embedded charts using simple JavaScript
        with open(output_file, 'w') as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write("<title>DRM Performance Visualization</title>\n")
            f.write('<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n')
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write(".chart-container { width: 800px; height: 400px; margin-bottom: 40px; }\n")
            f.write("h1, h2 { color: #333; }\n")
            f.write("</style>\n</head>\n<body>\n")
            
            # Add title
            f.write("<h1>Dynamic Resource Management Performance</h1>\n")
            
            # Resource Optimization Metrics
            f.write("<h2>Resource Optimization Performance</h2>\n")
            f.write('<div class="chart-container">\n')
            f.write('  <canvas id="optimizationChart"></canvas>\n')
            f.write('</div>\n')
            
            # Worker Utilization Chart
            if metrics.get('worker_utilization'):
                f.write("<h2>Worker Resource Utilization</h2>\n")
                f.write('<div class="chart-container">\n')
                f.write('  <canvas id="utilizationChart"></canvas>\n')
                f.write('</div>\n')
            
            # Scaling Decisions Timeline
            if metrics.get('scaling_decisions'):
                f.write("<h2>Scaling Decisions Timeline</h2>\n")
                f.write('<div class="chart-container">\n')
                f.write('  <canvas id="scalingChart"></canvas>\n')
                f.write('</div>\n')
            
            # JavaScript for charts
            f.write("<script>\n")
            
            # Resource Optimization Chart
            f.write("const optimizationCtx = document.getElementById('optimizationChart');\n")
            f.write("new Chart(optimizationCtx, {\n")
            f.write("  type: 'bar',\n")
            f.write("  data: {\n")
            f.write("    labels: ['Avg Allocation Time (ms)', 'Success Rate (%)'],\n")
            f.write("    datasets: [{\n")
            f.write("      label: 'Resource Optimization Metrics',\n")
            f.write("      data: [")
            f.write(f"{metrics['resource_optimization']['average_allocation_time_ms']:.2f}, ")
            f.write(f"{metrics['resource_optimization']['allocation_success_rate']*100:.1f}")
            f.write("],\n")
            f.write("      backgroundColor: ['rgba(54, 162, 235, 0.5)', 'rgba(75, 192, 192, 0.5)'],\n")
            f.write("      borderColor: ['rgba(54, 162, 235, 1)', 'rgba(75, 192, 192, 1)'],\n")
            f.write("      borderWidth: 1\n")
            f.write("    }]\n")
            f.write("  },\n")
            f.write("  options: {\n")
            f.write("    scales: { y: { beginAtZero: true } },\n")
            f.write("    plugins: { title: { display: true, text: 'Resource Optimization Performance' } }\n")
            f.write("  }\n")
            f.write("});\n")
            
            # Worker Utilization Chart (if data available)
            if metrics.get('worker_utilization'):
                # Process utilization data
                util_data = metrics['worker_utilization']
                timestamps = []
                cpu_util = []
                memory_util = []
                gpu_util = []
                
                for i, entry in enumerate(util_data):
                    # Use index if timestamp isn't available
                    timestamps.append(entry.get('timestamp', f"Sample {i}"))
                    cpu_util.append(entry.get('cpu_utilization', 0) * 100)
                    memory_util.append(entry.get('memory_utilization', 0) * 100)
                    gpu_util.append(entry.get('gpu_utilization', 0) * 100 if 'gpu_utilization' in entry else 0)
                
                f.write("const utilizationCtx = document.getElementById('utilizationChart');\n")
                f.write("new Chart(utilizationCtx, {\n")
                f.write("  type: 'line',\n")
                f.write("  data: {\n")
                f.write("    labels: ")
                json.dump([str(t) for t in timestamps], f)
                f.write(",\n")
                f.write("    datasets: [\n")
                f.write("      {\n")
                f.write("        label: 'CPU Utilization (%)',\n")
                f.write("        data: ")
                json.dump(cpu_util, f)
                f.write(",\n")
                f.write("        borderColor: 'rgba(54, 162, 235, 1)',\n")
                f.write("        backgroundColor: 'rgba(54, 162, 235, 0.1)',\n")
                f.write("        tension: 0.1\n")
                f.write("      },\n")
                f.write("      {\n")
                f.write("        label: 'Memory Utilization (%)',\n")
                f.write("        data: ")
                json.dump(memory_util, f)
                f.write(",\n")
                f.write("        borderColor: 'rgba(75, 192, 192, 1)',\n")
                f.write("        backgroundColor: 'rgba(75, 192, 192, 0.1)',\n")
                f.write("        tension: 0.1\n")
                f.write("      },\n")
                if any(v > 0 for v in gpu_util):
                    f.write("      {\n")
                    f.write("        label: 'GPU Utilization (%)',\n")
                    f.write("        data: ")
                    json.dump(gpu_util, f)
                    f.write(",\n")
                    f.write("        borderColor: 'rgba(255, 99, 132, 1)',\n")
                    f.write("        backgroundColor: 'rgba(255, 99, 132, 0.1)',\n")
                    f.write("        tension: 0.1\n")
                    f.write("      },\n")
                f.write("    ]\n")
                f.write("  },\n")
                f.write("  options: {\n")
                f.write("    scales: { y: { beginAtZero: true, max: 100 } },\n")
                f.write("    plugins: { title: { display: true, text: 'Worker Resource Utilization Over Time' } }\n")
                f.write("  }\n")
                f.write("});\n")
            
            # Scaling Decisions Chart (if data available)
            if metrics.get('scaling_decisions'):
                # Process scaling data
                scaling_data = metrics['scaling_decisions']
                scale_timestamps = []
                scale_values = []
                scale_colors = []
                scale_labels = []
                
                for i, decision in enumerate(scaling_data):
                    scale_timestamps.append(decision.get('timestamp', f"Decision {i}"))
                    action = decision.get('action', 'unknown')
                    count = decision.get('count', 0)
                    
                    # Normalize to a scale: positive for scale up, negative for scale down
                    if action == 'scale_up':
                        scale_values.append(count)
                        scale_colors.append('rgba(75, 192, 192, 0.8)')  # Green for scale up
                    elif action == 'scale_down':
                        scale_values.append(-count)
                        scale_colors.append('rgba(255, 99, 132, 0.8)')  # Red for scale down
                    else:
                        scale_values.append(0)
                        scale_colors.append('rgba(201, 203, 207, 0.8)')  # Gray for maintain
                    
                    scale_labels.append(f"{action}: {count}")
                
                f.write("const scalingCtx = document.getElementById('scalingChart');\n")
                f.write("new Chart(scalingCtx, {\n")
                f.write("  type: 'bar',\n")
                f.write("  data: {\n")
                f.write("    labels: ")
                json.dump([str(t) for t in scale_timestamps], f)
                f.write(",\n")
                f.write("    datasets: [{\n")
                f.write("      label: 'Scaling Decisions',\n")
                f.write("      data: ")
                json.dump(scale_values, f)
                f.write(",\n")
                f.write("      backgroundColor: ")
                json.dump(scale_colors, f)
                f.write(",\n")
                f.write("      borderColor: 'rgb(0, 0, 0)',\n")
                f.write("      borderWidth: 1\n")
                f.write("    }]\n")
                f.write("  },\n")
                f.write("  options: {\n")
                f.write("    scales: { y: { title: { display: true, text: 'Worker Count Change' } } },\n")
                f.write("    plugins: { \n")
                f.write("      title: { display: true, text: 'Scaling Decisions Timeline' },\n")
                f.write("      tooltip: {\n")
                f.write("        callbacks: {\n")
                f.write("          label: function(context) {\n")
                f.write("            return ")
                json.dump(scale_labels, f)
                f.write("[context.dataIndex];\n")
                f.write("          }\n")
                f.write("        }\n")
                f.write("      }\n")
                f.write("    }\n")
                f.write("  }\n")
                f.write("});\n")
            
            f.write("</script>\n")
            f.write("</body>\n</html>")
    
    def run_drm_tests(
        self, 
        component: Optional[str] = None,
        include_e2e: bool = True,
        output_dir: Optional[str] = None,
        report_formats: List[str] = ['json', 'md', 'html']
    ) -> int:
        """
        Run DRM-specific tests.
        
        Args:
            component: Specific DRM component to test (or None for all)
            include_e2e: Whether to include end-to-end tests
            output_dir: Directory to write reports
            report_formats: List of output formats
            
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        # 1. Discover DRM-specific tests
        discovered_tests = self.discover_drm_tests(component, include_e2e)
        if not discovered_tests:
            print("No DRM test files discovered.")
            return 1
        
        # 2. Submit tests and gather task IDs
        task_ids = []
        for test_file in discovered_tests:
            # Analyze requirements (with DRM-specific analysis)
            requirements = self.analyze_drm_test_requirements(test_file)
            
            # Submit test
            task_id = self.submit_test(test_file, requirements)
            task_ids.append(task_id)
        
        # 3. Wait for results
        results = self.wait_for_results(task_ids)
        
        # 4. Generate standard reports
        report_files = self.generate_report(results, output_dir, report_formats)
        
        # 5. Generate DRM-specific reports
        drm_report_files = self.generate_drm_specific_report(results, output_dir)
        report_files.update(drm_report_files)
        
        # 6. Report back to CI/CD system
        success = self.report_to_ci_system(results, report_files)
        
        # 7. Return appropriate exit code
        return 0 if success else 1


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='DRM CI/CD Integration for Distributed Testing Framework')
    
    # Coordinator connection options
    parser.add_argument('--coordinator', default='http://localhost:8080', help='Coordinator URL')
    parser.add_argument('--api-key', required=True, help='API key for authentication')
    
    # CI/CD provider options
    parser.add_argument('--provider', default='generic', 
                        choices=['github', 'gitlab', 'jenkins', 'generic'],
                        help='CI/CD provider')
    
    # DRM component options
    parser.add_argument('--component', choices=['dynamic_resource_manager', 'resource_performance_predictor', 
                                               'cloud_provider_manager', 'resource_optimizer', 
                                               'drm_integration', 'all'],
                        default='all',
                        help='Specific DRM component to test')
    
    # Test and report options
    parser.add_argument('--no-e2e', action='store_true', help='Exclude end-to-end tests')
    parser.add_argument('--simulate-scaling', action='store_true', help='Simulate scaling scenarios')
    parser.add_argument('--output-dir', help='Directory to write reports')
    parser.add_argument('--report-formats', nargs='+', default=['json', 'md', 'html'],
                        choices=['json', 'md', 'html'], 
                        help='Report formats to generate')
    
    # Execution options
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Maximum time to wait for test completion (seconds)')
    parser.add_argument('--poll-interval', type=int, default=15,
                        help='How often to poll for results (seconds)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize DRM CI/CD integration
    integration = DRMCICDIntegration(
        coordinator_url=args.coordinator,
        api_key=args.api_key,
        provider=args.provider,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        verbose=args.verbose,
        simulate_scaling=args.simulate_scaling
    )
    
    # Run DRM tests
    exit_code = integration.run_drm_tests(
        component=args.component,
        include_e2e=not args.no_e2e,
        output_dir=args.output_dir,
        report_formats=args.report_formats
    )
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()