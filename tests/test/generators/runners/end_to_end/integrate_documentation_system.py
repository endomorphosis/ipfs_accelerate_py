#!/usr/bin/env python3
"""
Documentation System Integration

This script integrates the enhanced documentation system with the Integrated Component Test Runner,
ensuring seamless generation of comprehensive documentation for all model and hardware combinations.
It handles:

1. Integration of enhanced documentation templates with the existing template database
2. Patching the model documentation generator to fix variable substitution issues
3. Modifying the integrated component test runner to use the enhanced documentation system
4. Adding tests to verify the integration works correctly
5. Implementing visualization capabilities for benchmark results in documentation
"""

import os
import sys
import logging
import shutil
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Import project modules
from model_documentation_generator import ModelDocGenerator
from integrated_component_test_runner import IntegratedComponentTester, EnhancedModelDocGenerator
from doc_template_fixer import monkey_patch_model_doc_generator, monkey_patch_template_renderer
from enhance_documentation_templates import create_documentation_template, add_enhanced_documentation_templates

# Constants
TEMPLATE_DB_PATH = os.path.join(script_dir, "template_database.duckdb")
OUTPUT_DIR = os.path.join(script_dir, "test_output")


def integrate_enhanced_doc_generator():
    """
    Integrate the enhanced documentation generator with the Integrated Component Test Runner.
    This modifies how documentation is generated in the test runner.
    """
    logger.info("Integrating enhanced documentation generator with test runner")
    
    # Apply the monkey patches to fix variable substitution issues
    monkey_patch_model_doc_generator()
    monkey_patch_template_renderer()
    
    # Enhance the EnhancedModelDocGenerator in the integrated component test runner
    original_generate_markdown = EnhancedModelDocGenerator.generate_markdown
    
    def enhanced_generate_markdown(self, test_results=None, benchmark_results=None, git_hash=None):
        """
        Enhanced version of generate_markdown that adds visualization capabilities for benchmark results.
        
        Args:
            test_results: Test results dictionary (optional)
            benchmark_results: Benchmark results dictionary (optional)
            git_hash: Git hash of the current commit (optional)
            
        Returns:
            Generated markdown documentation as string
        """
        # Call the original method to generate the base markdown
        base_markdown = original_generate_markdown(self, test_results, benchmark_results, git_hash)
        
        # Enhance the benchmark results section with visualizations if benchmark_results are available
        if benchmark_results and isinstance(benchmark_results, dict) and 'results_by_batch' in benchmark_results:
            # Create a new section for visualizations
            visualization_section = self._create_benchmark_visualizations(benchmark_results)
            
            # Insert the visualization section before the conclusion
            if "## Known Limitations" in base_markdown:
                base_markdown = base_markdown.replace(
                    "## Known Limitations", 
                    f"{visualization_section}\n\n## Known Limitations"
                )
            else:
                base_markdown += f"\n\n{visualization_section}"
        
        return base_markdown
    
    # Add a new method to create benchmark visualizations
    def _create_benchmark_visualizations(self, benchmark_results):
        """
        Create visualizations for benchmark results.
        
        Args:
            benchmark_results: Benchmark results dictionary
            
        Returns:
            Markdown string with visualization section
        """
        visualization_md = "## Performance Visualizations\n\n"
        
        # Extract batch results
        batch_results = benchmark_results.get('results_by_batch', {})
        
        if not batch_results:
            return visualization_md + "No benchmark data available for visualization."
        
        # Create ASCII charts for latency and throughput
        visualization_md += "### Latency by Batch Size\n\n"
        visualization_md += "```\n"
        # Get batch sizes and latencies
        batch_sizes = sorted([int(b) for b in batch_results.keys()])
        latencies = [batch_results[str(b)].get('average_latency_ms', 0) for b in batch_sizes]
        
        # Find the max latency for scaling
        max_latency = max(latencies) if latencies else 0
        chart_width = 40  # Width of the ASCII chart
        
        # Create a simple ASCII bar chart
        for i, batch_size in enumerate(batch_sizes):
            latency = latencies[i]
            bar_length = int((latency / max_latency) * chart_width) if max_latency > 0 else 0
            visualization_md += f"Batch {batch_size:2d}: {'█' * bar_length} {latency:.2f}ms\n"
        
        visualization_md += "```\n\n"
        
        # Create throughput chart
        visualization_md += "### Throughput by Batch Size\n\n"
        visualization_md += "```\n"
        # Get throughputs
        throughputs = [batch_results[str(b)].get('average_throughput_items_per_second', 0) for b in batch_sizes]
        
        # Find the max throughput for scaling
        max_throughput = max(throughputs) if throughputs else 0
        
        # Create a simple ASCII bar chart
        for i, batch_size in enumerate(batch_sizes):
            throughput = throughputs[i]
            bar_length = int((throughput / max_throughput) * chart_width) if max_throughput > 0 else 0
            visualization_md += f"Batch {batch_size:2d}: {'█' * bar_length} {throughput:.2f} items/s\n"
        
        visualization_md += "```\n\n"
        
        # Add interpretation of results
        visualization_md += "### Performance Analysis\n\n"
        
        # Find optimal batch size for throughput
        if throughputs:
            optimal_index = throughputs.index(max(throughputs))
            optimal_batch = batch_sizes[optimal_index]
            visualization_md += f"- **Optimal Batch Size**: {optimal_batch} (for maximum throughput)\n"
            
            # Check if latency increases with batch size
            if len(batch_sizes) > 1 and latencies[0] < latencies[-1]:
                latency_increase = (latencies[-1] / latencies[0]) if latencies[0] > 0 else 0
                visualization_md += f"- **Latency Scaling**: {latency_increase:.2f}x increase from batch 1 to batch {batch_sizes[-1]}\n"
            
            # Check throughput scaling
            if len(batch_sizes) > 1 and throughputs[0] > 0:
                throughput_scaling = throughputs[-1] / throughputs[0]
                visualization_md += f"- **Throughput Scaling**: {throughput_scaling:.2f}x from batch 1 to batch {batch_sizes[-1]}\n"
            
            # Hardware-specific analysis
            visualization_md += f"- **{self.hardware.upper()} Performance**: "
            if self.hardware in ["cuda", "rocm", "webgpu"]:
                visualization_md += "GPU acceleration shows best performance with larger batch sizes.\n"
            elif self.hardware == "cpu":
                visualization_md += "CPU processing shows balanced performance across batch sizes.\n"
            else:
                visualization_md += f"This hardware platform shows specific performance characteristics for {self.model_name}.\n"
        
        return visualization_md
    
    # Add the new methods to the EnhancedModelDocGenerator class
    EnhancedModelDocGenerator.generate_markdown = enhanced_generate_markdown
    EnhancedModelDocGenerator._create_benchmark_visualizations = _create_benchmark_visualizations
    
    logger.info("Enhanced documentation generator successfully integrated with test runner")


def modify_integrated_component_tester():
    """
    Modify the IntegratedComponentTester class to use the enhanced documentation system.
    """
    logger.info("Modifying IntegratedComponentTester to use enhanced documentation system")
    
    # Store the original method for reference
    original_run_test_with_docs = IntegratedComponentTester.run_test_with_docs
    
    def enhanced_run_test_with_docs(self, temp_dir: str, test_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced version of run_test_with_docs that uses our improved documentation system.
        
        Args:
            temp_dir: Directory containing generated test files
            test_results: Test results to include in documentation (optional)
            
        Returns:
            Dictionary with results including documentation path
        """
        logger.info(f"Running enhanced documentation generation for {self.model_name} on {self.hardware}")
        
        # Get file paths
        model_name_safe = self.model_name.replace('/', '_')
        skill_file = os.path.join(temp_dir, f"{model_name_safe}_{self.hardware}_skill.py")
        test_file = os.path.join(temp_dir, f"test_{model_name_safe}_{self.hardware}.py")
        benchmark_file = os.path.join(temp_dir, f"benchmark_{model_name_safe}_{self.hardware}.py")
        
        # Check if files exist
        if not all(os.path.exists(f) for f in [skill_file, test_file, benchmark_file]):
            logger.warning("Missing required component files. Cannot generate documentation.")
            return {"success": False, "documentation_path": None}
        
        # Run the benchmark to get performance data
        benchmark_results = self.run_benchmark(temp_dir)
        
        # Create documentation directory if it doesn't exist
        docs_output_dir = os.path.join(self.output_dir, "enhanced_docs_test", self.model_name.replace('/', '_'))
        os.makedirs(docs_output_dir, exist_ok=True)
        
        try:
            # Create documentation generator
            doc_generator = EnhancedModelDocGenerator(
                model_name=self.model_name,
                hardware=self.hardware,
                skill_path=skill_file,
                test_path=test_file,
                benchmark_path=benchmark_file,
                expected_results_path=None
            )
            
            # Generate markdown documentation
            md_content = doc_generator.generate_markdown(
                test_results=test_results, 
                benchmark_results=benchmark_results.get('benchmark_results'),
                git_hash=self.git_hash
            )
            
            # Write documentation to file
            doc_file = os.path.join(docs_output_dir, f"{model_name_safe}_{self.hardware}_docs.md")
            with open(doc_file, 'w') as f:
                f.write(md_content)
                
            logger.info(f"Documentation generated: {doc_file}")
            return {"success": True, "documentation_path": doc_file}
            
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            return {"success": False, "documentation_path": None, "error": str(e)}
    
    # Replace the original method
    IntegratedComponentTester.run_test_with_docs = enhanced_run_test_with_docs
    logger.info("IntegratedComponentTester successfully modified")


def test_integration():
    """
    Test the integration to ensure it works correctly.
    """
    logger.info("Testing integration with a sample test run")
    
    # Create a temporary test directory
    test_dir = os.path.join(OUTPUT_DIR, "integration_test_" + os.urandom(4).hex())
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Create a test instance
        tester = IntegratedComponentTester(
            model_name="bert-base-uncased",
            hardware="cuda",
            update_expected=False,
            generate_docs=True,
            template_db_path=TEMPLATE_DB_PATH,
            output_dir=OUTPUT_DIR,
            verbose=True
        )
        
        # Generate components
        skill_file, test_file, benchmark_file = tester.generate_components(test_dir)
        
        # Create mock test results
        test_results = {
            "success": True,
            "test_count": 5,
            "execution_time": 2.5,
            "stdout": "All tests passed successfully"
        }
        
        # Run test with documentation
        result = tester.run_test_with_docs(test_dir, test_results)
        
        if result["success"]:
            logger.info(f"Integration test successful! Documentation generated at: {result['documentation_path']}")
            
            # Verify documentation exists and contains expected content
            with open(result['documentation_path'], 'r') as f:
                content = f.read()
                if "bert-base-uncased" in content and "CUDA" in content and "Performance Visualizations" in content:
                    logger.info("Documentation content verification passed")
                else:
                    logger.warning("Documentation content missing expected sections")
        else:
            logger.error(f"Integration test failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Error during integration test: {e}")
    
    finally:
        # Clean up test directory
        if os.path.exists(test_dir) and not tester.keep_temp:
            shutil.rmtree(test_dir)


def main():
    """
    Main function to integrate the enhanced documentation system.
    """
    logger.info("Starting integration of enhanced documentation system")
    
    # Step 1: Add enhanced documentation templates to the database
    logger.info("Step 1: Adding enhanced documentation templates")
    try:
        add_enhanced_documentation_templates()
    except Exception as e:
        logger.error(f"Error adding enhanced documentation templates: {e}")
        return
    
    # Step 2: Apply patches to fix variable substitution issues
    logger.info("Step 2: Applying patches to fix variable substitution")
    try:
        monkey_patch_model_doc_generator()
        monkey_patch_template_renderer()
    except Exception as e:
        logger.error(f"Error applying patches: {e}")
        return
    
    # Step 3: Integrate enhanced documentation generator with test runner
    logger.info("Step 3: Integrating enhanced documentation generator")
    try:
        integrate_enhanced_doc_generator()
    except Exception as e:
        logger.error(f"Error integrating enhanced documentation generator: {e}")
        return
    
    # Step 4: Modify the integrated component tester
    logger.info("Step 4: Modifying integrated component tester")
    try:
        modify_integrated_component_tester()
    except Exception as e:
        logger.error(f"Error modifying integrated component tester: {e}")
        return
    
    # Step 5: Test the integration
    logger.info("Step 5: Testing integration")
    try:
        test_integration()
    except Exception as e:
        logger.error(f"Error testing integration: {e}")
        return
    
    logger.info("Integration of enhanced documentation system completed successfully")
    logger.info("You can now run the Integrated Component Test Runner with --generate-docs to create enhanced documentation")


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Integration of Enhanced Documentation System")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--test-only", action="store_true", help="Only run integration test")
    parser.add_argument("--skip-test", action="store_true", help="Skip integration test")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run the appropriate functions
    if args.test_only:
        # Apply necessary patches for testing
        monkey_patch_model_doc_generator()
        monkey_patch_template_renderer()
        integrate_enhanced_doc_generator()
        modify_integrated_component_tester()
        # Run the test
        test_integration()
    elif args.skip_test:
        add_enhanced_documentation_templates()
        monkey_patch_model_doc_generator()
        monkey_patch_template_renderer()
        integrate_enhanced_doc_generator()
        modify_integrated_component_tester()
    else:
        main()