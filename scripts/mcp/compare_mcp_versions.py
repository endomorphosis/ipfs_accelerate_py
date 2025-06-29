#!/usr/bin/env python3
"""
MCP Server Version Comparison Tool

This script helps compare multiple MCP server versions by running the comprehensive
test suite on each version and generating a detailed comparison report.

It identifies specific differences between versions to help diagnose why different
versions may not work correctly with the ipfs_accelerate_py module.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MCPVersionComparer:
    """Compare different MCP server versions by running comprehensive tests on each."""
    
    def __init__(self, versions: List[str], output_dir: str = "version_comparison_results",
                 mcp_server_script: str = "./start_mcp_server.sh", 
                 test_port_start: int = 8010):
        """Initialize the version comparer.
        
        Args:
            versions: List of MCP versions to compare (e.g., ["v1.0.0", "v1.1.0"])
            output_dir: Directory to store comparison results
            mcp_server_script: Path to the MCP server start script
            test_port_start: Starting port number for testing (each version uses a different port)
        """
        self.versions = versions
        self.output_dir = output_dir
        self.mcp_server_script = mcp_server_script
        self.test_port_start = test_port_start
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_by_version = {}
        self.comparison_report = {
            "timestamp": self.timestamp,
            "versions_compared": versions,
            "version_details": {},
            "tool_comparison": {},
            "test_comparison": {},
            "vfs_comparison": {},
            "summary": {
                "working_versions": [],
                "failing_versions": [],
                "key_differences": []
            }
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create version-specific directories
        for version in versions:
            version_dir = os.path.join(output_dir, version)
            if not os.path.exists(version_dir):
                os.makedirs(version_dir)
    
    def run_tests_on_versions(self) -> bool:
        """Run comprehensive tests on each MCP version.
        
        Returns:
            bool: True if all versions were tested, False otherwise
        """
        logger.info(f"Starting tests on {len(self.versions)} MCP versions")
        
        for i, version in enumerate(self.versions):
            port = self.test_port_start + i
            version_dir = os.path.join(self.output_dir, version)
            
            logger.info(f"Testing MCP version: {version} on port {port}")
            success = self.test_single_version(version, port, version_dir)
            
            if success:
                logger.info(f"Successfully tested version {version}")
            else:
                logger.warning(f"Failed to complete tests for version {version}")
        
        # Check if we have results for all versions
        for version in self.versions:
            if version not in self.results_by_version:
                logger.error(f"Missing test results for version {version}")
                return False
        
        return True
    
    def test_single_version(self, version: str, port: int, output_dir: str) -> bool:
        """Run tests on a single MCP version.
        
        Args:
            version: MCP version to test
            port: Port to use for this version's server
            output_dir: Directory to store this version's results
            
        Returns:
            bool: True if tests were run successfully, False otherwise
        """
        try:
            # Assume we need to checkout or switch to this version first
            # This is just a placeholder - you'd need to implement version switching logic
            # based on how your MCP versions are managed (git tags, different dirs, etc.)
            logger.info(f"Switching to MCP version {version}")
            # self.switch_to_version(version)  # Uncomment and implement if needed
            
            # Run the comprehensive tests
            test_output_dir = os.path.join(output_dir, "test_results")
            if not os.path.exists(test_output_dir):
                os.makedirs(test_output_dir)
            
            cmd = [
                self.mcp_server_script,
                "--port", str(port),
                "--test-level", "comprehensive",
                "--output-dir", test_output_dir
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Save command output
            with open(os.path.join(output_dir, "test_output.log"), "w") as f:
                f.write(f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}")
            
            # Find the test report JSON file (should be the most recent file in test_output_dir)
            report_files = [f for f in os.listdir(test_output_dir) 
                          if f.startswith("mcp_test_report_") and f.endswith(".json")]
            
            if not report_files:
                logger.error(f"No test report found for version {version}")
                return False
            
            # Sort by modification time (newest first)
            report_files.sort(key=lambda f: os.path.getmtime(os.path.join(test_output_dir, f)), reverse=True)
            report_file = os.path.join(test_output_dir, report_files[0])
            
            # Load the test results
            with open(report_file, "r") as f:
                test_results = json.load(f)
            
            # Store results for this version
            self.results_by_version[version] = test_results
            
            # Look for integration test results
            integration_files = [f for f in os.listdir(test_output_dir) 
                              if f.startswith("mcp_integration_results_") and f.endswith(".json")]
            
            if integration_files:
                integration_files.sort(key=lambda f: os.path.getmtime(os.path.join(test_output_dir, f)), reverse=True)
                integration_file = os.path.join(test_output_dir, integration_files[0])
                
                with open(integration_file, "r") as f:
                    integration_results = json.load(f)
                
                # Add integration results to the version's results
                self.results_by_version[version]["integration_results"] = integration_results
            
            # Look for VFS test results
            vfs_files = [f for f in os.listdir(test_output_dir) 
                       if f.startswith("vfs_test_results_") and f.endswith(".json")]
            
            if vfs_files:
                vfs_files.sort(key=lambda f: os.path.getmtime(os.path.join(test_output_dir, f)), reverse=True)
                vfs_file = os.path.join(test_output_dir, vfs_files[0])
                
                with open(vfs_file, "r") as f:
                    vfs_results = json.load(f)
                
                # Add VFS results to the version's results
                self.results_by_version[version]["vfs_results"] = vfs_results
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing version {version}: {e}")
            return False
    
    def analyze_results(self) -> None:
        """Analyze test results and identify key differences between versions."""
        logger.info("Analyzing test results across versions")
        
        # Categorize versions as working or failing
        for version, results in self.results_by_version.items():
            # Check if this version is generally working
            is_working = True
            
            # Basic server connectivity check
            if results.get("server", {}).get("status") != "running":
                is_working = False
            
            # Check test results if available
            tests = results.get("tests", [])
            if tests:
                fail_count = sum(1 for test in tests if test.get("result") == "fail")
                if fail_count > len(tests) / 3:  # More than 1/3 of tests failing
                    is_working = False
            
            # Check integration results if available
            integration = results.get("integration_results", {})
            if integration:
                integration_summary = integration.get("summary", {})
                if integration_summary.get("failed", 0) > integration_summary.get("passed", 0):
                    is_working = False
            
            # Record result
            if is_working:
                self.comparison_report["summary"]["working_versions"].append(version)
            else:
                self.comparison_report["summary"]["failing_versions"].append(version)
            
            # Add version details
            self.comparison_report["version_details"][version] = {
                "server_status": results.get("server", {}).get("status"),
                "server_version": results.get("server", {}).get("version", "unknown"),
                "test_summary": self._extract_test_summary(results),
                "has_integration_results": "integration_results" in results,
                "has_vfs_results": "vfs_results" in results
            }
        
        # Compare available tools across versions
        all_tools = set()
        tools_by_version = {}
        
        for version, results in self.results_by_version.items():
            tools = results.get("tools_registered", [])
            tools_by_version[version] = set(tools)
            all_tools.update(tools)
        
        # Find tool differences
        for tool in sorted(all_tools):
            versions_with_tool = []
            versions_without_tool = []
            
            for version in self.versions:
                if version in tools_by_version and tool in tools_by_version[version]:
                    versions_with_tool.append(version)
                else:
                    versions_without_tool.append(version)
            
            if versions_without_tool:  # Only record differences
                self.comparison_report["tool_comparison"][tool] = {
                    "present_in": versions_with_tool,
                    "missing_from": versions_without_tool
                }
                
                # If the tool is missing from failing versions but present in working versions,
                # this could be a key difference
                if (versions_without_tool and versions_with_tool and
                    all(v in self.comparison_report["summary"]["failing_versions"] for v in versions_without_tool) and
                    all(v in self.comparison_report["summary"]["working_versions"] for v in versions_with_tool)):
                    self.comparison_report["summary"]["key_differences"].append(
                        f"Tool '{tool}' is missing from failing versions but present in working versions"
                    )
        
        # Compare test results across versions
        all_tests = set()
        tests_by_version = {}
        
        for version, results in self.results_by_version.items():
            tests = {test.get("name"): test.get("result") for test in results.get("tests", [])}
            tests_by_version[version] = tests
            all_tests.update(tests.keys())
        
        # Find test result differences
        for test in sorted(all_tests):
            test_results = {}
            
            for version in self.versions:
                if version in tests_by_version and test in tests_by_version[version]:
                    test_results[version] = tests_by_version[version][test]
            
            # Check if results differ between versions
            results_values = list(test_results.values())
            if results_values and not all(r == results_values[0] for r in results_values):
                self.comparison_report["test_comparison"][test] = test_results
                
                # If test passes in working versions but fails in failing versions,
                # this could be a key difference
                passing_versions = [v for v, r in test_results.items() if r == "pass"]
                failing_versions = [v for v, r in test_results.items() if r == "fail"]
                
                if (passing_versions and failing_versions and
                    all(v in self.comparison_report["summary"]["working_versions"] for v in passing_versions) and
                    all(v in self.comparison_report["summary"]["failing_versions"] for v in failing_versions)):
                    self.comparison_report["summary"]["key_differences"].append(
                        f"Test '{test}' passes in working versions but fails in failing versions"
                    )
        
        # Compare VFS functionality
        vfs_comparison = {}
        for version, results in self.results_by_version.items():
            vfs_results = results.get("vfs_results", {})
            if vfs_results:
                vfs_tests = vfs_results.get("tests", {})
                vfs_comparison[version] = {
                    "available_tools": vfs_results.get("vfs_tools", {}).get("available_tools", []),
                    "test_results": {name: result.get("status") for name, result in vfs_tests.items()},
                    "summary": vfs_results.get("summary", {})
                }
        
        if vfs_comparison:
            self.comparison_report["vfs_comparison"] = vfs_comparison
            
            # Check if VFS functionality differs between versions
            working_versions_with_vfs = [v for v in self.comparison_report["summary"]["working_versions"] 
                                        if v in vfs_comparison]
            failing_versions_with_vfs = [v for v in self.comparison_report["summary"]["failing_versions"] 
                                        if v in vfs_comparison]
            
            if working_versions_with_vfs and failing_versions_with_vfs:
                # Compare VFS tool availability
                for version in failing_versions_with_vfs:
                    missing_tools = set()
                    for work_ver in working_versions_with_vfs:
                        missing_tools.update(set(vfs_comparison[work_ver]["available_tools"]) - 
                                           set(vfs_comparison[version]["available_tools"]))
                    
                    if missing_tools:
                        self.comparison_report["summary"]["key_differences"].append(
                            f"Version {version} is missing VFS tools: {', '.join(missing_tools)}"
                        )
                
                # Compare test results
                for version in failing_versions_with_vfs:
                    failing_tests = []
                    for test, result in vfs_comparison[version]["test_results"].items():
                        if result != "passed":
                            # Check if this test passes in working versions
                            if any(vfs_comparison[work_ver]["test_results"].get(test) == "passed" 
                                 for work_ver in working_versions_with_vfs):
                                failing_tests.append(test)
                    
                    if failing_tests:
                        self.comparison_report["summary"]["key_differences"].append(
                            f"Version {version} fails VFS tests: {', '.join(failing_tests)}"
                        )
    
    def _extract_test_summary(self, results: Dict[str, Any]) -> Dict[str, int]:
        """Extract and summarize test results from a version's test data."""
        passed = 0
        failed = 0
        skipped = 0
        
        # Check main test results
        for test in results.get("tests", []):
            result = test.get("result")
            if result == "pass":
                passed += 1
            elif result == "fail":
                failed += 1
            elif result == "skip":
                skipped += 1
        
        # Check integration results if available
        integration = results.get("integration_results", {}).get("summary", {})
        if integration:
            passed += integration.get("passed", 0)
            failed += integration.get("failed", 0)
            skipped += integration.get("skipped", 0)
        
        # Check VFS results if available
        vfs = results.get("vfs_results", {}).get("summary", {})
        if vfs:
            passed += vfs.get("passed", 0)
            failed += vfs.get("failed", 0)
            skipped += vfs.get("skipped", 0)
        
        return {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": passed + failed + skipped
        }
    
    def generate_report(self) -> str:
        """Generate and save a detailed comparison report.
        
        Returns:
            str: Path to the generated report file
        """
        report_file = os.path.join(self.output_dir, f"comparison_report_{self.timestamp}.json")
        report_md = os.path.join(self.output_dir, f"comparison_report_{self.timestamp}.md")
        
        # Save JSON report
        with open(report_file, "w") as f:
            json.dump(self.comparison_report, f, indent=2)
        
        # Generate a more readable Markdown report
        with open(report_md, "w") as f:
            f.write(f"# MCP Version Comparison Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Versions Compared\n\n")
            for version in self.versions:
                status = "✅ Working" if version in self.comparison_report["summary"]["working_versions"] else "❌ Failing"
                f.write(f"- **{version}**: {status}\n")
            
            f.write(f"\n## Key Differences\n\n")
            if self.comparison_report["summary"]["key_differences"]:
                for diff in self.comparison_report["summary"]["key_differences"]:
                    f.write(f"- {diff}\n")
            else:
                f.write("No key differences identified.\n")
            
            f.write(f"\n## Tool Availability Comparison\n\n")
            if self.comparison_report["tool_comparison"]:
                f.write("| Tool | Present In | Missing From |\n")
                f.write("|------|------------|-------------|\n")
                for tool, info in self.comparison_report["tool_comparison"].items():
                    present = ", ".join(info["present_in"])
                    missing = ", ".join(info["missing_from"])
                    f.write(f"| {tool} | {present} | {missing} |\n")
            else:
                f.write("No tool differences found.\n")
            
            f.write(f"\n## Test Result Comparison\n\n")
            if self.comparison_report["test_comparison"]:
                f.write("| Test | " + " | ".join(self.versions) + " |\n")
                f.write("|------|" + "|".join(["---" for _ in self.versions]) + "|\n")
                
                for test, results in self.comparison_report["test_comparison"].items():
                    row = [test]
                    for version in self.versions:
                        result = results.get(version, "N/A")
                        if result == "pass":
                            cell = "✅ Pass"
                        elif result == "fail":
                            cell = "❌ Fail"
                        else:
                            cell = "⚪ N/A"
                        row.append(cell)
                    f.write("| " + " | ".join(row) + " |\n")
            else:
                f.write("No test result differences found.\n")
            
            f.write(f"\n## Version Details\n\n")
            for version, details in self.comparison_report["version_details"].items():
                f.write(f"### {version}\n\n")
                f.write(f"- **Server Status**: {details['server_status']}\n")
                f.write(f"- **Server Version**: {details['server_version']}\n")
                
                summary = details["test_summary"]
                f.write(f"- **Test Summary**: {summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped (Total: {summary['total']})\n")
                f.write(f"- **Integration Tests**: {'Included' if details['has_integration_results'] else 'Not included'}\n")
                f.write(f"- **VFS Tests**: {'Included' if details['has_vfs_results'] else 'Not included'}\n\n")
        
        logger.info(f"Report generated: {report_md}")
        return report_md
    
    def compare_versions(self) -> str:
        """Run the full comparison workflow.
        
        Returns:
            str: Path to the generated report file
        """
        # Run tests on all versions
        if not self.run_tests_on_versions():
            logger.error("Failed to complete tests on all versions")
        
        # Analyze results
        self.analyze_results()
        
        # Generate report
        return self.generate_report()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare multiple MCP server versions")
    parser.add_argument("--versions", required=True, nargs="+", 
                        help="List of MCP versions to compare (e.g., v1.0.0 v1.1.0)")
    parser.add_argument("--output-dir", default="version_comparison_results",
                        help="Directory for comparison results")
    parser.add_argument("--server-script", default="./start_mcp_server.sh",
                        help="Path to the MCP server start script")
    parser.add_argument("--port-start", type=int, default=8010,
                        help="Starting port number for testing")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    comparer = MCPVersionComparer(
        versions=args.versions,
        output_dir=args.output_dir,
        mcp_server_script=args.server_script,
        test_port_start=args.port_start
    )
    
    report_path = comparer.compare_versions()
    print(f"Comparison report generated: {report_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
