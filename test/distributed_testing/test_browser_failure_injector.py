#!/usr/bin/env python3
"""
Test Script for Browser Failure Injector

This script thoroughly tests the browser_failure_injector to ensure
all failure types and intensities work correctly. It helps verify
that the injector can reliably create various browser failure scenarios
for testing recovery strategies.

Usage:
    python test_browser_failure_injector.py [--browser chrome] [--failure connection_failure]
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("browser_failure_injector_test")

# Set more verbose logging if environment variable is set
if os.environ.get("SELENIUM_BRIDGE_LOG_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

try:
    from selenium_browser_bridge import (
        BrowserConfiguration, SeleniumBrowserBridge, SELENIUM_AVAILABLE
    )
except ImportError:
    logger.error("Error importing selenium_browser_bridge. Make sure it exists at the expected path.")
    SELENIUM_AVAILABLE = False

try:
    from browser_failure_injector import (
        BrowserFailureInjector, FailureType
    )
    INJECTOR_AVAILABLE = True
except ImportError:
    logger.error("Error importing browser_failure_injector. Make sure it exists at the expected path.")
    INJECTOR_AVAILABLE = False
    
    # Define fallback FailureType for type checking
    from enum import Enum
    class FailureType(Enum):
        """Types of browser failures."""
        CONNECTION_FAILURE = "connection_failure"
        RESOURCE_EXHAUSTION = "resource_exhaustion"
        GPU_ERROR = "gpu_error"
        API_ERROR = "api_error"
        TIMEOUT = "timeout"
        CRASH = "crash"
        INTERNAL_ERROR = "internal_error"
        UNKNOWN = "unknown"

class BrowserFailureInjectorTest:
    """
    Test class for the Browser Failure Injector.
    
    This class provides a comprehensive test suite for verifying that the browser
    failure injector works correctly with different failure types and intensities.
    """
    
    def __init__(self, browser_name: str = "chrome", platform: str = "webgpu",
                 headless: bool = True, save_results: Optional[str] = None):
        """
        Initialize the failure injector test.
        
        Args:
            browser_name: Browser name to test (chrome, firefox, edge)
            platform: Platform to test (webgpu, webnn)
            headless: Whether to run in headless mode
            save_results: Path to save test results (or None)
        """
        self.browser_name = browser_name
        self.platform = platform
        self.headless = headless
        self.save_results = save_results
        
        # Test results
        self.results = {}
        
        # All supported failure types
        self.all_failure_types = [
            FailureType.CONNECTION_FAILURE,
            FailureType.RESOURCE_EXHAUSTION,
            FailureType.GPU_ERROR,
            FailureType.API_ERROR,
            FailureType.TIMEOUT,
            FailureType.INTERNAL_ERROR,
            FailureType.CRASH,
            FailureType.UNKNOWN
        ]
        
        # All supported intensities
        self.all_intensities = ["mild", "moderate", "severe"]
        
        logger.info(f"Initialized failure injector test with browser={browser_name}, platform={platform}")
    
    async def test_failure_type(self, failure_type: FailureType) -> Dict[str, Any]:
        """
        Test a specific failure type with all intensities.
        
        Args:
            failure_type: Failure type to test
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing failure type: {failure_type.value}")
        
        # Results for this failure type
        failure_results = {
            "failure_type": failure_type.value,
            "intensities": {},
            "success": True
        }
        
        # Test each intensity level
        for intensity in self.all_intensities:
            intensity_result = await self.test_failure_with_intensity(failure_type, intensity)
            failure_results["intensities"][intensity] = intensity_result
            
            # If any intensity fails, mark the whole failure type as failed
            if not intensity_result.get("success", False):
                failure_results["success"] = False
        
        return failure_results
    
    async def test_failure_with_intensity(self, failure_type: FailureType, intensity: str) -> Dict[str, Any]:
        """
        Test a specific failure type with a specific intensity.
        
        Args:
            failure_type: Failure type to test
            intensity: Intensity level to test (mild, moderate, severe)
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing {failure_type.value} with {intensity} intensity")
        
        # Result for this test
        result = {
            "failure_type": failure_type.value,
            "intensity": intensity,
            "success": False,
            "error": None,
            "start_time": time.time(),
            "end_time": None
        }
        
        # Create browser configuration
        config = BrowserConfiguration(
            browser_name=self.browser_name,
            platform=self.platform,
            headless=self.headless,
            timeout=30
        )
        
        # Create browser bridge
        bridge = SeleniumBrowserBridge(config)
        
        try:
            # Launch browser
            launch_success = await bridge.launch(allow_simulation=True)
            
            if not launch_success:
                result["error"] = f"Failed to launch {self.browser_name}"
                result["end_time"] = time.time()
                return result
            
            # Check if we're in simulation mode
            result["simulation_mode"] = getattr(bridge, 'simulation_mode', False)
            
            # Create failure injector
            injector = BrowserFailureInjector(bridge)
            
            # Inject the failure
            logger.info(f"Injecting {failure_type.value} failure with {intensity} intensity")
            injection_result = await injector.inject_failure(failure_type, intensity)
            
            # Store results
            result["injection_result"] = injection_result
            result["success"] = injection_result.get("success", False)
            
            # Get injector statistics
            stats = injector.get_failure_stats()
            result["injector_stats"] = stats
            
            # If the injector reports it was successful, but the browser crashed completely,
            # we should try to verify if the browser is still responsive
            if result["success"]:
                try:
                    # Simple check to see if the browser is still responding
                    # For some failure types like CRASH, this would fail (which is expected)
                    responsive = False
                    
                    try:
                        if hasattr(bridge, 'check_browser_responsive'):
                            responsive = await bridge.check_browser_responsive()
                    except Exception:
                        # Expected for some failure types
                        pass
                    
                    result["browser_responsive"] = responsive
                    
                    # For CRASH tests, we expect the browser to be unresponsive, so invert the result
                    if failure_type == FailureType.CRASH:
                        result["expected_unresponsive"] = True
                        if not responsive:
                            logger.info("Browser unresponsive after crash test (expected)")
                    
                except Exception as e:
                    logger.warning(f"Error checking browser responsiveness: {str(e)}")
            
            # Add browser metrics if available
            if hasattr(bridge, 'get_metrics'):
                try:
                    metrics = bridge.get_metrics()
                    result["browser_metrics"] = metrics
                except Exception as e:
                    logger.warning(f"Error getting browser metrics: {str(e)}")
            
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
        finally:
            # Record end time
            result["end_time"] = time.time()
            result["duration_seconds"] = result["end_time"] - result["start_time"]
            
            # Close browser
            if bridge:
                try:
                    await bridge.close()
                except Exception as e:
                    logger.warning(f"Error closing browser: {str(e)}")
        
        return result
    
    async def run_all_tests(self, specific_failure: Optional[str] = None) -> Dict[str, Any]:
        """
        Run tests for all failure types or a specific one.
        
        Args:
            specific_failure: Specific failure type to test (or None for all)
            
        Returns:
            Dictionary with all test results
        """
        # Build tests to run
        failure_types_to_test = []
        
        if specific_failure:
            # Find matching failure type
            try:
                for ft in self.all_failure_types:
                    if ft.value == specific_failure:
                        failure_types_to_test.append(ft)
                        break
                
                if not failure_types_to_test:
                    logger.error(f"Unknown failure type: {specific_failure}")
                    return {"error": f"Unknown failure type: {specific_failure}"}
                    
            except Exception as e:
                logger.error(f"Error finding failure type: {str(e)}")
                return {"error": str(e)}
        else:
            # Test all failure types
            failure_types_to_test = self.all_failure_types
        
        # Overall results
        all_results = {
            "browser": self.browser_name,
            "platform": self.platform,
            "headless": self.headless,
            "start_time": time.time(),
            "failure_types": {},
            "end_time": None,
            "success_count": 0,
            "failure_count": 0,
            "total_count": 0
        }
        
        # Track total test counts
        successful_tests = 0
        total_tests = 0
        
        # Run tests for each failure type
        for failure_type in failure_types_to_test:
            result = await self.test_failure_type(failure_type)
            all_results["failure_types"][failure_type.value] = result
            
            # Count successful and total tests
            intensity_count = len(result.get("intensities", {}))
            total_tests += intensity_count
            
            success_count = sum(1 for intensity, ir in result.get("intensities", {}).items() 
                              if ir.get("success", False))
            successful_tests += success_count
            
            # Print intermediate results
            print(f"\nResults for {failure_type.value}:")
            print(f"  Success: {success_count}/{intensity_count} intensities")
            for intensity, ir in result.get("intensities", {}).items():
                status = "✅ SUCCESS" if ir.get("success", False) else "❌ FAILED"
                error = f" - Error: {ir.get('error')}" if ir.get("error") else ""
                print(f"  {intensity}: {status}{error}")
            
        # Record final stats
        all_results["end_time"] = time.time()
        all_results["duration_seconds"] = all_results["end_time"] - all_results["start_time"]
        all_results["success_count"] = successful_tests
        all_results["failure_count"] = total_tests - successful_tests
        all_results["total_count"] = total_tests
        all_results["success_rate"] = successful_tests / total_tests if total_tests > 0 else 0
        
        # Store all results
        self.results = all_results
        
        # Save results if requested
        if self.save_results:
            self._save_results()
        
        return all_results
    
    def _save_results(self) -> None:
        """Save test results to a file."""
        if not self.save_results:
            return
            
        try:
            with open(self.save_results, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            print(f"\nResults saved to {self.save_results}")
            
            # Also generate a markdown summary
            markdown_path = self.save_results.replace('.json', '.md')
            if markdown_path == self.save_results:
                markdown_path += '.md'
                
            # Create markdown summary
            with open(markdown_path, 'w') as f:
                f.write(f"# Browser Failure Injector Test Results\n\n")
                
                f.write(f"## Configuration\n\n")
                f.write(f"- **Browser:** {self.results['browser']}\n")
                f.write(f"- **Platform:** {self.results['platform']}\n")
                f.write(f"- **Headless:** {self.results['headless']}\n")
                f.write(f"- **Duration:** {self.results['duration_seconds']:.2f} seconds\n\n")
                
                f.write(f"## Summary\n\n")
                f.write(f"- **Total Tests:** {self.results['total_count']}\n")
                f.write(f"- **Successful Tests:** {self.results['success_count']}\n")
                f.write(f"- **Failed Tests:** {self.results['failure_count']}\n")
                f.write(f"- **Success Rate:** {self.results['success_rate']:.2%}\n\n")
                
                f.write(f"## Results by Failure Type\n\n")
                f.write(f"| Failure Type | Mild | Moderate | Severe | Success Rate |\n")
                f.write(f"|--------------|------|----------|--------|-------------|\n")
                
                for failure_type, results in self.results["failure_types"].items():
                    intensities = results.get("intensities", {})
                    
                    # Format each intensity result
                    mild = "✅" if intensities.get("mild", {}).get("success", False) else "❌"
                    moderate = "✅" if intensities.get("moderate", {}).get("success", False) else "❌"
                    severe = "✅" if intensities.get("severe", {}).get("success", False) else "❌"
                    
                    # Calculate success rate
                    success_count = sum(1 for i, r in intensities.items() if r.get("success", False))
                    total_count = len(intensities)
                    success_rate = success_count / total_count if total_count > 0 else 0
                    
                    f.write(f"| {failure_type} | {mild} | {moderate} | {severe} | {success_rate:.2%} |\n")
                
                f.write(f"\n## Detailed Results\n\n")
                
                for failure_type, results in self.results["failure_types"].items():
                    f.write(f"### {failure_type}\n\n")
                    
                    for intensity, ir in results.get("intensities", {}).items():
                        status = "✅ SUCCESS" if ir.get("success", False) else "❌ FAILED"
                        f.write(f"#### {intensity.title()} Intensity: {status}\n\n")
                        
                        if ir.get("error"):
                            f.write(f"- **Error:** {ir['error']}\n")
                        
                        f.write(f"- **Duration:** {ir.get('duration_seconds', 0):.2f} seconds\n")
                        f.write(f"- **Simulation Mode:** {ir.get('simulation_mode', False)}\n")
                        
                        # Show browser responsiveness if available
                        if "browser_responsive" in ir:
                            f.write(f"- **Browser Responsive:** {ir['browser_responsive']}\n")
                        
                        f.write("\n")
                
                f.write(f"## Conclusion\n\n")
                
                if self.results['success_rate'] > 0.9:
                    f.write("The browser failure injector is working correctly for most failure types and intensities.\n")
                elif self.results['success_rate'] > 0.7:
                    f.write("The browser failure injector is working for most failure types, but some issues exist.\n")
                else:
                    f.write("The browser failure injector has significant issues that need to be addressed.\n")
                
            print(f"Markdown summary saved to {markdown_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def print_summary(self) -> None:
        """Print a summary of the test results."""
        if not self.results:
            print("\nNo test results available")
            return
        
        print("\n" + "=" * 80)
        print("Browser Failure Injector Test Summary")
        print("=" * 80)
        
        print(f"Browser:      {self.results['browser']}")
        print(f"Platform:     {self.results['platform']}")
        print(f"Total Tests:  {self.results['total_count']}")
        print(f"Successful:   {self.results['success_count']}")
        print(f"Failed:       {self.results['failure_count']}")
        print(f"Success Rate: {self.results['success_rate']:.2%}")
        print(f"Duration:     {self.results['duration_seconds']:.2f} seconds")
        
        print("\nResults by Failure Type:")
        print("-" * 60)
        print(f"{'Failure Type':<20} {'Mild':<10} {'Moderate':<10} {'Severe':<10}")
        print("-" * 60)
        
        for failure_type, results in self.results["failure_types"].items():
            intensities = results.get("intensities", {})
            
            # Format each intensity result
            mild = "✅ SUCCESS" if intensities.get("mild", {}).get("success", False) else "❌ FAILED"
            moderate = "✅ SUCCESS" if intensities.get("moderate", {}).get("success", False) else "❌ FAILED"
            severe = "✅ SUCCESS" if intensities.get("severe", {}).get("success", False) else "❌ FAILED"
            
            print(f"{failure_type:<20} {mild:<10} {moderate:<10} {severe:<10}")
        
        print("=" * 80)
        
        # Provide recommendations based on results
        if self.results['success_rate'] > 0.9:
            print("\nThe failure injector is working correctly for most failure types and intensities.")
        elif self.results['success_rate'] > 0.7:
            print("\nThe failure injector is working for most failure types, but some issues exist.")
            
            # Identify problematic failure types
            problematic = []
            for failure_type, results in self.results["failure_types"].items():
                success_count = sum(1 for i, r in results.get("intensities", {}).items() if r.get("success", False))
                total_count = len(results.get("intensities", {}))
                if total_count > 0 and success_count / total_count < 0.7:
                    problematic.append(failure_type)
            
            if problematic:
                print(f"Problematic failure types: {', '.join(problematic)}")
        else:
            print("\nThe failure injector has significant issues that need to be addressed.")


async def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Browser Failure Injector")
    parser.add_argument("--browser", default="chrome", choices=["chrome", "firefox", "edge"], 
                        help="Browser to test (chrome, firefox, edge)")
    parser.add_argument("--platform", default="webgpu", choices=["webgpu", "webnn"], 
                        help="Platform to test (webgpu, webnn)")
    parser.add_argument("--failure", 
                        help="Specific failure type to test (or omit for all failure types)")
    parser.add_argument("--no-headless", action="store_true", 
                        help="Run browser in visible mode (not headless)")
    parser.add_argument("--save-results", type=str, 
                        help="Path to save test results (JSON)")
    args = parser.parse_args()
    
    # Create default save path if not provided
    save_path = args.save_results
    if not save_path:
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        if args.failure:
            save_path = os.path.join(reports_dir, f"failure_injector_{args.browser}_{args.failure}_{timestamp}.json")
        else:
            save_path = os.path.join(reports_dir, f"failure_injector_{args.browser}_{timestamp}.json")
    
    # Check dependencies
    if not SELENIUM_AVAILABLE:
        logger.error("Selenium not available. Cannot run tests.")
        return 1
        
    if not INJECTOR_AVAILABLE:
        logger.error("Browser Failure Injector not available. Cannot run tests.")
        return 1
    
    # Create and run tests
    print("-" * 80)
    print(f"Running Browser Failure Injector tests with:")
    print(f"  Browser:      {args.browser}")
    print(f"  Platform:     {args.platform}")
    print(f"  Failure Type: {args.failure if args.failure else 'All'}")
    print(f"  Headless:     {not args.no_headless}")
    print("-" * 80)
    
    injector_test = BrowserFailureInjectorTest(
        browser_name=args.browser,
        platform=args.platform,
        headless=not args.no_headless,
        save_results=save_path
    )
    
    # Run tests
    await injector_test.run_all_tests(args.failure)
    
    # Print summary
    injector_test.print_summary()
    
    # Determine exit code based on results
    if injector_test.results.get("success_rate", 0) > 0.8:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)