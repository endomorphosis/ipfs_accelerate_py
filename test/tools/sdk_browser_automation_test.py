#!/usr/bin/env python3
"""
Comprehensive Browser Automation Test for SDK Dashboard

This script uses Playwright to test the complete SDK Dashboard application,
verifying that all inference pipelines work through the JavaScript SDK
with JSON-RPC communication.
"""

import anyio
import os
import subprocess
import sys
import time
import signal
from pathlib import Path
from typing import Optional

try:
    from playwright.async_api import async_playwright, Browser, Page
    HAVE_PLAYWRIGHT = True
except ImportError:
    HAVE_PLAYWRIGHT = False
    print("⚠️ Playwright not available. Install with: pip install playwright")

class SDKBrowserTester:
    """Browser automation tester for SDK Dashboard."""
    
    def __init__(self):
        """Initialize the browser tester."""
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.dashboard_process = None
        self.dashboard_url = "http://localhost:8080"
        self.jsonrpc_url = "http://localhost:8000"
        self.screenshots_dir = Path("data/screenshots/sdk_dashboard")
        
        # Create screenshots directory
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Test results
        self.test_results = {
            "server_connection": False,
            "dashboard_loaded": False,
            "sdk_loaded": False,
            "inference_tests": {},
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0
        }
    
    async def setup(self):
        """Setup browser and start servers."""
        if not HAVE_PLAYWRIGHT:
            raise RuntimeError("Playwright is required for browser testing")
        
        print("🚀 Setting up browser automation test...")
        
        # Start the SDK dashboard application
        await self.start_dashboard()
        
        # Setup Playwright browser
        playwright = await async_playwright().start()
        
        # Launch browser with options
        self.browser = await playwright.chromium.launch(
            headless=False,  # Set to False to see the browser in action
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        
        # Create a new page
        self.page = await self.browser.new_page()
        
        # Set viewport
        await self.page.set_viewport_size({"width": 1920, "height": 1080})
        
        print("✅ Browser setup complete")
    
    async def start_dashboard(self):
        """Start the SDK dashboard application."""
        print("🔧 Starting SDK Dashboard application...")
        
        # Start the dashboard in the background
        self.dashboard_process = subprocess.Popen(
            [sys.executable, "sdk_dashboard_app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        
        # Wait for servers to start
        print("⏳ Waiting for servers to start...")
        await anyio.sleep(8)  # Give servers time to start
        
        # Check if process is still running
        if self.dashboard_process.poll() is not None:
            stdout, stderr = self.dashboard_process.communicate()
            print(f"❌ Dashboard process failed to start:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            raise RuntimeError("Failed to start dashboard")
        
        print("✅ Dashboard application started")
    
    async def test_server_connection(self):
        """Test that servers are responding."""
        print("🔗 Testing server connections...")
        
        try:
            # Test dashboard server
            response = await self.page.goto(f"{self.dashboard_url}/health")
            if response and response.status == 200:
                self.test_results["server_connection"] = True
                print("✅ Dashboard server is responding")
            else:
                print(f"❌ Dashboard server error: {response.status if response else 'No response'}")
            
        except Exception as e:
            print(f"❌ Server connection failed: {e}")
    
    async def test_dashboard_loading(self):
        """Test that the dashboard loads correctly."""
        print("📊 Testing dashboard loading...")
        
        try:
            # Navigate to dashboard
            await self.page.goto(self.dashboard_url)
            
            # Wait for the page to load
            await self.page.wait_for_selector('.feature-highlight', timeout=10000)
            
            # Check if key elements are present
            title = await self.page.text_content('h1')
            if "Kitchen Sink AI Model Testing Interface" in title:
                self.test_results["dashboard_loaded"] = True
                print("✅ Dashboard loaded successfully")
                
                # Take screenshot
                await self.page.screenshot(
                    path=self.screenshots_dir / "01_dashboard_loaded.png",
                    full_page=True
                )
            else:
                print(f"❌ Dashboard title incorrect: {title}")
                
        except Exception as e:
            print(f"❌ Dashboard loading failed: {e}")
    
    async def test_sdk_initialization(self):
        """Test that the SDK initializes correctly."""
        print("🔧 Testing SDK initialization...")
        
        try:
            # Wait for the SDK to initialize
            await self.page.wait_for_function(
                "typeof window.MCPClient !== 'undefined'",
                timeout=10000
            )
            
            # Check if SDK is loaded
            sdk_loaded = await self.page.evaluate("typeof window.MCPClient !== 'undefined'")
            
            if sdk_loaded:
                self.test_results["sdk_loaded"] = True
                print("✅ MCP SDK loaded successfully")
                
                # Check server connection status
                await self.page.wait_for_selector('.server-status.connected', timeout=15000)
                
                server_status = await self.page.text_content('.server-status')
                if "Connected" in server_status:
                    print("✅ SDK connected to JSON-RPC server")
                else:
                    print(f"⚠️ SDK connection status: {server_status}")
                
                # Take screenshot
                await self.page.screenshot(
                    path=self.screenshots_dir / "02_sdk_connected.png",
                    full_page=True
                )
                
            else:
                print("❌ MCP SDK not loaded")
                
        except Exception as e:
            print(f"❌ SDK initialization failed: {e}")
    
    async def test_text_generation(self):
        """Test text generation functionality."""
        print("📝 Testing text generation...")
        
        try:
            # Click on Text Generation tab (it should be active by default)
            await self.page.wait_for_selector('#text-prompt')
            
            # Fill in the form
            await self.page.fill('#text-prompt', 'Once upon a time in a magical kingdom')
            await self.page.fill('#text-model', '')  # Use auto-selection
            
            # Adjust parameters
            await self.page.fill('#max-length', '150')
            await self.page.fill('#temperature', '0.8')
            
            # Take screenshot before submission
            await self.page.screenshot(
                path=self.screenshots_dir / "03_text_generation_form.png",
                full_page=True
            )
            
            # Submit form
            await self.page.click('button[type="submit"]', timeout=5000)
            
            # Wait for result
            await self.page.wait_for_selector('#text-generation-result .result-item', timeout=15000)
            
            # Check if result appears
            result_content = await self.page.text_content('#text-generation-result')
            if result_content and len(result_content.strip()) > 0:
                self.test_results["inference_tests"]["text_generation"] = True
                print("✅ Text generation working")
                
                # Take screenshot of result
                await self.page.screenshot(
                    path=self.screenshots_dir / "04_text_generation_result.png",
                    full_page=True
                )
            else:
                print("❌ Text generation failed - no result")
                
        except Exception as e:
            print(f"❌ Text generation test failed: {e}")
            self.test_results["inference_tests"]["text_generation"] = False
    
    async def test_text_classification(self):
        """Test text classification functionality."""
        print("🏷️ Testing text classification...")
        
        try:
            # Click on Text Classification tab
            await self.page.click('#v-pills-text-classification-tab')
            await self.page.wait_for_selector('#classification-text')
            
            # Fill in the form
            await self.page.fill('#classification-text', 'I love this product! It works amazingly well.')
            
            # Take screenshot before submission
            await self.page.screenshot(
                path=self.screenshots_dir / "05_text_classification_form.png",
                full_page=True
            )
            
            # Submit form
            await self.page.click('#text-classification-form button[type="submit"]')
            
            # Wait for result
            await self.page.wait_for_selector('#text-classification-result .classification-result', timeout=15000)
            
            # Check if result appears
            result_content = await self.page.text_content('#text-classification-result')
            if "Classification Result" in result_content:
                self.test_results["inference_tests"]["text_classification"] = True
                print("✅ Text classification working")
                
                # Take screenshot of result
                await self.page.screenshot(
                    path=self.screenshots_dir / "06_text_classification_result.png",
                    full_page=True
                )
            else:
                print("❌ Text classification failed - no result")
                
        except Exception as e:
            print(f"❌ Text classification test failed: {e}")
            self.test_results["inference_tests"]["text_classification"] = False
    
    async def test_text_embeddings(self):
        """Test text embeddings functionality."""
        print("🧮 Testing text embeddings...")
        
        try:
            # Click on Text Embeddings tab
            await self.page.click('#v-pills-text-embeddings-tab')
            await self.page.wait_for_selector('#embeddings-text')
            
            # Fill in the form
            await self.page.fill('#embeddings-text', 'This is a sample text for embedding generation.')
            
            # Take screenshot before submission
            await self.page.screenshot(
                path=self.screenshots_dir / "07_text_embeddings_form.png",
                full_page=True
            )
            
            # Submit form
            await self.page.click('#text-embeddings-form button[type="submit"]')
            
            # Wait for result
            await self.page.wait_for_selector('#text-embeddings-result .embeddings-result', timeout=15000)
            
            # Check if result appears
            result_content = await self.page.text_content('#text-embeddings-result')
            if "Text Embeddings" in result_content and "Dimensions:" in result_content:
                self.test_results["inference_tests"]["text_embeddings"] = True
                print("✅ Text embeddings working")
                
                # Take screenshot of result
                await self.page.screenshot(
                    path=self.screenshots_dir / "08_text_embeddings_result.png",
                    full_page=True
                )
            else:
                print("❌ Text embeddings failed - no result")
                
        except Exception as e:
            print(f"❌ Text embeddings test failed: {e}")
            self.test_results["inference_tests"]["text_embeddings"] = False
    
    async def test_model_recommendations(self):
        """Test model recommendations functionality."""
        print("💡 Testing model recommendations...")
        
        try:
            # Click on Recommendations tab
            await self.page.click('#v-pills-recommendations-tab')
            await self.page.wait_for_selector('#task-type')
            
            # Select task type
            await self.page.select_option('#task-type', 'text_generation')
            await self.page.select_option('#input-type', 'text')
            
            # Take screenshot before submission
            await self.page.screenshot(
                path=self.screenshots_dir / "09_recommendations_form.png",
                full_page=True
            )
            
            # Submit form
            await self.page.click('#model-recommendations-form button[type="submit"]')
            
            # Wait for result
            await self.page.wait_for_selector('#model-recommendations-result .recommendations-result', timeout=15000)
            
            # Check if result appears
            result_content = await self.page.text_content('#model-recommendations-result')
            if "Model Recommendations" in result_content:
                self.test_results["inference_tests"]["model_recommendations"] = True
                print("✅ Model recommendations working")
                
                # Take screenshot of result
                await self.page.screenshot(
                    path=self.screenshots_dir / "10_recommendations_result.png",
                    full_page=True
                )
            else:
                print("❌ Model recommendations failed - no result")
                
        except Exception as e:
            print(f"❌ Model recommendations test failed: {e}")
            self.test_results["inference_tests"]["model_recommendations"] = False
    
    async def test_model_manager(self):
        """Test model manager functionality."""
        print("🗄️ Testing model manager...")
        
        try:
            # Click on Model Manager tab
            await self.page.click('#v-pills-model-manager-tab')
            await self.page.wait_for_selector('#search-query')
            
            # Search for models
            await self.page.fill('#search-query', 'gpt')
            await self.page.fill('#search-limit', '5')
            
            # Take screenshot before submission
            await self.page.screenshot(
                path=self.screenshots_dir / "11_model_manager_form.png",
                full_page=True
            )
            
            # Submit form
            await self.page.click('#model-search-form button[type="submit"]')
            
            # Wait for result
            await self.page.wait_for_selector('#model-search-result .model-search-result', timeout=15000)
            
            # Check if result appears
            result_content = await self.page.text_content('#model-search-result')
            if "Model Search Results" in result_content:
                self.test_results["inference_tests"]["model_manager"] = True
                print("✅ Model manager working")
                
                # Take screenshot of result
                await self.page.screenshot(
                    path=self.screenshots_dir / "12_model_manager_result.png",
                    full_page=True
                )
            else:
                print("❌ Model manager failed - no result")
                
        except Exception as e:
            print(f"❌ Model manager test failed: {e}")
            self.test_results["inference_tests"]["model_manager"] = False
    
    async def test_vision_models(self):
        """Test vision models functionality."""
        print("👁️ Testing vision models...")
        
        try:
            # Click on Vision Models tab
            await self.page.click('#v-pills-vision-models-tab')
            await self.page.wait_for_selector('#image-generation-tab')
            
            # Click on Image Generation sub-tab
            await self.page.click('#image-generation-tab')
            await self.page.wait_for_selector('#image-prompt')
            
            # Fill in the form
            await self.page.fill('#image-prompt', 'A beautiful sunset over the ocean with sailing boats')
            await self.page.fill('#image-width', '512')
            await self.page.fill('#image-height', '512')
            
            # Take screenshot before submission
            await self.page.screenshot(
                path=self.screenshots_dir / "13_vision_models_form.png",
                full_page=True
            )
            
            # Submit form
            await self.page.click('#image-generation-form button[type="submit"]')
            
            # Wait for result
            await self.page.wait_for_selector('#image-generation-result .result-item', timeout=15000)
            
            # Check if result appears
            result_content = await self.page.text_content('#image-generation-result')
            if result_content and len(result_content.strip()) > 0:
                self.test_results["inference_tests"]["vision_models"] = True
                print("✅ Vision models working")
                
                # Take screenshot of result
                await self.page.screenshot(
                    path=self.screenshots_dir / "14_vision_models_result.png",
                    full_page=True
                )
            else:
                print("❌ Vision models failed - no result")
                
        except Exception as e:
            print(f"❌ Vision models test failed: {e}")
            self.test_results["inference_tests"]["vision_models"] = False
    
    async def test_specialized_tools(self):
        """Test specialized tools functionality."""
        print("⚙️ Testing specialized tools...")
        
        try:
            # Click on Specialized tab
            await self.page.click('#v-pills-specialized-tab')
            await self.page.wait_for_selector('#code-description')
            
            # Fill in the form
            await self.page.fill('#code-description', 'Create a function that sorts a list of numbers in ascending order')
            await self.page.select_option('#code-language', 'python')
            
            # Take screenshot before submission
            await self.page.screenshot(
                path=self.screenshots_dir / "15_specialized_tools_form.png",
                full_page=True
            )
            
            # Submit form
            await self.page.click('#code-generation-form button[type="submit"]')
            
            # Wait for result
            await self.page.wait_for_selector('#code-generation-result .code-result', timeout=15000)
            
            # Check if result appears
            result_content = await self.page.text_content('#code-generation-result')
            if "Generated Code" in result_content:
                self.test_results["inference_tests"]["specialized_tools"] = True
                print("✅ Specialized tools working")
                
                # Take screenshot of result
                await self.page.screenshot(
                    path=self.screenshots_dir / "16_specialized_tools_result.png",
                    full_page=True
                )
            else:
                print("❌ Specialized tools failed - no result")
                
        except Exception as e:
            print(f"❌ Specialized tools test failed: {e}")
            self.test_results["inference_tests"]["specialized_tools"] = False
    
    async def generate_summary_report(self):
        """Generate a summary report of all tests."""
        print("📊 Generating test summary report...")
        
        # Calculate test statistics
        self.test_results["total_tests"] = len(self.test_results["inference_tests"]) + 3  # +3 for basic tests
        self.test_results["passed_tests"] = (
            sum(1 for v in [
                self.test_results["server_connection"],
                self.test_results["dashboard_loaded"],
                self.test_results["sdk_loaded"]
            ] if v) +
            sum(1 for v in self.test_results["inference_tests"].values() if v)
        )
        self.test_results["failed_tests"] = self.test_results["total_tests"] - self.test_results["passed_tests"]
        
        # Create summary screenshot
        await self.page.goto(self.dashboard_url)
        await self.page.wait_for_selector('.feature-highlight')
        await self.page.screenshot(
            path=self.screenshots_dir / "17_final_dashboard_overview.png",
            full_page=True
        )
        
        # Generate report
        report = f"""
# SDK Dashboard Browser Automation Test Report

## Test Summary
- **Total Tests**: {self.test_results["total_tests"]}
- **Passed Tests**: {self.test_results["passed_tests"]}
- **Failed Tests**: {self.test_results["failed_tests"]}
- **Success Rate**: {(self.test_results["passed_tests"] / self.test_results["total_tests"] * 100):.1f}%

## Basic Functionality Tests
- Server Connection: {'✅ PASS' if self.test_results["server_connection"] else '❌ FAIL'}
- Dashboard Loading: {'✅ PASS' if self.test_results["dashboard_loaded"] else '❌ FAIL'}
- SDK Initialization: {'✅ PASS' if self.test_results["sdk_loaded"] else '❌ FAIL'}

## Inference Pipeline Tests
"""
        
        for test_name, result in self.test_results["inference_tests"].items():
            status = '✅ PASS' if result else '❌ FAIL'
            report += f"- {test_name.replace('_', ' ').title()}: {status}\n"
        
        report += f"""
## Screenshots Generated
All test screenshots saved to: {self.screenshots_dir}/

## Conclusion
{'🎉 ALL TESTS PASSED! The SDK Dashboard is fully functional.' if self.test_results["failed_tests"] == 0 else f'⚠️ Some tests failed. Please review the {self.test_results["failed_tests"]} failed test(s).'}
"""
        
        # Save report
        report_path = self.screenshots_dir / "test_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"📄 Full report saved to: {report_path}")
    
    async def cleanup(self):
        """Cleanup resources."""
        print("🧹 Cleaning up...")
        
        if self.browser:
            await self.browser.close()
        
        if self.dashboard_process:
            self.dashboard_process.terminate()
            try:
                self.dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.dashboard_process.kill()
        
        print("✅ Cleanup complete")
    
    async def run_all_tests(self):
        """Run all tests in sequence."""
        try:
            await self.setup()
            
            # Basic functionality tests
            await self.test_server_connection()
            await self.test_dashboard_loading()
            await self.test_sdk_initialization()
            
            # Inference pipeline tests
            if self.test_results["sdk_loaded"]:
                await self.test_text_generation()
                await self.test_text_classification()
                await self.test_text_embeddings()
                await self.test_model_recommendations()
                await self.test_model_manager()
                await self.test_vision_models()
                await self.test_specialized_tools()
            else:
                print("❌ Skipping inference tests - SDK not loaded")
            
            # Generate final report
            await self.generate_summary_report()
            
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
        finally:
            await self.cleanup()

async def main():
    """Main function to run the browser tests."""
    if not HAVE_PLAYWRIGHT:
        print("❌ Playwright is required for browser automation testing")
        print("📦 Install with: pip install playwright && playwright install chromium")
        return False
    
    print("🤖 Starting SDK Dashboard Browser Automation Tests")
    print("=" * 60)
    
    tester = SDKBrowserTester()
    await tester.run_all_tests()
    
    return tester.test_results["failed_tests"] == 0

if __name__ == "__main__":
    success = anyio.run(main())
    sys.exit(0 if success else 1)