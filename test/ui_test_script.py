#!/usr/bin/env python3
"""
Kitchen Sink UI Testing Script with Playwright
Tests the UI/UX and captures screenshots for debugging and improvement
"""

import anyio
import os
import sys
import json
import time
import subprocess
import signal
from pathlib import Path
from threading import Thread
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

try:
    from playwright.async_api import async_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️ Playwright not available - will run basic server tests")

import requests
import tempfile

class KitchenSinkUITester:
    """Comprehensive UI testing for the Kitchen Sink AI interface."""
    
    def __init__(self):
        self.server_process = None
        self.server_url = "http://127.0.0.1:8080"
        self.screenshots_dir = Path("data/test_screenshots/ui_test")
        self.console_logs = []
        self.error_logs = []
        self.test_results = {}
        
        # Create screenshots directory
        self.screenshots_dir.mkdir(exist_ok=True)
        
    async def start_server(self):
        """Start the kitchen sink server."""
        print("🚀 Starting Kitchen Sink server...")
        
        try:
            # Start server in background
            self.server_process = subprocess.Popen([
                sys.executable, "kitchen_sink_demo.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
            
            # Wait for server to start
            for i in range(30):  # 30 second timeout
                try:
                    response = requests.get(f"{self.server_url}/", timeout=2)
                    if response.status_code == 200:
                        print("✅ Server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
            
            print("❌ Server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop the kitchen sink server."""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print("✅ Server stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print("⚠️ Server forcefully killed")
            except Exception as e:
                print(f"⚠️ Error stopping server: {e}")
    
    async def capture_console_logs(self, page: Page):
        """Capture console messages and errors."""
        
        def handle_console(msg):
            log_entry = {
                "type": msg.type,
                "text": msg.text,
                "location": getattr(msg.location, "url", "unknown") if msg.location else "unknown",
                "timestamp": time.time()
            }
            self.console_logs.append(log_entry)
            print(f"📝 Console [{msg.type}]: {msg.text}")
            
        def handle_page_error(error):
            error_entry = {
                "message": str(error),
                "timestamp": time.time()
            }
            self.error_logs.append(error_entry)
            print(f"❌ Page Error: {error}")
        
        page.on("console", handle_console)
        page.on("pageerror", handle_page_error)
    
    async def take_screenshot(self, page: Page, name: str, full_page: bool = True):
        """Take a screenshot with the given name."""
        try:
            screenshot_path = self.screenshots_dir / f"{name}.png"
            await page.screenshot(path=str(screenshot_path), full_page=full_page)
            print(f"📸 Screenshot saved: {screenshot_path}")
            return str(screenshot_path)
        except Exception as e:
            print(f"❌ Failed to take screenshot {name}: {e}")
            return None
    
    async def test_page_load(self, page: Page):
        """Test initial page load and capture screenshots."""
        print("🔍 Testing page load...")
        
        try:
            # Navigate to the main page
            await page.goto(self.server_url, wait_until="networkidle")
            await page.wait_for_timeout(2000)  # Wait for dynamic content
            
            # Take initial screenshot
            await self.take_screenshot(page, "01_initial_load")
            
            # Check for basic elements
            title = await page.title()
            print(f"📄 Page title: {title}")
            
            # Check if tabs are present
            tabs = await page.query_selector_all(".nav-link")
            print(f"📋 Found {len(tabs)} tabs")
            
            # Check for status message
            status_element = await page.query_selector("#status-message")
            if status_element:
                status_text = await status_element.inner_text()
                print(f"ℹ️ Status: {status_text}")
            
            self.test_results["page_load"] = {
                "success": True,
                "title": title,
                "tab_count": len(tabs),
                "status": status_text if status_element else "No status"
            }
            
        except Exception as e:
            print(f"❌ Page load test failed: {e}")
            self.test_results["page_load"] = {"success": False, "error": str(e)}
    
    async def test_text_generation_tab(self, page: Page):
        """Test the text generation tab functionality."""
        print("🔍 Testing Text Generation tab...")
        
        try:
            # Click on text generation tab (should already be active)
            generation_tab = await page.query_selector("#generation-tab")
            if generation_tab:
                await generation_tab.click()
                await page.wait_for_timeout(1000)
            
            await self.take_screenshot(page, "02_text_generation_tab")
            
            # Test model autocomplete
            model_input = await page.query_selector("#gen-model")
            if model_input:
                await model_input.fill("gpt")
                await page.wait_for_timeout(2000)  # Wait for autocomplete
                await self.take_screenshot(page, "03_model_autocomplete")
                await model_input.fill("")  # Clear for auto-selection test
            
            # Fill in the prompt
            prompt_input = await page.query_selector("#gen-prompt")
            if prompt_input:
                await prompt_input.fill("Write a short story about a robot")
            
            # Adjust parameters
            max_length_slider = await page.query_selector("#gen-max-length")
            if max_length_slider:
                await max_length_slider.fill("50")
            
            temperature_slider = await page.query_selector("#gen-temperature")
            if temperature_slider:
                await temperature_slider.fill("0.8")
            
            await self.take_screenshot(page, "04_generation_form_filled")
            
            # Submit the form
            submit_button = await page.query_selector("#generation-form button[type='submit']")
            if submit_button:
                await submit_button.click()
                await page.wait_for_timeout(3000)  # Wait for response
                await self.take_screenshot(page, "05_generation_result")
            
            # Check if result appeared
            result_element = await page.query_selector("#generation-result")
            result_text = ""
            if result_element:
                result_text = await result_element.inner_text()
                print(f"📝 Generation result: {result_text[:100]}...")
            
            self.test_results["text_generation"] = {
                "success": bool(result_text),
                "result_length": len(result_text),
                "form_filled": True
            }
            
        except Exception as e:
            print(f"❌ Text generation test failed: {e}")
            self.test_results["text_generation"] = {"success": False, "error": str(e)}
    
    async def test_classification_tab(self, page: Page):
        """Test the text classification tab."""
        print("🔍 Testing Text Classification tab...")
        
        try:
            # Click on classification tab
            classification_tab = await page.query_selector("#classification-tab")
            if classification_tab:
                await classification_tab.click()
                await page.wait_for_timeout(1000)
            
            await self.take_screenshot(page, "06_classification_tab")
            
            # Fill in text to classify
            text_input = await page.query_selector("#class-text")
            if text_input:
                await text_input.fill("This product is absolutely amazing! I love it so much.")
            
            await self.take_screenshot(page, "07_classification_form_filled")
            
            # Submit the form
            submit_button = await page.query_selector("#classification-form button[type='submit']")
            if submit_button:
                await submit_button.click()
                await page.wait_for_timeout(3000)
                await self.take_screenshot(page, "08_classification_result")
            
            # Check for result
            result_element = await page.query_selector("#classification-result")
            result_text = ""
            if result_element:
                result_text = await result_element.inner_text()
                print(f"🏷️ Classification result: {result_text[:100]}...")
            
            self.test_results["classification"] = {
                "success": bool(result_text),
                "result_found": bool(result_text)
            }
            
        except Exception as e:
            print(f"❌ Classification test failed: {e}")
            self.test_results["classification"] = {"success": False, "error": str(e)}
    
    async def test_embeddings_tab(self, page: Page):
        """Test the embeddings tab."""
        print("🔍 Testing Text Embeddings tab...")
        
        try:
            # Click on embeddings tab
            embeddings_tab = await page.query_selector("#embeddings-tab")
            if embeddings_tab:
                await embeddings_tab.click()
                await page.wait_for_timeout(1000)
            
            await self.take_screenshot(page, "09_embeddings_tab")
            
            # Fill in text to embed
            text_input = await page.query_selector("#embed-text")
            if text_input:
                await text_input.fill("Machine learning is transforming technology")
            
            # Submit the form
            submit_button = await page.query_selector("#embeddings-form button[type='submit']")
            if submit_button:
                await submit_button.click()
                await page.wait_for_timeout(3000)
                await self.take_screenshot(page, "10_embeddings_result")
            
            # Check for result
            result_element = await page.query_selector("#embeddings-result")
            result_text = ""
            if result_element:
                result_text = await result_element.inner_text()
                print(f"🔢 Embeddings result: {result_text[:100]}...")
            
            self.test_results["embeddings"] = {
                "success": bool(result_text),
                "result_found": bool(result_text)
            }
            
        except Exception as e:
            print(f"❌ Embeddings test failed: {e}")
            self.test_results["embeddings"] = {"success": False, "error": str(e)}
    
    async def test_recommendations_tab(self, page: Page):
        """Test the recommendations tab."""
        print("🔍 Testing Model Recommendations tab...")
        
        try:
            # Click on recommendations tab
            recommendations_tab = await page.query_selector("#recommendations-tab")
            if recommendations_tab:
                await recommendations_tab.click()
                await page.wait_for_timeout(1000)
            
            await self.take_screenshot(page, "11_recommendations_tab")
            
            # Submit the form with default values
            submit_button = await page.query_selector("#recommendations-form button[type='submit']")
            if submit_button:
                await submit_button.click()
                await page.wait_for_timeout(3000)
                await self.take_screenshot(page, "12_recommendations_result")
            
            # Check for result
            result_element = await page.query_selector("#recommendations-result")
            result_text = ""
            if result_element:
                result_text = await result_element.inner_text()
                print(f"🎯 Recommendations result: {result_text[:100]}...")
            
            self.test_results["recommendations"] = {
                "success": bool(result_text),
                "result_found": bool(result_text)
            }
            
        except Exception as e:
            print(f"❌ Recommendations test failed: {e}")
            self.test_results["recommendations"] = {"success": False, "error": str(e)}
    
    async def test_models_tab(self, page: Page):
        """Test the models management tab."""
        print("🔍 Testing Models tab...")
        
        try:
            # Click on models tab
            models_tab = await page.query_selector("#models-tab")
            if models_tab:
                await models_tab.click()
                await page.wait_for_timeout(1000)
            
            await self.take_screenshot(page, "13_models_tab")
            
            # Check if models are listed
            table_body = await page.query_selector("#models-table-body")
            rows = []
            if table_body:
                rows = await table_body.query_selector_all("tr")
            
            print(f"📊 Found {len(rows)} model entries")
            
            # Test search functionality
            search_input = await page.query_selector("#model-search")
            if search_input:
                await search_input.fill("gpt")
                await page.wait_for_timeout(1000)
                await self.take_screenshot(page, "14_models_search")
                await search_input.fill("")  # Clear search
            
            self.test_results["models"] = {
                "success": True,
                "model_count": len(rows),
                "search_functional": bool(search_input)
            }
            
        except Exception as e:
            print(f"❌ Models test failed: {e}")
            self.test_results["models"] = {"success": False, "error": str(e)}
    
    async def test_responsive_design(self, page: Page):
        """Test responsive design at different screen sizes."""
        print("🔍 Testing responsive design...")
        
        try:
            # Test mobile view
            await page.set_viewport_size({"width": 375, "height": 667})
            await page.wait_for_timeout(1000)
            await self.take_screenshot(page, "15_mobile_view")
            
            # Test tablet view
            await page.set_viewport_size({"width": 768, "height": 1024})
            await page.wait_for_timeout(1000)
            await self.take_screenshot(page, "16_tablet_view")
            
            # Test desktop view
            await page.set_viewport_size({"width": 1920, "height": 1080})
            await page.wait_for_timeout(1000)
            await self.take_screenshot(page, "17_desktop_view")
            
            self.test_results["responsive"] = {"success": True}
            
        except Exception as e:
            print(f"❌ Responsive test failed: {e}")
            self.test_results["responsive"] = {"success": False, "error": str(e)}
    
    async def run_playwright_tests(self):
        """Run comprehensive Playwright tests."""
        if not PLAYWRIGHT_AVAILABLE:
            print("⚠️ Playwright not available - skipping browser tests")
            return
        
        print("🎭 Starting Playwright tests...")
        
        async with async_playwright() as p:
            try:
                # Try to launch browser
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    user_agent="Mozilla/5.0 (compatible; KitchenSinkTester/1.0)"
                )
                page = await context.new_page()
                
                # Set up console logging
                await self.capture_console_logs(page)
                
                # Run tests
                await self.test_page_load(page)
                await self.test_text_generation_tab(page)
                await self.test_classification_tab(page)
                await self.test_embeddings_tab(page)
                await self.test_recommendations_tab(page)
                await self.test_models_tab(page)
                await self.test_responsive_design(page)
                
                await browser.close()
                print("✅ Playwright tests completed")
                
            except Exception as e:
                print(f"❌ Playwright tests failed: {e}")
                # Try with minimal browser setup
                try:
                    browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
                    context = await browser.new_context()
                    page = await context.new_page()
                    
                    await self.test_page_load(page)
                    await self.take_screenshot(page, "basic_page_screenshot")
                    
                    await browser.close()
                    print("✅ Basic Playwright test completed")
                except Exception as e2:
                    print(f"❌ Even basic Playwright test failed: {e2}")
    
    def run_basic_api_tests(self):
        """Run basic API tests without browser."""
        print("🔧 Running basic API tests...")
        
        try:
            # Test models endpoint
            response = requests.get(f"{self.server_url}/api/models", timeout=10)
            print(f"📊 Models API: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"📈 Found {len(data.get('models', []))} models")
            
            # Test search endpoint
            response = requests.get(f"{self.server_url}/api/models/search?q=gpt", timeout=10)
            print(f"🔍 Search API: {response.status_code}")
            
            # Test inference endpoints
            test_data = {"prompt": "Test prompt", "max_length": 50}
            response = requests.post(f"{self.server_url}/api/inference/generate", 
                                   json=test_data, timeout=10)
            print(f"🤖 Generation API: {response.status_code}")
            
            self.test_results["api_tests"] = {"success": True}
            
        except Exception as e:
            print(f"❌ API tests failed: {e}")
            self.test_results["api_tests"] = {"success": False, "error": str(e)}
    
    def analyze_results(self):
        """Analyze test results and provide improvement recommendations."""
        print("\n" + "="*60)
        print("📊 TEST RESULTS ANALYSIS")
        print("="*60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                              if result.get("success", False))
        
        print(f"✅ Successful tests: {successful_tests}/{total_tests}")
        print(f"❌ Failed tests: {total_tests - successful_tests}/{total_tests}")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅" if result.get("success", False) else "❌"
            print(f"{status} {test_name}: {result}")
        
        print(f"\n📝 Console logs captured: {len(self.console_logs)}")
        print(f"❌ Error logs captured: {len(self.error_logs)}")
        
        if self.console_logs:
            print("\nConsole Messages:")
            for log in self.console_logs[:10]:  # Show first 10
                print(f"  {log['type']}: {log['text'][:100]}")
        
        if self.error_logs:
            print("\nError Messages:")
            for error in self.error_logs:
                print(f"  ❌ {error['message']}")
        
        # Generate improvement recommendations
        self.generate_improvement_recommendations()
    
    def generate_improvement_recommendations(self):
        """Generate UX/UI improvement recommendations based on test results."""
        print("\n" + "="*60)
        print("💡 UI/UX IMPROVEMENT RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        # Check for failed tests
        for test_name, result in self.test_results.items():
            if not result.get("success", False):
                recommendations.append(f"Fix {test_name} functionality - {result.get('error', 'Unknown error')}")
        
        # Check console errors
        if self.error_logs:
            recommendations.append("Fix JavaScript errors found in console")
        
        # Check specific functionality
        if "page_load" in self.test_results:
            page_result = self.test_results["page_load"]
            if page_result.get("tab_count", 0) < 5:
                recommendations.append("Ensure all 5 tabs are properly rendered")
        
        # General UX improvements
        recommendations.extend([
            "Add loading indicators for better user feedback",
            "Implement better error handling and user-friendly error messages",
            "Add keyboard navigation support for accessibility",
            "Implement progressive enhancement for slower connections",
            "Add visual feedback for form submissions",
            "Improve mobile responsiveness for smaller screens",
            "Add tooltips for better user guidance",
            "Implement dark mode toggle for better user experience"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Save recommendations to file
        with open("ui_improvement_recommendations.json", "w") as f:
            json.dump({
                "test_results": self.test_results,
                "console_logs": self.console_logs,
                "error_logs": self.error_logs,
                "recommendations": recommendations,
                "timestamp": time.time()
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to ui_improvement_recommendations.json")
    
    async def run_all_tests(self):
        """Run all available tests."""
        print("🚀 Starting comprehensive UI testing...")
        
        # Start server
        if not await self.start_server():
            print("❌ Cannot test UI without server")
            return
        
        try:
            # Run basic API tests
            self.run_basic_api_tests()
            
            # Run Playwright tests if available
            await self.run_playwright_tests()
            
        finally:
            # Stop server
            self.stop_server()
        
        # Analyze and report results
        self.analyze_results()

async def main():
    """Main test execution function."""
    tester = KitchenSinkUITester()
    await tester.run_all_tests()

if __name__ == "__main__":
    anyio.run(main())