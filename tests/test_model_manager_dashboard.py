#!/usr/bin/env python3
"""
Playwright Test for Model Manager Dashboard

Tests the Model Manager Browser tab functionality including:
- Loading and displaying models
- Searching and filtering models
- Viewing model details
- Using the MCP SDK integration
"""

import anyio
import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from playwright.async_api import async_playwright, Page, Browser, expect
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️ Playwright not available - install with: pip install playwright && playwright install")
    # Create stub types to prevent NameError during collection
    Page = None
    Browser = None
    async_playwright = None
    expect = None

import requests


class ModelManagerDashboardTest:
    """Test suite for Model Manager Dashboard."""
    
    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.server_url = "http://127.0.0.1:3001"
        self.screenshots_dir = Path("test_screenshots")
        self.test_results = {}
        
        # Create screenshots directory
        self.screenshots_dir.mkdir(exist_ok=True)
        
    async def start_server(self) -> bool:
        """Start the MCP dashboard server."""
        print("🚀 Starting MCP Dashboard server...")
        
        try:
            # Start server using the CLI command
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "ipfs_accelerate_py.cli_entry", "mcp", "start", "--port", "3001"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            
            # Wait for server to start
            for i in range(60):  # 60 second timeout
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=2)
                    if response.status_code == 200:
                        print("✅ Server started successfully")
                        time.sleep(2)  # Give it a moment to fully initialize
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
            
            print("❌ Server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop the MCP dashboard server."""
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
    
    async def test_page_load(self, page: Page):
        """Test that the dashboard page loads successfully."""
        print("\n📋 Testing page load...")
        
        try:
            await page.goto(self.server_url, wait_until="networkidle", timeout=30000)
            
            # Check page title
            title = await page.title()
            print(f"   Page title: {title}")
            
            # Check that main elements are present
            header = await page.query_selector("h1")
            assert header is not None, "Header not found"
            
            # Take screenshot
            await page.screenshot(path=self.screenshots_dir / "01_page_load.png")
            
            self.test_results['page_load'] = 'PASS'
            print("   ✅ Page load test passed")
            
        except Exception as e:
            self.test_results['page_load'] = f'FAIL: {e}'
            print(f"   ❌ Page load test failed: {e}")
            raise
    
    async def test_model_browser_tab(self, page: Page):
        """Test switching to the Model Browser tab."""
        print("\n📋 Testing Model Browser tab...")
        
        try:
            # Click on Model Browser tab
            model_browser_tab = await page.query_selector('button.tab-button:has-text("Model Browser")')
            assert model_browser_tab is not None, "Model Browser tab button not found"
            
            await model_browser_tab.click()
            await page.wait_for_timeout(1000)
            
            # Check that tab content is visible
            tab_content = await page.query_selector('#model-browser')
            assert tab_content is not None, "Model Browser tab content not found"
            
            # Check that tab is active
            is_visible = await tab_content.is_visible()
            assert is_visible, "Model Browser tab content not visible"
            
            # Take screenshot
            await page.screenshot(path=self.screenshots_dir / "02_model_browser_tab.png", full_page=True)
            
            self.test_results['model_browser_tab'] = 'PASS'
            print("   ✅ Model Browser tab test passed")
            
        except Exception as e:
            self.test_results['model_browser_tab'] = f'FAIL: {e}'
            print(f"   ❌ Model Browser tab test failed: {e}")
            await page.screenshot(path=self.screenshots_dir / "02_model_browser_tab_error.png")
            raise
    
    async def test_statistics_display(self, page: Page):
        """Test that model statistics are displayed."""
        print("\n📋 Testing statistics display...")
        
        try:
            # Wait for statistics to load
            await page.wait_for_selector('#mm-total-models', timeout=10000)
            
            # Check that stats are present
            total_models = await page.text_content('#mm-total-models')
            perf_models = await page.text_content('#mm-perf-models')
            compat_models = await page.text_content('#mm-compat-models')
            
            print(f"   Total models: {total_models}")
            print(f"   Performance models: {perf_models}")
            print(f"   Compatibility models: {compat_models}")
            
            # Verify we have some data (even if it's fallback)
            assert total_models is not None, "Total models not displayed"
            
            # Take screenshot
            await page.screenshot(path=self.screenshots_dir / "03_statistics.png")
            
            self.test_results['statistics_display'] = 'PASS'
            print("   ✅ Statistics display test passed")
            
        except Exception as e:
            self.test_results['statistics_display'] = f'FAIL: {e}'
            print(f"   ❌ Statistics display test failed: {e}")
            await page.screenshot(path=self.screenshots_dir / "03_statistics_error.png")
    
    async def test_models_list_load(self, page: Page):
        """Test that models list loads."""
        print("\n📋 Testing models list load...")
        
        try:
            # Wait for models list to load (or show error)
            await page.wait_for_selector('#mm-models-list', timeout=10000)
            
            # Check for either models or an error message
            models_list = await page.query_selector('#mm-models-list')
            content = await models_list.inner_html()
            
            # Check if we have model cards or a message
            has_models = 'model-card' in content
            has_message = 'alert' in content
            
            assert has_models or has_message, "No models or message displayed"
            
            if has_models:
                # Count model cards
                model_cards = await page.query_selector_all('.model-card')
                print(f"   Found {len(model_cards)} model cards")
            else:
                print("   No models found (expected if fallback database is empty)")
            
            # Take screenshot
            await page.screenshot(path=self.screenshots_dir / "04_models_list.png", full_page=True)
            
            self.test_results['models_list_load'] = 'PASS'
            print("   ✅ Models list load test passed")
            
        except Exception as e:
            self.test_results['models_list_load'] = f'FAIL: {e}'
            print(f"   ❌ Models list load test failed: {e}")
            await page.screenshot(path=self.screenshots_dir / "04_models_list_error.png")
    
    async def test_search_functionality(self, page: Page):
        """Test model search functionality."""
        print("\n📋 Testing search functionality...")
        
        try:
            # Find search input
            search_input = await page.query_selector('#mm-search-input')
            assert search_input is not None, "Search input not found"
            
            # Enter search term
            await search_input.fill('bert')
            
            # Click search button
            search_btn = await page.query_selector('#mm-search-btn')
            assert search_btn is not None, "Search button not found"
            await search_btn.click()
            
            # Wait for results
            await page.wait_for_timeout(2000)
            
            # Take screenshot
            await page.screenshot(path=self.screenshots_dir / "05_search_results.png", full_page=True)
            
            self.test_results['search_functionality'] = 'PASS'
            print("   ✅ Search functionality test passed")
            
        except Exception as e:
            self.test_results['search_functionality'] = f'FAIL: {e}'
            print(f"   ❌ Search functionality test failed: {e}")
            await page.screenshot(path=self.screenshots_dir / "05_search_error.png")
    
    async def test_filter_functionality(self, page: Page):
        """Test filter dropdowns functionality."""
        print("\n📋 Testing filter functionality...")
        
        try:
            # Test task filter
            task_filter = await page.query_selector('#mm-task-filter')
            assert task_filter is not None, "Task filter not found"
            await task_filter.select_option('text-classification')
            
            await page.wait_for_timeout(1000)
            
            # Test hardware filter
            hardware_filter = await page.query_selector('#mm-hardware-filter')
            assert hardware_filter is not None, "Hardware filter not found"
            await hardware_filter.select_option('cpu')
            
            await page.wait_for_timeout(2000)
            
            # Take screenshot
            await page.screenshot(path=self.screenshots_dir / "06_filters.png", full_page=True)
            
            self.test_results['filter_functionality'] = 'PASS'
            print("   ✅ Filter functionality test passed")
            
        except Exception as e:
            self.test_results['filter_functionality'] = f'FAIL: {e}'
            print(f"   ❌ Filter functionality test failed: {e}")
            await page.screenshot(path=self.screenshots_dir / "06_filters_error.png")
    
    async def test_model_details(self, page: Page):
        """Test viewing model details."""
        print("\n📋 Testing model details view...")
        
        try:
            # Look for a details button
            details_btn = await page.query_selector('button:has-text("Details")')
            
            if details_btn:
                await details_btn.click()
                await page.wait_for_timeout(1000)
                
                # Check for modal
                modal = await page.query_selector('.modal-overlay')
                if modal:
                    print("   Modal opened successfully")
                    
                    # Take screenshot
                    await page.screenshot(path=self.screenshots_dir / "07_model_details.png", full_page=True)
                    
                    # Close modal
                    close_btn = await page.query_selector('.modal-close')
                    if close_btn:
                        await close_btn.click()
                        await page.wait_for_timeout(500)
                else:
                    print("   No modal found (model details might use different display)")
            else:
                print("   No details button found (may not have models loaded)")
            
            self.test_results['model_details'] = 'PASS'
            print("   ✅ Model details test passed")
            
        except Exception as e:
            self.test_results['model_details'] = f'FAIL: {e}'
            print(f"   ❌ Model details test failed: {e}")
            await page.screenshot(path=self.screenshots_dir / "07_model_details_error.png")
    
    async def test_refresh_button(self, page: Page):
        """Test the refresh button."""
        print("\n📋 Testing refresh button...")
        
        try:
            # Click refresh button
            refresh_btn = await page.query_selector('#mm-refresh-btn')
            assert refresh_btn is not None, "Refresh button not found"
            await refresh_btn.click()
            
            # Wait for refresh
            await page.wait_for_timeout(2000)
            
            # Take screenshot
            await page.screenshot(path=self.screenshots_dir / "08_refresh.png", full_page=True)
            
            self.test_results['refresh_button'] = 'PASS'
            print("   ✅ Refresh button test passed")
            
        except Exception as e:
            self.test_results['refresh_button'] = f'FAIL: {e}'
            print(f"   ❌ Refresh button test failed: {e}")
            await page.screenshot(path=self.screenshots_dir / "08_refresh_error.png")
    
    async def run_all_tests(self):
        """Run all Playwright tests."""
        if not PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright not available - skipping tests")
            return False
        
        print("🎭 Starting Playwright tests for Model Manager Dashboard...")
        
        # Start server
        if not await self.start_server():
            print("❌ Failed to start server")
            return False
        
        try:
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 720}
                )
                page = await context.new_page()
                
                # Run tests
                await self.test_page_load(page)
                await self.test_model_browser_tab(page)
                await self.test_statistics_display(page)
                await self.test_models_list_load(page)
                await self.test_search_functionality(page)
                await self.test_filter_functionality(page)
                await self.test_model_details(page)
                await self.test_refresh_button(page)
                
                await browser.close()
                
            print("\n📊 Test Results:")
            print("=" * 50)
            for test_name, result in self.test_results.items():
                status_icon = "✅" if result == 'PASS' else "❌"
                print(f"{status_icon} {test_name}: {result}")
            print("=" * 50)
            
            # Count passes and fails
            passes = sum(1 for r in self.test_results.values() if r == 'PASS')
            total = len(self.test_results)
            print(f"\n📈 Summary: {passes}/{total} tests passed")
            
            return passes == total
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False
            
        finally:
            self.stop_server()


async def main():
    """Main test runner."""
    tester = ModelManagerDashboardTest()
    success = await tester.run_all_tests()
    
    print(f"\n{'='*50}")
    if success:
        print("✅ All Model Manager Dashboard tests passed!")
        print(f"📸 Screenshots saved to: {tester.screenshots_dir}")
        return 0
    else:
        print("❌ Some tests failed - check screenshots for details")
        print(f"📸 Screenshots saved to: {tester.screenshots_dir}")
        return 1


if __name__ == '__main__':
    exit_code = anyio.run(main())
    sys.exit(exit_code)
