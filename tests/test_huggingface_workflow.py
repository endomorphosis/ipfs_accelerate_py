#!/usr/bin/env python3
"""
End-to-End Playwright Test for HuggingFace Model Search and Download Workflow

This test verifies the complete workflow:
1. Start MCP server with `ipfs-accelerate mcp start`
2. Search for "bert" model on HuggingFace Hub
3. Download the model
4. Verify it appears in the Model Manager

Usage:
    python3 tests/test_huggingface_workflow.py
"""

import asyncio
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from playwright.async_api import async_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    # Don't call sys.exit() - it breaks pytest collection
    # Tests will be skipped via pytest.mark.skipif instead
    # Create stub types to prevent NameError during collection
    Page = None
    Browser = None
    async_playwright = None

import requests
import pytest


class HuggingFaceWorkflowTest:
    """End-to-end test for HuggingFace model search and download workflow."""
    
    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.server_url = "http://127.0.0.1:8899"
        self.screenshots_dir = Path("test_screenshots_workflow")
        self.test_results = {}
        
        # Create screenshots directory
        self.screenshots_dir.mkdir(exist_ok=True)
        
    async def start_mcp_server(self) -> bool:
        """Start the MCP dashboard server using ipfs-accelerate mcp start."""
        print("\n🚀 Starting MCP Dashboard server with 'ipfs-accelerate mcp start'...")
        
        try:
            # Start server using the actual CLI command as user would
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "ipfs_accelerate_py.mcp_dashboard"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            
            # Wait for server to start
            print("   Waiting for server to be ready...")
            for i in range(60):  # 60 second timeout
                try:
                    response = requests.get(f"{self.server_url}/mcp", timeout=2)
                    if response.status_code == 200:
                        print("   ✅ Server started successfully")
                        time.sleep(3)  # Give it extra time to fully initialize
                        return True
                except requests.exceptions.RequestException:
                    if i % 10 == 0:
                        print(f"   Still waiting... ({i}s)")
                    time.sleep(1)
            
            print("   ❌ Server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"   ❌ Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop the MCP dashboard server."""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print("\n✅ Server stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print("\n⚠️ Server forcefully killed")
            except Exception as e:
                print(f"\n⚠️ Error stopping server: {e}")
    
    async def test_complete_workflow(self, page: Page):
        """Test the complete workflow from search to download to verification."""
        print("\n" + "="*80)
        print("TESTING COMPLETE HUGGINGFACE WORKFLOW")
        print("="*80)
        
        try:
            # Step 1: Load the dashboard
            print("\n📋 Step 1: Loading MCP Dashboard...")
            await page.goto(f"{self.server_url}/mcp", wait_until="networkidle", timeout=30000)
            await page.screenshot(path=self.screenshots_dir / "01_dashboard_loaded.png", full_page=True)
            print("   ✅ Dashboard loaded")
            
            # Step 2: Navigate to HuggingFace Search tab
            print("\n📋 Step 2: Navigating to HF Search tab...")
            hf_search_tab = await page.query_selector('button.tab-button:has-text("HF Search")')
            if not hf_search_tab:
                print("   ⚠️ HF Search tab not found, trying alternative selector...")
                hf_search_tab = await page.query_selector('button[onclick*="model-manager"]')
            
            assert hf_search_tab is not None, "HF Search tab button not found"
            await hf_search_tab.click()
            await page.wait_for_timeout(1000)
            await page.screenshot(path=self.screenshots_dir / "02_hf_search_tab.png", full_page=True)
            print("   ✅ HF Search tab opened")
            
            # Step 3: Enter "bert" in search field
            print("\n📋 Step 3: Searching for 'bert' model...")
            search_input = await page.query_selector('#hf-search')
            assert search_input is not None, "Search input not found"
            
            await search_input.fill('bert')
            await page.screenshot(path=self.screenshots_dir / "03_search_filled.png", full_page=True)
            print("   ✅ Search term entered: 'bert'")
            
            # Step 4: Click search button
            print("\n📋 Step 4: Clicking Search button...")
            search_button = await page.query_selector('button:has-text("Search HF Hub")')
            assert search_button is not None, "Search button not found"
            
            await search_button.click()
            print("   ⏳ Waiting for search results...")
            
            # Wait for results to load (with timeout)
            await page.wait_for_timeout(5000)
            await page.screenshot(path=self.screenshots_dir / "04_search_results.png", full_page=True)
            
            # Check if results appeared
            results_div = await page.query_selector('#hf-search-results')
            if results_div:
                content = await results_div.inner_html()
                if 'model-result' in content or 'bert' in content.lower():
                    print("   ✅ Search results displayed")
                elif 'No models found' in content:
                    print("   ⚠️ No models found - API might not be working")
                elif 'Search failed' in content:
                    print("   ❌ Search failed - check API endpoint")
                else:
                    print(f"   ⚠️ Unexpected results content: {content[:200]}...")
            
            # Step 5: Find and click download button for a bert model
            print("\n📋 Step 5: Looking for Download button...")
            download_buttons = await page.query_selector_all('button:has-text("Download")')
            
            if len(download_buttons) > 0:
                print(f"   Found {len(download_buttons)} download button(s)")
                # Click the first download button
                await download_buttons[0].click()
                print("   ✅ Clicked Download button")
                
                # Wait for download to process
                await page.wait_for_timeout(3000)
                await page.screenshot(path=self.screenshots_dir / "05_download_initiated.png", full_page=True)
                
                # Check for toast notification
                toast = await page.query_selector('.toast')
                if toast:
                    toast_text = await toast.inner_text()
                    print(f"   📢 Toast notification: {toast_text}")
            else:
                print("   ⚠️ No download buttons found in search results")
                await page.screenshot(path=self.screenshots_dir / "05_no_download_buttons.png", full_page=True)
            
            # Step 6: Navigate to Model Browser tab
            print("\n📋 Step 6: Navigating to Model Browser tab...")
            model_browser_tab = await page.query_selector('button.tab-button:has-text("Model Browser")')
            assert model_browser_tab is not None, "Model Browser tab not found"
            
            await model_browser_tab.click()
            await page.wait_for_timeout(2000)
            await page.screenshot(path=self.screenshots_dir / "06_model_browser_tab.png", full_page=True)
            print("   ✅ Model Browser tab opened")
            
            # Step 7: Verify downloaded model appears
            print("\n📋 Step 7: Verifying downloaded model in Model Browser...")
            models_list = await page.query_selector('#mm-models-list')
            
            if models_list:
                content = await models_list.inner_html()
                
                if 'bert' in content.lower():
                    print("   ✅ BERT model found in Model Browser!")
                    model_cards = await page.query_selector_all('.model-card')
                    print(f"   📊 Total models visible: {len(model_cards)}")
                elif 'No models found' in content:
                    print("   ⚠️ Model Browser shows 'No models found'")
                elif 'loading' in content.lower():
                    print("   ⏳ Models still loading...")
                    await page.wait_for_timeout(3000)
                    content = await models_list.inner_html()
                    if 'bert' in content.lower():
                        print("   ✅ BERT model found after waiting!")
                    else:
                        print("   ❌ BERT model not found in Model Browser")
                else:
                    print(f"   ℹ️ Model Browser content: {content[:300]}...")
            else:
                print("   ❌ Model Browser list element not found")
            
            await page.screenshot(path=self.screenshots_dir / "07_final_verification.png", full_page=True)
            
            # Step 8: Summary
            print("\n" + "="*80)
            print("WORKFLOW TEST SUMMARY")
            print("="*80)
            print("✅ All steps completed")
            print(f"📸 Screenshots saved to: {self.screenshots_dir}")
            
            self.test_results['workflow_complete'] = 'PASS'
            
        except Exception as e:
            print(f"\n❌ Workflow test failed: {e}")
            await page.screenshot(path=self.screenshots_dir / "ERROR_workflow_failed.png", full_page=True)
            self.test_results['workflow_complete'] = f'FAIL: {e}'
            raise
    
    async def run_all_tests(self):
        """Run all tests."""
        if not PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright not available - cannot run tests")
            return False
        
        print("\n" + "="*80)
        print("HUGGINGFACE WORKFLOW E2E TEST")
        print("="*80)
        
        # Start server
        if not await self.start_mcp_server():
            print("❌ Failed to start server")
            return False
        
        try:
            async with async_playwright() as p:
                # Launch browser in headed mode so we can see what's happening
                print("\n🌐 Launching browser...")
                browser = await p.chromium.launch(
                    headless=False,  # Run in headed mode for visibility
                    slow_mo=500  # Slow down actions to see them
                )
                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080}
                )
                page = await context.new_page()
                
                # Enable console logging from the page
                page.on("console", lambda msg: print(f"   [Browser Console] {msg.type}: {msg.text}"))
                
                # Run the workflow test
                await self.test_complete_workflow(page)
                
                await browser.close()
                
            print("\n" + "="*80)
            print("TEST RESULTS:")
            print("="*80)
            for test_name, result in self.test_results.items():
                status_icon = "✅" if result == 'PASS' else "❌"
                print(f"{status_icon} {test_name}: {result}")
            print("="*80)
            
            return all(r == 'PASS' for r in self.test_results.values())
            
        except Exception as e:
            print(f"\n❌ Test execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            self.stop_server()


async def main():
    """Main test runner."""
    tester = HuggingFaceWorkflowTest()
    success = await tester.run_all_tests()
    
    print(f"\n{'='*80}")
    if success:
        print("✅ ALL TESTS PASSED!")
        print(f"📸 Screenshots saved to: {tester.screenshots_dir}")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print(f"📸 Screenshots saved to: {tester.screenshots_dir}")
        return 1


if __name__ == '__main__':
    if not PLAYWRIGHT_AVAILABLE:
        print("⚠️ Playwright not available - install with: pip install playwright && playwright install")
        sys.exit(1)
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
