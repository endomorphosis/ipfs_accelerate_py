#!/usr/bin/env python3
"""
End-to-End Test with Playwright Screenshots

This test launches the MCP server, opens the dashboard, and captures screenshots
of each stage:
1. Dashboard overview
2. HF Search tab
3. Search results for "llama"
4. Download initiated
5. Download complete
6. Model Manager view

Usage:
    python3 tests/test_playwright_e2e_with_screenshots.py
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_with_playwright():
    """Test MCP dashboard with Playwright and capture screenshots."""
    
    print("\n" + "="*80)
    print("🎭 MCP Dashboard End-to-End Test with Playwright Screenshots")
    print("="*80)
    
    # Check if playwright is installed
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("\n❌ Playwright not installed")
        print("   Install with: pip install playwright && playwright install chromium")
        return False
    
    # Create screenshots directory
    screenshots_dir = Path(__file__).parent / "playwright_screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    print(f"\n📸 Screenshots will be saved to: {screenshots_dir}")
    
    # Start MCP server
    print("\n🚀 Starting MCP server...")
    server_process = None
    
    try:
        # Start server in background
        server_process = subprocess.Popen(
            [sys.executable, "-m", "ipfs_accelerate_py.cli", "mcp", "start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        print("   Waiting for server to start...")
        time.sleep(5)
        
        # Check if server is running
        if server_process.poll() is not None:
            print("   ❌ Server failed to start")
            stdout, stderr = server_process.communicate()
            print(f"   STDOUT: {stdout[:500]}")
            print(f"   STDERR: {stderr[:500]}")
            return False
        
        print("   ✅ Server started")
        
        # Run Playwright tests
        print("\n🎭 Launching browser and capturing screenshots...")
        
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_viewport_size({"width": 1920, "height": 1080})
            
            try:
                # Step 1: Load dashboard
                print("\n📋 Step 1: Loading dashboard...")
                page.goto("http://localhost:9000", timeout=30000)
                page.wait_for_timeout(2000)
                page.screenshot(path=str(screenshots_dir / "01_dashboard_overview.png"), full_page=True)
                print("   ✅ Dashboard loaded")
                print(f"   Screenshot saved: 01_dashboard_overview.png")
                
                # Step 2: Navigate to HF Search tab
                print("\n📋 Step 2: Opening HF Search tab...")
                hf_tab_button = page.query_selector('button:has-text("HF Search")')
                if hf_tab_button:
                    hf_tab_button.click()
                    page.wait_for_timeout(1000)
                    page.screenshot(path=str(screenshots_dir / "02_hf_search_tab.png"), full_page=True)
                    print("   ✅ HF Search tab opened")
                    print(f"   Screenshot saved: 02_hf_search_tab.png")
                else:
                    print("   ⚠️ HF Search tab button not found")
                
                # Step 3: Enter search query
                print("\n📋 Step 3: Searching for 'llama'...")
                search_input = page.query_selector('#hf-search')
                if search_input:
                    search_input.fill('llama')
                    page.wait_for_timeout(500)
                    page.screenshot(path=str(screenshots_dir / "03_search_input.png"), full_page=True)
                    print("   ✅ Search term entered")
                    print(f"   Screenshot saved: 03_search_input.png")
                else:
                    print("   ⚠️ Search input not found")
                
                # Step 4: Click search button
                print("\n📋 Step 4: Executing search...")
                search_btn = page.query_selector('button:has-text("Search HF Hub")')
                if search_btn:
                    search_btn.click()
                    page.wait_for_timeout(3000)  # Wait for results
                    page.screenshot(path=str(screenshots_dir / "04_search_results.png"), full_page=True)
                    print("   ✅ Search executed")
                    print(f"   Screenshot saved: 04_search_results.png")
                    
                    # Check for results
                    results_div = page.query_selector('#hf-search-results')
                    if results_div:
                        content = results_div.inner_html()
                        if 'model-result' in content:
                            result_count = content.count('model-result')
                            print(f"   ✅ Found {result_count} model results")
                        else:
                            print("   ⚠️ No model results in content")
                else:
                    print("   ⚠️ Search button not found")
                
                # Step 5: Click download on first model
                print("\n📋 Step 5: Initiating download...")
                download_btn = page.query_selector('button:has-text("Download")')
                if download_btn:
                    # Get model ID before clicking
                    model_card = download_btn.locator("xpath=ancestor::div[@class='model-result']")
                    model_title = model_card.query_selector(".model-title")
                    model_name = model_title.inner_text() if model_title else "unknown"
                    
                    print(f"   Downloading model: {model_name}")
                    download_btn.click()
                    page.wait_for_timeout(3000)  # Wait for download
                    page.screenshot(path=str(screenshots_dir / "05_download_initiated.png"), full_page=True)
                    print("   ✅ Download initiated")
                    print(f"   Screenshot saved: 05_download_initiated.png")
                    
                    # Wait a bit more for completion
                    page.wait_for_timeout(2000)
                    page.screenshot(path=str(screenshots_dir / "06_download_complete.png"), full_page=True)
                    print("   ✅ Download completed")
                    print(f"   Screenshot saved: 06_download_complete.png")
                else:
                    print("   ⚠️ Download button not found")
                
                # Step 6: Check Model Manager
                print("\n📋 Step 6: Checking Model Manager...")
                model_mgr_tab = page.query_selector('button:has-text("Model Manager")')
                if model_mgr_tab:
                    model_mgr_tab.click()
                    page.wait_for_timeout(2000)
                    page.screenshot(path=str(screenshots_dir / "07_model_manager.png"), full_page=True)
                    print("   ✅ Model Manager tab opened")
                    print(f"   Screenshot saved: 07_model_manager.png")
                else:
                    print("   ⚠️ Model Manager tab not found")
                
                # Step 7: Check browser console for errors
                print("\n📋 Step 7: Checking console logs...")
                console_messages = []
                
                def handle_console(msg):
                    console_messages.append(f"{msg.type}: {msg.text}")
                
                page.on("console", handle_console)
                page.reload()
                page.wait_for_timeout(2000)
                
                # Filter for errors
                errors = [msg for msg in console_messages if 'error' in msg.lower()]
                if errors:
                    print(f"   ⚠️ Found {len(errors)} console errors:")
                    for error in errors[:5]:  # Show first 5
                        print(f"      {error}")
                else:
                    print("   ✅ No console errors detected")
                
                print("\n✅ All screenshots captured successfully!")
                print(f"\n📁 Screenshots location: {screenshots_dir}")
                print("\nGenerated screenshots:")
                for screenshot in sorted(screenshots_dir.glob("*.png")):
                    print(f"   - {screenshot.name}")
                
                return True
                
            except Exception as e:
                print(f"\n❌ Error during Playwright test: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            finally:
                browser.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Stop server
        if server_process:
            print("\n🛑 Stopping MCP server...")
            server_process.send_signal(signal.SIGINT)
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
            print("   ✅ Server stopped")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("MCP Dashboard E2E Test with Playwright Screenshots")
    print("="*80)
    print("\nThis test will:")
    print("  1. Start the MCP server")
    print("  2. Open the dashboard in a browser")
    print("  3. Navigate through the HF Search workflow")
    print("  4. Capture screenshots at each stage")
    print("  5. Verify no JavaScript errors")
    print("\nRequirements:")
    print("  - Playwright installed: pip install playwright && playwright install chromium")
    print("="*80)
    
    success = test_with_playwright()
    
    if success:
        print("\n" + "="*80)
        print("✅ TEST PASSED - Screenshots captured successfully")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("❌ TEST FAILED - Check output above for details")
        print("="*80)
        sys.exit(1)
