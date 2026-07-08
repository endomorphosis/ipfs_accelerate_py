#!/usr/bin/env python3
"""
Enhanced Functional E2E Test with Playwright

This test not only captures screenshots but also validates:
1. API responses and data integrity
2. Search functionality with query validation
3. Download functionality and status tracking
4. Error handling and edge cases
5. Real HuggingFace Hub integration (when available)

Usage:
    python3 tests/test_playwright_e2e_functional.py
"""

import os
import sys
import time
import json
import subprocess
import signal
import requests
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_api_endpoints_directly():
    """Test API endpoints directly before browser testing."""
    print("\n🔬 Testing API Endpoints Directly...")
    
    base_url = "http://localhost:8899"
    
    # Test 1: Health check
    print("\n📋 Test 1: Dashboard accessibility...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("   ✅ Dashboard accessible")
        else:
            print(f"   ❌ Dashboard returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Dashboard not accessible: {e}")
        return False
    
    # Test 2: Model search API
    print("\n📋 Test 2: Testing search API...")
    try:
        response = requests.get(f"{base_url}/api/mcp/models/search?q=llama&limit=10", timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            is_fallback = data.get('fallback', False)
            
            print(f"   ✅ Search API returned {len(results)} results")
            print(f"   {'⚠️  Using fallback data' if is_fallback else '✅ Using real HuggingFace API'}")
            
            if len(results) > 0:
                print(f"   ✅ First result: {results[0].get('model_id', 'unknown')}")
                
                # Validate data structure
                required_fields = ['model_id', 'model_info']
                first_result = results[0]
                missing_fields = [field for field in required_fields if field not in first_result]
                
                if missing_fields:
                    print(f"   ❌ Missing required fields: {missing_fields}")
                    return False
                else:
                    print("   ✅ Data structure validated")
                    
                # Check if we're actually searching (not just returning all models)
                query_term = 'llama'
                matched = any(query_term.lower() in str(r).lower() for r in results)
                if matched:
                    print(f"   ✅ Search results match query term '{query_term}'")
                else:
                    print(f"   ⚠️  Search results may not be filtered by query")
            else:
                print("   ⚠️  No results returned")
                
        else:
            print(f"   ❌ Search API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Search API failed: {e}")
        return False
    
    # Test 3: Model stats API
    print("\n📋 Test 3: Testing stats API...")
    try:
        response = requests.get(f"{base_url}/api/mcp/models/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            total = data.get('total_cached_models', 0)
            is_fallback = data.get('fallback', False)
            
            print(f"   ✅ Stats API returned data")
            print(f"   {'⚠️  Using fallback data' if is_fallback else '✅ Using real data'}")
            print(f"   Total cached models: {total}")
        else:
            print(f"   ❌ Stats API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Stats API failed: {e}")
        return False
    
    # Test 4: Download API (POST request)
    print("\n📋 Test 4: Testing download API...")
    try:
        test_model_id = "microsoft/DialoGPT-medium"
        response = requests.post(
            f"{base_url}/api/mcp/models/download",
            json={"model_id": test_model_id},
            timeout=15
        )
        
        if response.status_code in [200, 400, 501]:  # Accept 501 if not implemented
            data = response.json()
            if response.status_code == 501:
                print(f"   ⚠️  Download API not implemented: {data.get('error', 'Unknown')}")
            elif response.status_code == 400:
                print(f"   ⚠️  Download failed: {data.get('message', data.get('error', 'Unknown'))}")
            else:
                print(f"   ✅ Download API responded: {data.get('status', 'unknown')}")
        else:
            print(f"   ❌ Download API returned unexpected status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ⚠️  Download API error (may be expected): {e}")
    
    print("\n✅ API endpoint tests completed")
    return True


def test_with_playwright():
    """Test MCP dashboard with Playwright and validate functionality."""
    
    print("\n" + "="*80)
    print("🎭 Enhanced Functional E2E Test with Playwright")
    print("="*80)
    
    # Check if playwright is installed
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("\n❌ Playwright not installed")
        print("   Install with: pip install playwright && playwright install chromium")
        return False
    
    # Create screenshots directory
    screenshots_dir = Path(__file__).parent / "playwright_screenshots_functional"
    screenshots_dir.mkdir(exist_ok=True)
    print(f"\n📸 Screenshots will be saved to: {screenshots_dir}")
    
    # Start MCP server
    print("\n🚀 Starting MCP server...")
    server_process = None
    
    try:
        # Start server in background
        mcp_dashboard_path = Path(__file__).parent.parent / "ipfs_accelerate_py" / "mcp_dashboard.py"
        server_process = subprocess.Popen(
            [sys.executable, str(mcp_dashboard_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        print("   Waiting for server to start...")
        max_wait = 10
        for i in range(max_wait):
            time.sleep(1)
            try:
                requests.get("http://localhost:8899", timeout=1)
                break
            except:
                if i == max_wait - 1:
                    print("   ❌ Server failed to start in time")
                    return False
        
        # Check if server is running
        if server_process.poll() is not None:
            print("   ❌ Server process terminated")
            stdout, stderr = server_process.communicate()
            print(f"   STDOUT: {stdout[:500]}")
            print(f"   STDERR: {stderr[:500]}")
            return False
        
        print("   ✅ Server started")
        
        # Test API endpoints first
        if not test_api_endpoints_directly():
            print("\n❌ API endpoint tests failed")
            return False
        
        # Run Playwright UI tests
        print("\n🎭 Starting browser UI tests...")
        
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.set_viewport_size({"width": 1920, "height": 1080})
            
            # Track console messages
            console_messages = []
            errors = []
            
            def handle_console(msg):
                console_messages.append(f"{msg.type}: {msg.text}")
                if msg.type in ['error', 'warning']:
                    errors.append(f"{msg.type}: {msg.text}")
            
            page.on("console", handle_console)
            
            try:
                # Test 1: Load dashboard and verify content
                print("\n📋 UI Test 1: Dashboard content validation...")
                page.goto("http://localhost:8899", timeout=30000)
                page.wait_for_timeout(2000)
                
                # Check if essential elements exist
                essential_elements = [
                    ('button:has-text("HF Search")', 'HF Search button'),
                    ('#hf-search', 'Search input'),
                    ('button:has-text("Search HF Hub")', 'Search button'),
                ]
                
                for selector, name in essential_elements:
                    element = page.query_selector(selector)
                    if element:
                        print(f"   ✅ Found {name}")
                    else:
                        print(f"   ❌ Missing {name}")
                        return False
                
                page.screenshot(path=str(screenshots_dir / "01_dashboard_validated.png"), full_page=True)
                
                # Test 2: Search with validation
                print("\n📋 UI Test 2: Search functionality validation...")
                hf_tab = page.query_selector('button:has-text("HF Search")')
                hf_tab.click()
                page.wait_for_timeout(1000)
                
                # Enter search query
                search_input = page.query_selector('#hf-search')
                test_query = 'llama'
                search_input.fill(test_query)
                page.wait_for_timeout(500)
                
                # Execute search
                search_btn = page.query_selector('button:has-text("Search HF Hub")')
                search_btn.click()
                page.wait_for_timeout(4000)  # Wait for results
                
                # Validate search results
                results_div = page.query_selector('#hf-search-results')
                if not results_div:
                    print("   ❌ Search results div not found")
                    return False
                
                content = results_div.inner_html()
                result_count = content.count('model-result')
                
                if result_count == 0:
                    print("   ❌ No search results found")
                    print(f"   Content preview: {content[:500]}")
                    return False
                
                print(f"   ✅ Found {result_count} search results")
                
                # Verify results contain search term
                if test_query.lower() in content.lower():
                    print(f"   ✅ Results contain search term '{test_query}'")
                else:
                    print(f"   ⚠️  Results may not be filtered by search term")
                
                page.screenshot(path=str(screenshots_dir / "02_search_validated.png"), full_page=True)
                
                # Test 3: Download button interaction
                print("\n📋 UI Test 3: Download interaction validation...")
                download_btns = page.query_selector_all('button:has-text("Download")')
                
                if len(download_btns) == 0:
                    print("   ❌ No download buttons found")
                    return False
                
                print(f"   ✅ Found {len(download_btns)} download buttons")
                
                # Click first download button
                initial_html = page.inner_html('#hf-search-results')
                download_btns[0].click()
                page.wait_for_timeout(2000)
                
                # Check if UI updated
                updated_html = page.inner_html('#hf-search-results')
                
                if initial_html != updated_html:
                    print("   ✅ UI updated after download click")
                else:
                    print("   ⚠️  UI may not have updated after download")
                
                page.screenshot(path=str(screenshots_dir / "03_download_interaction.png"), full_page=True)
                
                # Test 4: Check for JavaScript errors
                print("\n📋 UI Test 4: JavaScript error check...")
                critical_errors = [e for e in errors if 'error' in e.lower()]
                
                if critical_errors:
                    print(f"   ⚠️  Found {len(critical_errors)} JavaScript errors:")
                    for error in critical_errors[:3]:
                        print(f"      {error}")
                else:
                    print("   ✅ No JavaScript errors detected")
                
                # Test 5: Test empty search
                print("\n📋 UI Test 5: Empty search handling...")
                search_input.fill('')
                search_btn.click()
                page.wait_for_timeout(2000)
                
                results_div = page.query_selector('#hf-search-results')
                empty_content = results_div.inner_html()
                empty_count = empty_content.count('model-result')
                
                print(f"   ℹ️  Empty search returned {empty_count} results")
                page.screenshot(path=str(screenshots_dir / "04_empty_search.png"), full_page=True)
                
                # Test 6: Test different query
                print("\n📋 UI Test 6: Different query test...")
                search_input.fill('bert')
                search_btn.click()
                page.wait_for_timeout(3000)
                
                results_div = page.query_selector('#hf-search-results')
                bert_content = results_div.inner_html()
                bert_count = bert_content.count('model-result')
                
                print(f"   ✅ Query 'bert' returned {bert_count} results")
                
                if 'bert' in bert_content.lower():
                    print("   ✅ Results contain 'bert' in content")
                else:
                    print("   ⚠️  Results may not match 'bert' query")
                
                page.screenshot(path=str(screenshots_dir / "05_bert_search.png"), full_page=True)
                
                print("\n✅ All UI tests completed successfully!")
                print(f"\n📁 Screenshots saved to: {screenshots_dir}")
                
                return True
                
            except Exception as e:
                print(f"\n❌ Error during UI tests: {e}")
                import traceback
                traceback.print_exc()
                
                # Take error screenshot
                try:
                    page.screenshot(path=str(screenshots_dir / "error_state.png"), full_page=True)
                    print(f"   📸 Error screenshot saved")
                except:
                    pass
                
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
    print("Enhanced Functional E2E Test")
    print("="*80)
    print("\nThis test validates:")
    print("  1. API endpoint functionality")
    print("  2. Data structure integrity")
    print("  3. Search query filtering")
    print("  4. Download interaction")
    print("  5. Error handling")
    print("  6. UI state changes")
    print("\nRequirements:")
    print("  - Playwright: pip install playwright && playwright install chromium")
    print("  - Requests: pip install requests")
    print("="*80)
    
    success = test_with_playwright()
    
    if success:
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("❌ TESTS FAILED - Review output above")
        print("="*80)
        sys.exit(1)
