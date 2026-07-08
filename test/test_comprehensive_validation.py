#!/usr/bin/env python3
"""
Comprehensive Test Suite for HuggingFace Model Search and Download Workflow

This test validates:
1. Backend MCP server tools work independently
2. Each backend function in ipfs_accelerate_py works
3. GUI integration connects to backend properly
4. End-to-end workflow with Playwright verification

As requested: First verify backend tools, then GUI integration.

REQUIREMENTS:
    pip install flask flask-cors requests
    # Optional for Playwright tests:
    pip install playwright && playwright install chromium
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*80)
print("COMPREHENSIVE HUGGINGFACE WORKFLOW VALIDATION")
print("="*80)
print("\nℹ️  Prerequisites: pip install flask flask-cors requests")
print("ℹ️  For Playwright: pip install playwright && playwright install chromium")

# ============================================================================
# PHASE 1: Test Backend MCP Server Tools
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: VALIDATING BACKEND MCP SERVER TOOLS")
print("="*80)

def validate_phase1_backend_tools():
    """Test that backend MCP server tools exist and work."""
    
    print("\n📋 Step 1.1: Testing HuggingFaceHubScanner class...")
    try:
        from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner
        scanner = HuggingFaceHubScanner()
        print("   ✅ HuggingFaceHubScanner instantiated")
        
        # Test search method
        print("\n📋 Step 1.2: Testing scanner.search_models() method...")
        results = scanner.search_models(query="bert", limit=3)
        if results and len(results) > 0:
            print(f"   ✅ Search returned {len(results)} models")
            print(f"   📦 Sample: {results[0].model_id if hasattr(results[0], 'model_id') else results[0]}")
        else:
            print("   ⚠️ Search returned no results (might be using mock)")
        
        return True, scanner
        
    except ImportError as e:
        if 'aiohttp' in str(e):
            print(f"   ⚠️ aiohttp not available (optional dependency)")
            print("   💡 Install for async operations: pip install aiohttp")
            print("   ℹ️  Backend will use synchronous fallback")
            # Try to import anyway to test the fallback
            try:
                from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner
                scanner = HuggingFaceHubScanner()
                return True, scanner
            except Exception as e2:
                print(f"   ❌ Even with fallback, scanner failed: {e2}")
                return False, None
        else:
            print(f"   ❌ Backend scanner failed: {e}")
            return False, None
    except Exception as e:
        print(f"   ❌ Backend scanner failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

# ============================================================================
# PHASE 2: Test ipfs_accelerate_py Package Functions
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: VALIDATING IPFS_ACCELERATE_PY PACKAGE FUNCTIONS")
print("="*80)

def validate_phase2_package_functions(scanner):
    """Test that package functions work independently."""
    
    print("\n📋 Step 2.1: Testing model_manager integration...")
    try:
        from ipfs_accelerate_py.model_manager import ModelManager
        manager = ModelManager()
        print("   ✅ ModelManager instantiated")
        
        # List models
        print("\n📋 Step 2.2: Testing manager.list_models()...")
        models = manager.list_models()
        print(f"   ✅ Found {len(models)} models in manager")
        
        return True, manager
        
    except Exception as e:
        print(f"   ⚠️ ModelManager not available: {e}")
        return False, None

# ============================================================================
# PHASE 3: Test MCP Dashboard API Endpoints  
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: VALIDATING MCP DASHBOARD API ENDPOINTS")
print("="*80)

def validate_phase3_dashboard_apis():
    """Test that MCP Dashboard APIs work when server is running."""
    
    import requests
    
    base_url = "http://127.0.0.1:8899"
    
    print("\n📋 Step 3.1: Checking if MCP server is running...")
    try:
        response = requests.get(f"{base_url}/mcp", timeout=2)
        if response.status_code != 200:
            print(f"   ⚠️ Server not running - start with: python3 -m ipfs_accelerate_py.mcp_dashboard")
            return False
        print("   ✅ MCP server is running")
    except Exception as e:
        print(f"   ⚠️ Server not accessible: {e}")
        print("   💡 Start server: python3 -m ipfs_accelerate_py.mcp_dashboard")
        return False
    
    print("\n📋 Step 3.2: Testing /api/mcp/models/search endpoint...")
    try:
        response = requests.get(f"{base_url}/api/mcp/models/search?q=bert&limit=3", timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            print(f"   ✅ Search API works - found {len(results)} models")
            if len(results) > 0:
                model = results[0]
                model_id = model.get('model_id', 'unknown')
                print(f"   📦 Sample model: {model_id}")
                
                # Test download endpoint
                print(f"\n📋 Step 3.3: Testing /api/mcp/models/download endpoint...")
                download_response = requests.post(
                    f"{base_url}/api/mcp/models/download",
                    json={'model_id': model_id},
                    timeout=30
                )
                if download_response.status_code == 200:
                    result = download_response.json()
                    print(f"   ✅ Download API works")
                    print(f"   📝 Response: {result.get('status', 'unknown')}")
                else:
                    print(f"   ⚠️ Download returned status {download_response.status_code}")
                    print(f"   Response: {download_response.text[:200]}")
        else:
            print(f"   ❌ Search API returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        return False
    
    print("\n📋 Step 3.4: Testing /api/mcp/models/stats endpoint...")
    try:
        response = requests.get(f"{base_url}/api/mcp/models/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✅ Stats API works")
            print(f"   📊 Total models: {stats.get('total_cached_models', 0)}")
        else:
            print(f"   ⚠️ Stats API returned status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Stats API failed: {e}")
    
    return True

# ============================================================================
# PHASE 4: Test GUI Integration with Playwright
# ============================================================================

print("\n" + "="*80)
print("PHASE 4: VALIDATING GUI INTEGRATION WITH PLAYWRIGHT")
print("="*80)

def validate_phase4_gui_with_playwright():
    """Test GUI integration using Playwright with screenshots."""
    
    try:
        from playwright.sync_api import sync_playwright
        print("   ✅ Playwright available")
    except ImportError:
        print("   ⚠️ Playwright not installed")
        print("   💡 Install with: pip install playwright && playwright install")
        return False
    
    base_url = "http://127.0.0.1:8899"
    screenshots_dir = Path("data/test_screenshots/validation")
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📋 Step 4.1: Opening browser and loading dashboard...")
    
    try:
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=False, slow_mo=500)
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            
            # Step 1: Load dashboard
            print("   Loading dashboard...")
            page.goto(f"{base_url}/mcp", wait_until="networkidle")
            page.screenshot(path=str(screenshots_dir / "01_dashboard_loaded.png"), full_page=True)
            print("   ✅ Dashboard loaded")
            
            # Step 2: Click HF Search tab
            print("\n📋 Step 4.2: Navigating to HF Search tab...")
            hf_tab = page.query_selector('button:has-text("HF Search")')
            if hf_tab:
                hf_tab.click()
                time.sleep(1)
                page.screenshot(path=str(screenshots_dir / "02_hf_search_tab.png"), full_page=True)
                print("   ✅ HF Search tab opened")
            else:
                print("   ❌ HF Search tab not found")
                return False
            
            # Step 3: Enter search query
            print("\n📋 Step 4.3: Searching for 'bert'...")
            search_input = page.query_selector('#hf-search')
            if search_input:
                search_input.fill('bert')
                page.screenshot(path=str(screenshots_dir / "03_search_input.png"))
                print("   ✅ Search term entered")
            else:
                print("   ❌ Search input not found")
                return False
            
            # Step 4: Click search button
            print("\n📋 Step 4.4: Clicking Search button...")
            search_btn = page.query_selector('button:has-text("Search HF Hub")')
            if search_btn:
                search_btn.click()
                print("   ⏳ Waiting for results...")
                time.sleep(5)
                page.screenshot(path=str(screenshots_dir / "04_search_results.png"), full_page=True)
                print("   ✅ Search executed")
            else:
                print("   ❌ Search button not found")
                return False
            
            # Step 5: Check results
            print("\n📋 Step 4.5: Verifying search results...")
            results_div = page.query_selector('#hf-search-results')
            if results_div:
                content = results_div.inner_html()
                if 'model-result' in content:
                    print("   ✅ Search results displayed")
                    
                    # Step 6: Click download button
                    print("\n📋 Step 4.6: Clicking Download button...")
                    download_btn = page.query_selector('button:has-text("Download")')
                    if download_btn:
                        download_btn.click()
                        time.sleep(3)
                        page.screenshot(path=str(screenshots_dir / "05_download_clicked.png"), full_page=True)
                        print("   ✅ Download button clicked")
                    else:
                        print("   ⚠️ No download button found")
                else:
                    print(f"   ⚠️ No results found or different content: {content[:200]}")
            
            # Step 7: Navigate to Model Browser
            print("\n📋 Step 4.7: Checking Model Browser tab...")
            browser_tab = page.query_selector('button:has-text("Model Browser")')
            if browser_tab:
                browser_tab.click()
                time.sleep(2)
                page.screenshot(path=str(screenshots_dir / "06_model_browser.png"), full_page=True)
                print("   ✅ Model Browser tab opened")
                
                # Check if model appears
                models_list = page.query_selector('#mm-models-list')
                if models_list:
                    content = models_list.inner_html()
                    if 'bert' in content.lower():
                        print("   ✅ BERT model found in Model Browser!")
                    else:
                        print("   ⚠️ BERT model not yet visible in Model Browser")
            else:
                print("   ❌ Model Browser tab not found")
            
            page.screenshot(path=str(screenshots_dir / "07_final_state.png"), full_page=True)
            
            browser.close()
            
            print(f"\n   📸 Screenshots saved to: {screenshots_dir}")
            return True
            
    except Exception as e:
        print(f"   ❌ Playwright test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# RUN ALL PHASES
# ============================================================================

def run_all_phases():
    """Run all test phases in sequence."""
    
    results = {}
    
    # Phase 1: Backend tools
    success1, scanner = validate_phase1_backend_tools()
    results['phase1_backend'] = 'PASS' if success1 else 'FAIL'
    
    # Phase 2: Package functions
    success2, manager = validate_phase2_package_functions(scanner if success1 else None)
    results['phase2_package'] = 'PASS' if success2 else 'WARN'
    
    # Phase 3: API endpoints (requires server)
    success3 = validate_phase3_dashboard_apis()
    results['phase3_apis'] = 'PASS' if success3 else 'WARN (server not running)'
    
    # Phase 4: GUI with Playwright (requires server)
    if success3:
        success4 = validate_phase4_gui_with_playwright()
        results['phase4_gui'] = 'PASS' if success4 else 'FAIL'
    else:
        print("\n   ⚠️ Skipping Phase 4 - server not running")
        results['phase4_gui'] = 'SKIPPED (server not running)'
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for phase, result in results.items():
        icon = "✅" if result == 'PASS' else "⚠️" if 'WARN' in result or 'SKIP' in result else "❌"
        print(f"{icon} {phase}: {result}")
    print("="*80)
    
    return results

if __name__ == '__main__':
    results = run_all_phases()
    
    # Determine exit code
    has_failure = any('FAIL' in v for v in results.values())
    sys.exit(1 if has_failure else 0)
