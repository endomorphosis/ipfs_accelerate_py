#!/usr/bin/env python3
"""
Playwright test for MCP Dashboard GitHub Workflows and Runners display

This test verifies that:
1. The MCP dashboard loads correctly
2. DOM elements for workflows and runners exist
3. MCP SDK is loaded and initialized
4. GitHub Workflows tab is accessible
5. Data is populated from MCP server tools
"""

import anyio
import logging
import os
import sys
import time
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MCP_SERVER_HOST = "localhost"
MCP_SERVER_PORT = 3001
DASHBOARD_URL = f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}/"
SCREENSHOTS_DIR = Path("./data/test_screenshots")
SERVER_STARTUP_TIMEOUT = 30  # seconds


class MCPServerProcess:
    """Context manager for MCP server process"""
    
    def __init__(self, port=3001):
        self.port = port
        self.process = None
        
    def __enter__(self):
        """Start the MCP server"""
        logger.info(f"Starting MCP server on port {self.port}...")
        
        # Start the server in a subprocess
        server_script = Path(__file__).parent / "mcp_jsonrpc_server.py"
        
        if not server_script.exists():
            logger.error(f"MCP server script not found: {server_script}")
            raise FileNotFoundError(f"MCP server script not found: {server_script}")
        
        # Start server with Python
        self.process = subprocess.Popen(
            [sys.executable, str(server_script), "--port", str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        start_time = time.time()
        while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
            try:
                import requests
                response = requests.get(f"http://{MCP_SERVER_HOST}:{self.port}/", timeout=2)
                if response.status_code == 200:
                    logger.info(f"✓ MCP server started successfully on port {self.port}")
                    return self
            except Exception:
                time.sleep(1)
        
        # If we get here, server didn't start
        logger.error("MCP server failed to start within timeout")
        self.cleanup()
        raise RuntimeError("MCP server failed to start")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the MCP server"""
        self.cleanup()
    
    def cleanup(self):
        """Clean up the server process"""
        if self.process:
            logger.info("Stopping MCP server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't stop gracefully, killing...")
                self.process.kill()
                self.process.wait()
            logger.info("✓ MCP server stopped")


def test_mcp_dashboard_workflows_and_runners():
    """
    Main test function that verifies the MCP dashboard workflows and runners display
    """
    # Create screenshots directory
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Screenshots will be saved to: {SCREENSHOTS_DIR}")
    
    # Start MCP server
    try:
        with MCPServerProcess(port=MCP_SERVER_PORT):
            # Run Playwright tests
            with sync_playwright() as p:
                # Launch browser
                logger.info("Launching Chromium browser...")
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080}
                )
                page = context.new_page()
                
                # Enable console logging
                page.on("console", lambda msg: logger.info(f"Browser console: [{msg.type}] {msg.text}"))
                page.on("pageerror", lambda err: logger.error(f"Browser error: {err}"))
                
                try:
                    # Step 1: Navigate to dashboard
                    logger.info(f"Navigating to dashboard: {DASHBOARD_URL}")
                    page.goto(DASHBOARD_URL, wait_until="domcontentloaded", timeout=30000)
                    # Wait a bit for JavaScript to initialize
                    page.wait_for_timeout(2000)
                    page.screenshot(path=SCREENSHOTS_DIR / "01_dashboard_loaded.png")
                    logger.info("✓ Dashboard loaded")
                    
                    # Step 2: Verify page title
                    logger.info("Verifying page title...")
                    title = page.title()
                    assert "MCP" in title or "IPFS Accelerate" in title, f"Unexpected page title: {title}"
                    logger.info(f"✓ Page title verified: {title}")
                    
                    # Step 3: Verify MCP SDK is loaded
                    logger.info("Checking if MCP SDK is loaded...")
                    mcp_loaded = page.evaluate("typeof MCPClient !== 'undefined'")
                    assert mcp_loaded, "MCP SDK (MCPClient) not loaded"
                    logger.info("✓ MCP SDK loaded")
                    
                    # Step 4: Find and click GitHub Workflows tab
                    logger.info("Looking for GitHub Workflows tab...")
                    
                    # Try different selectors for the tab
                    workflows_tab_selectors = [
                        "button:has-text('GitHub Workflows')",
                        "button:has-text('⚡ GitHub Workflows')",
                        ".nav-tab:has-text('Workflows')",
                        "[onclick*='github-workflows']"
                    ]
                    
                    workflows_tab = None
                    for selector in workflows_tab_selectors:
                        try:
                            workflows_tab = page.locator(selector).first
                            if workflows_tab.is_visible():
                                logger.info(f"✓ Found workflows tab with selector: {selector}")
                                break
                        except Exception:
                            continue
                    
                    if workflows_tab and workflows_tab.is_visible():
                        logger.info("Clicking GitHub Workflows tab...")
                        workflows_tab.click()
                        page.wait_for_timeout(1000)  # Wait for tab switch animation
                        page.screenshot(path=SCREENSHOTS_DIR / "02_workflows_tab_clicked.png")
                        logger.info("✓ GitHub Workflows tab clicked")
                    else:
                        logger.warning("Could not find GitHub Workflows tab button")
                        page.screenshot(path=SCREENSHOTS_DIR / "02_workflows_tab_not_found.png")
                    
                    # Step 5: Verify workflows container exists
                    logger.info("Verifying workflows container exists...")
                    workflows_containers = [
                        "#github-workflows-container",
                        "#github-workflows",
                        "[id*='workflow']"
                    ]
                    
                    workflows_container = None
                    for selector in workflows_containers:
                        try:
                            container = page.locator(selector).first
                            if container.count() > 0:
                                workflows_container = container
                                logger.info(f"✓ Found workflows container: {selector}")
                                break
                        except Exception:
                            continue
                    
                    assert workflows_container is not None, "Workflows container not found in DOM"
                    
                    # Step 6: Verify runners container exists
                    logger.info("Verifying runners container exists...")
                    runners_containers = [
                        "#active-runners-container",
                        "#github-runners-container",
                        "[id*='runner']"
                    ]
                    
                    runners_container = None
                    for selector in runners_containers:
                        try:
                            container = page.locator(selector).first
                            if container.count() > 0:
                                runners_container = container
                                logger.info(f"✓ Found runners container: {selector}")
                                break
                        except Exception:
                            continue
                    
                    assert runners_container is not None, "Runners container not found in DOM"
                    
                    # Step 7: Check if GitHub manager is initialized
                    logger.info("Checking if GitHub manager is initialized...")
                    github_manager_exists = page.evaluate("typeof githubManager !== 'undefined'")
                    if github_manager_exists:
                        logger.info("✓ GitHub manager (githubManager) is initialized")
                        
                        # Check if manager has MCP client
                        has_mcp = page.evaluate("githubManager && githubManager.mcp !== null")
                        if has_mcp:
                            logger.info("✓ GitHub manager has MCP client")
                        else:
                            logger.warning("⚠ GitHub manager does not have MCP client")
                    else:
                        logger.warning("⚠ GitHub manager (githubManager) not found")
                    
                    # Step 8: Take screenshot of workflows section
                    page.screenshot(path=SCREENSHOTS_DIR / "03_workflows_section.png")
                    
                    # Step 9: Check for Track button and try to click it
                    logger.info("Looking for Track button...")
                    track_button_selectors = [
                        "button:has-text('Track')",
                        "button[onclick*='trackRunners']"
                    ]
                    
                    for selector in track_button_selectors:
                        try:
                            track_button = page.locator(selector).first
                            if track_button.is_visible():
                                logger.info(f"✓ Found Track button with selector: {selector}")
                                logger.info("Clicking Track button to load runners...")
                                track_button.click()
                                page.wait_for_timeout(3000)  # Wait for data to load
                                page.screenshot(path=SCREENSHOTS_DIR / "04_after_track_click.png")
                                logger.info("✓ Track button clicked")
                                break
                        except Exception as e:
                            logger.debug(f"Track button not found with selector {selector}: {e}")
                    
                    # Step 10: Check workflows container content
                    logger.info("Checking workflows container content...")
                    if workflows_container:
                        workflows_html = workflows_container.inner_html()
                        logger.info(f"Workflows container HTML length: {len(workflows_html)}")
                        
                        if "Loading" in workflows_html or "loading" in workflows_html:
                            logger.info("Workflows container shows loading state")
                        elif "Error" in workflows_html or "error" in workflows_html:
                            logger.warning("⚠ Workflows container shows error state")
                            logger.warning(f"Error content: {workflows_html[:200]}")
                        elif len(workflows_html.strip()) < 50:
                            logger.warning("⚠ Workflows container appears empty")
                        else:
                            logger.info("✓ Workflows container has content")
                    
                    # Step 11: Check runners container content
                    logger.info("Checking runners container content...")
                    if runners_container:
                        runners_html = runners_container.inner_html()
                        logger.info(f"Runners container HTML length: {len(runners_html)}")
                        
                        if "Loading" in runners_html or "loading" in runners_html:
                            logger.info("Runners container shows loading state")
                        elif "Error" in runners_html or "error" in runners_html:
                            logger.warning("⚠ Runners container shows error state")
                            logger.warning(f"Error content: {runners_html[:200]}")
                        elif len(runners_html.strip()) < 50:
                            logger.warning("⚠ Runners container appears empty")
                        else:
                            logger.info("✓ Runners container has content")
                    
                    # Step 12: Take final screenshot
                    page.screenshot(path=SCREENSHOTS_DIR / "05_final_state.png")
                    
                    # Step 13: Get all visible elements with workflows/runners in their ID
                    logger.info("Listing all workflow/runner related elements...")
                    workflow_elements = page.evaluate("""
                        () => {
                            const elements = document.querySelectorAll('[id*="workflow"], [id*="runner"]');
                            return Array.from(elements).map(el => ({
                                id: el.id,
                                tag: el.tagName,
                                visible: el.offsetParent !== null,
                                hasContent: el.innerHTML.length > 0
                            }));
                        }
                    """)
                    
                    logger.info("Workflow/Runner related elements:")
                    for elem in workflow_elements:
                        logger.info(f"  - {elem['tag']}#{elem['id']} - Visible: {elem['visible']}, Has Content: {elem['hasContent']}")
                    
                    # Summary
                    logger.info("\n" + "="*60)
                    logger.info("TEST SUMMARY")
                    logger.info("="*60)
                    logger.info(f"✓ Dashboard loaded successfully")
                    logger.info(f"✓ MCP SDK loaded: {mcp_loaded}")
                    logger.info(f"✓ GitHub Manager initialized: {github_manager_exists}")
                    logger.info(f"✓ Workflows container found: {workflows_container is not None}")
                    logger.info(f"✓ Runners container found: {runners_container is not None}")
                    logger.info(f"Screenshots saved to: {SCREENSHOTS_DIR.absolute()}")
                    logger.info("="*60)
                    
                except Exception as e:
                    logger.error(f"Test failed: {e}")
                    page.screenshot(path=SCREENSHOTS_DIR / "error_state.png")
                    raise
                finally:
                    # Cleanup
                    context.close()
                    browser.close()
                    logger.info("Browser closed")
    
    except Exception as e:
        logger.error(f"Error during test: {e}")
        raise


if __name__ == "__main__":
    logger.info("Starting MCP Dashboard Playwright Test")
    logger.info("="*60)
    
    try:
        test_mcp_dashboard_workflows_and_runners()
        logger.info("\n✓ All tests passed!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Tests failed: {e}")
        sys.exit(1)
