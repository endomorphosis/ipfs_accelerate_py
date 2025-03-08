#!/usr/bin/env python3
"""
Browser WebNN/WebGPU Capability Checker

This script helps verify that your browser has proper WebNN/WebGPU support
before running real benchmarks. It launches a browser to detect and report hardware
acceleration capabilities.

Usage:
    python check_browser_webnn_webgpu.py --browser chrome
    python check_browser_webnn_webgpu.py --browser firefox --platform webgpu
    python check_browser_webnn_webgpu.py --browser edge --platform webnn
    python check_browser_webnn_webgpu.py --check-all

Features:
    - Checks WebNN and WebGPU hardware acceleration support
    - Tests all installed browsers or a specific browser
    - Generates a detailed report of browser capabilities
    - Identifies simulation vs. real hardware implementation
    - Provides recommendations for optimal browser selection
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

# Import BrowserAutomation if available
try:
    from fixed_web_platform.browser_automation import (
        BrowserAutomation,
        find_browser_executable
    )
    BROWSER_AUTOMATION_AVAILABLE = True
except ImportError:
    logger.warning("BrowserAutomation not available. Using basic browser detection.")
    BROWSER_AUTOMATION_AVAILABLE = False

# Constants
SUPPORTED_BROWSERS = ["chrome", "firefox", "edge", "safari", "all"]
SUPPORTED_PLATFORMS = ["webnn", "webgpu", "all"]

def find_available_browsers():
    """Find all available browsers on the system."""
    available_browsers = {}
    
    for browser in ["chrome", "firefox", "edge", "safari"]:
        if BROWSER_AUTOMATION_AVAILABLE:
            path = find_browser_executable(browser)
            if path:
                available_browsers[browser] = path
        else:
            # Fallback to basic detection if BrowserAutomation not available
            found = False
            
            # Browser-specific checks
            if browser == "chrome":
                paths = [
                    "google-chrome", "/usr/bin/google-chrome",
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                ]
            elif browser == "firefox":
                paths = [
                    "firefox", "/usr/bin/firefox",
                    r"C:\Program Files\Mozilla Firefox\firefox.exe",
                    "/Applications/Firefox.app/Contents/MacOS/firefox"
                ]
            elif browser == "edge":
                paths = [
                    "microsoft-edge", "/usr/bin/microsoft-edge",
                    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
                ]
            elif browser == "safari":
                paths = [
                    "/Applications/Safari.app/Contents/MacOS/Safari"
                ]
            else:
                paths = []
            
            # Check each path
            for path in paths:
                try:
                    if os.path.exists(path) and os.access(path, os.X_OK):
                        available_browsers[browser] = path
                        found = True
                        break
                    elif os.name != 'nt' and not path.startswith('/'):
                        # Try using 'which' on Linux/macOS
                        result = subprocess.run(
                            ["which", path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            available_browsers[browser] = result.stdout.strip()
                            found = True
                            break
                except Exception:
                    continue
    
    return available_browsers

def create_capability_detection_html():
    """Create HTML file for detecting browser capabilities."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        html_path = f.name
        
        html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>WebNN/WebGPU Capability Checker</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .result { margin: 10px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
        .success { color: green; }
        .error { color: red; }
        .warning { color: orange; }
        .info { color: blue; }
        pre { white-space: pre-wrap; background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>WebNN/WebGPU Capability Checker</h1>
    
    <div id="summary" class="result">
        <p>Checking browser capabilities...</p>
    </div>
    
    <div id="webgpu-result" class="result">
        <h2>WebGPU</h2>
        <p>Checking WebGPU support...</p>
    </div>
    
    <div id="webnn-result" class="result">
        <h2>WebNN</h2>
        <p>Checking WebNN support...</p>
    </div>
    
    <div id="details" class="result">
        <h2>Detailed Information</h2>
        <div id="browser-info">
            <h3>Browser Information</h3>
            <pre id="browser-details">Loading...</pre>
        </div>
        <div id="gpu-info">
            <h3>GPU Information</h3>
            <pre id="gpu-details">Loading...</pre>
        </div>
    </div>
    
    <script>
        // Store capability check results
        const results = {
            webgpu: {
                supported: false,
                real: false,
                details: {},
                simulation: true,
                error: null
            },
            webnn: {
                supported: false,
                real: false,
                details: {},
                simulation: true,
                error: null
            },
            browser: {
                userAgent: navigator.userAgent,
                platform: navigator.platform,
                vendor: navigator.vendor,
                language: navigator.language,
                hardware_concurrency: navigator.hardwareConcurrency || 'unknown',
                device_memory: navigator.deviceMemory || 'unknown'
            }
        };
        
        // Update UI with capability check results
        function updateUI() {
            const summary = document.getElementById('summary');
            const webgpuResult = document.getElementById('webgpu-result');
            const webnnResult = document.getElementById('webnn-result');
            const browserDetails = document.getElementById('browser-details');
            const gpuDetails = document.getElementById('gpu-details');
            
            // Update browser details
            browserDetails.textContent = JSON.stringify(results.browser, null, 2);
            
            // Update WebGPU results
            if (results.webgpu.error) {
                webgpuResult.innerHTML = `
                    <h2>WebGPU</h2>
                    <div class="error">
                        <p>❌ WebGPU is not supported</p>
                        <p>Error: ${results.webgpu.error}</p>
                    </div>
                `;
            } else if (results.webgpu.supported) {
                webgpuResult.innerHTML = `
                    <h2>WebGPU</h2>
                    <div class="${results.webgpu.real ? 'success' : 'warning'}">
                        <p>${results.webgpu.real ? '✅ Real WebGPU hardware acceleration available' : '⚠️ WebGPU supported but using simulation'}</p>
                        <p>Implementation: ${results.webgpu.real ? 'HARDWARE' : 'SIMULATION'}</p>
                        <pre>${JSON.stringify(results.webgpu.details, null, 2)}</pre>
                    </div>
                `;
                
                // Update GPU details
                gpuDetails.textContent = JSON.stringify(results.webgpu.details, null, 2);
            } else {
                webgpuResult.innerHTML = `
                    <h2>WebGPU</h2>
                    <div class="error">
                        <p>❌ WebGPU is not supported in this browser</p>
                    </div>
                `;
            }
            
            // Update WebNN results
            if (results.webnn.error) {
                webnnResult.innerHTML = `
                    <h2>WebNN</h2>
                    <div class="error">
                        <p>❌ WebNN is not supported</p>
                        <p>Error: ${results.webnn.error}</p>
                    </div>
                `;
            } else if (results.webnn.supported) {
                webnnResult.innerHTML = `
                    <h2>WebNN</h2>
                    <div class="${results.webnn.real ? 'success' : 'warning'}">
                        <p>${results.webnn.real ? '✅ Real WebNN hardware acceleration available' : '⚠️ WebNN supported but using simulation'}</p>
                        <p>Implementation: ${results.webnn.real ? 'HARDWARE' : 'SIMULATION'}</p>
                        <pre>${JSON.stringify(results.webnn.details, null, 2)}</pre>
                    </div>
                `;
            } else {
                webnnResult.innerHTML = `
                    <h2>WebNN</h2>
                    <div class="error">
                        <p>❌ WebNN is not supported in this browser</p>
                    </div>
                `;
            }
            
            // Update summary
            const webgpuStatus = results.webgpu.supported 
                ? (results.webgpu.real ? "✅ Real Hardware" : "⚠️ Simulation") 
                : "❌ Not Supported";
                
            const webnnStatus = results.webnn.supported 
                ? (results.webnn.real ? "✅ Real Hardware" : "⚠️ Simulation") 
                : "❌ Not Supported";
                
            summary.innerHTML = `
                <h2>Capability Summary</h2>
                <p><strong>WebGPU:</strong> ${webgpuStatus}</p>
                <p><strong>WebNN:</strong> ${webnnStatus}</p>
                <p><strong>Browser:</strong> ${results.browser.userAgent}</p>
                <p><strong>Hardware Concurrency:</strong> ${results.browser.hardware_concurrency} cores</p>
                <p><strong>Device Memory:</strong> ${results.browser.device_memory} GB</p>
            `;
            
            // Store results in localStorage for retrieval
            localStorage.setItem('capability_check_results', JSON.stringify(results));
        }
        
        // Check WebGPU support
        async function checkWebGPU() {
            try {
                if (!navigator.gpu) {
                    results.webgpu.error = "WebGPU API not available in this browser";
                    return;
                }
                
                const adapter = await navigator.gpu.requestAdapter();
                if (!adapter) {
                    results.webgpu.error = "No WebGPU adapter found";
                    return;
                }
                
                const info = await adapter.requestAdapterInfo();
                results.webgpu.supported = true;
                results.webgpu.details = {
                    vendor: info.vendor,
                    architecture: info.architecture,
                    device: info.device,
                    description: info.description
                };
                
                // Check for simulation vs real hardware
                // Real hardware typically has meaningful vendor and device info
                results.webgpu.real = !!(info.vendor && info.vendor !== 'Software' && 
                                       info.device && info.device !== 'Software Adapter');
                results.webgpu.simulation = !results.webgpu.real;
                
                // Get additional features (requires requesting a device)
                try {
                    const device = await adapter.requestDevice();
                    
                    // Query features
                    const features = [];
                    for (const feature of device.features.values()) {
                        features.push(feature);
                    }
                    
                    results.webgpu.details.features = features;
                    
                    // Query limits
                    results.webgpu.details.limits = {};
                    for (const [key, value] of Object.entries(device.limits)) {
                        results.webgpu.details.limits[key] = value;
                    }
                    
                    // Test compute shaders
                    results.webgpu.details.compute_shaders = 
                        device.features.has('shader-f16') || 
                        features.includes('shader-f16');
                    
                } catch (deviceError) {
                    results.webgpu.details.device_error = deviceError.message;
                }
                
            } catch (error) {
                results.webgpu.error = error.message;
            } finally {
                updateUI();
            }
        }
        
        // Check WebNN support
        async function checkWebNN() {
            try {
                if (!('ml' in navigator)) {
                    results.webnn.error = "WebNN API not available in this browser";
                    return;
                }
                
                try {
                    const context = await navigator.ml.createContext();
                    results.webnn.supported = true;
                    
                    const device = await context.queryDevice();
                    results.webnn.details = {
                        device: device,
                        contextType: (context && context.type) || 'unknown'
                    };
                    
                    // Check for simulation vs real hardware
                    // Real hardware typically uses GPU or dedicated NPU
                    const contextType = context && context.type;
                    results.webnn.real = contextType && contextType !== 'cpu';
                    results.webnn.simulation = contextType === 'cpu';
                    
                } catch (contextError) {
                    results.webnn.error = contextError.message;
                }
                
            } catch (error) {
                results.webnn.error = error.message;
            } finally {
                updateUI();
            }
        }
        
        // Run all checks
        async function runChecks() {
            try {
                // Update browser details
                document.getElementById('browser-details').textContent = 
                    JSON.stringify(results.browser, null, 2);
                
                // Run platform checks in parallel
                await Promise.all([
                    checkWebGPU(),
                    checkWebNN()
                ]);
                
                // Final UI update
                updateUI();
                
            } catch (error) {
                console.error("Error running capability checks:", error);
                
                // Update summary with error
                document.getElementById('summary').innerHTML = `
                    <h2>Capability Checks Failed</h2>
                    <div class="error">
                        <p>Error: ${error.message}</p>
                    </div>
                `;
            }
        }
        
        // Run the checks when page loads
        runChecks();
    </script>
</body>
</html>
"""
        f.write(html_content.encode('utf-8'))
    
    return html_path

async def check_browser_capabilities(browser, platform, headless=False):
    """Check WebNN/WebGPU capabilities for a specific browser."""
    if not BROWSER_AUTOMATION_AVAILABLE:
        logger.error("BrowserAutomation not available. Cannot check capabilities.")
        return None
    
    logger.info(f"Checking {platform} capabilities for {browser}")
    
    # Create HTML file for capability detection
    html_file = create_capability_detection_html()
    if not html_file:
        logger.error("Failed to create capability detection HTML")
        return None
    
    try:
        # Create browser automation instance
        automation = BrowserAutomation(
            platform=platform,
            browser_name=browser,
            headless=headless,
            model_type="text"
        )
        
        # Launch browser
        success = await automation.launch()
        if not success:
            logger.error(f"Failed to launch {browser}")
            return None
        
        try:
            # Wait for capability checks to complete
            await asyncio.sleep(3)
            
            # Get capability check results
            if hasattr(automation, 'driver') and automation.driver:
                try:
                    # Execute JavaScript to get results from localStorage
                    result = automation.driver.execute_script("""
                        return localStorage.getItem('capability_check_results');
                    """)
                    
                    if result:
                        try:
                            return json.loads(result)
                        except json.JSONDecodeError:
                            logger.error("Invalid JSON in capability check results")
                    else:
                        logger.error("No capability check results found")
                except Exception as e:
                    logger.error(f"Error getting capability check results: {e}")
            
            return None
            
        finally:
            # Close browser
            await automation.close()
            
    except Exception as e:
        logger.error(f"Error checking browser capabilities: {e}")
        return None
    finally:
        # Remove temporary HTML file
        if html_file and os.path.exists(html_file):
            try:
                os.unlink(html_file)
            except Exception:
                pass

def format_capability_report(browser, capabilities, platform):
    """Format capability check results as a readable report."""
    if not capabilities:
        return f"\n=== {browser.upper()} ===\nFailed to check capabilities\n"
    
    report = f"\n=== {browser.upper()} ===\n"
    
    # Add browser info
    browser_info = capabilities.get("browser", {})
    report += f"User Agent: {browser_info.get('userAgent', 'Unknown')}\n"
    report += f"Platform: {browser_info.get('platform', 'Unknown')}\n"
    report += f"Cores: {browser_info.get('hardware_concurrency', 'Unknown')}\n"
    report += f"Memory: {browser_info.get('device_memory', 'Unknown')}\n\n"
    
    # Add WebGPU info if requested
    if platform in ["webgpu", "all"]:
        webgpu = capabilities.get("webgpu", {})
        report += "WebGPU:\n"
        
        if webgpu.get("error"):
            report += f"  Status: ❌ Not supported (Error: {webgpu.get('error')})\n"
        elif webgpu.get("supported"):
            if webgpu.get("real"):
                report += "  Status: ✅ REAL HARDWARE ACCELERATION\n"
            else:
                report += "  Status: ⚠️ Simulation (no hardware acceleration)\n"
            
            # Add details
            details = webgpu.get("details", {})
            report += f"  Vendor: {details.get('vendor', 'Unknown')}\n"
            report += f"  Device: {details.get('device', 'Unknown')}\n"
            report += f"  Architecture: {details.get('architecture', 'Unknown')}\n"
            
            # Add compute shader support
            compute_shaders = details.get("compute_shaders", False)
            report += f"  Compute Shaders: {'✅ Supported' if compute_shaders else '❌ Not supported'}\n"
            
            # Add limits
            limits = details.get("limits", {})
            if limits:
                report += "  Key Limits:\n"
                for key, value in limits.items():
                    if key in ["maxComputeWorkgroupSizeX", "maxComputeWorkgroupSizeY", 
                              "maxComputeWorkgroupSizeZ", "maxStorageBufferBindingSize"]:
                        report += f"    {key}: {value}\n"
        else:
            report += "  Status: ❌ Not supported\n"
        
        report += "\n"
    
    # Add WebNN info if requested
    if platform in ["webnn", "all"]:
        webnn = capabilities.get("webnn", {})
        report += "WebNN:\n"
        
        if webnn.get("error"):
            report += f"  Status: ❌ Not supported (Error: {webnn.get('error')})\n"
        elif webnn.get("supported"):
            if webnn.get("real"):
                report += "  Status: ✅ REAL HARDWARE ACCELERATION\n"
            else:
                report += "  Status: ⚠️ Simulation (CPU fallback)\n"
            
            # Add details
            details = webnn.get("details", {})
            report += f"  Context Type: {details.get('contextType', 'Unknown')}\n"
        else:
            report += "  Status: ❌ Not supported\n"
        
        report += "\n"
    
    # Add recommendations
    report += "Recommendation:\n"
    
    webgpu_real = capabilities.get("webgpu", {}).get("real", False)
    webnn_real = capabilities.get("webnn", {}).get("real", False)
    
    if webgpu_real and webnn_real:
        report += "  ✅ EXCELLENT - Full hardware acceleration for both WebGPU and WebNN\n"
        report += f"  This browser ({browser}) is recommended for all model types\n"
    elif webgpu_real:
        report += "  ✅ GOOD - Real WebGPU hardware acceleration available\n"
        if browser == "firefox":
            report += "  Recommended for audio models (best compute shader performance)\n"
        else:
            report += "  Recommended for vision and multimodal models\n"
    elif webnn_real:
        report += "  ✅ GOOD - Real WebNN hardware acceleration available\n"
        report += "  Recommended for text embedding models\n"
    elif capabilities.get("webgpu", {}).get("supported") or capabilities.get("webnn", {}).get("supported"):
        report += "  ⚠️ LIMITED - APIs supported but using simulation/CPU fallback\n"
        report += "  Performance will be limited compared to real hardware acceleration\n"
    else:
        report += "  ❌ NOT RECOMMENDED - No WebNN or WebGPU support\n"
        report += "  Consider using a different browser with better support\n"
    
    return report

async def check_all_browsers(platform, headless=False):
    """Check capabilities for all available browsers."""
    available_browsers = find_available_browsers()
    
    if not available_browsers:
        logger.error("No supported browsers found on this system")
        return False
    
    logger.info(f"Found {len(available_browsers)} browsers: {', '.join(available_browsers.keys())}")
    
    reports = []
    results = {}
    
    # Check each browser
    for browser, path in available_browsers.items():
        logger.info(f"Checking {browser} ({path})...")
        
        capabilities = await check_browser_capabilities(browser, platform, headless)
        report = format_capability_report(browser, capabilities, platform)
        reports.append(report)
        results[browser] = capabilities
    
    # Print all reports
    print("\n" + "="*50)
    print(f"BROWSER CAPABILITY REPORT - {platform.upper()}")
    print("="*50)
    
    for report in reports:
        print(report)
    
    # Print summary recommendations
    print("="*50)
    print("SUMMARY RECOMMENDATIONS")
    print("="*50)
    
    # For text models
    print("\nFor TEXT models:")
    recommended_text = []
    for browser, capabilities in results.items():
        if capabilities and capabilities.get("webnn", {}).get("real"):
            recommended_text.append(browser)
        elif capabilities and capabilities.get("webgpu", {}).get("real"):
            recommended_text.append(browser)
    
    if recommended_text:
        print(f"  Recommended browsers: {', '.join(recommended_text)}")
        print(f"  Best choice: {recommended_text[0]}")
    else:
        print("  No browsers with hardware acceleration found")
    
    # For vision models
    print("\nFor VISION models:")
    recommended_vision = []
    for browser, capabilities in results.items():
        if capabilities and capabilities.get("webgpu", {}).get("real"):
            recommended_vision.append(browser)
    
    if recommended_vision:
        print(f"  Recommended browsers: {', '.join(recommended_vision)}")
        print(f"  Best choice: {recommended_vision[0]}")
    else:
        print("  No browsers with hardware acceleration found")
    
    # For audio models
    print("\nFor AUDIO models:")
    recommended_audio = []
    for browser, capabilities in results.items():
        if browser == "firefox" and capabilities and capabilities.get("webgpu", {}).get("real"):
            recommended_audio.insert(0, browser)  # Firefox is preferred for audio
        elif capabilities and capabilities.get("webgpu", {}).get("real"):
            recommended_audio.append(browser)
    
    if recommended_audio:
        print(f"  Recommended browsers: {', '.join(recommended_audio)}")
        print(f"  Best choice: {recommended_audio[0]}")
    else:
        print("  No browsers with hardware acceleration found")
    
    print("\n" + "="*50)
    
    return True

async def main_async(args):
    """Run the browser capability check asynchronously."""
    if args.check_all:
        return await check_all_browsers(args.platform, args.headless)
    elif args.browser == "all":
        return await check_all_browsers(args.platform, args.headless)
    else:
        capabilities = await check_browser_capabilities(args.browser, args.platform, args.headless)
        report = format_capability_report(args.browser, capabilities, args.platform)
        
        print("\n" + "="*50)
        print(f"BROWSER CAPABILITY REPORT - {args.browser.upper()}")
        print("="*50)
        print(report)
        print("="*50)
        
        return capabilities is not None

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check browser WebNN/WebGPU capabilities")
    
    parser.add_argument("--browser", choices=SUPPORTED_BROWSERS, default="chrome",
                      help="Browser to check (or 'all' for all available browsers)")
    
    parser.add_argument("--platform", choices=SUPPORTED_PLATFORMS, default="all",
                      help="Platform to check")
    
    parser.add_argument("--headless", action="store_true",
                      help="Run browser in headless mode")
    
    parser.add_argument("--check-all", action="store_true",
                      help="Check all available browsers")
    
    args = parser.parse_args()
    
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(0 if main() else 1)