#!/usr/bin/env python3
"""
Simple WebNN Test

A simplified test for WebNN implementation using Selenium and browser automation.
"""

import os
import sys
import time
import json
import asyncio
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple HTML to test WebNN support
WEBNN_TEST_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>WebNN Test</title>
</head>
<body>
    <h1>WebNN Test</h1>
    <div id="status">Checking WebNN support...</div>
    
    <script>
        async function checkWebNN() {
            const statusElement = document.getElementById('status');
            
            try {
                if (!('ml' in navigator)) {
                    statusElement.textContent = 'WebNN is not supported in this browser';
                    return false;
                }
                
                // Check for specific backends
                const backends = [];
                
                // Try CPU backend
                try {
                    const cpuContext = await navigator.ml.createContext({ devicePreference: 'cpu' });
                    if (cpuContext) {
                        backends.push('cpu');
                    }
                } catch (e) {
                    console.error('CPU backend not available:', e);
                }
                
                // Try GPU backend
                try {
                    const gpuContext = await navigator.ml.createContext({ devicePreference: 'gpu' });
                    if (gpuContext) {
                        backends.push('gpu');
                    }
                } catch (e) {
                    console.error('GPU backend not available:', e);
                }
                
                if (backends.length > 0) {
                    statusElement.textContent = 'WebNN is available with backends: ' + backends.join(', ');
                    
                    // Store the result as a global variable so we can access it from Selenium
                    window.webnnTestResult = {
                        supported: true,
                        backends: backends,
                        timestamp: new Date().toISOString()
                    };
                    
                    return true;
                } else {
                    statusElement.textContent = 'WebNN has no available backends';
                    
                    // Store the result as a global variable
                    window.webnnTestResult = {
                        supported: false,
                        error: 'No backends available',
                        timestamp: new Date().toISOString()
                    };
                    
                    return false;
                }
            } catch (error) {
                statusElement.textContent = 'WebNN error: ' + error.message;
                
                // Store the error as a global variable
                window.webnnTestResult = {
                    supported: false,
                    error: error.message,
                    timestamp: new Date().toISOString()
                };
                
                return false;
            }
        }
        
        // Run the test
        checkWebNN().then(result => {
            console.log('WebNN test result:', result);
        });
    </script>
</body>
</html>
"""

async def test_webnn():
    """Test WebNN support in Chrome."""
    # Create a temporary HTML file
    html_path = os.path.join(os.getcwd(), "webnn_test.html")
    with open(html_path, "w") as f:
        f.write(WEBNN_TEST_HTML)
    
    logger.info(f"Created test HTML file: {html_path}")
    
    # Set up Chrome
    try:
        logger.info("Setting up Chrome with WebNN support")
        
        # Set up Chrome options
        options = ChromeOptions()
        options.add_argument("--enable-features=WebNN")
        options.add_argument("--no-sandbox")
        
        # Set up Chrome service
        service = ChromeService(ChromeDriverManager().install())
        
        # Start Chrome
        driver = webdriver.Chrome(service=service, options=options)
        
        # Load the test page
        file_url = f"file://{html_path}"
        logger.info(f"Loading test page: {file_url}")
        driver.get(file_url)
        
        # Wait for the test to complete
        time.sleep(3)
        
        # Get the status element
        status_element = driver.find_element(By.ID, "status")
        status_text = status_element.text
        
        logger.info(f"Status: {status_text}")
        
        # Get the test result from JavaScript
        js_result = driver.execute_script("return window.webnnTestResult")
        
        if js_result:
            logger.info(f"WebNN test result: {json.dumps(js_result, indent=2)}")
            
            if js_result.get("supported", False):
                logger.info("WebNN is supported!")
            else:
                logger.warning("WebNN is not supported")
        else:
            logger.warning("No WebNN test result found")
        
        # Close the browser
        driver.quit()
        
        # Clean up
        os.remove(html_path)
        
        # Return success based on test result
        return js_result and js_result.get("supported", False)
        
    except Exception as e:
        logger.error(f"Error testing WebNN: {e}")
        
        # Clean up
        if os.path.exists(html_path):
            os.remove(html_path)
            
        return False

def main():
    """Main function."""
    logger.info("Testing WebNN support in Chrome")
    
    # Run the test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(test_webnn())
    
    if result:
        logger.info("WebNN test passed")
        return 0
    else:
        logger.error("WebNN test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())