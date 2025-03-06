
/**
 * Test Runner for Web Audio Model Testing
 */

// Report test results back to server
async function reportTestResults(results) {
    try {
        const response = await fetch('/report-results', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(results)
        });
        
        if (!response.ok) {
            throw new Error(`Failed to report results: ${response.statusText}`);
        }
        
        console.log('Test results reported successfully');
        return true;
    } catch (error) {
        console.error('Failed to report test results:', error);
        
        // If server reporting fails, try to display results on page
        displayResultsOnPage(results);
        return false;
    }
}

// Display results on the page for manual viewing
function displayResultsOnPage(results) {
    const resultsDiv = document.getElementById('results') || 
                       document.createElement('div');
    
    if (!document.getElementById('results')) {
        resultsDiv.id = 'results';
        document.body.appendChild(resultsDiv);
    }
    
    // Format results as JSON with nice indentation
    resultsDiv.innerHTML = `
        <h2>Test Results</h2>
        <pre>${JSON.stringify(results, null, 2)}</pre>
    `;
}

// Run a test case and measure performance
async function runTestCase(testCase, options) {
    const startTime = performance.now();
    let success = false;
    let error = null;
    let metrics = {};
    
    try {
        const result = await testCase(options);
        success = true;
        metrics = result.metrics || {};
    } catch (err) {
        error = err.message || String(err);
        console.error('Test case failed:', err);
    }
    
    const endTime = performance.now();
    
    return {
        success,
        error,
        executionTime: endTime - startTime,
        ...metrics
    };
}

// Run all test cases for a model with different configurations
async function runAllTests(model, testCases, configurations) {
    const results = {
        model,
        timestamp: new Date().toISOString(),
        browser: getBrowserInfo(),
        platform: getPlatformInfo(),
        webnnSupport: 'ml' in navigator,
        webgpuSupport: !!navigator.gpu,
        tests: []
    };
    
    for (const testCase of testCases) {
        for (const config of configurations) {
            console.log(`Running test: ${testCase.name} with config:`, config);
            
            const testResult = await runTestCase(testCase, {
                model,
                ...config
            });
            
            results.tests.push({
                testName: testCase.name,
                configuration: config,
                result: testResult
            });
        }
    }
    
    // Report results
    await reportTestResults(results);
    
    return results;
}

// Get browser information
function getBrowserInfo() {
    const userAgent = navigator.userAgent;
    let browserName = "Unknown";
    let browserVersion = "";
    
    // Extract browser name and version from user agent
    if (userAgent.indexOf("Firefox") > -1) {
        browserName = "Firefox";
        browserVersion = userAgent.match(/Firefox\/([\d.]+)/)[1];
    } else if (userAgent.indexOf("Chrome") > -1) {
        browserName = "Chrome";
        browserVersion = userAgent.match(/Chrome\/([\d.]+)/)[1];
    } else if (userAgent.indexOf("Safari") > -1) {
        browserName = "Safari";
        browserVersion = userAgent.match(/Version\/([\d.]+)/)[1];
    } else if (userAgent.indexOf("Edge") > -1 || userAgent.indexOf("Edg") > -1) {
        browserName = "Edge";
        browserVersion = userAgent.match(/Edge?\/([\d.]+)/)[1];
    }
    
    return {
        name: browserName,
        version: browserVersion,
        userAgent
    };
}

// Get platform information
function getPlatformInfo() {
    return {
        os: navigator.platform,
        language: navigator.language,
        hardwareConcurrency: navigator.hardwareConcurrency || 'unknown',
        deviceMemory: navigator.deviceMemory || 'unknown'
    };
}

// Export utilities
export {
    reportTestResults,
    displayResultsOnPage,
    runTestCase,
    runAllTests,
    getBrowserInfo,
    getPlatformInfo
};
