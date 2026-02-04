# Comprehensive Playwright E2E Testing Implementation Plan

## Executive Summary

This document outlines the comprehensive implementation of Playwright-based end-to-end testing for the IPFS Accelerate Dashboard, with full log correlation between dashboard actions and MCP server operations.

## Implementation Status: ✅ COMPLETE

All phases have been implemented and are ready for use.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Playwright Test Runner                       │
│  ┌────────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Test Specs    │  │   Fixtures   │  │    Utilities      │  │
│  │  - Core Tests  │  │  - Dashboard │  │  - Log Correlator │  │
│  │  - Runners     │  │  - MCP Server│  │  - Screenshots    │  │
│  │  - Models      │  │              │  │  - Reports        │  │
│  │  - Inference   │  │              │  │                   │  │
│  └────────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Browser (Chromium/Firefox/WebKit)             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              IPFS Accelerate Dashboard (HTML/JS)           │ │
│  │  ┌──────────┐  ┌─────────────┐  ┌──────────────────────┐ │ │
│  │  │ MCP SDK  │→│  Dashboard   │→│   UI Components      │ │ │
│  │  │ Client   │  │  Controller  │  │   - Tabs             │ │ │
│  │  └──────────┘  └─────────────┘  │   - Forms            │ │ │
│  │       ↓                          │   - Results Display  │ │ │
│  │   Console Logs                   └──────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ JSON-RPC
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Server (Python)                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Flask Dashboard Server                                    │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │ │
│  │  │  JSON-RPC    │→│   MCP Tools   │→│  Server Logs    │ │ │
│  │  │  Endpoint    │  │   - Inference │  │  (structured)   │ │ │
│  │  └──────────────┘  │   - Runners   │  └─────────────────┘ │ │
│  │                    │   - Models    │                       │ │
│  │                    │   - Workflows │                       │ │
│  │                    └──────────────┘                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implemented Components

### 1. Test Infrastructure ✅

#### Configuration Files
- **playwright.config.ts**: Main Playwright configuration
  - Multi-browser support (Chromium, Firefox, WebKit)
  - Mobile viewport testing
  - Screenshot and video recording
  - HTML/JSON/JUnit reporters
  - Web server integration

- **tsconfig.json**: TypeScript configuration
- **package.json**: Dependencies and npm scripts

#### Directory Structure
```
test/e2e/
├── fixtures/
│   ├── dashboard.fixture.ts      # Dashboard testing utilities
│   └── mcp-server.fixture.ts     # MCP server log capture
├── tests/
│   ├── 01-dashboard-core.spec.ts
│   ├── 02-github-runners.spec.ts
│   ├── 03-model-download.spec.ts
│   ├── 04-model-inference.spec.ts
│   └── 05-comprehensive.spec.ts
└── utils/
    ├── log-correlator.ts          # Log correlation engine
    ├── screenshot-manager.ts      # Screenshot utilities
    └── report-generator.ts        # Report generation
```

### 2. Test Fixtures ✅

#### Dashboard Fixture (`dashboard.fixture.ts`)
Provides:
- Console log capture (log, info, warn, error, debug)
- Page error tracking
- Screenshot management with auto-incrementing
- Tab navigation helpers
- MCP SDK readiness verification
- Console log filtering and search
- MCP tool invocation utilities

**Example Usage:**
```typescript
test('my test', async ({ page, dashboard }) => {
  await page.goto('/');
  await dashboard.waitForMCPReady();
  await dashboard.navigateToTab('Model Manager');
  await dashboard.takeScreenshot('model-manager');
  
  const logs = dashboard.getConsoleLogs('error');
  expect(logs.length).toBe(0);
});
```

#### MCP Server Fixture (`mcp-server.fixture.ts`)
Provides:
- Server log capture
- Structured log parsing (JSON detection)
- Log pattern matching
- Time-based log filtering
- Server lifecycle management

### 3. Utility Modules ✅

#### Log Correlator (`log-correlator.ts`)
**Features:**
- Correlate dashboard and server logs by timestamp proximity
- Pre-defined correlation patterns for common operations
- Time delta analysis
- Correlation report generation
- Sequential pattern matching

**Common Patterns:**
- MCP SDK initialization ↔ Server start
- Model download ↔ Download progress logs
- AI inference ↔ Inference request logs
- GitHub workflow ↔ Workflow queue creation
- Runner provisioning ↔ Runner creation logs
- Model search ↔ HuggingFace API calls
- Hardware info ↔ System detection logs
- Network peers ↔ Peer connection logs

**Example Usage:**
```typescript
const correlator = new LogCorrelator();
const patterns = LogCorrelator.getCommonPatterns();

const correlations = correlator.findCorrelations(
  dashboardLogs,
  serverLogs,
  patterns
);

console.log(correlator.generateReport());
```

#### Screenshot Manager (`screenshot-manager.ts`)
**Features:**
- Baseline/current/diff directory management
- Screenshot comparison
- Responsive design testing (multiple viewports)
- Annotated screenshots with element highlights
- Visual regression testing

**Standard Viewports:**
- Desktop 1080p (1920x1080)
- Desktop Laptop (1366x768)
- Tablet Portrait (768x1024)
- Mobile iPhone (375x667)
- Mobile Large (414x896)

**Example Usage:**
```typescript
const screenshotMgr = new ScreenshotManager('my-test');

await screenshotMgr.captureAndCompare(page, 'initial-state');
await screenshotMgr.captureResponsive(page, 'responsive', 
  ScreenshotManager.getStandardViewports()
);
await screenshotMgr.captureAnnotated(page, 'highlighted', [
  { selector: '#important-element', label: 'Key Feature' }
]);
```

#### Report Generator (`report-generator.ts`)
**Features:**
- JSON and HTML report generation
- Test result aggregation
- Screenshot embedding
- Log correlation display
- Summary statistics
- Detailed test breakdowns

### 4. Test Suites ✅

#### 01-dashboard-core.spec.ts
**Tests:**
- Dashboard loading and MCP SDK initialization
- Navigation through all 13 tabs
- Console log capture and validation
- Server status display
- Responsive design (5 viewports)

**Tabs Tested:**
1. Overview
2. AI Inference
3. Advanced AI
4. Model Manager
5. IPFS Manager
6. Network & Status
7. Queue Monitor
8. GitHub Workflows
9. Runner Management
10. SDK Playground
11. MCP Tools
12. Coverage Analysis
13. System Logs

#### 02-github-runners.spec.ts
**Tests:**
- GitHub Workflows tab display and workflow loading
- Runner management interface
- MCP tool calls for runner operations
- Log correlation between dashboard and server
- End-to-end runner provisioning workflow

**Log Correlation Points:**
- Workflow tab click → gh_create_workflow_queues call
- Runner list load → gh_list_runners call
- Runner actions → Server log entries

#### 03-model-download.spec.ts
**Tests:**
- Model Manager tab and search interface
- Model search functionality
- Model details display
- Download initiation
- Download progress tracking
- Log correlation for downloads

**Log Correlation Points:**
- Model search → HuggingFace API calls
- Download button → Download API request
- Progress updates → Server download logs

#### 04-model-inference.spec.ts
**Tests:**
- AI Inference tab display
- Model selection interface
- Inference parameter configuration
- Inference execution
- Result display
- Advanced AI operations
- Log correlation for inference

**Log Correlation Points:**
- Inference start → Server inference request log
- Model loading → Model load logs
- Inference complete → Result logs

#### 05-comprehensive.spec.ts
**Tests:**
- Complete workflow: dashboard → runners → models → inference
- All tab functionality verification
- Stress testing (rapid navigation)
- MCP tool execution end-to-end
- Multi-step workflow validation

### 5. CI/CD Integration ✅

#### GitHub Actions Workflow
**File:** `.github/workflows/playwright-e2e.yml`

**Features:**
- Matrix strategy for multi-browser testing
- Python and Node.js setup
- Automated server startup
- Test execution
- Artifact upload (reports, screenshots)
- Test result publishing
- Report merging

**Triggered On:**
- Push to main/develop
- Pull requests
- Manual workflow dispatch

### 6. Documentation ✅

#### README.md
Comprehensive documentation including:
- Installation instructions
- Running tests (all variants)
- Test structure explanation
- Test scenarios overview
- Log correlation patterns
- Screenshot locations
- Report viewing
- CI/CD integration
- Environment variables
- Troubleshooting guide
- Development guidelines
- Best practices

---

## Usage Examples

### Basic Test Run
```bash
# Install dependencies
npm install
npm run install:browsers

# Run all tests
npm test

# View report
npm run report
```

### Specific Test Suites
```bash
# Core functionality only
npm run test:core

# GitHub runners
npm run test:runners

# Models (download + inference)
npm run test:models

# Comprehensive workflows
npm run test:comprehensive
```

### Browser-Specific
```bash
# Chromium
npm run test:chromium

# Firefox
npm run test:firefox

# WebKit (Safari)
npm run test:webkit

# Mobile browsers
npm run test:mobile
```

### Debug Mode
```bash
# Interactive debugging
npm run test:debug

# Visible browser
npm run test:headed

# Interactive UI
npm run test:ui
```

---

## Test Scenarios in Detail

### Scenario 1: GitHub Runner Provisioning with Log Correlation

```typescript
test('runner provisioning with logs', async ({ page }) => {
  const consoleLogs = [];
  
  page.on('console', msg => {
    consoleLogs.push({
      type: msg.type(),
      text: msg.text(),
      timestamp: new Date().toISOString(),
    });
  });
  
  // Navigate to Runner Management
  await page.goto('/');
  await page.locator('button.nav-tab:has-text("Runner Management")').click();
  
  // Trigger runner action
  await page.locator('button:has-text("Load Runners")').click();
  await page.waitForTimeout(3000);
  
  // Verify logs show MCP tool call
  const runnerLogs = consoleLogs.filter(log =>
    /gh_list_runners|runner/i.test(log.text)
  );
  
  expect(runnerLogs.length).toBeGreaterThan(0);
});
```

**Expected Log Correlation:**
```
Dashboard Console: [info] Calling MCP tool: gh_list_runners
↓ (within 2000ms)
MCP Server Log: [INFO] Executing tool: gh_list_runners with params: {...}
↓ (within 3000ms)
MCP Server Log: [INFO] gh_list_runners completed: found 5 runners
↓ (within 1000ms)
Dashboard Console: [info] Loaded 5 runners
```

### Scenario 2: AI Model Download with Progress Tracking

```typescript
test('model download with progress', async ({ page }) => {
  const screenshotMgr = new ScreenshotManager('model-download');
  const downloadLogs = [];
  
  page.on('console', msg => {
    if (/download/i.test(msg.text())) {
      downloadLogs.push(msg.text());
    }
  });
  
  await page.goto('/');
  await page.locator('button.nav-tab:has-text("Model Manager")').click();
  
  await screenshotMgr.captureAndCompare(page, 'before-download');
  
  // Initiate download
  await page.locator('button:has-text("Download")').first().click();
  await page.waitForTimeout(2000);
  
  await screenshotMgr.captureAndCompare(page, 'download-started');
  
  // Verify download logs
  const progressLogs = downloadLogs.filter(log => 
    /progress|percent|downloaded/i.test(log)
  );
  
  console.log('Download progress logs:', progressLogs);
});
```

**Expected Log Sequence:**
1. Download button click captured
2. Dashboard console: "Downloading model: model-name"
3. Server log: "Model download initiated"
4. Progress updates in both dashboard and server
5. Completion log in both places
6. Screenshots at each stage

### Scenario 3: AI Inference with Result Validation

```typescript
test('inference with result validation', async ({ page }) => {
  const consoleLogs = [];
  
  page.on('console', msg => consoleLogs.push(msg));
  
  await page.goto('/');
  await page.locator('button.nav-tab:has-text("AI Inference")').click();
  
  // Set up inference
  await page.locator('textarea').fill('Test prompt');
  
  // Clear previous logs
  consoleLogs.length = 0;
  const startTime = Date.now();
  
  // Run inference
  await page.locator('button:has-text("Run Inference")').click();
  await page.waitForTimeout(5000);
  
  const endTime = Date.now();
  
  // Analyze logs in time window
  const inferenceLogs = consoleLogs.filter(log =>
    /inference|generate|complete/i.test(log.text())
  );
  
  // Verify expected sequence
  const patterns = [
    /inference.*start/i,
    /model.*load/i,
    /inference.*complete/i,
  ];
  
  for (const pattern of patterns) {
    const found = inferenceLogs.some(log => pattern.test(log.text()));
    expect(found).toBeTruthy();
  }
});
```

---

## Log Correlation Patterns in Detail

### Pattern 1: MCP SDK Initialization
**Dashboard Pattern:** `/MCP SDK client initialized/i`  
**Server Pattern:** `/MCP.*server.*start/i`  
**Max Time Delta:** 5000ms

**Validation:**
- Dashboard: MCP client object exists
- Server: Server started on specified port
- Correlation: Both events within 5 seconds

### Pattern 2: Model Download
**Dashboard Pattern:** `/Downloading model.*(\w+)/i`  
**Server Pattern:** `/download.*model/i`  
**Max Time Delta:** 10000ms

**Validation:**
- Dashboard: Download UI shows progress
- Server: Download service logs show file transfer
- Correlation: Progress updates align temporally

### Pattern 3: AI Inference
**Dashboard Pattern:** `/Running inference/i`  
**Server Pattern:** `/inference.*request/i`  
**Max Time Delta:** 10000ms

**Validation:**
- Dashboard: Inference button clicked
- Server: Inference engine processes request
- Result: Output appears in dashboard
- Correlation: Complete chain within time window

### Pattern 4: GitHub Workflow
**Dashboard Pattern:** `/GitHub.*workflow/i`  
**Server Pattern:** `/gh_create_workflow_queues|workflow.*created/i`

**Validation:**
- Dashboard: Workflow tab shows queues
- Server: MCP tool gh_create_workflow_queues executed
- Correlation: Queue creation matches display

### Pattern 5: Runner Provisioning
**Dashboard Pattern:** `/runner.*provision/i`  
**Server Pattern:** `/runner.*created|provision.*runner/i`

**Validation:**
- Dashboard: Runner UI updates
- Server: Runner management tool logs
- Correlation: Runner state changes match

---

## Screenshot Management

### Automatic Screenshots
Taken at key points:
1. Dashboard loaded
2. Tab navigation (each tab)
3. Before/after actions
4. Error states
5. Final state

### Visual Regression
- **Baseline**: First run creates baseline
- **Current**: Each run captures current state
- **Diff**: Differences highlighted if found

### Directory Structure
```
test-results/
├── screenshots/
│   ├── 01_dashboard-loaded.png
│   ├── 02_tab-ai-inference.png
│   └── ...
└── visual-regression/
    ├── baseline/
    ├── current/
    └── diff/
```

---

## Report Generation

### HTML Report
Comprehensive HTML report with:
- Test summary (passed/failed/skipped)
- Execution duration
- Console logs for each test
- Server logs for each test
- Log correlations with time deltas
- Embedded screenshots
- Interactive navigation

### JSON Report
Machine-readable format with:
- Detailed test results
- Log data
- Correlation data
- Timing information
- Perfect for further analysis

### JUnit XML
For CI/CD integration:
- Compatible with standard CI tools
- Test result publishing
- Historical tracking

---

## Extending the Test Suite

### Adding New Tests

1. **Create test file:**
```typescript
// test/e2e/tests/06-my-feature.spec.ts
import { test, expect } from '@playwright/test';
import { ScreenshotManager } from '../utils/screenshot-manager';

test.describe('My Feature', () => {
  test('should work correctly', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('my-feature');
    
    await page.goto('/');
    await screenshotMgr.captureAndCompare(page, 'initial');
    
    // Test implementation
  });
});
```

2. **Add correlation pattern:**
```typescript
// In log-correlator.ts
{
  dashboardPattern: /my.*feature/i,
  serverPattern: /feature.*executed/i,
  description: 'My feature execution',
}
```

3. **Update CI workflow:**
```yaml
# Add to test matrix or create separate job
```

### Creating Custom Fixtures

```typescript
// test/e2e/fixtures/my-fixture.ts
import { test as base } from '@playwright/test';

export interface MyFixture {
  myHelper: () => Promise<void>;
}

export const test = base.extend<{ myFixture: MyFixture }>({
  myFixture: async ({}, use) => {
    const fixture: MyFixture = {
      myHelper: async () => {
        // Implementation
      },
    };
    
    await use(fixture);
  },
});
```

---

## Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order

### 2. Waiting Strategies
```typescript
// ❌ Bad: Fixed waits
await page.waitForTimeout(5000);

// ✅ Good: Conditional waits
await page.waitForSelector('#element');
await page.waitForFunction(() => window.ready);
```

### 3. Log Correlation
```typescript
// ✅ Good: Time-based correlation
const startTime = Date.now();
// Action
const endTime = Date.now();
const relevantLogs = logs.filter(log =>
  logTime >= startTime && logTime <= endTime
);
```

### 4. Screenshot Strategy
```typescript
// Take screenshots at meaningful points
await screenshotMgr.captureAndCompare(page, 'before-action');
// Action
await screenshotMgr.captureAndCompare(page, 'after-action');
// Use full-page for overview
await screenshotMgr.captureAndCompare(page, 'overview', { fullPage: true });
```

### 5. Error Handling
```typescript
try {
  await someAction();
} catch (error) {
  await screenshotMgr.captureAndCompare(page, 'error-state');
  console.log('Logs at error:', consoleLogs);
  throw error;
}
```

---

## Performance Considerations

### Test Execution Time
- Average test suite: 5-10 minutes
- Per test: 30-60 seconds
- Can be parallelized across browsers

### Resource Usage
- Memory: ~500MB per browser instance
- Disk: ~100MB for screenshots/videos per run
- Network: Depends on API calls

### Optimization Tips
1. Run tests in parallel when possible
2. Use selective test execution during development
3. Clean up old test results regularly
4. Use headed mode only when debugging

---

## Troubleshooting Guide

### Common Issues

#### 1. Server Not Starting
**Symptom:** Tests fail immediately with connection errors

**Solution:**
```bash
# Start server manually first
python -m ipfs_accelerate_py.mcp_dashboard --port 3001

# Then run tests with existing server
# Set in playwright.config.ts:
webServer: { reuseExistingServer: true }
```

#### 2. Tests Timing Out
**Symptom:** Tests exceed timeout limits

**Solution:**
```typescript
// Increase timeouts in playwright.config.ts
timeout: 180 * 1000,  // 3 minutes
```

#### 3. Log Correlation Failures
**Symptom:** No correlations found

**Solution:**
1. Check MCP server is logging correctly
2. Verify timestamp formats match
3. Adjust maxTimeDelta in patterns
4. Check log patterns match actual logs

#### 4. Screenshot Comparison Failures
**Symptom:** Visual regression tests fail unexpectedly

**Solution:**
1. Review diff images in test-results/visual-regression/diff/
2. Update baseline if changes are intentional
3. Mask dynamic elements (timestamps, etc.)

---

## Future Enhancements

### Planned Improvements
1. ✅ Video recording for failed tests
2. ⏳ Real-time log streaming from MCP server
3. ⏳ Performance metrics collection
4. ⏳ Accessibility testing integration
5. ⏳ Load testing capabilities
6. ⏳ API response time tracking
7. ⏳ Memory leak detection
8. ⏳ Network traffic analysis

### Integration Opportunities
1. Grafana dashboards for test metrics
2. Slack notifications for test failures
3. Automated issue creation for failures
4. Historical trend analysis
5. Flaky test detection

---

## Conclusion

This comprehensive Playwright E2E testing suite provides:

✅ **Complete Coverage**: Tests all dashboard features  
✅ **Log Correlation**: Verifies end-to-end workflows  
✅ **Visual Documentation**: Screenshot capture at all stages  
✅ **Multi-Browser**: Chrome, Firefox, Safari support  
✅ **CI/CD Ready**: GitHub Actions integration  
✅ **Detailed Reports**: HTML, JSON, JUnit formats  
✅ **Developer Friendly**: Clear documentation and examples  
✅ **Extensible**: Easy to add new tests and features  

The test suite is production-ready and can be integrated into your CI/CD pipeline immediately.

---

## Support and Contribution

### Getting Help
1. Check this documentation
2. Review test-results/ directory
3. Check GitHub Issues
4. Contact the team

### Contributing
1. Follow existing patterns
2. Add appropriate documentation
3. Include screenshots
4. Verify CI passes
5. Submit pull request

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-04  
**Status:** Complete and Ready for Use
