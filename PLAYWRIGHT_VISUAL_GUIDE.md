# Playwright E2E Testing Suite - Visual Guide

## 🎯 Testing Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PLAYWRIGHT TEST RUNNER                    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Test Specs  │  │   Fixtures   │  │    Utilities     │ │
│  │              │  │              │  │                  │ │
│  │ • Core       │  │ • Dashboard  │  │ • Log Correlator │ │
│  │ • Runners    │  │ • MCP Server │  │ • Screenshots    │ │
│  │ • Models     │  │              │  │ • Reports        │ │
│  │ • Inference  │  │              │  │                  │ │
│  │ • E2E        │  │              │  │                  │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              BROWSERS (Chromium/Firefox/WebKit)              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         IPFS Accelerate Dashboard (HTML/JS)            │ │
│  │                                                        │ │
│  │  ┌──────────┐  ┌─────────────┐  ┌─────────────────┐ │ │
│  │  │ MCP SDK  │→ │  Dashboard  │→ │  UI Components  │ │ │
│  │  │ Client   │  │  Controller │  │  - Tabs         │ │ │
│  │  └──────────┘  └─────────────┘  │  - Forms        │ │ │
│  │       ↓                          │  - Results      │ │ │
│  │  Console Logs                    └─────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │ JSON-RPC
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP SERVER (Python)                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Flask Dashboard Server                    │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │ │
│  │  │  JSON-RPC    │→ │   MCP Tools  │→ │ Server Logs│ │ │
│  │  │  Endpoint    │  │  - Inference │  │ (captured) │ │ │
│  │  └──────────────┘  │  - Runners   │  └────────────┘ │ │
│  │                    │  - Models    │                  │ │
│  │                    │  - Workflows │                  │ │
│  │                    └──────────────┘                  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Test Flow Diagram

```
┌─────────────┐
│  Start Test │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Navigate to Page    │
│ - goto('/')        │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Wait for MCP Ready  │
│ - SDK initialized   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐     ┌──────────────┐
│ Perform Action      │────→│ Take         │
│ - Click button      │     │ Screenshot   │
│ - Fill form         │     └──────────────┘
│ - Navigate tab      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐     ┌──────────────┐
│ Capture Logs        │────→│ Dashboard    │
│ - Console logs      │     │ Console Logs │
│ - Network requests  │     └──────────────┘
└──────┬──────────────┘
       │                    ┌──────────────┐
       │                    │ MCP Server   │
       ├───────────────────→│ Logs         │
       │                    └──────────────┘
       ▼
┌─────────────────────┐
│ Correlate Logs      │
│ - Match patterns    │
│ - Verify timing     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Assert Results      │
│ - UI state correct  │
│ - Logs match        │
│ - No errors         │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Generate Report     │
│ - Screenshots       │
│ - Logs              │
│ - Correlations      │
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│  Test Done  │
└─────────────┘
```

## 🔄 Log Correlation Flow

```
┌────────────────┐
│ User Action in │
│ Dashboard      │
└────────┬───────┘
         │
         ▼
┌────────────────────────┐
│ Dashboard Console Log  │
│ "Downloading model X"  │
│ Timestamp: T0          │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ JSON-RPC Request       │
│ POST /jsonrpc          │
│ tools/call             │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ MCP Server Log         │
│ "Model download start" │
│ Timestamp: T0 + 500ms  │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ MCP Server Log         │
│ "Download progress"    │
│ Timestamp: T0 + 2000ms │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Dashboard Console Log  │
│ "Download complete"    │
│ Timestamp: T0 + 5000ms │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Log Correlator         │
│ - Finds matching logs  │
│ - Calculates delta     │
│ - Validates sequence   │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Correlation Report     │
│ ✓ All logs matched     │
│ ✓ Within time window   │
└────────────────────────┘
```

## 📸 Screenshot Capture Points

```
Test Execution Timeline
├─ 00:00 - Dashboard Loaded        → Screenshot #1
├─ 00:02 - Tab Navigation          → Screenshot #2
├─ 00:03 - Before Action           → Screenshot #3
├─ 00:05 - Action In Progress      → Screenshot #4
├─ 00:08 - After Action            → Screenshot #5
└─ 00:10 - Final State             → Screenshot #6

Each Screenshot Includes:
✓ Full page capture
✓ Console logs up to that point
✓ Network requests
✓ Current timestamp
✓ Browser viewport info
```

## 🎭 Test Suite Organization

```
e2e/
│
├── fixtures/                    ← Reusable test helpers
│   ├── dashboard.fixture.ts    ← Dashboard utilities
│   └── mcp-server.fixture.ts   ← Server log capture
│
├── tests/                       ← Actual test specs
│   ├── 01-dashboard-core.spec.ts
│   │   └── Tests: Loading, SDK, Tabs, Logs
│   │
│   ├── 02-github-runners.spec.ts
│   │   └── Tests: Workflows, Runners, Provisioning
│   │
│   ├── 03-model-download.spec.ts
│   │   └── Tests: Search, Download, Progress
│   │
│   ├── 04-model-inference.spec.ts
│   │   └── Tests: Selection, Execution, Results
│   │
│   └── 05-comprehensive.spec.ts
│       └── Tests: E2E Workflows, Stress Test
│
└── utils/                       ← Utility modules
    ├── log-correlator.ts       ← Log matching engine
    ├── screenshot-manager.ts   ← Screenshot utilities
    └── report-generator.ts     ← Report creation
```

## 🔍 How Tests Validate Functionality

```
┌──────────────────────────────────────────────────────────┐
│                    TEST VALIDATION                        │
└──────────────────────────────────────────────────────────┘

1. UI Validation
   ├─ Element exists          → await expect(element).toBeVisible()
   ├─ Element has text        → await expect(element).toContainText()
   └─ Element is interactive  → await element.click()

2. Console Log Validation
   ├─ Capture all logs        → page.on('console', ...)
   ├─ Filter by pattern       → logs.filter(log => /pattern/.test())
   └─ Validate sequence       → LogMatcher.matchSequence()

3. Server Log Validation
   ├─ Capture server output   → mcpServer.serverLogs
   ├─ Parse structured logs   → JSON.parse(logData)
   └─ Match with dashboard    → correlator.findCorrelations()

4. Network Validation
   ├─ Capture requests        → page.on('request', ...)
   ├─ Verify endpoints called → requests.filter(url => /api/)
   └─ Check response data     → await response.json()

5. Screenshot Validation
   ├─ Capture current state   → screenshotMgr.capture()
   ├─ Compare with baseline   → pixelmatch comparison
   └─ Generate diff           → highlight differences

6. Correlation Validation
   ├─ Match log patterns      → LogCorrelator patterns
   ├─ Verify timing           → time delta < maxDelta
   └─ Generate report         → correlator.generateReport()
```

## 📈 Report Generation Flow

```
Test Results
├─ Test 1 (Passed)
│  ├─ Screenshots: 6
│  ├─ Console Logs: 42
│  ├─ Server Logs: 28
│  └─ Correlations: 8
│
├─ Test 2 (Failed)
│  ├─ Screenshots: 4
│  ├─ Console Logs: 35
│  ├─ Server Logs: 22
│  ├─ Correlations: 5
│  └─ Error: Assertion failed
│
└─ Test 3 (Skipped)

        ↓

Report Generator
├─ Aggregate results
├─ Embed screenshots
├─ Format logs
├─ Calculate statistics
└─ Generate HTML/JSON

        ↓

Output Files
├─ test-results/html-report/index.html
├─ test-results/test-results.json
├─ test-results/junit.xml
└─ test-results/screenshots/*.png
```

## 🚀 CI/CD Pipeline

```
GitHub Push/PR
       │
       ▼
┌─────────────────┐
│ GitHub Actions  │
│ Workflow Start  │
└────────┬────────┘
         │
         ├─────────────────────────────────┐
         │                                 │
         ▼                                 ▼
┌────────────────┐              ┌────────────────┐
│ Job: Chromium  │              │ Job: Firefox   │
│                │              │                │
│ 1. Setup       │              │ 1. Setup       │
│ 2. Install     │              │ 2. Install     │
│ 3. Start Server│              │ 3. Start Server│
│ 4. Run Tests   │              │ 4. Run Tests   │
│ 5. Upload      │              │ 5. Upload      │
└────────┬───────┘              └────────┬───────┘
         │                                │
         └─────────────┬──────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Job: WebKit    │
              │                │
              │ 1. Setup       │
              │ 2. Install     │
              │ 3. Start Server│
              │ 4. Run Tests   │
              │ 5. Upload      │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │ Merge Reports  │
              │ Publish Results│
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │ Artifacts      │
              │ - HTML Report  │
              │ - Screenshots  │
              │ - JUnit XML    │
              └────────────────┘
```

## 🎨 Legend

```
┌────────┐
│ Symbol │ Meaning
├────────┼─────────────────────────
│   →    │ Flow direction
│   ↓    │ Data flow down
│   ├─   │ Branch/Connection
│   └─   │ End branch
│   ▼    │ Sequential step
│   ✓    │ Success/Complete
│   ✗    │ Failure/Error
└────────┴─────────────────────────
```

## 📚 Quick Reference

### Common Patterns

```typescript
// Navigate and capture
await page.goto('/');
await screenshotMgr.capture(page, 'loaded');

// Wait for element
await expect(page.locator('#element')).toBeVisible();

// Capture logs
page.on('console', msg => logs.push(msg));

// Correlate logs
const matches = correlator.findCorrelations(
  dashboardLogs,
  serverLogs,
  patterns
);

// Assert correlation
expect(matches.length).toBeGreaterThan(0);
```

### Test Structure

```typescript
test.describe('Feature', () => {
  test('should work', async ({ page }) => {
    // Setup
    const mgr = new ScreenshotManager('test');
    
    // Action
    await page.goto('/');
    await page.click('button');
    
    // Capture
    await mgr.capture(page, 'after-click');
    
    // Assert
    await expect(page.locator('.result')).toBeVisible();
  });
});
```

---

**This visual guide helps understand the testing architecture and flow. For detailed usage, see the comprehensive documentation.**
