# Playwright E2E Testing Suite for IPFS Accelerate Dashboard

## Overview

This comprehensive Playwright testing suite provides end-to-end testing for the IPFS Accelerate Dashboard with full log correlation between dashboard actions and MCP server operations.

## Features

- ✅ **Comprehensive Dashboard Testing**: Tests all 13 dashboard tabs
- ✅ **MCP Log Correlation**: Matches dashboard events with MCP server logs
- ✅ **Screenshot Capture**: Automated visual documentation of tests
- ✅ **Console Log Validation**: Captures and validates JavaScript console logs
- ✅ **Network Request Tracking**: Monitors all API calls
- ✅ **Visual Regression**: Screenshot comparison capabilities
- ✅ **Multi-Browser Support**: Tests on Chromium, Firefox, and WebKit
- ✅ **Mobile Testing**: Responsive design validation
- ✅ **Detailed Reports**: HTML and JSON test reports

## Installation

### Prerequisites

- Node.js >= 18.0.0
- Python >= 3.8
- IPFS Accelerate Dashboard server

### Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Playwright browsers
npm run install:browsers

# Install system dependencies (Linux only)
npm run install:deps
```

### Python Dependencies

The dashboard server must be running. Install Python dependencies:

```bash
pip install -r requirements_dashboard.txt
```

## Running Tests

### All Tests

```bash
npm test
```

### Specific Test Suites

```bash
# Core dashboard functionality
npm run test:core

# GitHub runners provisioning
npm run test:runners

# AI model download and inference
npm run test:models

# Comprehensive workflow tests
npm run test:comprehensive

# IPFS operations
npm run test:ipfs

# Advanced features (workflows, multiplex, CLI)
npm run test:advanced

# System monitoring (hardware, logs, metrics)
npm run test:system

# Distributed & backend (P2P, Copilot, backends)
npm run test:distributed

# Complete tool coverage (all 100+ tools)
npm run test:complete
```

### Browser-Specific Tests

```bash
# Chromium only
npm run test:chromium

# Firefox only
npm run test:firefox

# WebKit (Safari) only
npm run test:webkit

# Mobile browsers
npm run test:mobile
```

### Debug Mode

```bash
# Interactive debug mode
npm run test:debug

# Headed mode (visible browser)
npm run test:headed

# Interactive UI mode
npm run test:ui
```

## Test Structure

```
e2e/
├── fixtures/           # Test fixtures and utilities
│   ├── dashboard.fixture.ts      # Dashboard-specific helpers
│   └── mcp-server.fixture.ts     # MCP server log capture
├── tests/              # Test specifications
│   ├── 01-dashboard-core.spec.ts       # Core functionality
│   ├── 02-github-runners.spec.ts       # GitHub runners
│   ├── 03-model-download.spec.ts       # Model downloads
│   ├── 04-model-inference.spec.ts      # AI inference
│   └── 05-comprehensive.spec.ts        # Full workflows
└── utils/              # Utility modules
    ├── log-correlator.ts          # Log correlation engine
    ├── screenshot-manager.ts      # Screenshot utilities
    └── report-generator.ts        # Report generation
```

## Test Scenarios

### 1. Dashboard Core (01-dashboard-core.spec.ts)

- ✅ Dashboard loading and MCP SDK initialization
- ✅ Tab navigation (all 13 tabs)
- ✅ Console log capture and validation
- ✅ Server status display
- ✅ Responsive design testing

### 2. GitHub Runners (02-github-runners.spec.ts)

- ✅ GitHub Workflows tab display
- ✅ Runner management interface
- ✅ MCP tool calls for runner operations
- ✅ Log correlation between dashboard and server
- ✅ End-to-end runner provisioning workflow

### 3. Model Download (03-model-download.spec.ts)

- ✅ Model Manager tab and search interface
- ✅ Model search functionality
- ✅ Model details display
- ✅ Download initiation
- ✅ Download progress tracking
- ✅ Log correlation for downloads

### 4. Model Inference (04-model-inference.spec.ts)

- ✅ AI Inference tab display
- ✅ Model selection interface
- ✅ Inference parameter configuration
- ✅ Inference execution
- ✅ Result display
- ✅ Advanced AI operations
- ✅ Log correlation for inference

### 5. Comprehensive Workflows (05-comprehensive.spec.ts)

- ✅ Complete workflow: dashboard → runners → models → inference
- ✅ All tab functionality verification
- ✅ Stress testing (rapid navigation)
- ✅ MCP tool execution end-to-end

### 6. IPFS Operations (06-ipfs-operations.spec.ts)

- ✅ IPFS Manager tab functionality
- ✅ File operations (add, cat, ls, mkdir, pin)
- ✅ Network operations (id, swarm peers, pubsub, DHT)
- ✅ IPFS tool integration via MCP

### 7. Advanced Features (07-advanced-features.spec.ts)

- ✅ Multiplex inference configuration
- ✅ Endpoint registration and management
- ✅ CLI endpoint tools
- ✅ Queue history and monitoring
- ✅ Distributed inference capabilities
- ✅ Workflow management (create, list, execute, templates)
- ✅ HuggingFace model search integration

### 8. System Monitoring (08-system-monitoring.spec.ts)

- ✅ Hardware information retrieval
- ✅ Model acceleration options
- ✅ Model benchmarking
- ✅ System logs retrieval and filtering
- ✅ Error log filtering
- ✅ Performance metrics display
- ✅ Coverage analysis
- ✅ MCP tools display

### 9. Distributed & Backend (09-distributed-backend.spec.ts)

- ✅ P2P scheduler status
- ✅ Task submission to P2P network
- ✅ Peer state management
- ✅ Merkle clock operations
- ✅ Copilot command suggestions
- ✅ Copilot SDK sessions
- ✅ Backend listing and configuration
- ✅ Docker container management
- ✅ Complete feature coverage validation

### 10. Complete Tool Coverage (10-complete-tool-coverage.spec.ts)

- ✅ Docker tools (execute, build, list, stop, pull)
- ✅ Backend management (status, selection, routing, tasks)
- ✅ Hardware tools (info, test, recommend)
- ✅ Shared tools (generate, classify, IPFS, models, network)
- ✅ CLI adapter tools (register, list, execute)
- ✅ Verification of all 100+ MCP tools
- ✅ Actual MCP tool invocations with arguments

## Log Correlation

The test suite automatically correlates dashboard actions with MCP server logs using common patterns:

| Dashboard Action | MCP Server Log Pattern | Description |
|-----------------|------------------------|-------------|
| SDK Initialization | `MCP.*server.*start` | MCP SDK initialization |
| Model Download | `download.*model` | Model download |
| AI Inference | `inference.*request` | AI inference |
| GitHub Workflow | `gh_create_workflow_queues` | GitHub workflow creation |
| Runner Provisioning | `runner.*created` | Runner provisioning |
| Model Search | `search.*huggingface` | Model search |
| Hardware Info | `hardware.*detected` | Hardware info |
| Network Peers | `peer.*connected` | Network peer status |

## Screenshots

Screenshots are automatically captured during tests and saved to:

```
test-results/
├── screenshots/           # Test run screenshots
├── visual-regression/     # Visual regression baselines
│   ├── baseline/
│   ├── current/
│   └── diff/
└── html-report/          # HTML test reports
```

## Reports

After running tests, view reports:

```bash
# Open HTML report
npm run report

# Reports are also available at:
# - test-results/html-report/index.html
# - test-results/test-results.json
# - test-results/junit.xml
```

## Configuration

Edit `playwright.config.ts` to customize:

- Base URL (default: `http://localhost:3001`)
- Timeout values
- Screenshot/video settings
- Browser configurations
- Viewport sizes

## CI/CD Integration

### GitHub Actions

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: |
          npm install
          npx playwright install --with-deps
      - name: Start MCP server
        run: |
          python -m ipfs_accelerate_py.mcp_dashboard --port 3001 &
          sleep 10
      - name: Run tests
        run: npm test
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report
          path: test-results/
```

## Environment Variables

```bash
# Dashboard URL (default: http://localhost:3001)
export DASHBOARD_URL=http://localhost:3001

# MCP Server settings
export MCP_SERVER_PORT=3001
export MCP_SERVER_HOST=localhost

# CI mode (enables retries and different settings)
export CI=true
```

## Troubleshooting

### Server Not Starting

If the dashboard server doesn't start automatically:

1. Start it manually:
   ```bash
   python -m ipfs_accelerate_py.mcp_dashboard --port 3001
   ```

2. Set `reuseExistingServer: true` in `playwright.config.ts`

### Tests Timing Out

Increase timeouts in `playwright.config.ts`:

```typescript
timeout: 120 * 1000,  // 2 minutes
navigationTimeout: 60 * 1000,  // 1 minute
```

### Browser Installation Issues

```bash
# Reinstall browsers
npx playwright install --with-deps chromium firefox webkit
```

### Log Correlation Issues

If logs aren't correlating:

1. Verify MCP server is running with verbose logging
2. Check `test-results/` for captured logs
3. Adjust `maxTimeDelta` in log correlation patterns

## Development

### Adding New Tests

1. Create a new spec file in `e2e/tests/`
2. Import required fixtures and utilities
3. Use the dashboard fixture for console log capture
4. Use the screenshot manager for visual documentation

Example:

```typescript
import { test, expect } from '@playwright/test';
import { ScreenshotManager } from '../utils/screenshot-manager';

test.describe('My New Feature', () => {
  test('should do something', async ({ page }) => {
    const screenshotMgr = new ScreenshotManager('my-feature');
    
    await page.goto('/');
    await screenshotMgr.captureAndCompare(page, 'initial-state');
    
    // Your test code here
  });
});
```

### Extending Fixtures

Add custom fixtures in `e2e/fixtures/`:

```typescript
export const test = base.extend<{ myFixture: MyFixture }>({
  myFixture: async ({}, use) => {
    // Setup
    const fixture = { /* ... */ };
    await use(fixture);
    // Teardown
  },
});
```

## Best Practices

1. **Always use screenshots**: Document visual state at key points
2. **Correlate logs**: Use log correlation utilities to verify end-to-end flow
3. **Wait appropriately**: Use `waitForTimeout` judiciously, prefer `waitForSelector`
4. **Handle async**: Properly await all async operations
5. **Isolate tests**: Each test should be independent
6. **Clean up**: Use fixtures for setup/teardown

## Contributing

When adding tests:

1. Follow existing naming conventions
2. Add appropriate log correlation patterns
3. Include screenshots for visual verification
4. Update this README with new test scenarios
5. Ensure tests pass in CI environment

## License

AGPL-3.0 - See LICENSE file

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review test-results/ for detailed logs
3. Open an issue on GitHub
