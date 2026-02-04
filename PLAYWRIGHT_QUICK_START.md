# Playwright E2E Testing - Quick Start Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Node.js 18+
- Python 3.8+
- Git

### Step 1: Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Playwright browsers
npm run install:browsers

# Install Python dependencies (if not already installed)
pip install -r requirements_dashboard.txt
```

### Step 2: Start the Dashboard Server

In a separate terminal:

```bash
python -m ipfs_accelerate_py.mcp_dashboard --port 3001
```

Wait for the server to start (you should see "Running on http://localhost:3001")

### Step 3: Run Tests

```bash
# Run all tests
npm test

# Or run specific test suites
npm run test:core          # Core dashboard tests
npm run test:runners       # GitHub runners tests
npm run test:models        # Model download/inference tests
npm run test:comprehensive # Full workflow tests
```

### Step 4: View Results

```bash
# Open HTML report in browser
npm run report

# Or manually open:
# test-results/html-report/index.html
```

## 📸 Screenshots

Screenshots are automatically saved to `test-results/screenshots/`

## 📊 What Gets Tested

### ✅ Core Dashboard
- Dashboard loading
- MCP SDK initialization
- All 13 tab navigation
- Console log validation
- Responsive design

### ✅ GitHub Runners
- Workflows tab display
- Runner management UI
- MCP tool calls
- Log correlation with server

### ✅ AI Models
- Model search
- Model download
- Download progress tracking
- Log correlation

### ✅ AI Inference
- Inference interface
- Model selection
- Parameter configuration
- Inference execution
- Result display
- Log correlation

### ✅ Comprehensive Workflows
- End-to-end workflows
- Multi-step operations
- Stress testing

## 🔍 Log Correlation

Tests automatically correlate:
- Dashboard console logs
- MCP server logs
- Network requests
- User actions

Example correlation:
```
Dashboard: "Downloading model: bert-base"
  ↓ (within 2000ms)
Server: "Model download initiated: bert-base"
  ↓ (within 5000ms)
Server: "Download progress: 50%"
  ↓
Dashboard: "Download complete"
```

## 🐛 Debugging

### Run in headed mode (visible browser)
```bash
npm run test:headed
```

### Run in debug mode (step through)
```bash
npm run test:debug
```

### Run in UI mode (interactive)
```bash
npm run test:ui
```

## 🎯 Test Specific Features

```bash
# Test only Chromium
npm run test:chromium

# Test only Firefox
npm run test:firefox

# Test only WebKit (Safari)
npm run test:webkit

# Test mobile viewports
npm run test:mobile
```

## 📝 Common Issues

### Issue: Server not starting
**Solution:**
```bash
# Check if port 3001 is in use
lsof -ti:3001 | xargs kill -9

# Start server manually
python -m ipfs_accelerate_py.mcp_dashboard --port 3001
```

### Issue: Tests timing out
**Solution:** Increase timeouts in `playwright.config.ts`:
```typescript
timeout: 180 * 1000,  // 3 minutes
```

### Issue: Browser not installed
**Solution:**
```bash
npx playwright install --with-deps chromium firefox webkit
```

## 📂 Directory Structure

```
e2e/
├── fixtures/               # Test utilities
│   ├── dashboard.fixture.ts
│   └── mcp-server.fixture.ts
├── tests/                  # Test specifications
│   ├── 01-dashboard-core.spec.ts
│   ├── 02-github-runners.spec.ts
│   ├── 03-model-download.spec.ts
│   ├── 04-model-inference.spec.ts
│   └── 05-comprehensive.spec.ts
└── utils/                  # Helper utilities
    ├── log-correlator.ts
    ├── screenshot-manager.ts
    └── report-generator.ts

test-results/               # Test output
├── screenshots/            # Test screenshots
├── visual-regression/      # Visual regression data
├── html-report/           # HTML test report
├── test-results.json      # JSON test results
└── junit.xml              # JUnit XML results
```

## 🤝 CI/CD Integration

Tests automatically run in GitHub Actions on:
- Push to main/develop
- Pull requests
- Manual workflow dispatch

View results in GitHub Actions tab.

## 📚 Next Steps

1. **Read the full documentation**: `e2e/README.md`
2. **Review implementation plan**: `PLAYWRIGHT_IMPLEMENTATION_PLAN.md`
3. **Add custom tests**: Follow patterns in `e2e/tests/`
4. **Customize**: Modify `playwright.config.ts` as needed

## 💡 Pro Tips

1. **Use screenshots liberally**: They help debug failures
2. **Check console logs**: Most issues show up there first
3. **Correlate logs**: Use log correlation to verify end-to-end flow
4. **Run tests often**: Catch issues early
5. **Keep tests isolated**: Each test should be independent

## 🎉 Success Criteria

Your tests are working correctly if:
- ✅ All tests pass
- ✅ No error logs in console (or < 5)
- ✅ Screenshots show expected UI state
- ✅ Log correlations are found
- ✅ HTML report generates successfully

## 📞 Support

- **Documentation**: `e2e/README.md`
- **Implementation**: `PLAYWRIGHT_IMPLEMENTATION_PLAN.md`
- **Issues**: GitHub Issues

---

**Happy Testing! 🎭**
