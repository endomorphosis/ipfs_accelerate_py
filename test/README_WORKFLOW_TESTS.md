# HuggingFace Workflow Tests

This directory contains tests to verify the complete HuggingFace model search and download workflow.

## Comprehensive Validation Approach

As recommended, we validate the system in phases:

1. **Backend MCP Server Tools** - Verify core functionality works
2. **Package Functions** - Test ipfs_accelerate_py functions independently  
3. **API Endpoints** - Validate HTTP APIs when server runs
4. **GUI Integration** - Test browser interface with Playwright

## Test Files

### 🎯 `test_comprehensive_validation.py` - **RECOMMENDED**
**Systematic validation of all layers from backend to GUI.**

This test follows the recommended approach: verify backend tools first, then GUI.

**What it tests:**
- **Phase 1**: HuggingFaceHubScanner backend class and methods
- **Phase 2**: ModelManager and ipfs_accelerate_py package functions
- **Phase 3**: MCP Dashboard API endpoints (`/api/mcp/models/*`)
- **Phase 4**: GUI integration with Playwright (with screenshots)

**Usage:**
```bash
# Test backend only (no server needed)
python3 tests/test_comprehensive_validation.py
# Will test Phases 1-2, skip 3-4 if server not running

# Full test with server running
python3 -m ipfs_accelerate_py.mcp_dashboard  # Terminal 1
python3 tests/test_comprehensive_validation.py  # Terminal 2
```

**Output:**
- Phase-by-phase validation results
- Screenshots in `data/test_screenshots/validation/` directory
- Clear indication of which layers work

---

### 1. `test_workflow_simple.py` - API Test (No Playwright Required)
Simple validation that tests the API endpoints directly without requiring Playwright.

**Usage:**
```bash
# Start the MCP server first
python3 -m ipfs_accelerate_py.mcp_dashboard

# In another terminal, run the test
python3 tests/test_workflow_simple.py
```

**What it tests:**
- Server is running and accessible
- HuggingFace model search API works
- Model download API works
- Downloaded models appear in stats

---

### 2. `test_huggingface_workflow.py` - Full E2E Playwright Test
Comprehensive end-to-end test that uses Playwright to test the complete UI workflow.

**Prerequisites:**
```bash
pip install playwright
playwright install chromium
```

**Usage:**
```bash
# The test will start the server automatically
python3 tests/test_huggingface_workflow.py
```

**What it tests:**
1. Starts MCP server with `ipfs-accelerate mcp start`
2. Opens dashboard in browser
3. Navigates to "🔍 HF Search" tab
4. Searches for "bert" model
5. Clicks Download button
6. Navigates to "📚 Model Browser" tab
7. Verifies the downloaded model appears

**Output:**
- Console output showing each step
- Screenshots saved to `data/test_screenshots/workflow/` directory
- Browser window shows the actual interactions (runs in headed mode)

---

### 3. `test_model_manager_dashboard.py` - Original Model Manager Test
Tests the Model Manager Browser tab functionality.

**Usage:**
```bash
python3 tests/test_model_manager_dashboard.py
```

---

### 4. `test_mcp_start_command.py` - CLI Command Test
Tests that `ipfs-accelerate mcp start` command works correctly.

**Usage:**
```bash
python3 tests/test_mcp_start_command.py
```

---

## Recommended Testing Workflow

### Step 1: Validate Backend Tools
```bash
# Test backend components independently
python3 tests/test_comprehensive_validation.py
# Should pass Phase 1 (Backend) and Phase 2 (Package)
```

### Step 2: Validate with Server Running
```bash
# Terminal 1: Start server
python3 -m ipfs_accelerate_py.mcp_dashboard

# Terminal 2: Run comprehensive test
python3 tests/test_comprehensive_validation.py
# Should pass all 4 phases with screenshots
```

### Step 3: Manual Verification
Open http://127.0.0.1:8899/mcp and:
1. Click "🔍 HF Search" tab
2. Search for "bert"
3. Click "⬇️ Download"
4. Check "📚 Model Browser" tab

---

## Architecture Verification

The system is built in layers:

```
┌─────────────────────────────────────┐
│  GUI (dashboard.html + .js files)  │ ← Phase 4
├─────────────────────────────────────┤
│  HTTP APIs (/api/mcp/models/*)     │ ← Phase 3
├─────────────────────────────────────┤
│  MCPDashboard (mcp_dashboard.py)   │ ← Phase 3
├─────────────────────────────────────┤
│  HuggingFaceHubScanner             │ ← Phase 1
│  ModelManager                       │ ← Phase 2
└─────────────────────────────────────┘
```

Each layer can be tested independently, then integrated.

---

## Troubleshooting

### Backend tests fail (Phase 1)
- Check if `huggingface_hub_scanner.py` exists
- Verify imports work: `python3 -c "from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner"`

### Package tests fail (Phase 2)
- Check if `model_manager.py` exists
- May use mock if actual implementation unavailable (non-fatal)

### API tests fail (Phase 3)
- Ensure server is running: `python3 -m ipfs_accelerate_py.mcp_dashboard`
- Check server logs for errors
- Verify port 8899 is available

### GUI tests fail (Phase 4)
- Install Playwright: `pip install playwright && playwright install`
- Check screenshots in `data/test_screenshots/validation/` for debugging
- Verify JavaScript console in browser (F12)

### Search returns no results
- Check if HuggingFace API is accessible
- May be using mock/fallback data
- Check server logs for API errors

### Download fails
- Verify backend has write permissions
- Check server logs for detailed error messages
- Ensure enough disk space
