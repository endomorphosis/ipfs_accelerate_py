# HuggingFace Workflow Tests

This directory contains tests to verify the complete HuggingFace model search and download workflow.

## Test Files

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
- Screenshots saved to `test_screenshots_workflow/` directory
- Browser window shows the actual interactions (runs in headed mode)

---

### 3. `test_model_manager_dashboard.py` - Original Model Manager Test
Tests the Model Manager Browser tab functionality.

**Usage:**
```bash
python3 tests/test_model_manager_dashboard.py
```

---

## Complete Workflow Test

To verify the entire workflow works:

1. **Start the MCP server:**
   ```bash
   python3 -m ipfs_accelerate_py.mcp_dashboard
   ```
   Server will be available at: http://127.0.0.1:8899/mcp

2. **Run the simple API test:**
   ```bash
   python3 tests/test_workflow_simple.py
   ```

3. **Run the full Playwright E2E test:**
   ```bash
   python3 tests/test_huggingface_workflow.py
   ```

## Manual Testing

You can also test manually:

1. Open http://127.0.0.1:8899/mcp in your browser
2. Click the "🔍 HF Search" tab
3. Enter "bert" in the search box
4. Click "🔍 Search HF Hub"
5. Click "⬇️ Download" on any bert model
6. Switch to "📚 Model Browser" tab
7. Verify the model appears in the list

## Troubleshooting

### Server won't start
- Check if port 8899 is available
- Check for errors in console output

### Search returns no results
- Check API endpoint in browser console
- Verify `/api/mcp/models/search` is accessible

### Download fails
- Check browser console for errors
- Verify `/api/mcp/models/download` endpoint works
- Check server logs for backend errors

### Playwright tests fail
- Ensure Playwright is installed: `pip install playwright`
- Install browser: `playwright install chromium`
- Check screenshots in test_screenshots_workflow/ for debugging
