# Playwright E2E Test Fix Summary

## Issue
The Playwright end-to-end test was failing with the following error:
```
/home/barberb/ipfs_accelerate_py/.venv/bin/python3: No module named ipfs_accelerate_py.cli
```

## Root Cause
The test was trying to run the MCP server using `python3 -m ipfs_accelerate_py.cli mcp start`, but:
1. The `cli.py` file is located at the project root, not inside the `ipfs_accelerate_py` package
2. The package wasn't installed in development mode in the virtual environment
3. The test was using the wrong port (9000 instead of 8899)

## Fixes Applied

### 1. Package Installation
Installed the package in development mode:
```bash
pip install -e .
```

### 2. Server Start Command
Changed the test to run the MCP dashboard directly instead of using the CLI:
```python
# Before:
server_process = subprocess.Popen(
    [sys.executable, "-m", "ipfs_accelerate_py.cli", "mcp", "start"],
    ...
)

# After:
mcp_dashboard_path = Path(__file__).parent.parent / "ipfs_accelerate_py" / "mcp_dashboard.py"
server_process = subprocess.Popen(
    [sys.executable, str(mcp_dashboard_path)],
    ...
)
```

### 3. Port Configuration
Updated the test to use the correct default port (8899):
```python
# Before:
page.goto("http://localhost:9000", timeout=30000)

# After:
page.goto("http://localhost:8899", timeout=30000)
```

### 4. Playwright API Fix
Fixed the model name extraction to use the correct Playwright API:
```python
# Before:
model_card = download_btn.locator("xpath=ancestor::div[@class='model-result']")

# After:
model_results = page.query_selector_all('.model-result')
if model_results:
    first_model = model_results[0]
    model_title_elem = first_model.query_selector('.model-title')
    model_name = model_title_elem.inner_text() if model_title_elem else "unknown"
```

## Test Results

✅ **Test now passes successfully!**

The test captures 6 screenshots:
1. `01_dashboard_overview.png` - Dashboard landing page
2. `02_hf_search_tab.png` - HuggingFace Search tab
3. `03_search_input.png` - Search input with "llama" query
4. `04_search_results.png` - Search results showing 2 models
5. `05_download_initiated.png` - Download button clicked
6. `06_download_complete.png` - Download completed

## Test Output
```
✅ TEST PASSED - Screenshots captured successfully

📁 Screenshots location: /home/barberb/ipfs_accelerate_py/tests/playwright_screenshots

Generated screenshots:
   - 01_dashboard_overview.png
   - 02_hf_search_tab.png
   - 03_search_input.png
   - 04_search_results.png
   - 05_download_initiated.png
   - 06_download_complete.png
```

## Running the Test

To run the test:
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Install dependencies
pip install playwright
playwright install chromium

# Run the test
python3 tests/test_playwright_e2e_with_screenshots.py
```

## Notes
- The test successfully starts the MCP server, navigates the dashboard, performs a search, and captures screenshots
- No JavaScript errors were detected during the test
- The "Model Manager" tab was not found in the dashboard (marked as ⚠️ warning, not an error)
