# Playwright Testing Guide for HuggingFace Search

## Prerequisites

```bash
# Install dependencies
pip install flask flask-cors huggingface_hub requests playwright

# Install Playwright browsers
playwright install chromium
```

## Running the Tests

### Option 1: Using Existing Test Suite

```bash
cd /home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py
python tests/test_comprehensive_validation.py
```

This will:
1. Start the MCP dashboard on port 9000
2. Navigate to the dashboard in a browser
3. Test the HF Search functionality
4. Save screenshots to `test_screenshots/`

### Option 2: Manual Testing Steps

#### 1. Start the MCP Dashboard

```bash
# Using the CLI
ipfs-accelerate mcp start

# Or directly
cd /home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py
python -m ipfs_accelerate_py.mcp_dashboard
```

The dashboard will start on: http://0.0.0.0:9000

#### 2. Test the Search Functionality

1. **Open Browser**: Navigate to http://localhost:9000
2. **Click "HF Search" tab**: Should open the HuggingFace search interface
3. **Enter search query**: Type "bert" in the search box
4. **Click "Search HF Hub"**: Should display results
5. **Verify results**: Should see models like:
   - bert-base-uncased
   - bert-large-uncased
   - distilbert-base-uncased
6. **Test download**: Click download button on a model

### Expected Behavior

#### First Search (Cache Empty)
```
1. User enters "bert" and clicks Search
2. Status: "Searching HuggingFace Hub..."
3. Backend calls HuggingFace API
4. Results populate (1-2 seconds)
5. Models displayed with metadata:
   - Model ID
   - Downloads count
   - Likes count
   - Description
   - Tags
```

#### Second Search (Cached)
```
1. User searches "bert" again
2. Results appear immediately (<50ms)
3. No API call made (uses cache)
```

## Playwright Test Script

If you want to create a custom test:

```python
from playwright.sync_api import sync_playwright
import time

def test_hf_search():
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        # Navigate to dashboard
        page.goto('http://localhost:9000')
        page.screenshot(path='01_dashboard.png')
        
        # Click HF Search tab
        page.click('button:has-text("HF Search")')
        time.sleep(1)
        page.screenshot(path='02_hf_search_tab.png')
        
        # Enter search query
        page.fill('#hf-search', 'bert')
        page.screenshot(path='03_search_input.png')
        
        # Click search button
        page.click('button:has-text("Search HF Hub")')
        time.sleep(5)  # Wait for results
        page.screenshot(path='04_search_results.png', full_page=True)
        
        # Check for results
        results = page.query_selector_all('.model-result')
        print(f"Found {len(results)} results")
        
        # Take final screenshot
        page.screenshot(path='05_final.png', full_page=True)
        
        browser.close()

if __name__ == '__main__':
    test_hf_search()
```

## Verification Checklist

After running tests, verify:

- [ ] Dashboard loads without errors
- [ ] HF Search tab is accessible
- [ ] Search input field is present
- [ ] Search button works
- [ ] Results are displayed (not empty)
- [ ] Each result shows:
  - [ ] Model ID/name
  - [ ] Download count
  - [ ] Likes count
  - [ ] Description
  - [ ] Action buttons (Download, View)
- [ ] Second search is faster (cached)
- [ ] Screenshots captured successfully

## Troubleshooting

### No Results Returned

**Check logs for:**
```
Cache has 0 results, fetching from HuggingFace API
Retrieved X models from HuggingFace API
```

**If you see:**
```
Error searching HuggingFace API: ...
```

**Possible causes:**
1. No internet connection to huggingface.co
2. Rate limiting (wait a minute and retry)
3. API unavailable (check https://status.huggingface.co/)

### Dashboard Won't Start

```bash
# Check if port is in use
lsof -i :9000

# Try different port
python -m ipfs_accelerate_py.mcp_dashboard --port 8899
```

### Import Errors

```bash
# Verify all dependencies installed
pip list | grep -E "(flask|huggingface|playwright)"

# Reinstall if needed
pip install -r requirements_dashboard.txt
```

## Screenshot Locations

Tests save screenshots to:
```
tests/test_screenshots/
├── 01_dashboard_loaded.png
├── 02_hf_search_tab.png
├── 03_search_input.png
├── 04_search_results.png
├── 05_download_clicked.png
└── final_status.png
```

## API Endpoints Being Tested

1. **GET /api/mcp/models/search**
   - Query params: `q`, `task`, `hardware`, `limit`
   - Returns: Search results with model metadata

2. **POST /api/mcp/models/download**
   - Body: `{"model_id": "bert-base-uncased"}`
   - Returns: Download status and path

3. **GET /api/mcp/models/stats**
   - Returns: Model statistics and cache info

## Success Criteria

✅ Test passes if:
1. Dashboard loads successfully
2. Search returns real model data (not mock/empty)
3. Results include popular BERT models
4. Download functionality works
5. Performance is acceptable (<2s first search, <50ms cached)
6. Screenshots show working UI

## Notes

- First run may be slower (downloading models metadata)
- Subsequent searches use cache (much faster)
- Cache persists across restarts
- Real HuggingFace API used (not mock data)
