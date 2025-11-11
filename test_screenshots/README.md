# Test Screenshots

This directory contains screenshots captured during Playwright testing of the MCP Dashboard.

## Screenshots

### 01_dashboard_loaded.png
**Size**: 633KB (1920x1080)
**Description**: Initial dashboard state after loading
- Shows the main dashboard interface
- All navigation tabs visible
- MCP SDK loaded and initialized
- Server status indicators visible

### 02_workflows_tab_clicked.png
**Size**: 564KB (1920x1080)
**Description**: GitHub Workflows tab after being clicked
- GitHub Workflows tab is active
- Workflows section is visible
- Shows the tab switching functionality works

### 03_workflows_section.png
**Size**: 564KB (1920x1080)
**Description**: Workflows section rendered
- Shows workflows container properly positioned
- Active runners section visible
- Track button present

### 04_after_track_click.png
**Size**: 567KB (1920x1080)
**Description**: State after clicking Track button
- Demonstrates interactive functionality
- Shows button click works
- Runners tracking initiated

### 05_final_state.png
**Size**: 567KB (1920x1080)
**Description**: Final state with all elements loaded
- Complete dashboard functionality visible
- All containers properly rendered
- No errors in the interface

### error_state.png (deprecated)
**Description**: Screenshot from a failed test run (kept for reference)

## How These Were Generated

These screenshots were automatically captured by the Playwright test suite in `test_mcp_dashboard_playwright.py`:

```python
page.goto(DASHBOARD_URL)
page.screenshot(path=SCREENSHOTS_DIR / "01_dashboard_loaded.png")
```

## Verification

These screenshots verify:
- ✅ Dashboard loads correctly
- ✅ GitHub Workflows tab is accessible
- ✅ DOM elements are properly rendered
- ✅ Interactive elements (buttons, tabs) work
- ✅ No visual errors or broken layouts

## Regenerating Screenshots

To regenerate these screenshots:

```bash
cd /home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py
python test_mcp_dashboard_playwright.py
```

New screenshots will be saved to this directory, overwriting the existing ones.

## Notes

- Screenshots are excluded from git via `.gitignore`
- They are large files (500-600KB each)
- They provide visual proof that the dashboard works correctly
- Useful for regression testing and visual comparison
