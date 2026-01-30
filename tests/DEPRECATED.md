# ⚠️ DEPRECATED - This Directory Has Been Moved

## Notice

This `tests/` directory has been **deprecated** and is no longer used.

All test files have been moved to the `test/` directory, which is the designated production location for tests.

## Migration Details

- **Date**: January 30, 2026
- **Reason**: Consolidate all tests into the standard `test/` directory structure
- **Status**: ✅ Migration Complete

## What Was Moved

The following items were migrated from `tests/` to `test/`:

### Test Files (84 files)
- All `test_*.py` files including:
  - Core component tests (test_accelerate.py, test_integration.py, etc.)
  - MCP tests (test_mcp_*.py)
  - P2P tests (test_p2p_*.py)
  - Model tests (test_model_*.py)
  - And many more...

### Supporting Files
- `run_all_tests.py` - Test runner
- `run_mcp.py` - MCP runner
- `ui_test_script.py` - UI testing script
- `playwright_pipeline_screenshots.py` - Screenshot automation
- Database files (`*.db`, `*.wal`)
- Screenshot directories (now `playwright_screenshots_legacy/` and `playwright_screenshots_functional_legacy/`)
- Documentation files (with `_LEGACY` suffix to avoid conflicts)

## Where to Find Tests Now

All tests are now located in the `test/` directory:

```bash
cd test/
```

The `test/` directory follows a structured organization:

```
test/
├── test_*.py              # Migrated legacy tests (now in root of test/)
├── models/                # Model-specific tests
├── hardware/              # Hardware-specific tests
├── api/                   # API-specific tests
├── distributed_testing/   # Distributed testing framework
├── common/                # Shared utilities
└── README.md              # Full documentation
```

## Running Tests

Instead of running tests from `tests/`, use the `test/` directory:

```bash
# Old way (deprecated)
# python tests/test_accelerate.py

# New way
python test/test_accelerate.py

# Or use pytest directly
pytest test/test_accelerate.py

# Or run all tests
pytest test/
```

## Configuration

The pytest configuration has been updated in `pytest.ini` to use the `test/` directory.

## Need Help?

- See `test/README.md` for comprehensive test documentation
- See `test/README_LEGACY_TESTS.md` for the old tests documentation
- Refer to the root `README.md` for project overview

## Future Plans

This `tests/` directory will be removed in a future release. Please update any scripts, documentation, or tooling that references this old location.
