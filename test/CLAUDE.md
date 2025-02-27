# IPFS Accelerate Python Framework - Development Guide

## Build & Test Commands
- Run single test: `python -m test.apis.test_<api_name>` or `python -m test.skills.test_<skill_name>`
- Run all tests in a directory: Use Python's unittest discovery `python -m unittest discover -s test/apis` or `python -m unittest discover -s test/skills`
- Tests compare collected results with expected results in JSON files

## Code Style Guidelines
- Use snake_case for variables, functions, methods, modules
- Use PEP 8 formatting standards
- Include comprehensive docstrings for classes and methods
- Use absolute imports with sys.path.append for module resolution
- Standard imports first, then third-party libraries
- Standardized error handling with try/except blocks and detailed error messages
- Test results stored in JSON files with consistent naming
- Unittest-based testing with async support via asyncio.run()
- Mocking external dependencies in tests with unittest.mock
- Tests include result collection, comparison with expected results, and detailed error reporting