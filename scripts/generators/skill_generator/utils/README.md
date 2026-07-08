# Utility Functions

This directory contains utility functions for the refactored generator suite.

## Current Utilities

Currently, this directory is empty. Planned utilities include:

- Logging utilities
- Progress tracking
- File handling utilities
- Template validation utilities
- Model verification utilities
- Hardware verification utilities

## Adding Utilities

When adding a new utility function:

1. Create a Python file with a descriptive name
2. Add detailed docstrings to all functions
3. Add type hints for all parameters and return values
4. Add error handling for all edge cases
5. Add unit tests for the utility function
6. Update this README with a description of the utility

## Usage

To use a utility function from another module:

```python
from utils.utility_module import utility_function

result = utility_function(parameter)
```