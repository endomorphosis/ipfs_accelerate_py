# Endpoint Handler Fix for IPFS Accelerate Python

This directory contains scripts to fix the endpoint_handler implementation in the IPFS Accelerate Python framework. The fix resolves the "'dict' object is not callable" error that occurs when trying to use endpoint handlers for local models.

## Problem Description

The current issue is that the `endpoint_handler` property in `ipfs_accelerate.py` returns a dictionary instead of a callable function. This causes errors when tests try to use endpoints with code like:

```python
endpoint_handler = accelerator.endpoint_handler(skill_handler, model_name, "cpu:0")
response = await endpoint_handler(input_data)  # Error: 'dict' object is not callable
```

## Solution

The fix modifies the `endpoint_handler` property to return a callable method that supports both dictionary-style access and function-style calls:

1. When accessed without arguments (`accelerator.endpoint_handler`), it returns the resources dictionary for direct attribute access.
2. When called with arguments (`accelerator.endpoint_handler(model, type)`), it returns a callable function.

## Fix Files

This directory contains the following files for fixing the issue:

### 1. `endpoint_handler_fix.py`

Contains the code for permanently fixing the `ipfs_accelerate.py` module. This file is used by the other scripts as a reference implementation.

### 2. `implement_endpoint_handler_fix.py`

This script:
- Creates a class that can apply the fix to any ipfs_accelerate_py instance
- Dynamically applies the fix at runtime without modifying the module
- Tests the fix with all models defined in mapped_models.json
- Outputs a summary of successful and failed models

Usage:
```bash
python implement_endpoint_handler_fix.py
```

### 3. `apply_endpoint_handler_fix.py`

This script:
- Finds the `ipfs_accelerate.py` module in your installation
- Makes a backup of the file
- Applies the fix by patching the module directly
- Verifies the fix was applied correctly

Usage:
```bash
python apply_endpoint_handler_fix.py
```

### 4. `run_local_endpoints_with_fix.py`

A script that applies the fix dynamically and runs tests for all local endpoints.

Usage:
```bash
python run_local_endpoints_with_fix.py
```

## Implementation Details

The fix implements three key methods:

1. **endpoint_handler property** - Returns the get_endpoint_handler method to make it callable
2. **get_endpoint_handler method** - Logic to return either a dictionary or callable function based on arguments
3. **_create_mock_handler method** - Creates appropriate mock handlers for different model types

The implementation includes:
- Model type detection for appropriate response format
- Support for both sync and async functions
- Proper error handling and fallback to mock implementations
- Backward compatibility for dictionary access

## How to Apply the Fix

### Option 1: Dynamic Fix (No Module Modification)

For testing or temporary use, you can apply the fix dynamically:

```python
from implement_endpoint_handler_fix import EndpointHandlerFixer

# Create an accelerator instance
accelerator = ipfs_accelerate_py(resources, metadata)

# Apply the fix
fixer = EndpointHandlerFixer()
fixer.apply_endpoint_handler_fix(accelerator)

# Now you can use the endpoint_handler as a callable
handler = accelerator.endpoint_handler(model, endpoint_type)
response = await handler(input_data)
```

### Option 2: Permanent Fix (Module Modification)

To apply the fix permanently:

```bash
python apply_endpoint_handler_fix.py
```

This will:
1. Make a backup of your `ipfs_accelerate.py` file
2. Apply the fix to the module
3. Verify the fix works correctly

## Testing After the Fix

After applying the fix, you can test the endpoints with:

```bash
python test_local_endpoints.py
```

## Reverting the Fix

If you need to revert the permanent fix, you can use the backup file created by `apply_endpoint_handler_fix.py`:

```bash
# The script will print the backup path, for example:
cp /path/to/ipfs_accelerate.py.bak.20250301123456 /path/to/ipfs_accelerate.py
```