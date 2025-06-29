# Test Migration Guide

This guide provides step-by-step instructions for migrating existing tests to the new test framework structure.

## Migration Process Overview

1. Analyze existing tests
2. Create destination directories
3. Choose migration strategy (copy or move)
4. Use migration script 
5. Verify migrated tests
6. Track migration progress

## Prerequisites

Before starting migration:

1. Ensure you have the test framework installed
2. Run the environment verification to check dependencies:
   ```bash
   python verify_test_environment.py
   ```

## Step 1: Analyze Existing Tests

The `migrate_tests.py` script can analyze existing tests without migrating them:

```bash
python migrate_tests.py --analyze-only
```

This will generate a report of existing tests and their suitable destinations in the new structure.

## Step 2: Create Destination Directories

The migration script will automatically create destination directories, but you can also create them manually:

```bash
python migrate_tests.py --create-dirs-only
```

## Step 3: Choose Migration Strategy

You have two options for migration:

1. **Copy Strategy**: Copy tests to the new location while keeping the original
   * Good for initial validation without breaking existing workflows
   * Use the `--copy` flag with the migration script

2. **Move Strategy**: Move tests to the new location
   * Removes tests from the original location
   * Use the `--move` flag with the migration script

## Step 4: Use Migration Script

Migrate specific tests or test categories:

```bash
# Migrate all BERT tests with copy strategy
python migrate_tests.py --pattern "*bert*" --copy

# Migrate WebGPU tests with move strategy
python migrate_tests.py --pattern "*webgpu*" --move

# Migrate a specific test file
python migrate_tests.py --file test_bert_base_uncased.py --move

# Migrate all tests in a specific directory
python migrate_tests.py --dir apis/ --copy
```

For batch migration, use pattern matching:

```bash
# Migrate all model tests
python migrate_tests.py --pattern "test_*base*.py" --copy

# Migrate all API tests
python migrate_tests.py --pattern "test_*api*.py" --copy
```

## Step 5: Verify Migrated Tests

After migration, verify that the tests still work:

```bash
# Run tests in the new location
python run.py --test-type model --model bert

# Run specific migrated test
python -m pytest test/models/text/bert/test_bert_base_uncased.py -v
```

## Step 6: Track Migration Progress

Track the migration progress using the tracking tool:

```bash
python track_migration_progress.py
```

This will generate a report of migrated tests and remaining tests.

## Common Migration Patterns

Here are common patterns for migrating specific test types:

### Model Tests

```bash
# Text models (BERT, T5, GPT, etc.)
python migrate_tests.py --pattern "test_bert*.py" --dest models/text/bert/
python migrate_tests.py --pattern "test_t5*.py" --dest models/text/t5/
python migrate_tests.py --pattern "test_gpt*.py" --dest models/text/gpt/

# Vision models (ViT, DETR, etc.)
python migrate_tests.py --pattern "test_vit*.py" --dest models/vision/vit/
python migrate_tests.py --pattern "test_detr*.py" --dest models/vision/detr/

# Audio models (Whisper, etc.)
python migrate_tests.py --pattern "test_whisper*.py" --dest models/audio/whisper/
```

### Hardware Tests

```bash
# WebGPU tests
python migrate_tests.py --pattern "test_webgpu*.py" --dest hardware/webgpu/
python migrate_tests.py --pattern "test_webgpu_compute*.py" --dest hardware/webgpu/compute_shaders/

# WebNN tests
python migrate_tests.py --pattern "test_webnn*.py" --dest hardware/webnn/

# CUDA tests
python migrate_tests.py --pattern "test_cuda*.py" --dest hardware/cuda/

# ROCm tests
python migrate_tests.py --pattern "test_rocm*.py" --dest hardware/rocm/
```

### API Tests

```bash
# HuggingFace API tests
python migrate_tests.py --pattern "test_hf_*.py" --dest api/huggingface/

# LLM provider tests
python migrate_tests.py --pattern "test_openai*.py" --dest api/llm_providers/
python migrate_tests.py --pattern "test_claude*.py" --dest api/llm_providers/
python migrate_tests.py --pattern "test_ollama*.py" --dest api/llm_providers/

# Internal API tests
python migrate_tests.py --pattern "test_ipfs_api*.py" --dest api/internal/
```

## Handling Common Migration Issues

### Import Path Issues

The migration script attempts to fix import paths, but you may need to update them manually. Common patterns:

1. Relative imports to absolute imports:
   ```python
   # Before
   from ..utils import helper_function
   
   # After
   from test.common.utils import helper_function
   ```

2. Updating model helper imports:
   ```python
   # Before
   from utils.model_helpers import load_model
   
   # After
   from test.common.model_helpers import load_model
   ```

### Test Configuration Issues

If tests depend on specific configurations, ensure they're available in the new location:

1. Check for local config files that need to be migrated
2. Update paths in test code that reference local resources
3. Update pytest fixture references

### CI Integration

After migrating a significant portion of tests, update CI workflows to use the new structure:

1. Use the new entry point: `python run.py`
2. Update test paths in CI scripts
3. Consider running both old and new tests during the transition period

## Phased Migration Approach

We recommend a phased approach to migration:

1. Start with less critical tests to validate the migration process
2. Migrate one test category at a time (e.g., all BERT tests)
3. Verify each batch thoroughly before proceeding
4. Keep original tests until confident in the new structure
5. Update CI to run both old and new tests during transition
6. Once all tests are migrated and verified, remove old test structure

## Getting Help

If you encounter issues during migration:

1. Check the migration log: `migration_log.txt`
2. Run with verbose logging: `python migrate_tests.py --verbose`
3. Contact the test framework team for assistance