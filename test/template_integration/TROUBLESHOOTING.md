# Troubleshooting Guide

This guide helps resolve common issues encountered when using the template integration system.

## Syntax Errors

### Indentation Issues

**Problem:** Generated test files have indentation errors, especially after if statements or in try/except blocks.

```python
if not os.path.exists(test_image_path):
dummy_image = Image.new('RGB', (224, 224), color='white')  # Indentation error
```

**Solution:**
1. Use the updated `fix_template_issues.py` script:
   ```bash
   python fix_template_issues.py
   ```

2. For manual fixes, update the special handling code in `MODEL_CONFIG` to use proper indentation:
   ```python
   "special_handling": """
       # Create a dummy image for testing if needed
       if not os.path.exists(test_image_path):
           dummy_image = Image.new('RGB', (224, 224), color='white')
           dummy_image.save(test_image_path)
   """
   ```

3. Ensure the `customize_template()` function in `model_template_fixes.py` properly detects and maintains indentation.

### Import Errors

**Problem:** Missing or duplicate imports in generated test files.

**Solution:**
1. Check the `custom_imports` list in the model configuration.
2. Remove duplicate imports.
3. Ensure imports are properly formatted.
4. Use the import deduplication feature in `customize_template()`.

### Template Mismatch

**Problem:** Using the wrong template for a model architecture.

**Solution:**
1. Check the architecture type in the model configuration.
2. Verify the template file exists in `TEMPLATES_DIR`.
3. Update `ARCHITECTURE_TYPES` if necessary.
4. Check architecture-to-template mapping in `get_template_path()`.

## Runtime Errors

### Missing Dependencies

**Problem:** Generated tests fail due to missing dependencies.

**Solution:**
1. Add missing dependencies to `requirements.txt`.
2. Add proper error handling in the test to gracefully handle missing dependencies.
3. Add mock implementations for critical dependencies.

### Registry Issues

**Problem:** Models don't appear in the registry or have incorrect metadata.

**Solution:**
1. Check the registry entry in the generated test.
2. Verify that `update_architecture_types()` was called.
3. Manually update the registry if necessary.

### Hardware Detection Issues

**Problem:** Tests fail to detect available hardware correctly.

**Solution:**
1. Verify the hardware detection code in the template.
2. Test with different hardware configurations.
3. Add more robust detection code if necessary.

## Integration Workflow Issues

### Analysis Step Fails

**Problem:** The analysis step fails to identify issues correctly.

**Solution:**
1. Verify the manual test files exist.
2. Check the analysis logic in `analyze_manual_models()`.
3. Run analysis with verbose logging.

### Generation Step Fails

**Problem:** The generation step fails to create valid test files.

**Solution:**
1. Check the model configuration.
2. Verify the template files exist.
3. Run generation for a single model to isolate the issue.
4. Use the `--generate-model` option with `--verify`.

### Verification Step Fails

**Problem:** The verification step fails to identify syntax issues.

**Solution:**
1. Run verification manually on the generated file.
2. Check the verification logic in `verify_test_file()`.
3. Use Python's built-in compiler to check syntax.

## Model-Specific Issues

### Vision Models (layoutlmv2, layoutlmv3)

**Problem:** Image creation code has indentation issues.

**Solution:**
1. Use model-specific handling in `customize_template()`.
2. Format the image handling code with proper indentation.
3. Insert at the correct position after the try block.

### Speech Models (clvp, seamless_m4t_v2)

**Problem:** Audio creation code has indentation issues.

**Solution:**
1. Use model-specific handling in `customize_template()`.
2. Format the audio handling code with proper indentation.
3. Pay special attention to try/except indentation within the audio creation code.

### Encoder-Decoder Models (bigbird, xlm_prophetnet)

**Problem:** Special handling code fails to apply correctly.

**Solution:**
1. Check the special handling code in the model configuration.
2. Ensure the architecture type is correct.
3. Verify the template file is appropriate for the model.

## Common Error Messages

### "expected an indented block after 'if' statement"

**Problem:** Indentation issue in conditional statements.

**Solution:**
1. Check indentation after if statements.
2. Ensure all lines in conditional blocks are properly indented.
3. Use the `fix_template_issues.py` script.

### "unexpected indent"

**Problem:** A line is indented when it shouldn't be.

**Solution:**
1. Check indentation transitions.
2. Ensure block ends are properly aligned.
3. Check for mixing of tabs and spaces.

### "undefined name"

**Problem:** Using a variable or function that hasn't been defined.

**Solution:**
1. Add missing import.
2. Define the variable or function before using it.
3. Check for typos in variable names.

## Advanced Troubleshooting

### Debugging Template Processing

To debug the template processing:

1. Add debug output to `customize_template()`:
   ```python
   logger.debug(f"Processing template for {model_name}")
   logger.debug(f"Template content length: {len(template_content)}")
   ```

2. Add indentation debugging:
   ```python
   logger.debug(f"Indentation level detected: {indentation}")
   logger.debug(f"Special handling lines: {len(special_handling_lines)}")
   ```

3. Add post-processing verification:
   ```python
   # After processing
   for i, line in enumerate(content.split('\n')):
       if "if " in line and not line.endswith(":"):
           logger.warning(f"Potential issue at line {i+1}: {line}")
   ```

### Manual Intervention

If automated fixes fail, you can manually edit the generated files:

1. Generate the file without verification:
   ```bash
   python model_template_fixes.py --generate-model MODEL
   ```

2. Manually edit the file to fix indentation issues.

3. Verify the file manually:
   ```bash
   python -m py_compile /path/to/fixed_tests/test_hf_MODEL.py
   ```

4. Use the fixed file as a reference to update the model configuration.

## Contacting Support

If you can't resolve an issue using this guide, please create an issue in the repository with:

1. The exact error message
2. The model name and architecture
3. The generated file (if possible)
4. The steps to reproduce the issue

The team will respond to issues as quickly as possible.