# HuggingFace Test Troubleshooting Guide

This guide helps you resolve common issues encountered when working with the HuggingFace test automation toolkit.

## Common Issues

### Indentation Errors

#### Symptoms

- Syntax errors in generated test files
- `IndentationError: unexpected indent`
- `IndentationError: unindent does not match any outer indentation level`

#### Solutions

1. **Run the indentation fixer**:

   ```bash
   python test_integration.py --fix
   ```

2. **Fix a specific file**:

   ```bash
   python complete_indentation_fix.py path/to/test_hf_model.py
   ```

3. **Manual fixes**: If automated fixes fail, look for:
   - Mixed tabs and spaces (always use 4 spaces per level)
   - Mismatched parentheses, brackets, or braces
   - Broken method boundaries (missing newlines between methods)

### Missing Dependencies

#### Symptoms

- `ImportError: No module named 'transformers'`
- `ImportError: No module named 'torch'`
- Errors about missing libraries when running tests

#### Solutions

1. **Install core dependencies**:

   ```bash
   pip install torch transformers
   ```

2. **Install all dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Check mock implementations**: The test files include mock implementations for missing dependencies. Ensure these are working correctly if you don't want to install all dependencies.

### Hardware Detection Issues

#### Symptoms

- Tests fail with CUDA errors despite having CUDA installed
- MPS (Metal Performance Shaders) not detected on Apple Silicon
- OpenVINO not working

#### Solutions

1. **Verify hardware availability**:

   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
   ```

2. **Update hardware detection code**: In some environments, the hardware detection might need customization. Check `check_hardware()` function in template files.

3. **Force CPU testing**:

   ```bash
   python test_integration.py --run --core --force-cpu
   ```

### Test Generation Failures

#### Symptoms

- Tests cannot be generated for certain model types
- Errors when running the generator

#### Solutions

1. **Check architecture mapping**: Ensure the model type is correctly mapped to an architecture type in the `ARCHITECTURE_TYPES` dictionary.

2. **Create a custom template**: If a model has unique requirements, create a custom template in the `templates` directory.

3. **Manually copy and modify**: Use an existing test file for a similar architecture as a starting point.

### Syntax Verification Failures

#### Symptoms

- Tests can't be executed due to syntax errors
- Errors in the verification step

#### Solutions

1. **Run Python's built-in syntax checker**:

   ```bash
   python -m py_compile test_hf_model.py
   ```

2. **Use a linter**:

   ```bash
   flake8 test_hf_model.py
   ```

3. **Fix specific issues**: Look for:
   - String formatting issues (f-strings vs old style)
   - Missing or extra parentheses
   - Incorrect dictionary or list syntax

### Test Execution Failures

#### Symptoms

- Tests run but fail with errors
- Model loading issues
- Memory errors

#### Solutions

1. **Check model availability**: Ensure the specified model is available on the Hugging Face Model Hub.

2. **Inspect error details**: Look for specific error messages in the output.

3. **Run with smaller models**: Some tests may fail due to memory limitations. Try with smaller models:

   ```bash
   python test_hf_model.py --model "distilbert-base-uncased"
   ```

4. **Force CPU execution**:

   ```bash
   python test_hf_model.py --cpu-only
   ```

### Report Generation Issues

#### Symptoms

- Coverage report cannot be generated
- Missing or incomplete reports

#### Solutions

1. **Check result files**: Ensure test results were saved to the expected directory.

2. **Manually generate report**:

   ```bash
   python test_integration.py --report
   ```

3. **Create individual test summaries**:

   ```bash
   python test_hf_model.py --save
   ```

## Architecture-Specific Issues

### Encoder-Only (BERT, RoBERTa, etc.)

- **Common issue**: Mask token handling
- **Solution**: Ensure the test text includes a proper mask token (`[MASK]` for BERT, `<mask>` for RoBERTa)

### Decoder-Only (GPT-2, LLaMA, etc.)

- **Common issue**: Padding token configuration
- **Solution**: Ensure `tokenizer.pad_token = tokenizer.eos_token` is included in the test

### Encoder-Decoder (T5, BART, etc.)

- **Common issue**: Decoder input handling
- **Solution**: Ensure decoder inputs are properly initialized

### Vision Models (ViT, Swin, etc.)

- **Common issue**: Image tensor shape
- **Solution**: Verify image inputs have correct shape (batch_size, channels, height, width)

### Multimodal Models (CLIP, BLIP, etc.)

- **Common issue**: Multi-input handling
- **Solution**: Ensure both text and image inputs are properly formatted

## CI/CD Integration Issues

### GitHub Actions

- **Common issue**: Workflow not triggered
- **Solution**: Check workflow file placement in `.github/workflows/`

### Missing Dependencies in CI

- **Common issue**: CI environment missing required packages
- **Solution**: Ensure requirements.txt is complete and properly installed in the workflow

## Performance Optimization

If tests are running slowly:

1. **Limit model loading**: Use smaller models or model configurations
2. **Reduce number of runs**: Modify the `num_runs` parameter in test methods
3. **Run only necessary tests**: Use specific test methods instead of running all tests
4. **Use batching**: For multiple tests, use the batch processing features

## Contributing to the Framework

If you encounter an issue not covered in this guide:

1. Open an issue in the repository
2. Include detailed error information
3. Provide steps to reproduce the issue
4. Suggest a fix if possible

## Contacting Support

For additional help:

1. Check the documentation in the `docs` directory
2. Refer to the [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index)
3. Open an issue in the [repository](https://github.com/yourusername/yourrepository)

## License

Apache 2.0