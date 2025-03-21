# HuggingFace Test Troubleshooting Guide

This guide helps you resolve common issues encountered when working with the HuggingFace test automation toolkit, especially when expanding test coverage to all 300+ model classes.

## Common Issues

### Hyphenated Model Names

#### Symptoms

- `SyntaxError` when running test files with hyphenated model names (e.g., "gpt-j", "xlm-roberta")
- Python variables or identifiers containing hyphens, causing syntax errors
- Class name capitalization issues (e.g., "GptjForCausalLM" vs "GPTJForCausalLM")

#### Solutions

1. **Convert model names to valid identifiers**:

   Use the `to_valid_identifier()` function to convert hyphenated names to valid Python identifiers:

   ```python
   def to_valid_identifier(text):
       """Convert text to a valid Python identifier."""
       text = text.replace("-", "_")
       text = re.sub(r'[^a-zA-Z0-9_]', '', text)
       if text and text[0].isdigit():
           text = '_' + text
       return text
   ```

2. **Fix class name capitalization issues**:

   Maintain a dictionary of special capitalization cases:

   ```python
   CLASS_NAME_FIXES = {
       "GptjForCausalLM": "GPTJForCausalLM",
       "GptneoForCausalLM": "GPTNeoForCausalLM",
       "XlmRobertaForMaskedLM": "XLMRobertaForMaskedLM"
   }
   ```

3. **Run the regeneration script**:

   ```bash
   python regenerate_fixed_tests.py
   ```

4. **Manually verify generated files**:

   ```bash
   cd fixed_tests
   python test_hf_gpt_j.py --list-models
   python test_hf_xlm_roberta.py --list-models
   ```

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
- Missing model families in registry
- Wrong architecture selected for models

#### Solutions

1. **Check architecture mapping**: Ensure the model type is correctly mapped to an architecture type in the `ARCHITECTURE_TYPES` dictionary:

   ```python
   ARCHITECTURE_TYPES = {
       "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta"],
       "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt"],
       "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "longt5", "led", "marian", "mt5", "flan"],
       # Other architecture types...
   }
   ```

2. **Update model registry**: Add missing models to the `MODEL_REGISTRY`:

   ```python
   MODEL_REGISTRY["new-model"] = {
       "family_name": "New-Model",
       "description": "Description of the model",
       "default_model": "org/new-model-base",
       "class": "NewModelForSequenceClassification",
       "test_class": "TestNewModelModels",
       "module_name": "test_hf_new_model",
       "tasks": ["sequence-classification"],
       "inputs": {
           "text": "Sample input text for the model."
       }
   }
   ```

3. **Use discovery scripts**: Run the model discovery script to find missing models:

   ```bash
   python find_models.py --update-registry
   ```

4. **Create a custom template**: If a model has unique requirements, create a custom template in the `templates` directory.

5. **Fix template selection logic**: Ensure the template selection function is correctly matching models to their architecture:

   ```python
   def get_template_for_architecture(model_type, templates_dir="templates"):
       arch_type = get_architecture_type(model_type)
       template_path = template_map.get(arch_type)
       if not template_path or not os.path.exists(template_path):
           logger.warning(f"Template not found for {arch_type}, using fallback")
           return fallback_template
       return template_path
   ```

6. **Manually copy and modify**: Use an existing test file for a similar architecture as a starting point.

7. **Generate minimal test**: Use the minimal test generator:

   ```bash
   python create_minimal_test.py --model "model-name"
   ```

### Syntax Verification Failures

#### Symptoms

- Tests can't be executed due to syntax errors
- Errors in the verification step
- Unterminated string literals
- Imbalanced triple quotes in docstrings

#### Solutions

1. **Add validation to the generator**:

   Use Python's `compile()` function to verify syntax:

   ```python
   try:
       compile(content, output_file, 'exec')
       logger.info(f"✅ Syntax is valid for {output_file}")
   except SyntaxError as e:
       logger.error(f"❌ Syntax error in generated file: {e}")
       # Show the problematic line for debugging
       if hasattr(e, 'lineno') and e.lineno is not None:
           lines = content.split('\n')
           line_no = e.lineno - 1  # 0-based index
           if 0 <= line_no < len(lines):
               logger.error(f"Problematic line {e.lineno}: {lines[line_no].rstrip()}")
   ```

2. **Fix syntax errors automatically**:

   Add a syntax error fixing function:

   ```python
   def fix_syntax_errors(content):
       """Fix common syntax errors like unterminated string literals."""
       # Fix extra quotes ("""")
       content = content.replace('""""', '"""')
       
       # Check for unclosed triple quotes
       triple_quotes_count = content.count('"""')
       if triple_quotes_count % 2 != 0:
           logger.info(f"Odd number of triple quotes found: {triple_quotes_count}, fixing...")
           # Try to find and fix the problem
           # ...
       
       return content
   ```

3. **Run Python's built-in syntax checker**:

   ```bash
   python -m py_compile test_hf_model.py
   ```

4. **Use a linter**:

   ```bash
   flake8 test_hf_model.py
   ```

5. **Verify syntax of all generated files**:

   ```bash
   python verify_python_syntax.py --dir fixed_tests
   ```

6. **Fix specific issues**: Look for:
   - String formatting issues (f-strings vs old style)
   - Missing or extra parentheses
   - Incorrect dictionary or list syntax
   - Unterminated string literals, especially in docstrings
   - Imbalanced triple quotes

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

## Systematic Approach for Complete Coverage

To efficiently expand test coverage to all 300+ HuggingFace model classes, follow this systematic approach:

### Phase 1: Model Discovery and Categorization

1. **Create an inventory of all models**:

   ```bash
   python find_models.py --catalog
   ```

2. **Categorize models by architecture**:

   ```bash
   python find_models.py --categorize
   ```

3. **Prioritize models based on importance**:
   - Tier 1: Core models (BERT, GPT, T5, ViT, etc.)
   - Tier 2: Commonly used models
   - Tier 3: Specialized or niche models

### Phase 2: Template Refinement

1. **Create or update templates for each architecture type**:
   - Encoder-only: `encoder_only_template.py`
   - Decoder-only: `decoder_only_template.py`
   - Encoder-decoder: `encoder_decoder_template.py`
   - Vision: `vision_template.py`
   - Multimodal: `multimodal_template.py`
   - Speech: `speech_template.py`

2. **Add architecture-specific test methods**:
   - Ensure proper input handling for each architecture
   - Add model-specific validation logic

### Phase 3: Automated Test Generation

1. **Generate tests for each tier**:

   ```bash
   # Generate tests for top-priority models first
   python generate_batch_tests.py --tier 1 --output-dir fixed_tests
   ```

2. **Track progress and identify failures**:

   ```bash
   python test_generator_fixed.py --report
   ```

3. **Fix common issues and regenerate tests**:

   ```bash
   python regenerate_fixed_tests.py --fix-common-issues
   ```

### Phase 4: Integration with CI/CD

1. **Add tests to CI pipeline**:

   ```bash
   python test_toolkit.py --setup-ci
   ```

2. **Create a dashboard for test coverage**:

   ```bash
   python visualize_test_coverage.py
   ```

## Common Error Patterns and Solutions

| Error Pattern | Likely Cause | Solution |
|---------------|--------------|----------|
| `SyntaxError: invalid syntax` | Hyphenated model name used as identifier | Convert to valid identifier with `to_valid_identifier()` |
| `expected an indented block after 'try' statement` | Import issues with missing modules | Use proper fallback in try/except blocks |
| `ModuleNotFoundError: No module named 'transformers'` | Missing dependencies | Add proper mock objects and HAS_* flags |
| `AttributeError: module 'transformers' has no attribute 'GptjForCausalLM'` | Incorrect class name capitalization | Add entry to CLASS_NAME_FIXES dictionary |
| `NameError: name 'gpt_j_pipeline' is not defined` | Variable naming inconsistency | Ensure consistent variable naming patterns |
| Triple quote imbalance | Docstring formatting issues | Fix with `fix_syntax_errors()` function |

## Contributing to the Framework

If you encounter an issue not covered in this guide:

1. Open an issue in the repository
2. Include detailed error information
3. Provide steps to reproduce the issue
4. Consider contributing a fix with a pull request

## Troubleshooting Command Reference

```bash
# Fix hyphenated model names in tests
python regenerate_fixed_tests.py

# Fix indentation issues
python fix_indentation_and_apply_template.py --dir fixed_tests

# Validate syntax of all files in a directory
python verify_python_syntax.py --dir fixed_tests

# Generate a minimal test for a specific model
python create_minimal_test.py --model "bert-base-uncased"

# Fix a specific test file
python fix_single_file.py --file fixed_tests/test_hf_gpt_j.py

# Show test coverage statistics
python visualize_test_coverage.py --report
```

## Contacting Support

For additional help:

1. Check the documentation in the `docs` directory
2. Refer to the [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index)
3. Open an issue in the repository

## License

Apache 2.0