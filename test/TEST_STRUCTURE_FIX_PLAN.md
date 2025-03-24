# Test Structure Fix Plan

## Issue Summary

The manually created test files for the remaining models do not follow the template structure used by the test generator system. This leads to several problems:

1. **Inconsistent Testing**: Manual tests don't use the same validation logic as generated tests
2. **Limited Hardware Support**: Manual tests lack proper hardware detection and fallbacks
3. **Missing Mock Systems**: Manual tests don't properly implement CI/CD mock objects
4. **No Result Collection**: Manual tests don't save standardized results
5. **Code Debt**: Any changes to testing patterns require manual updates to these files

## Affected Models

The manually created models that need to be regenerated using the template system:

1. `layoutlmv2` (vision-text model)
2. `layoutlmv3` (vision-text model)
3. `clvp` (speech model)
4. `hf_bigbird` (encoder-decoder model)
5. `seamless_m4t_v2` (speech model)
6. `xlm_prophetnet` (encoder-decoder model)

## Fix Steps

### 1. Map Manual Models to Correct Architecture Types

Model mapping to ensure proper template selection:

| Model | Architecture Type | Template to Use |
|-------|------------------|-----------------|
| layoutlmv2 | vision-encoder-text-decoder | vision_text_template.py |
| layoutlmv3 | vision-encoder-text-decoder | vision_text_template.py |
| clvp | speech | speech_template.py |
| hf_bigbird | encoder-decoder | encoder_decoder_template.py |
| seamless_m4t_v2 | speech | speech_template.py |
| xlm_prophetnet | encoder-decoder | encoder_decoder_template.py |

### 2. Add Model Entries to Architecture Types Dictionary

Update `ARCHITECTURE_TYPES` in the generator system to include these models:

```python
ARCHITECTURE_TYPES = {
    "encoder-decoder": [
        # Existing models...
        "bigbird", "xlm_prophetnet"
    ],
    "vision-encoder-text-decoder": [
        # Existing models...
        "layoutlmv2", "layoutlmv3"
    ],
    "speech": [
        # Existing models...
        "clvp", "seamless_m4t_v2"
    ]
}
```

### 3. Generate Tests Using Template System

Use the regenerate_test_file function to create properly templated test files:

```python
for model in ["layoutlmv2", "layoutlmv3", "clvp", "bigbird", "seamless_m4t_v2", "xlm_prophetnet"]:
    output_path = f"fixed_tests/test_hf_{model}.py"
    regenerate_test_file(output_path, force=True, verify=True)
```

### 4. Verify Generated Tests

After generation, verify each test file:
- Ensure syntax correctness
- Verify hardware detection
- Check mock object functionality
- Test result collection

### 5. Update Model Registry

Add specific model details to MODEL_REGISTRY for accurate handling:

```python
MODEL_REGISTRY.update({
    "layoutlmv2": {
        "description": "LayoutLMv2 model for document understanding",
        "class": "LayoutLMv2ForSequenceClassification",
        "default_model": "microsoft/layoutlmv2-base-uncased",
        "architecture": "vision-encoder-text-decoder"
    },
    # Add entries for other models
})
```

### 6. Update Documentation

Update HF_MODEL_COVERAGE_ROADMAP.md to reflect the changes:
- Mark all models as implemented
- Update the implementation counts (154/154 models)
- Update model architecture details

## Benefits of Fixing

Regenerating these models using the template system will:

1. **Maintain Consistency**: All tests will follow the same pattern
2. **Reduce Code Debt**: Future changes need only update templates
3. **Improve CI/CD**: Proper mock objects for CI environments
4. **Better Hardware Support**: Tests will work across different hardware
5. **Standardized Results**: Output will follow the same format for all models

## Implementation Timeline

1. Update architecture mapping (15 minutes)
2. Generate template-based tests (30 minutes)
3. Verify generated tests (30 minutes)
4. Update documentation (15 minutes)

Total time: Approximately 1.5 hours