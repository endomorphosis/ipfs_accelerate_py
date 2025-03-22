# Hyphenated Model Validation System

This validation system provides comprehensive validation for the hyphenated model test solution, addressing the need for model-specific validation rules, actual model inference validation, and comprehensive reporting with actionable insights.

## Components

The validation system consists of several components that work together:

1. **Syntax and Architecture Validator**: Validates test files against model-specific architecture requirements
2. **Model Inference Validator**: Tests actual model inference capabilities when possible
3. **Comprehensive Reporting System**: Generates reports with actionable recommendations
4. **CI/CD Integration**: Integrates with GitHub Actions workflows

## Key Files

- `validate_hyphenated_model_solution.py`: Main validation script for syntax and architecture rules
- `validate_model_inference.py`: Model-specific inference validation
- `validate_all_hyphenated_models.py`: Complete validation suite that combines all components
- `test_inference_validation.py`: Test script for specific model inference validation
- `ci_templates/github-actions-model-validation.yml`: GitHub Actions workflow for CI/CD integration

## Usage

### Basic Validation

To validate all hyphenated model test files:

```bash
python validate_hyphenated_model_solution.py --all --report
```

This will check all test files against architecture-specific validation rules and generate a report.

### Inference Validation

To validate model inference capabilities:

```bash
python validate_model_inference.py --model gpt-j --use-small
```

This will test the inference capabilities of a specific model using a smaller variant for faster testing.

### Complete Validation Suite

To run the complete validation suite:

```bash
python validate_all_hyphenated_models.py --regenerate --inference
```

This will:
1. Regenerate all test files for hyphenated models
2. Run syntax and architecture validation
3. Run inference validation with small model variants
4. Generate a comprehensive report

### Testing a Specific Model

To test inference validation for a specific model:

```bash
python test_inference_validation.py gpt-j --use-small
```

## Architecture-Specific Validation Rules

The system applies model-specific validation rules based on the model's architecture type:

### Encoder-Only Models (BERT, XLM-RoBERTa, etc.)
- Required methods: test_pipeline, test_from_pretrained, run_tests
- Required tasks: fill-mask
- Required variables: HAS_TRANSFORMERS, HAS_TORCH, HAS_TOKENIZERS

### Decoder-Only Models (GPT-J, GPT-Neo, etc.)
- Required methods: test_pipeline, test_from_pretrained, run_tests
- Required tasks: text-generation
- Required variables: HAS_TRANSFORMERS, HAS_TORCH, HAS_TOKENIZERS

### Encoder-Decoder Models (T5, BART, etc.)
- Required methods: test_pipeline, test_from_pretrained, run_tests
- Required tasks: text2text-generation, translation
- Required variables: HAS_TRANSFORMERS, HAS_TORCH, HAS_TOKENIZERS, HAS_SENTENCEPIECE

### Vision Models (ViT, Swin, etc.)
- Required methods: test_pipeline, test_from_pretrained, run_tests
- Required tasks: image-classification
- Required variables: HAS_TRANSFORMERS, HAS_TORCH, HAS_TOKENIZERS

### Vision-Text Models (CLIP, Vision-Text-Dual-Encoder, etc.)
- Required methods: test_pipeline, test_from_pretrained, run_tests
- Required tasks: image-to-text, zero-shot-image-classification
- Required variables: HAS_TRANSFORMERS, HAS_TORCH, HAS_TOKENIZERS

### Speech Models (Wav2Vec2, Speech-to-Text, etc.)
- Required methods: test_pipeline, test_from_pretrained, run_tests
- Required tasks: automatic-speech-recognition, audio-classification
- Required variables: HAS_TRANSFORMERS, HAS_TORCH, HAS_TOKENIZERS

## Mock Detection Validation

The system also validates that all test files properly implement mock detection for CI/CD environments:

- Checks for proper mock detection variables: HAS_TRANSFORMERS, HAS_TORCH, HAS_TOKENIZERS, HAS_SENTENCEPIECE
- Verifies that the run_tests method properly detects mock usage
- Ensures metadata includes information about mock detection
- Validates that the main function displays mock usage status to users

## Model Class Validation

The system validates model classes for proper capitalization according to HuggingFace conventions:

- GPT2LMHeadModel (not Gpt2LMHeadModel)
- GPTJForCausalLM (not GptjForCausalLM)
- XLMRoBERTaForMaskedLM (not XlmRobertaForMaskedLM)
- ViTForImageClassification (not VitForImageClassification)

## Comprehensive Reports

The validation system generates comprehensive reports with:

- Overall validation status
- Architecture-specific validation results
- Model-specific validation results
- Actionable recommendations for fixing issues
- Inference test results when available

Reports are saved as markdown files in the `validation_reports` directory.

## CI/CD Integration

The system includes a GitHub Actions workflow for CI/CD integration:

- Runs syntax validation on all test files
- Validates architecture-specific requirements
- Optionally runs inference validation with small model variants
- Generates a comprehensive report
- Uploads validation reports as artifacts

## Further Customization

To add support for additional models or architectures:

1. Update the `ARCHITECTURE_TYPES` dictionary in `validate_hyphenated_model_solution.py`
2. Add model-specific class mappings in `validate_model_inference.py`
3. Add small model variants for faster testing in `validate_model_inference.py`

## Troubleshooting

If you encounter issues with the validation system:

1. Check that you have the required dependencies installed: transformers, torch, tokenizers
2. Ensure that the test files exist in the `fixed_tests` directory
3. For inference validation issues, try using `--use-small` to use smaller model variants
4. Check the validation reports for specific issues and recommendations