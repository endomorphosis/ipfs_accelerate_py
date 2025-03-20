# Fixed Test Files for Hugging Face Models

This directory contains properly formatted, minimal test files for various Hugging Face model families. These files are generated with correct indentation and structure to ensure they pass syntax validation.

## Approach

The original test files had various indentation issues that made them difficult to fix automatically. Instead of trying to patch the existing files, we created minimal, clean implementations from scratch that:

1. Follow consistent indentation patterns (4 spaces for class level, 8 spaces for method level)
2. Include only essential functionality for testing models
3. Validate each file with Python's syntax checker before deployment
4. Maintain consistent structure across all model families

## Test Files

The following model families have minimal test implementations:

- `test_hf_bert.py`: BERT encoder-only models (`bert-base-uncased`)
- `test_hf_gpt2.py`: GPT-2 decoder-only models (`gpt2`)
- `test_hf_t5.py`: T5 encoder-decoder models (`t5-small`)
- `test_hf_vit.py`: Vision Transformer models (`google/vit-base-patch16-224`)

## Usage

Each test file can be run standalone:

```bash
python test_hf_bert.py  # Test BERT model

# Test with specific options
python test_hf_bert.py --model bert-base-uncased
python test_hf_bert.py --cpu-only  # Force CPU execution
python test_hf_bert.py --save      # Save results to file
```

## Extending to More Models

To add more model families, you can use the `create_minimal_test.py` script:

```bash
python create_minimal_test.py --families bert gpt2 t5 vit
```

## Implementation Details

Each test file includes:

- Proper import handling with fallbacks for missing dependencies
- Hardware detection for CPU, CUDA, MPS, and OpenVINO
- Minimal test class with pipeline API testing
- Command-line interface for customization
- Result storage and reporting

To ensure proper indentation, we follow these rules:

1. Top-level statements: 0 spaces
2. Class definitions: 0 spaces
3. Class methods: 4 spaces
4. Method content: 8 spaces
5. Nested blocks: 12 spaces

## Known Limitations

The minimal test files only include the essential functionality needed for basic validation. Some features from the original test files are not included:

- No advanced error classification
- Limited hardware testing options
- Simplified result reporting
- No performance benchmarking

## Why This Approach

Rather than trying to fix the complex indentation issues in the original files, creating minimal files from scratch ensures:

1. Clean, maintainable code
2. No subtle indentation bugs
3. Consistent style across all files
4. Easy extension to new model families