# Fixed HuggingFace Test Files

This directory contains fixed versions of HuggingFace test files with proper indentation and architecture-specific implementations. The tests are regenerated using architecture-specific templates that handle the unique requirements of each model type.

## Testing Progress (March 2025)

Current testing coverage:

| Category | Architecture | Models Tested | Status |
|----------|--------------|---------------|--------|
| text-encoders | encoder_only | albert, roberta, bert, electra, distilbert, hubert | ✅ 100% pass |
| text-decoders | decoder_only | bloom, gpt2, gptj, llama, gpt_neox, opt, gpt_neo | ✅ 100% pass |
| text-encoder-decoders | encoder_decoder | mt5, bart, pegasus, t5, mbart | ✅ 100% pass |
| vision | encoder_only | swin, vit, deit, beit, convnext | ✅ 100% pass |
| audio | encoder_only | wav2vec2, whisper | ✅ 100% pass |
| multimodal | encoder_decoder | clip, llava, blip | ✅ 100% pass |

All tests successful on CPU hardware platform. Testing is underway for additional hardware platforms (CUDA, ROCm, MPS, OpenVINO, Qualcomm, WebNN, WebGPU).

## Structure

- `*.py` - Fixed test files with proper indentation
- `*.py.bak` - Backup of original files (if available)
- `collected_results/` - Test results in JSON format

## Core Model Tests

The following core model tests have been fixed:

1. `test_hf_albert.py`
2. `test_hf_bart.py`
3. `test_hf_beit.py`
4. `test_hf_bert.py`
5. `test_hf_blip.py`
6. `test_hf_bloom.py`
7. `test_hf_clip.py`
8. `test_hf_convnext.py`
9. `test_hf_deit.py`
10. `test_hf_detr.py`
11. `test_hf_distilbert.py`
12. `test_hf_electra.py`
13. `test_hf_gpt2.py`
14. `test_hf_gpt_neo.py`
15. `test_hf_gpt_neox.py`
16. `test_hf_gptj.py`
17. `test_hf_hubert.py`
18. `test_hf_llama.py`
19. `test_hf_llava.py`
20. `test_hf_mbart.py`
21. `test_hf_mt5.py`
22. `test_hf_opt.py`
23. `test_hf_pegasus.py`
24. `test_hf_roberta.py`
25. `test_hf_swin.py`
26. `test_hf_t5.py`
27. `test_hf_vit.py`
28. `test_hf_wav2vec2.py`
29. `test_hf_whisper.py`

## Architecture Templates

Each test file implements architecture-specific handling through dedicated templates:

- **Encoder-Only** (BERT, RoBERTa, etc.):
  - Bidirectional attention patterns
  - Mask token handling for masked language modeling
  - Consistent tokenizer interface
  - Token prediction extraction

- **Decoder-Only** (GPT-2, LLaMA, etc.):
  - Autoregressive behavior
  - Padding token configuration (often setting pad_token = eos_token)
  - Causal attention patterns
  - Text generation capabilities

- **Encoder-Decoder** (T5, BART, etc.):
  - Separate encoder and decoder components
  - Decoder input initialization
  - Sequence-to-sequence capabilities
  - Translation task handling

- **Vision** (ViT, Swin, etc.):
  - Image preprocessing with proper tensor shapes
  - Image processor instead of tokenizer
  - Pixel values handling
  - Classification task implementation

- **Speech** (Whisper, Wav2Vec2, etc.):
  - Audio preprocessing and feature extraction
  - Mel spectrogram conversion
  - Automatic speech recognition task handling
  - Audio processor configuration

- **Vision-Text** (CLIP, BLIP, etc.):
  - Dual-stream architecture for images and text
  - Combined image-text processing
  - Contrastive learning implementations
  - Cross-modal alignment

## Test File Structure

Each generated test file follows a consistent structure:

1. **Hardware Detection** - Identifies available hardware (CPU, CUDA, MPS, OpenVINO)
2. **Dependency Management** - Gracefully handles missing dependencies with mock objects
3. **Mock Detection** - Clearly indicates when tests use real inference (🚀) vs. mock objects (🔷), with detailed dependency reporting
4. **Model Registry** - Maps model IDs to configurations and default parameters
5. **Test Class** - Implements architecture-specific testing methods
6. **Pipeline Testing** - Tests using the `transformers.pipeline()` API
7. **Direct Testing** - Tests using the low-level `from_pretrained()` API
8. **Hardware-specific Testing** - Tests on specialized hardware like OpenVINO
9. **Utility Functions** - Provides result saving and command-line interface

## Running Tests

Run a specific test:

```bash
python test_hf_bert.py  # Test with default model
python test_hf_bert.py --model "bert-base-uncased"  # Test specific model
python test_hf_bert.py --all-hardware  # Test on all available hardware
python test_hf_bert.py --list-models  # List available models of this architecture
python test_hf_bert.py --save  # Save results to JSON
```

## Regenerating Tests

To regenerate these fixed tests, use the script:

```bash
python ../regenerate_fixed_tests.py --model bert --verify  # Regenerate single model test
python ../regenerate_fixed_tests.py --all --verify  # Regenerate all tests
```

## Collected Results

The `collected_results/` directory contains JSON files with test results:

- Hardware details (CUDA version, device count, etc.)
- Model metadata (size, parameters, etc.)
- Performance metrics (load time, inference time, etc.)
- Success/failure status
- Error information if applicable
- Test environment information:
  - Available dependencies (`has_transformers`, `has_torch`, etc.)
  - Mock vs. real inference status (`using_real_inference`, `using_mocks`)
  - Test type indicator (`REAL INFERENCE` or `MOCK OBJECTS (CI/CD)`)

## Benefits of Fixed Tests

1. **Consistent Syntax** - All tests follow proper Python indentation and syntax rules
2. **Architecture Awareness** - Each test handles the specific requirements of its model family
3. **Hardware Optimization** - Tests automatically detect and use the best available hardware
4. **Graceful Degradation** - Tests continue to work even with missing dependencies
5. **Mock Detection** - Clear visual indicators (🚀 vs 🔷) distinguish between real inference and mock objects
6. **Test Transparency** - Users always know if tests are running with real or mocked dependencies, with detailed dependency reporting
7. **Comprehensive Environment Metadata** - Test results include detailed information about dependencies and mock status
8. **Comprehensive Reports** - Detailed reporting makes debugging and comparison easy

## Implementation Notes

These fixed test files maintain the core functionality of the original tests while addressing indentation issues and implementing architecture-specific handling to ensure proper Python syntax and execution. The templates are designed to be flexible and adaptable to different models within the same architecture family.

## Recent Fixes

The following issues were fixed in the latest update:

1. **Comprehensive Coverage** - Added tests for all 29 model families defined in MODEL_CATEGORIES
2. **GPT-2 Model Class** - Changed `AutoModelLMHeadModel` to `AutoModelForCausalLM` for correct model loading
3. **Architecture-specific Templates** - Implemented separate templates for each model architecture
4. **Test Validation** - Verified that all tests run successfully across all architectures
5. **ctypes.util Import** - Fixed import to use `import ctypes.util` instead of `import ctypes` for WebGPU detection
6. **Template Paths** - Enhanced template path handling to locate templates correctly regardless of working directory
7. **Mock Detection System** - Added visual indicators (🚀 vs. 🔷) to clearly show when tests use real inference vs. mock objects, with comprehensive dependency reporting
8. **Metadata Enrichment** - Added detailed test environment information to result JSON data for better tracking including specific dependency status


All tests are now passing with proper model initialization and execution.

## Comprehensive Testing Support

### Distributed Testing

All test files now support distributed testing execution:

```bash
# Update test files to support distributed testing
python ../update_for_distributed_testing.py --dir ./ --verify

# Run tests in distributed mode
python ../run_distributed_tests.py --model-family bert --workers 4
python ../run_distributed_tests.py --all --workers 8
```

Distributed testing features:
- Hardware-aware task distribution
- Parallel execution across multiple workers
- Result aggregation and performance reporting
- Fault tolerance with automatic retries
- Comprehensive dashboard for visualization

For more details, see [DISTRIBUTED_TESTING_README.md](../DISTRIBUTED_TESTING_README.md).

### Hardware Compatibility Testing

All tests now support comprehensive hardware compatibility testing:

```bash
# Run tests on all available hardware
python test_hf_bert.py --all-hardware

# Generate hardware compatibility matrix
python ../create_hardware_compatibility_matrix.py --architectures encoder-only decoder-only
```

Hardware compatibility features:
- Support for multiple hardware platforms (CPU, CUDA, MPS, OpenVINO, WebNN, WebGPU)
- Performance benchmarking with metrics collection
- Comprehensive compatibility matrix with DuckDB integration
- Hardware fallback mechanisms for graceful degradation
- Detailed performance reports and recommendations

For more details, see [HARDWARE_COMPATIBILITY_README.md](../HARDWARE_COMPATIBILITY_README.md).

### Mock Detection System

All tests include a comprehensive mock detection system:

- Visual indicators (🚀 vs. 🔷) for real vs. mock inference
- Detailed dependency reporting for transparent testing
- Granular tracking of specific missing dependencies
- Consistent mock behavior across all model types
- Test coverage for partial dependency scenarios

For more details on mock detection enhancements, see [MOCK_DETECTION_README.md](../MOCK_DETECTION_README.md).

This work completes Priority #2 from CLAUDE.md: "Comprehensive HuggingFace Model Testing (300+ classes)" and contributes to Priority #1: "Complete Distributed Testing Framework"



## Recent Updates (2025-03-20 20:31)

### Fixed Hyphenated Model Names

Fixed issues with hyphenated model names:
- Resolved syntax errors in files with hyphenated model names (gpt-j, gpt-neo, xlm-roberta, etc.)
- Replaced hyphens with underscores in variable names, class names, and identifiers
- Fixed registry key consistency across all test files
- Ensured proper Python syntax validation for all fixed files
- Created comprehensive fixing tool suite for automated repairs

### Added New Models

Recently added models:
- Added 20 new test files out of 23 attempted
- Updated testing coverage information
- Verified compatibility with architecture-specific templates
- All new tests include hardware detection and acceleration support

Previously added models:
- Added 9 new test files out of 19 attempted
- Updated testing coverage information
- Verified compatibility with architecture-specific templates
- All new tests include hardware detection and acceleration support
