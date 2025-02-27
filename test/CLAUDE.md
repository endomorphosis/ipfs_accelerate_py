# IPFS Accelerate Python Framework - Development Guide

## Build & Test Commands
- Run single test: `python -m test.apis.test_<api_name>` or `python -m test.skills.test_<skill_name>`
- Run all tests in a directory: Use Python's unittest discovery `python -m unittest discover -s test/apis` or `python -m unittest discover -s test/skills`
- Tests compare collected results with expected results in JSON files
- Run a test file directly: `python3 /home/barberb/ipfs_accelerate_py/test/skills/test_hf_<skill_name>.py`

## Code Style Guidelines
- Use snake_case for variables, functions, methods, modules
- Use PEP 8 formatting standards
- Include comprehensive docstrings for classes and methods
- Use absolute imports with sys.path.append for module resolution
- Standard imports first, then third-party libraries
- Standardized error handling with try/except blocks and detailed error messages
- Test results stored in JSON files with consistent naming
- Unittest-based testing with async support via asyncio.run()
- Mocking external dependencies in tests with unittest.mock
- Tests include result collection, comparison with expected results, and detailed error reporting

## Test File Standardization Pattern
Follow this pattern when updating test files for consistent structure:

1. **Imports Section**:
   - Standard library imports first
   - Third-party imports next
   - Absolute path setup with `sys.path.insert(0, "/home/barberb/ipfs_accelerate_py")`
   - Try/except pattern for importing optional dependencies like transformers

2. **Utility Functions**:
   - Add fallback implementations for specialized input handling
   - Include clear docstrings

3. **Class Structure**:
   - `__init__` with resources and metadata parameters
   - `test()` method organized by hardware platform
   - `__test__()` method for result collection, comparison, and storage

4. **Test Results Format**:
   - Include implementation type in status messages: `"Success (REAL)"` or `"Success (MOCK)"`
   - Store structured examples with input, output, timestamp, and implementation type
   - Use consistent metadata structure
   - Exclude variable data (timestamps, outputs) when comparing expected vs. collected

5. **Hardware Testing Sections**:
   - Test each platform (CPU, CUDA, OpenVINO, Apple, Qualcomm) in separate try/except blocks
   - Use clear implementation_type markers
   - Handle platform-specific exceptions gracefully
   - Store results in consistent format

6. **Result Storage and Comparison**:
   - Add metadata with environment information
   - Create proper directory structure
   - Use proper filtering to exclude variable fields in comparisons
   - Automatically update expected results with proper messaging

Files that need standardization:
- ✅ test_hf_clip.py (standardized)
- ✅ test_hf_whisper.py (standardized)
- ✅ test_hf_llava_next.py (standardized)
- ✅ test_hf_bert.py (standardized)
- ✅ test_hf_clap.py (standardized)
- ✅ test_hf_llama.py (standardized)
- ✅ test_hf_xclip.py (standardized)
- ✅ test_default_embed.py (standardized)
- ✅ test_default_lm.py (standardized)
- ✅ test_hf_t5.py (standardized)
- ✅ test_hf_llava.py (standardized)
- ✅ test_hf_wav2vec2.py (standardized)

Standardization Progress:
- Completed: 12/12 files (test_hf_clip, test_hf_bert, test_hf_t5, test_hf_llava, test_hf_clap, test_hf_xclip, test_hf_llama, test_hf_whisper, test_hf_llava_next, test_hf_wav2vec2, test_default_embed, test_default_lm)
- All files have been standardized! 🎉

Standardization Achievements:
- All files now follow consistent import ordering pattern
- All test files include proper organization by hardware platform
- Example collection with implementation type markers is implemented in all files
- All test outputs include structured examples with timestamps, input/output data, and implementation type markers
- Result storage and comparison has been improved across all files
- All files now handle REAL vs MOCK implementations clearly and gracefully

Key Standardization Elements Implemented:
1. **Consistent Imports**:
   - Standard library imports first
   - Third-party imports next
   - Absolute paths with sys.path.insert
   - Try/except pattern for optional dependencies

2. **Implementation Type Markers**:
   - All status messages clearly indicate if using real implementations or mocks
   - Consistent "(REAL)" or "(MOCK)" suffix on all success/failure messages
   - Proper fallback to mock implementations when needed

3. **Structured Examples**:
   - All test outputs include examples with:
     - Input data
     - Output data (or shape if output is large)
     - Timestamp
     - Elapsed time measurements
     - Implementation type marker
     - Platform identifier

4. **Hardware Platform Organization**:
   - Each platform (CPU, CUDA, OpenVINO, Apple, Qualcomm) has its own section
   - Consistent error handling and reporting across platforms
   - Graceful fallbacks when hardware isn't available

5. **Result Comparison**:
   - Proper directory structure for expected and collected results
   - Filtering of variable data (timestamps, outputs) when comparing
   - Metadata with complete environment information
   - Automatic expected results updates with clear messaging
   - Graceful handling of implementation type differences

## Hugging Face Models Compatibility Status

Based on test results, here's the current compatibility matrix for Hugging Face models:

| Model               | CPU Status     | OpenVINO Status | Notes                                                   |
|---------------------|----------------|-----------------|----------------------------------------------------------|
| BERT                | Success (REAL) | Success (MOCK)  | OpenVINO implementation needs development                |
| CLIP                | Success (REAL) | Success (MOCK)  | OpenVINO implementation needs development                |
| LLAMA               | Success (REAL) | Success (REAL)  | Both CPU and OpenVINO have real implementations         |
| LLaVA               | Success (REAL) | Error           | OpenVINO error: missing 'openvino_cli_convert' argument |
| T5                  | Success (REAL) | Error           | OpenVINO error: invalid model identifier                 |
| WAV2VEC2            | Success (REAL) | Success (MOCK)  | OpenVINO implementation needs development                |
| Whisper             | Success (REAL) | Success (MOCK)  | OpenVINO implementation needs development                |
| XCLIP               | Success (MOCK) | Success (MOCK)  | No real implementation available for any platform        |
| CLAP                | Success (MOCK) | Error           | OpenVINO error: "list index out of range"                |
| Sentence Embeddings | Success (REAL) | Success (MOCK)  | OpenVINO implementation needs development                |
| Language Model      | Success (REAL) | Success (MOCK)  | OpenVINO implementation needs development                |
| LLaVA-Next          | Success (REAL) | Success (REAL)  | Both CPU and OpenVINO have real implementations         |

## OpenVINO Compatibility Plan

Priority issues to fix for OpenVINO compatibility:

1. **LLaVA**: Fix missing 'openvino_cli_convert' argument in init_openvino. This appears to be a required positional argument that's not being passed correctly.

2. **T5**: Resolve invalid model identifier error. This indicates the model path or configuration is incorrect for OpenVINO.

3. **CLAP**: Fix the "list index out of range" error, which suggests an array access issue in the model initialization or conversion.

Next steps for models with mock implementations:

1. Implement real OpenVINO support for:
   - BERT
   - CLIP
   - WAV2VEC2
   - Whisper
   - Sentence Embeddings
   - Language Model

2. Leverage patterns from successful models:
   - Examine LLAMA and LLaVA-Next implementations which have working REAL OpenVINO support
   - Use consistent model conversion and initialization patterns across all models

Implementation recommendations:

1. Create standard functions for:
   - Model conversion to OpenVINO format (with proper error handling)
   - Model caching and loading with appropriate weight formats
   - Handling input/output tensors appropriately for each model type

2. Add comprehensive error handling with graceful fallback to CPU when OpenVINO fails

3. Update documentation with compatibility status for each hardware platform

Next Steps:
- Consider adding performance benchmarking across platforms
- Add additional hardware acceleration backends as they become available
- Keep test files updated as model implementations evolve