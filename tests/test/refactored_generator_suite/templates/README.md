# IPFS Accelerate Model Generator Templates

This directory contains templates for generating model skillsets for different model architectures. These templates are used by the model generator to create consistent model implementations across all hardware platforms.

## Template Structure

Each template follows a consistent structure with hardware-specific initialization and handlers:

1. **Base Structure**:
   - `MockHandler` class for simulating hardware without real implementations
   - Hardware detection and optimization
   - Consistent logging and error handling
   - Public API methods with standard interfaces

2. **Hardware Platforms Supported**:
   - CPU: Standard CPU implementation
   - CUDA: NVIDIA GPU implementation
   - OpenVINO: Intel hardware acceleration
   - MPS: Apple Silicon GPU implementation
   - ROCm: AMD GPU implementation
   - Qualcomm: Qualcomm AI Engine/Hexagon DSP implementation
   - WebNN: Web Neural Network API (browser)
   - WebGPU: Web GPU API (browser)

3. **Implementation Pattern**:
   - Hardware-specific initialization methods (`init_cpu`, `init_cuda`, etc.)
   - Hardware-specific handler creation methods (`create_cpu_handler`, etc.)
   - Automatic fallback to CPU when preferred hardware is unavailable
   - Comprehensive test cases for validation
   - Benchmarking functionality across hardware platforms

## Available Templates

- **encoder_only_template.py**: For BERT, RoBERTa, and other encoder-only models
  - Feature extraction and masked language modeling
  - Token-level and sequence-level embeddings

- **decoder_only_template.py**: For GPT-2, LLaMA, and other decoder-only models
  - Text generation with configurable parameters
  - Token generation with variable length outputs

- **vision_template.py**: For ViT, Swin, and other vision models
  - Image classification with multiple class predictions
  - Support for various image input formats

- **encoder_decoder_template.py**: For T5, BART, and other encoder-decoder models
  - Text-to-text generation (translation, summarization)
  - Encoder-decoder architecture with cross-attention

- **speech_template.py**: For Whisper, Wav2Vec2, and other speech models
  - Audio processing and feature extraction
  - Speech recognition capabilities

- **vision_text_template.py**: For CLIP, BLIP, and other vision-text models
  - Multi-modal processing of images and text
  - Image-text similarity and matching

## Usage

Templates are used by the model generator to create model-specific implementations:

```bash
python -m generators.model_generator --model <model_name> --device <device>
```

## Test Cases

Each template includes a set of test cases for validation across all hardware platforms:

```python
self.test_cases = [
    {
        "description": "Test on CPU platform",
        "platform": "CPU",
        "input": "Sample input",
        "expected": {"success": True}
    },
    # More test cases for other platforms
]
```

## Hardware Fallback

All templates implement automatic fallback to CPU when the preferred hardware is not available:

```python
if self.device != "cuda":
    logger.warning("CUDA not available, falling back to CPU")
    self.device = "cpu"
```

## Mock Mode

Templates support a mock mode for testing without actual hardware:

```python
MOCK_MODE = os.environ.get("MOCK_MODE", "False").lower() == "true"
```

## Command-Line Interface

Each generated skillset includes a command-line interface for testing:

```bash
python <generated_skillset>.py --model <model_id> --platform <platform> [--mock]
```

## Template Updates (March 23, 2025)

- Added `MockHandler` class for consistent mock implementations
- Implemented hardware-specific initialization methods for all platforms
- Added hardware-specific handler creation methods 
- Improved device lifecycle management with proper fallbacks
- Enhanced public API methods with consistent interfaces
- Added robust error handling and logging
- Implemented comprehensive benchmark functionality
- Added test cases for validation across all platforms
- Ensured proper template syntax and indentation handling
- Added command-line interface for direct testing