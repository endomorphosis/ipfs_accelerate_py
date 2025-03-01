# IPFS Accelerate Model Testing Framework

This directory contains the framework and tools for testing the various model implementations within the IPFS Accelerate Python library. The testing framework is designed to validate model functionality across different hardware platforms (CPU, CUDA, OpenVINO) and implementation types (REAL vs MOCK).

## Model Test Coverage

All models defined in `mapped_models.json` have corresponding test implementations as of March 2025.

### Model Test Status

| Model Type | Test File | CPU Support | CUDA Support | OpenVINO Support |
|------------|-----------|-------------|--------------|------------------|
| **Language Models** |
| BERT | `skills/test_hf_bert.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| DistilBERT | `skills/test_hf_distilbert.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| RoBERTa | `skills/test_hf_roberta.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| GPT Neo | `skills/test_hf_gpt_neo.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| GPTJ | `skills/test_hf_gptj.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| BART | `skills/test_hf_bart.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| MT5 | `skills/test_hf_mt5.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| mBART | `skills/test_hf_mbart.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| ELECTRA | `skills/test_hf_electra.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| Longformer | `skills/test_hf_longformer.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| DeBERTa-v2 | `skills/test_hf_deberta_v2.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| GPT2 | `skills/test_hf_gpt2.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| DPR | `skills/test_hf_dpr.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| MobileBERT | `skills/test_hf_mobilebert.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| MPNet | `skills/test_hf_mpnet.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| CamemBERT | `skills/test_hf_camembert.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| FlauBERT | `skills/test_hf_flaubert.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| CodeGen | `skills/test_hf_codegen.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| XLM-RoBERTa | `skills/test_hf_xlm_roberta.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| ALBERT | `skills/test_hf_albert.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| OPT | `skills/test_hf_opt.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| BLOOM | `skills/test_hf_bloom.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| SqueezeBERT | `skills/test_hf_squeezebert.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| LayoutLM | `skills/test_hf_layoutlm.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| DeBERTa | `skills/test_hf_deberta.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| LLAMA | `skills/test_hf_llama.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| Pegasus | `skills/test_hf_pegasus.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| LED | `skills/test_hf_led.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| Qwen2 | `skills/test_hf_qwen2.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| **Vision Models** |
| ViT | `skills/test_hf_vit.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| DeiT | `skills/test_hf_deit.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| DETR | `skills/test_hf_detr.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| Swin | `skills/test_hf_swin.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| ConvNeXt | `skills/test_hf_convnext.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| CLIP | `skills/test_hf_clip.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| VideoMAE | `skills/test_hf_videomae.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| XCLIP | `skills/test_hf_xclip.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| **Audio Models** |
| Wav2Vec2 | `skills/test_hf_wav2vec2.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| Whisper | `skills/test_hf_whisper.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| HuBERT | `skills/test_hf_hubert.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| CLAP | `skills/test_hf_clap.py` | ✅ REAL | ✅ REAL | 🟠 PARTIAL |
| **Multimodal Models** |
| LLaVA | `skills/test_hf_llava.py` | ✅ REAL | ✅ REAL | ⚠️ MOCK |
| LLaVA-Next | `skills/test_hf_llava_next.py` | ✅ REAL | ✅ REAL | ⚠️ MOCK |
| Qwen2-VL | `skills/test_hf_qwen2_vl.py` | ✅ REAL | ✅ REAL | ⚠️ MOCK |
| **Conversational Models** |
| BlenderBot | `skills/test_hf_blenderbot.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| BlenderBot-Small | `skills/test_hf_blenderbot_small.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| **Default Models** |
| Default Embedding | `skills/test_default_embed.py` | ✅ REAL | ✅ REAL | ✅ REAL |
| Default LM | `skills/test_default_lm.py` | ✅ REAL | ✅ REAL | ✅ REAL |

## Running Model Tests

To run the tests for a specific model:

```bash
# Run a specific model test
python3 skills/test_hf_bert.py

# Run multiple tests in parallel
python3 run_skills_tests.py --models bert,roberta,gpt2

# Run all model tests
python3 run_skills_tests.py --all

# Run tests for a specific type
python3 run_skills_tests.py --type language  # Options: language, vision, audio, multimodal
```

### Test options

The test scripts support various options:

```bash
python3 skills/test_hf_bert.py --real  # Force real implementation
python3 skills/test_hf_bert.py --mock  # Force mock implementation
python3 skills/test_hf_bert.py --platform cpu  # Test only CPU platform
python3 skills/test_hf_bert.py --platform cuda  # Test only CUDA platform
python3 skills/test_hf_bert.py --platform openvino  # Test only OpenVINO platform
```

## Test Structure

Each model test file follows a common structure:

1. **Imports and Setup**: Import necessary libraries and set up test environment
2. **CUDA Implementation**: Override `init_cuda` method for CUDA support
3. **Init Method**: Initialize model and test inputs
4. **Test Method**: Run tests for CPU, CUDA, and OpenVINO platforms
5. **Result Collection**: Generate structured test results

Example test method structure:

```python
def test(self):
    results = {}
    
    # Test basic initialization
    results["init"] = "Success" if self.model is not None else "Failed initialization"
    
    # CPU tests
    try:
        # Initialize CPU endpoint
        endpoint, processor, handler = self.model.init_cpu(...)
        
        # Run inference
        output = handler(self.test_input)
        
        # Record results
        results["cpu_handler"] = "Success (REAL)" if output is not None else "Failed CPU handler"
    except Exception as e:
        results["cpu_error"] = str(e)
        
    # CUDA tests (similar structure)
    if torch.cuda.is_available():
        try:
            # Initialize CUDA endpoint
            # Run inference
            # Record results
        except Exception as e:
            results["cuda_error"] = str(e)
    
    # Return structured results
    return {
        "status": results,
        "examples": self.examples,
        "metadata": {
            "model_name": self.model_name,
            # Additional metadata
        }
    }
```

## Creating New Model Tests

To add a test for a new model:

1. Copy an existing test file for a similar model type
2. Update the imports to reference the correct model class
3. Modify the model initialization and handling code
4. Adjust test inputs to be appropriate for the model type
5. Run the test to generate expected results

## Implementation Types

Tests detect different implementation types:

- **REAL**: Using actual model weights and performing real inference
- **MOCK**: Using simulated responses without loading model weights
- **PARTIAL**: Using a hybrid approach with some real components

The implementation type is determined by examining:
- Whether model weights are actually loaded
- The structure of the model outputs
- Performance characteristics like memory usage
- Special markers in the response structure

## Hardware Platform Support

Tests automatically detect and use available hardware:

- **CPU**: Always supported as the baseline platform
- **CUDA**: Used when NVIDIA GPUs are available
- **OpenVINO**: Used when Intel hardware acceleration is available
- **Other platforms**: Apple Silicon and Qualcomm support is present in some tests

## Common Troubleshooting

- **Model download issues**: Tests will create minimal test models when download fails
- **CUDA errors**: Ensure CUDA toolkit and drivers are properly installed
- **Memory errors**: Tests include fallback mechanisms for memory constraints
- **Import errors**: Tests mock missing dependencies to enable partial testing

## Performance Considerations

The test suite is designed to operate with minimal resources:

- Small model variants are preferred when available
- Tests create minimal models when downloads are not possible
- Batch size is optimized for each platform
- Tests include cleanup to free memory after testing

## Results Format

Test results are stored in two formats:

1. **JSON files**: Complete test results in `collected_results` directories
2. **Expected results**: Baseline for comparison in `expected_results` directories

The JSON structure follows this pattern:

```json
{
  "status": {
    "init": "Success",
    "cpu_init": "Success (REAL)",
    "cpu_handler": "Success (REAL)",
    "cuda_init": "Success (REAL)",
    "cuda_handler": "Success (REAL)"
  },
  "examples": [
    {
      "input": "Test input",
      "output": {
        "result": "Test output",
        "performance_metrics": {
          "inference_time": 0.123,
          "memory_usage_mb": 256
        }
      },
      "platform": "CPU",
      "implementation_type": "REAL"
    }
  ],
  "metadata": {
    "model_name": "model-name",
    "test_timestamp": "2025-03-01T12:00:00.000Z"
  }
}
```