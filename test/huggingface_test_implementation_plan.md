# Hugging Face Model Test Implementation Plan

## Overview

This document outlines our approach to testing Hugging Face models in the IPFS Accelerate Python framework. Instead of generating individual tests for each model, our optimized approach focuses on testing transformer classes, allowing multiple models to be tested with the same code.

## Test Architecture - Class-Based Approach

The key insight is that models sharing the same architecture class can be tested with the same code. For example, all BERT variants use `BertForMaskedLM`, all GPT variants use `GPTForCausalLM`, etc.

### Benefits of a Class-Based Approach:

1. **Efficiency**: One test file per transformer class instead of per model
2. **Maintainability**: Updates to test logic only need to be made in one place
3. **Consistency**: All models of the same class are tested identically
4. **Scalability**: Adding new models of existing classes requires no new code

## Transformers Class Map

We will organize tests based on core Hugging Face architecture classes:

| Test File | Transformer Class | Example Models |
|-----------|-------------------|----------------|
| test_hf_bert.py | BertForMaskedLM | bert-base-uncased, bert-large-uncased, distilbert-base-uncased |
| test_hf_gpt.py | GPT2LMHeadModel | gpt2, gpt2-medium, distilgpt2 |
| test_hf_t5.py | T5ForConditionalGeneration | t5-small, t5-base, t5-large |
| test_hf_llama.py | LlamaForCausalLM | meta-llama/Llama-2-7b, meta-llama/Llama-2-13b |
| test_hf_vit.py | ViTForImageClassification | google/vit-base-patch16-224, facebook/deit-base |
| test_hf_whisper.py | WhisperForConditionalGeneration | openai/whisper-tiny, openai/whisper-base |
| test_hf_sam.py | SamModel | facebook/sam-vit-base, facebook/sam-vit-large |
| test_hf_llava.py | LlavaForConditionalGeneration | llava-hf/llava-1.5-7b-hf |

## Implementation Approach

Each test file will:

1. **Define a base test class** for the transformer architecture
2. **Accept a model_id parameter** to specify which model to test
3. **Handle common dependencies** required by all models of that class
4. **Provide appropriate test inputs** for the model type
5. **Include test methods** for both pipeline and direct API usage

### Example Structure:

```python
class TestBertModels:
    """Base test class for all BERT-family models"""
    
    def __init__(self, model_id=None):
        self.model_id = model_id or "bert-base-uncased"  # Default model
        self.task = "fill-mask"
        self.dependencies = ["tokenizers>=0.11.0", "sentencepiece"]
        
    def run_tests(self, hardware="cpu"):
        """Run tests for specified model on specified hardware"""
        results = {}
        results.update(self.test_pipeline(hardware))
        results.update(self.test_from_pretrained(hardware))
        return results
        
    def test_pipeline(self, hardware="cpu"):
        # Test code that works for all BERT models
        # ...
        
    def test_from_pretrained(self, hardware="cpu"):
        # Test code that works for all BERT models
        # ...
```

## Command-Line Usage

The test framework will support:

```bash
# Test specific model with specific class
python test_hf_bert.py --model bert-base-uncased

# Test all models of a specific class
python test_hf_bert.py --all-models

# Test all models on all available hardware
python test_hf_bert.py --all-models --all-hardware

# Test specific model and report detailed performance metrics
python test_hf_bert.py --model bert-base-uncased --performance
```

## Dependency Management

Dependencies will be tracked per class rather than per model:

```python
CLASS_DEPENDENCIES = {
    "BertForMaskedLM": ["tokenizers>=0.11.0", "sentencepiece"],
    "GPT2LMHeadModel": ["regex"],
    "LlamaForCausalLM": ["sentencepiece", "tokenizers>=0.13.3", "accelerate>=0.20.3"],
    # ...
}
```

Additional model-specific dependencies can be specified in a configuration file:

```json
{
    "meta-llama/Llama-2-7b-hf": {
        "additional_deps": ["bitsandbytes>=0.39.0"],
        "requires_remote_code": true
    }
}
```

## Test Discovery and Registration

To maintain a comprehensive catalog of testable models:

1. Create a registry of model class mappings
2. Use the Hugging Face API to find popular models for each class
3. Generate a comprehensive model database with class information
4. Allow tests to query compatible models for their class

## Testing Matrix

Instead of testing every model individually, we will:

1. Ensure each transformer class has test coverage
2. Select representative models from each class for regular testing
3. Validate new models only need to be tested if they introduce new architecture classes

## Reporting Structure

Reports will be organized by class rather than by model:

```
test_results/
├── by_class/
│   ├── bert_models.json
│   ├── gpt_models.json
│   └── t5_models.json
├── by_hardware/
│   ├── cpu_results.json
│   ├── cuda_results.json
│   └── openvino_results.json
└── summary.json
```

## Implementation Status

Current test implementation status:

| Model Class | Models Covered | Implementation |
|-------------|----------------|---------------|
| BertForMaskedLM | bert-base, distilbert | Completed |
| GPT2LMHeadModel | gpt2, gpt2-medium | Completed |
| LlamaForCausalLM | Llama-2-7b | In Progress |
| ViTForImageClassification | vit-base | Completed |
| WhisperForConditionalGeneration | whisper-tiny | Completed |
| SamModel | sam-vit-base | In Progress |
| T5ForConditionalGeneration | t5-small | Completed |
| LlavaForConditionalGeneration | llava-1.5-7b | In Progress |

## Next Steps

1. **Create Base Test Classes**: Develop base test classes for each transformer architecture
2. **Build Model Registry**: Create a comprehensive registry mapping models to classes
3. **Implement Test Runner**: Develop a unified test runner for all model classes
4. **Enhance Reporting**: Create consolidated reports showing coverage by class
5. **Optimize Dependencies**: Refine dependency management by architecture class

## Conclusion

By organizing tests by transformer class rather than individual models, we can significantly reduce code duplication, improve maintainability, and ensure consistent test coverage. This approach allows us to efficiently test hundreds of models with a small set of test files while maintaining high test quality.