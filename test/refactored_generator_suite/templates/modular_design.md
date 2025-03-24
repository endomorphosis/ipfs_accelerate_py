# Modular Template Design for IPFS Accelerate Python

This document outlines the modular template design for the refactored generator suite in IPFS Accelerate Python.

## Overview

The modular template design separates concerns into three main components:

1. **Hardware Templates**: Hardware-specific code for different backends (CPU, CUDA, OpenVINO, Apple MPS, QNN, ROCm)
2. **Architecture Templates**: Architecture-specific code for different model types (encoder-only, decoder-only, encoder-decoder, vision, speech, etc.)
3. **Pipeline Templates**: Pipeline-specific code for different processing types (text, image, audio, etc.)

These components are composed together by the `TemplateComposer` to generate complete model implementations.

## Template Interfaces

### Hardware Templates

The `BaseHardwareTemplate` abstract class defines the interface for hardware-specific templates:

```python
class BaseHardwareTemplate(ABC):
    @abstractmethod
    def get_import_statements(self) -> str: ...
    @abstractmethod
    def get_hardware_init_code(self, model_class_name: str, task_type: str) -> str: ...
    @abstractmethod
    def get_handler_creation_code(self, model_class_name: str, task_type: str) -> str: ...
    @abstractmethod
    def get_inference_code(self, task_type: str) -> str: ...
    @abstractmethod
    def get_cleanup_code(self) -> str: ...
    @abstractmethod
    def get_mock_code(self, model_class_name: str, task_type: str) -> str: ...
    def get_hardware_detection_code(self) -> str: ...
    def is_compatible_with_architecture(self, arch_type: str) -> bool: ...
    def get_fallback_hardware(self) -> str: ...
```

### Architecture Templates

The `BaseArchitectureTemplate` abstract class defines the interface for architecture-specific templates:

```python
class BaseArchitectureTemplate(ABC):
    @abstractmethod
    def get_model_class(self, task_type: str) -> str: ...
    @abstractmethod
    def get_processor_class(self, task_type: str) -> str: ...
    @abstractmethod
    def get_input_processing_code(self, task_type: str) -> str: ...
    @abstractmethod
    def get_output_processing_code(self, task_type: str) -> str: ...
    @abstractmethod
    def get_mock_processor_code(self) -> str: ...
    @abstractmethod
    def get_mock_output_code(self) -> str: ...
    def get_model_config(self, model_name: str) -> str: ...
```

### Pipeline Templates

The `BasePipelineTemplate` abstract class defines the interface for pipeline-specific templates:

```python
class BasePipelineTemplate(ABC):
    @abstractmethod
    def get_import_statements(self) -> str: ...
    @abstractmethod
    def get_preprocessing_code(self, task_type: str) -> str: ...
    @abstractmethod
    def get_postprocessing_code(self, task_type: str) -> str: ...
    @abstractmethod
    def get_result_formatting_code(self, task_type: str) -> str: ...
    @abstractmethod
    def get_mock_input_code(self) -> str: ...
    @abstractmethod
    def get_mock_output_code(self) -> str: ...
    def get_pipeline_utilities(self) -> str: ...
    def is_compatible_with_architecture(self, arch_type: str) -> bool: ...
    def is_compatible_with_task(self, task_type: str) -> bool: ...
```

## Template Composition

The `TemplateComposer` class combines hardware, architecture, and pipeline templates to generate complete model implementations:

```python
class TemplateComposer:
    def __init__(self, 
                 hardware_templates: Dict[str, BaseHardwareTemplate],
                 architecture_templates: Dict[str, BaseArchitectureTemplate],
                 pipeline_templates: Dict[str, BasePipelineTemplate],
                 output_dir: str): ...
                 
    def select_hardware_template(self, hardware_type: str) -> BaseHardwareTemplate: ...
    def select_architecture_template(self, arch_type: str) -> BaseArchitectureTemplate: ...
    def select_pipeline_template(self, pipeline_type: str) -> BasePipelineTemplate: ...
    
    def select_templates_for_model(self, 
                                  model_name: str, 
                                  arch_type: str,
                                  hardware_types: List[str]) -> Tuple[BaseArchitectureTemplate, 
                                                                     List[BaseHardwareTemplate], 
                                                                     BasePipelineTemplate]: ...
                                                                     
    def generate_model_implementation(self,
                                     model_name: str,
                                     arch_type: str,
                                     hardware_types: List[str],
                                     force: bool = False) -> Tuple[bool, str]: ...
```

## Supported Hardware Types

The modular template system supports the following hardware types:

1. **CPU**: Central Processing Unit (default fallback)
2. **CUDA**: NVIDIA GPU with CUDA
3. **ROCm**: AMD GPU with ROCm
4. **OpenVINO**: Intel hardware acceleration
5. **MPS**: Apple Metal Performance Shaders for Apple Silicon
6. **QNN**: Qualcomm Neural Network for Qualcomm processors

## Supported Architecture Types

The modular template system supports the following architecture types:

1. **encoder-only**: Encoder-only models like BERT
2. **decoder-only**: Decoder-only models like GPT
3. **encoder-decoder**: Encoder-decoder models like T5
4. **vision**: Vision models like ViT
5. **vision-encoder-text-decoder**: Vision-text models like CLIP
6. **speech**: Speech models like Whisper

## Supported Pipeline Types

The pipeline templates correspond directly to Hugging Face pipeline types:

1. **text-classification**: Text classification processing
2. **token-classification**: Token classification (NER, POS) processing
3. **question-answering**: Question answering processing
4. **text-generation**: Causal text generation processing
5. **text2text-generation**: Sequence-to-sequence generation processing
6. **summarization**: Text summarization processing
7. **translation**: Translation processing
8. **fill-mask**: Masked language modeling processing
9. **image-classification**: Image classification processing
10. **image-segmentation**: Image segmentation processing
11. **object-detection**: Object detection processing
12. **image-to-text**: Image captioning processing
13. **visual-question-answering**: Visual question answering processing
14. **audio-classification**: Audio classification processing
15. **automatic-speech-recognition**: Speech recognition processing
16. **text-to-speech**: Text to speech processing

## Generated Files

The generated files follow a specific naming convention:

1. Files are prefixed with "hf_" followed by the model type (e.g., "hf_bert.py")
2. The model type is derived from:
   - The model's config.json using AutoConfig if available
   - The model name if autodetection is not available
3. Files are placed in the `refactored_generator_suite/generated_reference/` directory

## Design Considerations

1. **Modularity**: Each component is independent and can be replaced without affecting others
2. **Extensibility**: New hardware, architecture, or pipeline types can be added easily
3. **Fallback**: Graceful degradation when hardware is not available
4. **Mock Support**: Mock implementations for testing and development
5. **Composition**: Different templates can be combined to create new implementations

## Future Enhancements

1. **Dynamic Template Selection**: Automatically select templates based on model capabilities
2. **Pipeline Optimization**: Optimize pipelines for specific hardware
3. **Quantization Support**: Add support for quantized models
4. **Custom Pipelines**: Support for custom/specialized pipelines
5. **Streaming Inference**: Support for streaming inference pipelines