# Generator System Refactoring Plan

## 1. Introduction

This document outlines a comprehensive plan for refactoring the test generator system, which is responsible for creating standardized test files for various HuggingFace model architectures. The current implementation has several strengths but also faces challenges with code organization, maintainability, and extensibility. The refactored system will address these challenges while preserving and enhancing the existing functionality.

## 2. Current System Analysis

### 2.1 System Overview

The current generator system is primarily implemented in `test_generator_fixed.py` and uses a set of architecture-specific templates to generate test files. The system supports various model types (encoder-only, decoder-only, vision, speech, multimodal, etc.) and includes features such as:

- Hardware detection for different devices (CUDA, ROCm, MPS, OpenVINO, WebNN, WebGPU)
- Mock detection for CI/CD environments
- Task-specific input generation
- Advanced model selection based on hardware, framework, and task requirements
- Template-based code generation with architecture-specific customizations
- Syntax validation and indentation fixing

### 2.2 Key Components

1. **Main Generator Script**: `test_generator_fixed.py` - Handles command-line arguments, template selection, and generation logic
2. **Templates**: Architecture-specific templates in the `templates/` directory
   - `encoder_only_template.py` - For BERT-like models
   - `decoder_only_template.py` - For GPT-like models
   - `encoder_decoder_template.py` - For T5-like models
   - `vision_template.py` - For vision models
   - `speech_template.py` - For speech models
   - `multimodal_template.py` - For multimodal models
3. **Model Selection**: Integration with `advanced_model_selection.py` and model registries
4. **Hardware Detection**: Built-in hardware detection with fallbacks
5. **Utility Functions**: For fixing indentation and syntax issues

### 2.3 Current Issues

1. **Code Duplication**: Significant code duplication across templates and between the generator and templates
2. **Inconsistent Architecture**: Lack of a clear separation between template rendering and model selection logic
3. **Limited Extensibility**: Difficulty adding new model types or templates
4. **Maintenance Complexity**: Changes often need to be made in multiple places
5. **Testing Challenges**: Lack of comprehensive test coverage for the generator system
6. **Limited Configuration**: Hard-coded configurations scattered throughout the codebase
7. **Debugging Difficulty**: Complex error tracing when templates fail to generate correctly
8. **Poor Separation of Concerns**: Business logic mixed with template rendering
9. **File Size Issues**: Very large files making navigation and modification difficult
10. **Ad-hoc Error Handling**: Inconsistent error handling across components

## 3. Refactoring Goals

### 3.1 Primary Objectives

1. **Unified Architecture**: Create a cohesive architecture with clear component responsibilities
2. **Reduced Duplication**: Eliminate code duplication through inheritance and composition
3. **Enhanced Extensibility**: Make adding new model types and templates straightforward
4. **Improved Testing**: Enable comprehensive testing of all components
5. **Better Configuration Management**: Centralize and standardize configuration
6. **Clearer Dependency Management**: Standardize dependency handling and mocking
7. **Enhanced Developer Experience**: Improve error reporting and debugging capabilities
8. **API Consistency**: Provide consistent interfaces across components
9. **Documentation**: Improve inline and external documentation
10. **Integration with CI/CD**: Support automated testing and validation

### 3.2 Secondary Objectives

1. **Performance Improvements**: Optimize template rendering and validation
2. **Reporting Enhancements**: Improve reporting capabilities for generated tests
3. **Visualization**: Add visualization for test coverage
4. **Integration with Benchmark System**: Connect with the benchmark system for performance testing
5. **Enhanced Template Features**: Add more sophisticated template capabilities
6. **Multi-Framework Support**: Better support for PyTorch, TensorFlow, ONNX, etc.

## 4. Core Architecture

### 4.1 Architectural Principles

1. **Component-Based Design**: Separate responsibilities into well-defined components
2. **Registry Pattern**: Use registries for discoverable components
3. **Template Method Pattern**: Define core algorithms and allow specialization
4. **Strategy Pattern**: Inject strategies for specialized behaviors
5. **Facade Pattern**: Provide simplified interfaces to complex subsystems
6. **Command Pattern**: Encapsulate operations as objects for better testability
7. **Factory Pattern**: Create objects without specifying exact classes

### 4.2 Core Components

#### 4.2.1 Generator Core

The `GeneratorCore` will be the central component responsible for orchestrating the generation process:

```python
class GeneratorCore:
    def __init__(self, config, registry):
        self.config = config
        self.registry = registry
        
    def generate(self, model_type, options):
        template = self.registry.get_template(model_type)
        model_info = self.registry.get_model_info(model_type)
        hardware_info = self.registry.get_hardware_info()
        
        context = self.build_context(model_type, model_info, hardware_info, options)
        result = template.render(context)
        
        return self.validate_and_fix(result)
```

#### 4.2.2 Component Registry

The `ComponentRegistry` will manage all discoverable components:

```python
class ComponentRegistry:
    def __init__(self):
        self.templates = {}
        self.model_info_providers = {}
        self.hardware_detectors = {}
        self.syntax_fixers = {}
        
    def register_template(self, architecture, template):
        self.templates[architecture] = template
        
    def get_template(self, model_type):
        architecture = self.map_model_to_architecture(model_type)
        return self.templates.get(architecture)
        
    # Additional registration and lookup methods
```

#### 4.2.3 Template Base

The `TemplateBase` will define the interface for all templates:

```python
class TemplateBase:
    def __init__(self, config):
        self.config = config
        
    def render(self, context):
        """Render the template with the provided context"""
        raise NotImplementedError
        
    def get_imports(self):
        """Get the imports required by this template"""
        raise NotImplementedError
        
    def get_metadata(self):
        """Get metadata about this template"""
        raise NotImplementedError
```

#### 4.2.4 Hardware Detection

The `HardwareDetector` will provide a unified interface for hardware detection:

```python
class HardwareDetector:
    def detect_all(self):
        """Detect all available hardware"""
        results = {}
        results.update(self.detect_cuda())
        results.update(self.detect_rocm())
        results.update(self.detect_mps())
        results.update(self.detect_openvino())
        results.update(self.detect_webnn())
        results.update(self.detect_webgpu())
        return results
        
    def detect_cuda(self):
        """Detect CUDA availability and properties"""
        # Implementation
        
    # Additional detection methods
```

#### 4.2.5 Model Selection

The `ModelSelector` will handle model selection based on various criteria:

```python
class ModelSelector:
    def __init__(self, model_registry):
        self.model_registry = model_registry
        
    def select_model(self, model_type, task=None, hardware=None, max_size=None, framework=None):
        """Select the best model based on the provided criteria"""
        candidates = self.get_candidates(model_type)
        candidates = self.filter_by_task(candidates, task)
        candidates = self.filter_by_hardware(candidates, hardware)
        candidates = self.filter_by_size(candidates, max_size)
        candidates = self.filter_by_framework(candidates, framework)
        
        if not candidates:
            return self.get_fallback(model_type)
            
        return self.rank_candidates(candidates)[0]
        
    # Additional selection methods
```

#### 4.2.6 Dependency Management

The `DependencyManager` will handle dependency checking and mocking:

```python
class DependencyManager:
    def __init__(self, mock_config=None):
        self.mock_config = mock_config or {}
        self.dependencies = self.check_dependencies()
        
    def check_dependencies(self):
        """Check all dependencies and return their status"""
        results = {}
        results['torch'] = self.check_torch()
        results['transformers'] = self.check_transformers()
        results['tokenizers'] = self.check_tokenizers()
        results['sentencepiece'] = self.check_sentencepiece()
        # Additional dependencies
        return results
        
    def check_torch(self):
        """Check torch availability"""
        if self.should_mock('torch'):
            return {'available': False, 'mocked': True}
            
        try:
            import torch
            return {
                'available': True, 
                'mocked': False,
                'version': torch.__version__,
                'cuda': torch.cuda.is_available()
            }
        except ImportError:
            return {'available': False, 'mocked': False}
            
    # Additional dependency checking methods
    
    def should_mock(self, dependency):
        """Check if a dependency should be mocked"""
        env_var = f"MOCK_{dependency.upper()}"
        return os.environ.get(env_var, 'False').lower() == 'true' or self.mock_config.get(dependency, False)
```

#### 4.2.7 Configuration Management

The `ConfigManager` will handle configuration loading and access:

```python
class ConfigManager:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path):
        """Load configuration from the specified path or default locations"""
        # Implementation
        
    def get(self, key, default=None):
        """Get a configuration value"""
        # Implementation
        
    def set(self, key, value):
        """Set a configuration value"""
        # Implementation
        
    def merge(self, additional_config):
        """Merge additional configuration"""
        # Implementation
```

#### 4.2.8 Syntax Validation and Fixing

The `SyntaxValidator` will handle syntax validation and fixing:

```python
class SyntaxValidator:
    def validate(self, content):
        """Validate the syntax of the generated content"""
        # Implementation
        
    def fix(self, content):
        """Fix syntax issues in the generated content"""
        return self.fix_indentation(self.fix_quotes(self.fix_imports(content)))
        
    def fix_indentation(self, content):
        """Fix indentation issues"""
        # Implementation
        
    def fix_quotes(self, content):
        """Fix quote issues"""
        # Implementation
        
    def fix_imports(self, content):
        """Fix import issues"""
        # Implementation
```

#### 4.2.9 Result Collection

The `ResultCollector` will handle collecting and formatting results:

```python
class ResultCollector:
    def __init__(self, config):
        self.config = config
        
    def collect(self, test_results):
        """Collect and format test results"""
        # Implementation
        
    def format_as_json(self, results):
        """Format results as JSON"""
        # Implementation
        
    def format_as_markdown(self, results):
        """Format results as Markdown"""
        # Implementation
        
    def save(self, results, path):
        """Save results to the specified path"""
        # Implementation
```

### 4.3 Project Structure

```
refactored_generator_suite/
├── README.md                   # Overview and usage documentation
├── generator_core/             # Core generator components
│   ├── __init__.py             # Package exports
│   ├── generator.py            # Main generator implementation
│   ├── registry.py             # Component registry
│   ├── config.py               # Configuration management
│   ├── cli.py                  # Command-line interface
│   └── utils.py                # Utility functions
├── templates/                  # Template implementations
│   ├── __init__.py             # Package exports
│   ├── base.py                 # Base template class
│   ├── encoder_only.py         # Encoder-only template
│   ├── decoder_only.py         # Decoder-only template
│   ├── encoder_decoder.py      # Encoder-decoder template
│   ├── vision.py               # Vision template
│   ├── speech.py               # Speech template
│   └── multimodal.py           # Multimodal template
├── model_selection/            # Model selection components
│   ├── __init__.py             # Package exports
│   ├── selector.py             # Model selector implementation
│   ├── registry.py             # Model registry
│   └── filters.py              # Selection filters
├── hardware/                   # Hardware detection components
│   ├── __init__.py             # Package exports
│   ├── detector.py             # Hardware detector implementation
│   └── configurator.py         # Hardware configuration
├── dependencies/               # Dependency management
│   ├── __init__.py             # Package exports
│   ├── manager.py              # Dependency manager
│   └── mocks.py                # Mock implementations
├── syntax/                     # Syntax validation and fixing
│   ├── __init__.py             # Package exports
│   ├── validator.py            # Syntax validator
│   ├── fixer.py                # Syntax fixer implementation
│   └── indentation.py          # Indentation fixing utilities
├── results/                    # Result collection and reporting
│   ├── __init__.py             # Package exports
│   ├── collector.py            # Result collector
│   ├── formatter.py            # Result formatter
│   └── reporter.py             # Result reporter
├── tests/                      # Test suite
│   ├── __init__.py             # Package exports
│   ├── test_generator.py       # Generator tests
│   ├── test_templates.py       # Template tests
│   ├── test_model_selection.py # Model selection tests
│   ├── test_hardware.py        # Hardware detection tests
│   ├── test_dependencies.py    # Dependency management tests
│   ├── test_syntax.py          # Syntax validation tests
│   └── fixtures/               # Test fixtures
├── examples/                   # Example implementations
│   ├── simple_generator.py     # Simple usage example
│   ├── advanced_generator.py   # Advanced usage example
│   ├── custom_template.py      # Custom template example
│   └── integration_example.py  # Integration example
├── configs/                    # Configuration files
│   ├── default.yaml            # Default configuration
│   ├── templates.yaml          # Template configurations
│   └── models.yaml             # Model configurations
├── scripts/                    # Utility scripts
│   ├── generate_test.py        # Test generation script
│   ├── validate_template.py    # Template validation script
│   └── benchmark_generator.py  # Generator benchmarking script
├── docs/                       # Documentation
│   ├── README.md               # Documentation overview
│   ├── USAGE.md                # Usage guide
│   ├── TEMPLATES.md            # Template development guide
│   ├── MODELS.md               # Model configuration guide
│   └── ARCHITECTURE.md         # Architectural overview
└── setup.py                    # Package setup script
```

## 5. Template System

### 5.1 Template Base Class

The new template system will use a base class that defines common functionality:

```python
class TemplateBase:
    def __init__(self, config):
        self.config = config
        
    def render(self, context):
        """Render the template with the provided context"""
        template_str = self.get_template_str()
        return self._render_template(template_str, context)
        
    def get_template_str(self):
        """Get the template string"""
        raise NotImplementedError
        
    def _render_template(self, template_str, context):
        """Render the template string with the provided context"""
        # Implementation using string.Template or similar
        
    def get_imports(self):
        """Get the imports required by this template"""
        return [
            "import os",
            "import sys",
            "import json",
            "import time",
            "import datetime",
            "import logging",
            "import argparse",
            "from unittest.mock import patch, MagicMock, Mock",
            "from typing import Dict, List, Any, Optional, Union",
            "from pathlib import Path"
        ]
        
    def get_metadata(self):
        """Get metadata about this template"""
        return {
            "name": self.__class__.__name__,
            "version": "1.0.0",
            "description": "Base template",
            "supported_architectures": []
        }
```

### 5.2 Template Implementation

Each architecture will have its own template implementation:

```python
class EncoderOnlyTemplate(TemplateBase):
    def get_template_str(self):
        """Get the encoder-only template string"""
        return """
import os
import sys
import json
import time
import datetime
import logging
import argparse
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Hardware detection
{{ hardware_detection }}

# Dependency management
{{ dependency_management }}

# Model registry
{{ model_registry }}

# Test class implementation
class {{ model_type|capitalize }}Test:
    def __init__(self, model_name, output_dir=None, device=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device or select_device()
        
    def run(self):
        """Run all tests for this model"""
        results = {}
        results["pipeline"] = self.test_pipeline()
        results["model"] = self.test_model()
        
        {% if has_openvino %}
        results["openvino"] = self.test_openvino()
        {% endif %}
        
        return results
        
    def test_pipeline(self):
        """Test the pipeline API"""
        # Implementation
        
    def test_model(self):
        """Test the model API"""
        # Implementation
        
    {% if has_openvino %}
    def test_openvino(self):
        """Test OpenVINO integration"""
        # Implementation
    {% endif %}

# Main function
def main():
    """Main function"""
    # Implementation

if __name__ == "__main__":
    main()
"""
        
    def get_metadata(self):
        """Get metadata about this template"""
        metadata = super().get_metadata()
        metadata.update({
            "name": "EncoderOnlyTemplate",
            "description": "Template for encoder-only models",
            "supported_architectures": ["encoder-only"],
            "supported_models": ["bert", "roberta", "distilbert", "electra", "camembert", "xlm-roberta", "deberta"]
        })
        return metadata
```

### 5.3 Template Context Building

The context for templates will be built from various sources:

```python
def build_context(model_type, model_info, hardware_info, options):
    """Build the context for template rendering"""
    context = {
        "model_type": model_type,
        "model_info": model_info,
        "hardware_info": hardware_info,
        "has_cuda": hardware_info.get("cuda", {}).get("available", False),
        "has_rocm": hardware_info.get("rocm", {}).get("available", False),
        "has_mps": hardware_info.get("mps", {}).get("available", False),
        "has_openvino": hardware_info.get("openvino", {}).get("available", False),
        "has_webnn": hardware_info.get("webnn", {}).get("available", False),
        "has_webgpu": hardware_info.get("webgpu", {}).get("available", False),
        "options": options,
        "timestamp": datetime.datetime.now().isoformat(),
        "uuid": str(uuid.uuid4())
    }
    
    # Add dependency information
    context.update(options.get("dependencies", {}))
    
    # Add template-specific context
    context.update(options.get("template_context", {}))
    
    return context
```

## 6. Model Selection System

### 6.1 Model Registry

The model registry will manage model metadata and selection:

```python
class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.architectures = {}
        
    def register_model(self, model_id, metadata):
        """Register a model with its metadata"""
        self.models[model_id] = metadata
        
        # Update architecture mapping
        architecture = metadata.get("architecture", "unknown")
        if architecture not in self.architectures:
            self.architectures[architecture] = []
        self.architectures[architecture].append(model_id)
        
    def get_model(self, model_id):
        """Get model metadata by ID"""
        return self.models.get(model_id)
        
    def get_models_by_architecture(self, architecture):
        """Get all models for a specific architecture"""
        return [self.models[model_id] for model_id in self.architectures.get(architecture, [])]
        
    def get_architectures(self):
        """Get all registered architectures"""
        return list(self.architectures.keys())
        
    def get_model_ids(self):
        """Get all registered model IDs"""
        return list(self.models.keys())
```

### 6.2 Model Selection Filters

The model selection system will use filters to narrow down candidates:

```python
class ModelFilter:
    def filter(self, models, criteria):
        """Filter models based on criteria"""
        raise NotImplementedError

class TaskFilter(ModelFilter):
    def filter(self, models, task):
        """Filter models based on task compatibility"""
        if not task:
            return models
            
        return [model for model in models if task in model.get("recommended_tasks", [])]

class HardwareFilter(ModelFilter):
    def filter(self, models, hardware_profile):
        """Filter models based on hardware compatibility"""
        if not hardware_profile:
            return models
            
        # Implement hardware-specific filtering logic
        
class SizeFilter(ModelFilter):
    def filter(self, models, max_size_mb):
        """Filter models based on size constraints"""
        if not max_size_mb:
            return models
            
        return [model for model in models if self._get_model_size_mb(model) <= max_size_mb]
        
    def _get_model_size_mb(self, model):
        """Extract model size in MB from model metadata"""
        # Implementation
        
class FrameworkFilter(ModelFilter):
    def filter(self, models, framework):
        """Filter models based on framework compatibility"""
        if not framework:
            return models
            
        return [model for model in models if framework in model.get("frameworks", [])]
```

### 6.3 Model Selector

The model selector will use the filters to select the best model:

```python
class ModelSelector:
    def __init__(self, registry):
        self.registry = registry
        self.filters = {
            "task": TaskFilter(),
            "hardware": HardwareFilter(),
            "size": SizeFilter(),
            "framework": FrameworkFilter()
        }
        
    def select(self, model_type, criteria=None):
        """Select the best model based on criteria"""
        criteria = criteria or {}
        
        # Get all models of this type
        architecture = self._map_model_to_architecture(model_type)
        candidates = self.registry.get_models_by_architecture(architecture)
        
        # Filter candidates based on criteria
        for filter_name, filter_value in criteria.items():
            if filter_name in self.filters and filter_value:
                candidates = self.filters[filter_name].filter(candidates, filter_value)
                
        # If no candidates remain, return a default
        if not candidates:
            return self._get_default_model(model_type)
            
        # Rank and return the best candidate
        return self._rank_candidates(candidates)[0]
        
    def _map_model_to_architecture(self, model_type):
        """Map a model type to its architecture"""
        # Implementation
        
    def _get_default_model(self, model_type):
        """Get a default model for the given type"""
        # Implementation
        
    def _rank_candidates(self, candidates):
        """Rank candidates by preference"""
        # Implementation
```

## 7. Hardware Detection System

### 7.1 Hardware Detector Base

The hardware detection system will use a base class for all detectors:

```python
class HardwareDetectorBase:
    def detect(self):
        """Detect hardware availability and properties"""
        raise NotImplementedError
        
    def get_metadata(self):
        """Get metadata about this detector"""
        return {
            "name": self.__class__.__name__,
            "version": "1.0.0",
            "description": "Base hardware detector"
        }
```

### 7.2 Specific Hardware Detectors

Each hardware type will have its own detector:

```python
class CUDADetector(HardwareDetectorBase):
    def detect(self):
        """Detect CUDA availability and properties"""
        result = {"available": False, "version": None, "devices": []}
        
        try:
            import torch
            
            result["available"] = torch.cuda.is_available()
            
            if result["available"]:
                result["version"] = torch.version.cuda
                result["device_count"] = torch.cuda.device_count()
                
                # Get device properties
                devices = []
                for i in range(result["device_count"]):
                    props = torch.cuda.get_device_properties(i)
                    devices.append({
                        "index": i,
                        "name": props.name,
                        "total_memory": props.total_memory,
                        "major": props.major,
                        "minor": props.minor
                    })
                    
                result["devices"] = devices
        except ImportError:
            pass
            
        return result
        
    def get_metadata(self):
        """Get metadata about this detector"""
        metadata = super().get_metadata()
        metadata.update({
            "name": "CUDADetector",
            "description": "Detector for NVIDIA CUDA GPUs"
        })
        return metadata
```

### 7.3 Hardware Detection Manager

The hardware detection manager will coordinate all detectors:

```python
class HardwareDetectionManager:
    def __init__(self):
        self.detectors = {}
        
    def register_detector(self, hardware_type, detector):
        """Register a hardware detector"""
        self.detectors[hardware_type] = detector
        
    def detect_all(self):
        """Detect all hardware"""
        results = {}
        
        for hardware_type, detector in self.detectors.items():
            try:
                results[hardware_type] = detector.detect()
            except Exception as e:
                results[hardware_type] = {
                    "available": False,
                    "error": str(e)
                }
                
        return results
        
    def detect(self, hardware_type):
        """Detect specific hardware"""
        if hardware_type not in self.detectors:
            return {"available": False, "error": f"No detector for {hardware_type}"}
            
        try:
            return self.detectors[hardware_type].detect()
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
```

## 8. Dependency Management System

### 8.1 Dependency Manager

The dependency manager will handle checking and mocking dependencies:

```python
class DependencyManager:
    def __init__(self, mock_config=None):
        self.mock_config = mock_config or {}
        self.cached_results = {}
        
    def check(self, dependency_name):
        """Check a dependency's availability"""
        if dependency_name in self.cached_results:
            return self.cached_results[dependency_name]
            
        checker_method = getattr(self, f"check_{dependency_name}", None)
        if not checker_method:
            result = {"available": False, "error": f"No checker for {dependency_name}"}
        else:
            try:
                result = checker_method()
            except Exception as e:
                result = {"available": False, "error": str(e)}
                
        self.cached_results[dependency_name] = result
        return result
        
    def check_all(self):
        """Check all dependencies"""
        results = {}
        for dependency_name in self.get_all_dependency_names():
            results[dependency_name] = self.check(dependency_name)
        return results
        
    def get_all_dependency_names(self):
        """Get all dependency names"""
        methods = dir(self)
        return [m[6:] for m in methods if m.startswith("check_") and callable(getattr(self, m))]
        
    def should_mock(self, dependency_name):
        """Check if a dependency should be mocked"""
        env_var = f"MOCK_{dependency_name.upper()}"
        return os.environ.get(env_var, "False").lower() == "true" or self.mock_config.get(dependency_name, False)
        
    def check_torch(self):
        """Check torch availability"""
        if self.should_mock("torch"):
            return {"available": False, "mocked": True}
            
        try:
            import torch
            return {
                "available": True,
                "mocked": False,
                "version": torch.__version__,
                "cuda": torch.cuda.is_available() if hasattr(torch, "cuda") else False
            }
        except ImportError:
            return {"available": False, "mocked": False}
        
    # Additional dependency checkers (transformers, tokenizers, sentencepiece, etc.)
```

### 8.2 Mock Providers

The mock providers will create standardized mocks for dependencies:

```python
class MockProvider:
    def get_mock(self):
        """Get a mock for this dependency"""
        raise NotImplementedError
        
class TorchMockProvider(MockProvider):
    def get_mock(self):
        """Get a mock for torch"""
        mock = MagicMock()
        
        # Configure common torch attributes and methods
        mock.__version__ = "MOCK"
        
        # Mock CUDA
        mock.cuda = MagicMock()
        mock.cuda.is_available = lambda: False
        mock.cuda.device_count = lambda: 0
        
        # Mock device handling
        mock.device = lambda device_str: device_str
        
        # Mock tensor creation
        def mock_tensor(data, *args, **kwargs):
            result = MagicMock()
            result.shape = getattr(data, "shape", None)
            result.dtype = kwargs.get("dtype", None)
            result.device = kwargs.get("device", "cpu")
            return result
            
        mock.tensor = mock_tensor
        
        return mock
        
class TransformersMockProvider(MockProvider):
    def get_mock(self):
        """Get a mock for transformers"""
        mock = MagicMock()
        
        # Configure common transformers attributes and methods
        mock.__version__ = "MOCK"
        
        # Mock pipeline
        def mock_pipeline(task, model=None, **kwargs):
            pipeline_mock = MagicMock()
            pipeline_mock.return_value = [{"generated_text": "Mock generated text"}]
            return pipeline_mock
            
        mock.pipeline = mock_pipeline
        
        # Mock model loading
        model_mock = MagicMock()
        mock.AutoModelForSequenceClassification.from_pretrained = lambda *args, **kwargs: model_mock
        mock.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: model_mock
        mock.AutoModelForMaskedLM.from_pretrained = lambda *args, **kwargs: model_mock
        
        # Mock tokenizer loading
        tokenizer_mock = MagicMock()
        tokenizer_mock.encode = lambda text, **kwargs: [101, 102, 103]
        tokenizer_mock.decode = lambda ids, **kwargs: "Decoded text"
        mock.AutoTokenizer.from_pretrained = lambda *args, **kwargs: tokenizer_mock
        
        return mock
```

## 9. Syntax Validation and Fixing System

### 9.1 Syntax Validator

The syntax validator will check syntax correctness:

```python
class SyntaxValidator:
    def validate(self, content):
        """Validate the syntax of generated content"""
        try:
            # Try to compile the code
            compile(content, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, {
                "line": e.lineno,
                "offset": e.offset,
                "text": e.text,
                "message": str(e)
            }
```

### 9.2 Syntax Fixer

The syntax fixer will fix common syntax issues:

```python
class SyntaxFixer:
    def __init__(self):
        self.fixers = [
            self.fix_quotes,
            self.fix_indentation,
            self.fix_unbalanced_delimiters,
            self.fix_dangling_commas,
            self.fix_unbalanced_blocks
        ]
        
    def fix(self, content):
        """Apply all fixers to the content"""
        for fixer in self.fixers:
            content = fixer(content)
        return content
        
    def fix_quotes(self, content):
        """Fix quote-related issues"""
        # Replace multiple consecutive quotes
        content = content.replace('""""', '"""')
        content = content.replace("''''", "'''")
        
        # Fix unterminated triple quotes
        content = self._fix_unterminated_triple_quotes(content)
        
        return content
        
    def fix_indentation(self, content):
        """Fix indentation issues"""
        lines = content.split('\n')
        fixed_lines = []
        
        # Track indentation level
        current_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines or comment-only lines
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue
                
            # Adjust indentation for blocks that increase level
            if stripped.endswith(':'):
                fixed_lines.append(' ' * (4 * current_level) + stripped)
                current_level += 1
                continue
                
            # Handle dedentation
            if stripped in ('else:', 'elif', 'except:', 'finally:', 'except', 'elif:'):
                current_level = max(0, current_level - 1)
                fixed_lines.append(' ' * (4 * current_level) + stripped)
                current_level += 1
                continue
                
            # Normal line
            fixed_lines.append(' ' * (4 * current_level) + stripped)
            
        return '\n'.join(fixed_lines)
        
    # Additional fixing methods
```

### 9.3 Advanced Indentation Fixer

The advanced indentation fixer will use AST parsing for more accurate fixing:

```python
class ASTIndentationFixer:
    def fix(self, content):
        """Fix indentation using AST parsing"""
        try:
            import ast
            
            # Try to parse the AST
            tree = ast.parse(content)
            
            # Generate properly formatted code
            import astor
            fixed_content = astor.to_source(tree)
            
            return fixed_content
        except SyntaxError:
            # Fall back to regex-based fixer if AST parsing fails
            return self._fix_with_regex(content)
            
    def _fix_with_regex(self, content):
        """Fix indentation using regex patterns"""
        # Implementation
```

## 10. Implementation Plan

### 10.1 Phase 1: Core Architecture

1. Create project structure
2. Implement `GeneratorCore` and `ComponentRegistry`
3. Implement `TemplateBase` and a simple template
4. Implement basic command-line interface
5. Create configuration management system
6. Implement basic hardware detection
7. Create simple test generation with minimal dependencies

### 10.2 Phase 2: Template System

1. Implement architecture-specific templates
2. Add template context building
3. Create template validator
4. Implement template inheritance and composition
5. Add template rendering system
6. Create template testing framework
7. Migrate existing templates to new system

### 10.3 Phase 3: Model Selection System

1. Implement model registry
2. Create model selection filters
3. Implement model selector
4. Add task-specific model selection
5. Implement hardware-aware model selection
6. Create framework-specific model selection
7. Add fallback mechanisms

### 10.4 Phase 4: Hardware Detection System

1. Implement hardware detector base
2. Create hardware-specific detectors (CUDA, ROCm, MPS, etc.)
3. Implement hardware detection manager
4. Add advanced hardware property detection
5. Create hardware configuration system
6. Implement hardware-specific optimizations
7. Add hardware detection tests

### 10.5 Phase 5: Dependency Management System

1. Implement dependency manager
2. Create mock providers
3. Add dependency checking and mocking
4. Implement environment variable controls
5. Create dependency testing framework
6. Add advanced dependency handling
7. Implement mock validation

### 10.6 Phase 6: Syntax Validation and Fixing System

1. Implement syntax validator
2. Create basic syntax fixer
3. Add advanced indentation fixer
4. Implement syntax testing framework
5. Create AST-based fixing
6. Add validator and fixer integration
7. Create syntax fixing commands

### 10.7 Phase 7: Integration and Testing

1. Integrate all components
2. Create comprehensive test suite
3. Implement documentation system
4. Add examples
5. Create benchmarking system
6. Add CI/CD integration
7. Create migration tools for existing code

### 10.8 Phase 8: Extension and Optimization

1. Add performance optimizations
2. Create plugin system
3. Implement visualization tools
4. Add advanced reporting
5. Create dashboard integration
6. Implement advanced configuration
7. Add extensibility features

## 11. Migration Strategy

### 11.1 Steps for Migration

1. Create the new architecture alongside the existing system
2. Implement core functionality in the new system
3. Create compatibility layers for existing code
4. Gradually migrate functionality from old to new system
5. Add tests to ensure no regressions
6. Update documentation and examples
7. Train team on new system
8. Switch over to new system completely
9. Remove compatibility layers

### 11.2 Compatibility Considerations

1. Maintain same command-line interface for backward compatibility
2. Ensure output format remains compatible with existing tools
3. Support existing template formats during transition
4. Provide migration guides for custom template developers
5. Add compatibility checks for integrated systems

## 12. Testing Strategy

### 12.1 Unit Testing

1. Test each component in isolation
2. Use mock objects for dependencies
3. Verify edge cases and error handling
4. Ensure high test coverage
5. Add tests for configuration variations

### 12.2 Integration Testing

1. Test component interactions
2. Verify end-to-end functionality
3. Test with real hardware when available
4. Check compatibility with external systems
5. Verify performance characteristics

### 12.3 Regression Testing

1. Ensure no functionality is lost during migration
2. Verify output compatibility
3. Check for performance regressions
4. Test with existing client code
5. Verify documentation accuracy

## 13. Documentation

### 13.1 Architecture Documentation

1. Create overall architecture diagrams
2. Document component interactions
3. Explain design decisions
4. Provide extension points
5. Document configuration options

### 13.2 User Documentation

1. Create usage guides
2. Document command-line options
3. Provide examples
4. Create troubleshooting guide
5. Add FAQ

### 13.3 Developer Documentation

1. Document extension points
2. Create template development guide
3. Explain code organization
4. Document best practices
5. Add contribution guide

## 14. Conclusion

This comprehensive refactoring plan outlines a path to transform the existing generator system into a more maintainable, extensible, and robust system. By focusing on clear component responsibilities, reduced duplication, and improved testing, the refactored system will provide a solid foundation for future enhancements while preserving the valuable functionality of the current system.

The architecture proposed here emphasizes separation of concerns, clear interfaces, and extensibility, enabling easier maintenance and future additions. By implementing this plan in phases, we can ensure a smooth transition while minimizing disruption to existing workflows.