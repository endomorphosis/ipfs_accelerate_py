# Template System Guide for End-to-End Testing Framework

This guide explains the template database and rendering system that powers the End-to-End Testing Framework. The template system provides a flexible, efficient way to generate model skills, tests, benchmarks, and documentation from a centralized database of templates.

## Overview

The template system consists of two main components:

1. **Template Database**: A DuckDB-based database for storing and managing templates
2. **Template Renderer**: A rendering engine that applies variables to templates and generates output files

This system enables the End-to-End Testing Framework to focus on fixing generators rather than individual files, making maintenance more efficient and consistent across the entire test suite.

### Key Benefits

- **Consistency**: Ensures components are generated using standardized templates
- **Maintainability**: Centralizes template management in a database instead of scattered files
- **Flexibility**: Supports template inheritance and hardware-specific template variations
- **Efficiency**: Automates the generation of multiple components from a single source
- **Version Control**: Tracks template versions for accountability and versioning
- **Reproducibility**: Makes test generation reproducible and deterministic

## Template Database

The Template Database stores templates, model mappings, and hardware compatibility information in a DuckDB database. It provides:

- **Template Storage**: Centralized storage for all templates with versioning and inheritance
- **Model Family Mapping**: Automatic mapping of model names to model families
- **Hardware Compatibility**: Information about hardware compatibility for different model families
- **Variable Extraction**: Automatic extraction of variables from templates
- **Dependency Management**: Tracking dependencies between templates for inheritance and inclusion
- **Model-Family Inference**: Automatic detection of model families from model names
- **Versioning Support**: Tracking template versions over time for evolving standards
- **Active/Inactive Tracking**: Ability to deprecate templates without deleting them

### Database Schema

The Template Database uses the following tables:

- **templates**: Stores template content with metadata
  - `template_id`: Unique identifier for the template
  - `template_name`: Human-readable name
  - `template_type`: Type (skill, test, benchmark, documentation, helper)
  - `model_family`: Model family the template is for
  - `hardware_platform`: Hardware platform (NULL for platform-agnostic templates)
  - `template_content`: The actual template text with variable placeholders
  - `description`: Human-readable description
  - `version`: Version string (e.g., "1.0.0")
  - `parent_template_id`: Reference to parent template for inheritance
  - `created_at`: Creation timestamp
  - `updated_at`: Last update timestamp
  - `is_active`: Whether the template is active or deprecated

- **template_variables**: Tracks variables used in templates
  - `variable_id`: Unique identifier for the variable
  - `template_id`: Reference to the template
  - `variable_name`: Name of the variable (without ${} syntax)
  - `variable_description`: Human-readable description
  - `default_value`: Optional default value
  - `is_required`: Whether the variable is required
  - `created_at`: Creation timestamp

- **model_mappings**: Maps model names to model families
  - `mapping_id`: Unique identifier for the mapping
  - `model_name`: Name of the model (e.g., "bert-base-uncased")
  - `model_family`: Model family (e.g., "text_embedding")
  - `description`: Human-readable description
  - `created_at`: Creation timestamp
  - `updated_at`: Last update timestamp

- **template_dependencies**: Tracks dependencies between templates
  - `dependency_id`: Unique identifier for the dependency
  - `template_id`: Reference to the dependent template
  - `depends_on_template_id`: Reference to the dependency template
  - `dependency_type`: Type of dependency (include, extend)
  - `created_at`: Creation timestamp

- **hardware_compatibility**: Defines hardware compatibility for model families
  - `compatibility_id`: Unique identifier for the compatibility entry
  - `model_family`: Model family
  - `hardware_platform`: Hardware platform
  - `compatibility_level`: Compatibility level (full, limited, none)
  - `description`: Human-readable description
  - `created_at`: Creation timestamp
  - `updated_at`: Last update timestamp

### Template Types

The system supports the following template types:

- **skill**: Templates for model skill implementations
- **test**: Templates for model tests
- **benchmark**: Templates for model benchmarks
- **documentation**: Templates for model documentation
- **helper**: Helper templates that can be included in other templates

### Model Families

Templates are organized by model family:

- **text_embedding**: Text embedding models (BERT, RoBERTa, etc.)
- **text_generation**: Text generation models (LLaMA, T5, GPT, etc.)
- **vision**: Vision models (ViT, ResNet, etc.)
- **audio**: Audio models (Whisper, Wav2Vec2, etc.)
- **multimodal**: Multimodal models (CLIP, LLaVA, etc.)
- **audio_classification**: Audio classification models
- **vision_classification**: Vision classification models
- **object_detection**: Object detection models (DETR, YOLO, etc.)

### Hardware Platforms

Templates can be hardware-specific for the following platforms:

- **cpu**: CPU implementation
- **cuda**: NVIDIA CUDA implementation
- **rocm**: AMD ROCm implementation
- **mps**: Apple Metal Performance Shaders implementation
- **openvino**: Intel OpenVINO implementation
- **qnn**: Qualcomm Neural Networks implementation
- **webnn**: WebNN implementation
- **webgpu**: WebGPU implementation

## Template Renderer

The Template Renderer applies variables to templates and generates output files. It provides:

- **Template Rendering**: Applies variables to templates and renders them to files
- **Component Generation**: Generates complete component sets (skill, test, benchmark, documentation)
- **Variable Management**: Adds model-family-specific and hardware-specific variables
- **Hardware Compatibility**: Checks hardware compatibility for models
- **Metadata Header Generation**: Adds generation metadata to output files
- **Smart Variable Substitution**: Supports transformations like `${model_name.replace('-', '_')}` 
- **Batch Rendering**: Render multiple components at once
- **Inheritance Resolution**: Resolves template inheritance chains
- **Dependency Handling**: Processes includes and other dependencies between templates

### Model-Family Specific Variables

The renderer automatically adds the following variables based on model family:

| Model Family | Added Variables | 
|--------------|-----------------|
| text_embedding | `input_type: "text"`, `output_type: "embedding"`, `typical_sequence_length: 128`, `typical_output_dims: 768` |
| text_generation | `input_type: "text"`, `output_type: "text"`, `typical_sequence_length: 1024` |
| vision | `input_type: "image"`, `output_type: "embedding"`, `typical_output_dims: 768` |
| audio | `input_type: "audio"`, `output_type: "text"` |
| multimodal | `input_type: "multiple"`, `output_type: "multiple"` |

### Hardware-Specific Variables

The renderer automatically adds the following variables based on hardware platform:

| Hardware | Added Variables |
|----------|----------------|
| cpu | `hardware_specific_optimizations`, `memory_management: "host_memory"`, `precision: "float32"`, `threading_model: "parallel"` |
| cuda | `hardware_specific_optimizations`, `memory_management: "device_memory"`, `precision: "float16"`, `threading_model: "cuda_streams"` |
| webgpu | `hardware_specific_optimizations`, `memory_management: "device_memory"`, `precision: "float16"`, `threading_model: "browser_worker"` |

### Rendering Process

The rendering process involves the following steps:

1. **Model Family Detection**: Determine the model family based on the model name
2. **Template Selection**: Select the appropriate template based on model family, template type, and hardware platform
3. **Variable Collection**: Collect variables from model family, hardware platform, and custom variables
4. **Template Rendering**: Apply variables to the template
5. **Output Generation**: Write the rendered template to a file

## Using the Template System

### Initializing the Template Database

Initialize the template database with default templates:

```python
from template_database import add_default_templates
add_default_templates("./template_database.duckdb")
```

### Adding Templates

Add new templates to the database:

```python
from template_database import TemplateDatabase

# Create database instance
db = TemplateDatabase("./template_database.duckdb")

# Add template
template_id = db.add_template(
    template_name="text_embedding_skill",
    template_type="skill",
    model_family="text_embedding",
    template_content="...",  # Template content with ${variable} placeholders
    description="Basic template for text embedding model skills"
)
```

### Rendering Templates

Render templates for specific models and hardware platforms:

```python
from template_renderer import TemplateRenderer

# Create renderer
renderer = TemplateRenderer(db_path="./template_database.duckdb")

# Render a template
rendered_content = renderer.render_template(
    model_name="bert-base-uncased",
    template_type="skill",
    hardware_platform="cuda",
    variables={"batch_size": 4}
)

# Render a complete component set
generated_files = renderer.render_component_set(
    model_name="bert-base-uncased",
    hardware_platform="cuda",
    output_dir="./generated"
)
```

### Command-Line Interface

The template system includes command-line interfaces for managing and rendering templates:

#### Template Database CLI

```bash
# Initialize database with default templates
python template_database.py --init

# List templates in the database
python template_database.py --list

# Add a template to the database
python template_database.py --add-template --template-name "my_template" --template-type "skill" --model-family "text_embedding" --template-file "./my_template.py"

# Generate template files
python template_database.py --generate --model "bert-base-uncased" --output-dir "./generated"
```

#### Template Renderer CLI

```bash
# Initialize database with default templates
python template_renderer.py --initialize-db

# List compatible hardware platforms for a model
python template_renderer.py --model "bert-base-uncased" --list-compatible-hardware

# Render a specific template type
python template_renderer.py --model "bert-base-uncased" --hardware "cuda" --template-type "skill" --output-dir "./generated"

# Render all template types
python template_renderer.py --model "bert-base-uncased" --hardware "cuda" --output-dir "./generated"
```

## Integration with End-to-End Testing Framework

The template system is integrated with the End-to-End Testing Framework through the `IntegratedComponentTester.generate_components()` method. This method:

1. Initializes the template database if it doesn't exist
2. Creates a template renderer
3. Renders a complete component set for the specified model and hardware platform
4. Validates the generated files
5. Falls back to legacy template methods if the template system fails

### Example Usage

```python
# Create an integrated component tester
tester = IntegratedComponentTester(
    model_name="bert-base-uncased",
    hardware="cuda",
    template_db_path="./template_database.duckdb"
)

# Generate components
skill_file, test_file, benchmark_file = tester.generate_components(temp_dir="./temp")

# Run tests and benchmarks
test_success, test_results = tester.run_test(skill_file, test_file)
benchmark_success, benchmark_results = tester.run_benchmark(benchmark_file)

# Save results
results_dir = tester.save_results(
    test_success, test_results,
    benchmark_success, benchmark_results,
    (skill_file, test_file, benchmark_file)
)
```

## Template Specification

Templates use a simple variable substitution syntax with variables enclosed in `${...}`:

```python
#!/usr/bin/env python3
'''
Skill implementation for ${model_name} on ${hardware_type} hardware.
'''

import torch
import numpy as np

class ${model_name.replace('-', '_').replace('/', '_')}Skill:
    '''
    Model skill for ${model_name} on ${hardware_type} hardware.
    Model type: ${model_family}
    '''
    
    def __init__(self):
        self.model_name = "${model_name}"
        self.hardware = "${hardware_type}"
        self.model_type = "${model_family}"
        # ...
```

The system provides the following built-in variables:

- `${model_name}`: Name of the model
- `${model_family}`: Model family (text_embedding, vision, etc.)
- `${hardware_type}`: Hardware platform (cpu, cuda, etc.)
- `${batch_size}`: Default batch size for testing
- `${test_id}`: Unique test ID
- `${timestamp}`: Current timestamp

Additional variables are provided based on model family and hardware platform.

## Extending the Template System

### Adding New Model Families

To add a new model family:

1. Add the family to `MODEL_FAMILIES` in the template database
2. Add model mappings for the new family
3. Add hardware compatibility information for the new family
4. Create templates for the new family
5. Add family-specific variables to the `_add_model_family_variables` method in the renderer
6. Update the model family inference logic in `_infer_model_family` if needed

**Example**: Adding a "reinforcement_learning" family:

```python
# Add to MODEL_FAMILIES
MODEL_FAMILIES = [
    "text_embedding", "text_generation", "vision", "audio", "multimodal",
    "audio_classification", "vision_classification", "object_detection", 
    "reinforcement_learning"  # New family
]

# Update model family inference
def _infer_model_family(self, model_name: str) -> Optional[str]:
    model_name = model_name.lower()
    
    # Existing checks...
    
    # Reinforcement learning models
    if any(x in model_name for x in ["rl-", "ppo-", "dqn-", "a2c-"]):
        return "reinforcement_learning"
        
    return None

# Add model-family variables
def _add_model_family_variables(self, model_family: str, variables: Dict[str, Any]) -> None:
    # Existing code...
    
    # Reinforcement learning models
    elif model_family == "reinforcement_learning":
        variables.update({
            "input_type": "environment",
            "output_type": "action",
            "typical_sequence_length": None,
            "typical_output_dims": None,
            "common_use_case": "training agents, decision making"
        })
```

### Adding New Hardware Platforms

To add a new hardware platform:

1. Add the platform to `HARDWARE_PLATFORMS` in the template database
2. Add hardware compatibility information for the platform
3. Add hardware-specific variables in the renderer's `_add_hardware_specific_variables` method
4. Create hardware-specific templates

**Example**: Adding a "jetson" platform for NVIDIA Jetson devices:

```python
# Add to HARDWARE_PLATFORMS
HARDWARE_PLATFORMS = [
    "cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu", 
    "jetson"  # New platform
]

# Add hardware-specific variables
def _add_hardware_specific_variables(self, hardware_platform: str, variables: Dict[str, Any]) -> None:
    # Existing code...
    
    # Jetson-specific variables
    elif hardware_platform == "jetson":
        variables.update({
            "hardware_specific_optimizations": "- TensorRT optimizations\n- Power-aware scheduling\n- Small batch processing",
            "memory_management": "unified_memory",
            "precision": "mixed_precision",
            "threading_model": "tensorrt_streams",
            "initialization_code": "import torch\nimport tensorrt as trt\ndevice = 'cuda'"
        })
```

### Adding Template Inheritance

Templates can inherit from parent templates:

1. Create a base template
2. Create a child template with a parent_template_id
3. Use the renderer to apply inheritance during rendering

**Example**: Creating a specialized BERT template that inherits from the text_embedding template:

```python
# First, get the parent template ID
parent_template = db.get_template(
    model_family="text_embedding",
    template_type="skill"
)
parent_id = parent_template["template_id"]

# Create the specialized child template
template_id = db.add_template(
    template_name="bert_specialized_skill",
    template_type="skill",
    model_family="text_embedding",
    template_content="""
# This template extends the base text_embedding template
# and adds BERT-specific functionality

def get_bert_attention_weights(self):
    """Get attention weights from BERT model."""
    return self.model.encoder.layer[-1].attention.self.get_attention_matrix()
""",
    parent_template_id=parent_id
)
```

### Adding Custom Variable Transformations

Templates support various transformations on variables:

1. Simple transformation: `${variable.method()}`
2. Method chaining: `${variable.method1().method2()}`
3. Regular expression transformations: `${variable.replace('-', '_')}`
4. Conditional transformations: `${variable or 'default'}`
5. String formatting: `${f"Model-{variable}"}`

When adding new variables, consider what transformations users might need and document them in comments.

## Best Practices

### Template Organization

- Organize templates by model family and hardware platform
- Use inheritance for common functionality
- Keep templates modular and reusable

### Variable Naming

- Use clear, descriptive variable names
- Follow a consistent naming convention
- Document variables with comments

### Testing

- Test templates with different model families and hardware platforms
- Verify that rendered files are syntactically correct
- Validate generated files with the model validator

## Troubleshooting

### Common Issues

- **Template not found**: Check model family mapping and template existence
- **Variable not replaced**: Verify variable name and syntax
- **Validation fails**: Check template compatibility with model family and hardware
- **DuckDB import error**: Install DuckDB with `pip install duckdb`
- **Parent template error**: Ensure parent template exists before creating children
- **Hardware compatibility error**: Verify hardware compatibility for the model family
- **Template validation error**: Ensure template syntax is correct and all required variables have values
- **Variable transformation error**: Check that variable transformations are valid Python code

### Error Messages and Solutions

| Error Message | Possible Cause | Solution |
|--------------|----------------|----------|
| `No template found for {model_family} {template_type} on {hardware_platform}` | Missing template for the specified combination | Add a template for the model family and hardware platform, or use a generic template |
| `Could not determine model family for {model_name}` | Model name not mapped to a family | Add a model mapping or update family inference logic |
| `Invalid template type: {template_type}` | Template type not in allowed list | Use one of the allowed template types |
| `Invalid model family: {model_family}` | Model family not in allowed list | Add the model family to the MODEL_FAMILIES list |
| `Invalid hardware platform: {hardware_platform}` | Hardware platform not in allowed list | Add the hardware platform to the HARDWARE_PLATFORMS list |
| `Template contains unreplaced variables: {missing_vars}` | Some variables weren't replaced during rendering | Provide values for all required variables |
| `DuckDB is required for template database functionality` | DuckDB package not installed | Install DuckDB with `pip install duckdb` |

### Debugging

- Enable verbose logging with `verbose=True` or `--verbose` flag
- Use the command-line interface to list templates and verify they exist
- Check database initialization with `python template_database.py --init --verbose`
- Render templates with explicit variables to identify missing ones
- Test with simpler templates first, then add complexity
- Verify hardware compatibility with `template_renderer.py --model "model-name" --list-compatible-hardware`
- Check generated files manually for syntax errors or missing variables
- Use `--keep-temp` flag with the integrated component tester to preserve temporary files for inspection
- Run validation directly with `model_validator.validate_skill(file_path)` to see detailed validation errors

### Advanced Troubleshooting

If you're still having issues, try these steps:

1. **Direct database inspection**:
   ```python
   import duckdb
   conn = duckdb.connect('path/to/template_database.duckdb')
   print(conn.execute("SELECT * FROM templates").fetchall())
   ```

2. **Manual template rendering**:
   ```python
   from template_database import TemplateDatabase
   db = TemplateDatabase('path/to/template_database.duckdb')
   template = db.get_template(model_family="text_embedding", template_type="skill")
   print(template["template_content"])
   ```

3. **Test variable substitution manually**:
   ```python
   content = template["template_content"]
   variables = {"model_name": "bert-base-uncased", "hardware_type": "cpu"}
   for var_name, var_value in variables.items():
       content = content.replace(f"${{{var_name}}}", str(var_value))
   print(content)
   ```

4. **Check database integrity**:
   ```python
   from template_database import TemplateDatabase
   db = TemplateDatabase('path/to/template_database.duckdb')
   db.list_templates()  # Should return list of templates
   ```

## Examples of Generated Components

### Example Skill Component

```python
#!/usr/bin/env python3
# Generated by TemplateRenderer on 2025-03-16 10:15:22
# Model: bert-base-uncased
# Template: text_embedding_skill (f8a7e0c2-9b14-4d23-8fe1-a3b4c5d6e7f8)
# Hardware: cuda
# Type: skill

"""
Skill implementation for bert-base-uncased on cuda hardware.
Generated by integrated component test runner.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Union

class BertBaseUncasedSkill:
    """
    Model skill for bert-base-uncased on cuda hardware.
    Model type: text_embedding
    Input type: text
    Output type: embedding
    
    This skill provides text embedding functionality using the bert-base-uncased model.
    """
    
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.hardware = "cuda"
        self.model_type = "text_embedding"
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def setup(self, **kwargs) -> bool:
        """Set up the model and tokenizer."""
        try:
            # CUDA setup logic
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = device
            
            # Load model and tokenizer
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(device)
            self.model.eval()
            
            return True
        except Exception as e:
            print(f"Error setting up model: {e}")
            return False
    
    def run(self, inputs: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Run the model on inputs.
        
        Args:
            inputs: Input text or list of texts
            **kwargs: Additional parameters:
                - pooling_strategy: How to pool token embeddings (mean, cls, etc.)
                - batch_size: Override default batch size
        
        Returns:
            Dictionary with model outputs including embeddings
        """
        pooling_strategy = kwargs.get('pooling_strategy', 'mean')
        
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Model not set up. Call setup() first.")
                
            if isinstance(inputs, str):
                inputs = [inputs]
                
            # Tokenize inputs
            encoded_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
            encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
            
            # Run model
            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                
            # Process outputs based on pooling strategy
            if pooling_strategy == 'cls':
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif pooling_strategy == 'mean':
                # Mean pooling
                attention_mask = encoded_inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            else:
                # Default to last_hidden_state
                embeddings = outputs.last_hidden_state.cpu().numpy()
            
            return {
                "embeddings": embeddings,
                "embedding_dim": embeddings.shape[-1],
                "model_name": self.model_name,
                "hardware": self.hardware
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "model_name": self.model_name,
                "hardware": self.hardware
            }
    
    def cleanup(self) -> bool:
        """Clean up resources."""
        try:
            # Free CUDA memory
            if self.model and hasattr(torch, 'cuda') and torch.cuda.is_available():
                del self.model
                torch.cuda.empty_cache()
            
            self.model = None
            self.tokenizer = None
            return True
        except Exception as e:
            print(f"Error during cleanup: {e}")
            return False
```

### Example Test Component

```python
#!/usr/bin/env python3
# Generated by TemplateRenderer on 2025-03-16 10:15:22
# Model: bert-base-uncased
# Template: text_embedding_test (a1b2c3d4-e5f6-7890-a1b2-c3d4e5f6a7b8)
# Hardware: cpu
# Type: test

"""
Test for bert-base-uncased model on cpu hardware.
Generated by integrated component test runner.
"""

import unittest
import numpy as np
from bert_base_uncased_cpu_skill import BertBaseUncasedSkill

class TestBertBaseUncasedCpu(unittest.TestCase):
    """Test suite for bert-base-uncased model on cpu hardware."""
    
    def setUp(self):
        """Set up the test environment."""
        self.skill = BertBaseUncasedSkill()
        self.setup_success = self.skill.setup()
        
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'skill'):
            self.skill.cleanup()
    
    def test_setup(self):
        """Test model setup."""
        self.assertTrue(self.setup_success, "Model setup should succeed")
        self.assertIsNotNone(self.skill.model, "Model should be initialized")
        self.assertIsNotNone(self.skill.tokenizer, "Tokenizer should be initialized")
    
    def test_embedding_dimensions(self):
        """Test embedding dimensions."""
        if not self.setup_success:
            self.skipTest("Model setup failed")
        
        # Run model on test input
        result = self.skill.run("This is a test input")
        
        # Check result structure
        self.assertIn("embeddings", result, "Result should contain embeddings")
        self.assertIn("embedding_dim", result, "Result should contain embedding_dim")
        
        # Check embedding dimensions
        embeddings = result["embeddings"]
        self.assertEqual(embeddings.shape[0], 1, "Should have 1 embedding (batch size 1)")
        self.assertEqual(embeddings.shape[1], 768, "Embedding dimension should be 768")
        self.assertEqual(result["embedding_dim"], 768, "embedding_dim should be 768")
    
    def test_batch_processing(self):
        """Test batch processing."""
        if not self.setup_success:
            self.skipTest("Model setup failed")
        
        # Prepare batch input
        batch_input = ["First test sentence.", "Second test sentence.", "Third test sentence."]
        
        # Run model on batch
        result = self.skill.run(batch_input)
        
        # Check result
        self.assertIn("embeddings", result, "Result should contain embeddings")
        embeddings = result["embeddings"]
        self.assertEqual(embeddings.shape[0], 3, "Should have 3 embeddings (batch size 3)")
        self.assertEqual(embeddings.shape[1], 768, "Embedding dimension should be 768")
    
    def test_embedding_properties(self):
        """Test embedding properties."""
        if not self.setup_success:
            self.skipTest("Model setup failed")
        
        # Run model on test input
        result = self.skill.run("This is a test input")
        
        # Check embedding properties
        embeddings = result["embeddings"]
        self.assertTrue(np.all(np.isfinite(embeddings)), "Embeddings should all be finite")
        self.assertTrue(-100 < np.min(embeddings) < 100, "Embedding values should be within reasonable range")
        self.assertTrue(-100 < np.max(embeddings) < 100, "Embedding values should be within reasonable range")
    
    def test_pooling_strategies(self):
        """Test different pooling strategies."""
        if not self.setup_success:
            self.skipTest("Model setup failed")
        
        # Run model with different pooling strategies
        cls_result = self.skill.run("This is a test input", pooling_strategy='cls')
        mean_result = self.skill.run("This is a test input", pooling_strategy='mean')
        
        # Check result structure
        self.assertIn("embeddings", cls_result, "CLS result should contain embeddings")
        self.assertIn("embeddings", mean_result, "Mean result should contain embeddings")
        
        # Embeddings should be different for different pooling strategies
        cls_embeddings = cls_result["embeddings"]
        mean_embeddings = mean_result["embeddings"]
        self.assertFalse(np.allclose(cls_embeddings, mean_embeddings), 
                         "CLS and Mean pooling should produce different embeddings")

if __name__ == "__main__":
    unittest.main()
```

### Example Benchmark Component

```python
#!/usr/bin/env python3
# Generated by TemplateRenderer on 2025-03-16 10:15:22
# Model: bert-base-uncased
# Template: text_embedding_benchmark (c7d8e9f0-a1b2-c3d4-e5f6-a7b8c9d0e1f2)
# Hardware: cuda
# Type: benchmark

"""
Benchmark for bert-base-uncased model on cuda hardware.
Generated by integrated component test runner.
"""

import os
import time
import json
import argparse
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from bert_base_uncased_cuda_skill import BertBaseUncasedSkill

def benchmark_bert_base_uncased_on_cuda(batch_sizes: List[int] = [1, 2, 4, 8, 16],
                                      iterations: int = 10,
                                      warmup: int = 3,
                                      output_dir: str = "./benchmark_results") -> Dict[str, Any]:
    """
    Benchmark bert-base-uncased model on cuda hardware.
    
    Args:
        batch_sizes: List of batch sizes to benchmark
        iterations: Number of iterations for each batch size
        warmup: Number of warmup iterations before timing
        output_dir: Directory to store results
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"Benchmarking bert-base-uncased on cuda hardware")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Iterations: {iterations}")
    print(f"Warmup: {warmup}")
    
    # Initialize skill
    skill = BertBaseUncasedSkill()
    setup_start = time.time()
    setup_success = skill.setup()
    setup_time = time.time() - setup_start
    
    if not setup_success:
        print("Failed to set up model")
        return {
            "success": False,
            "model_name": "bert-base-uncased",
            "hardware": "cuda",
            "error": "Failed to set up model"
        }
    
    print(f"Model setup completed in {setup_time:.4f} seconds")
    
    # Generate test input text (medium-length sentence)
    base_text = "This is a benchmark test input for the bert-base-uncased model on cuda hardware."
    
    # Results will be stored here
    results_by_batch = {}
    
    # Memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    # Run benchmarks for each batch size
    for batch_size in batch_sizes:
        print(f"\nBenchmarking with batch size {batch_size}")
        
        # Create input batch
        if batch_size == 1:
            inputs = base_text
        else:
            inputs = [base_text] * batch_size
            
        # Warmup runs
        print(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            _ = skill.run(inputs)
            
        # Reset memory stats after warmup
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # Actual timing runs
        print(f"Running {iterations} timed iterations...")
        latencies = []
        
        for i in range(iterations):
            start_time = time.time()
            result = skill.run(inputs)
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            print(f"  Iteration {i+1}/{iterations}: {latency:.2f} ms")
            
        # Calculate statistics
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Calculate throughput
        throughput = batch_size * 1000 / avg_latency  # items per second
        
        # Get memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        else:
            memory_usage = None
            
        # Store results for this batch size
        results_by_batch[str(batch_size)] = {
            "average_latency_ms": float(avg_latency),
            "min_latency_ms": float(min_latency),
            "max_latency_ms": float(max_latency),
            "p50_latency_ms": float(p50_latency),
            "p95_latency_ms": float(p95_latency),
            "p99_latency_ms": float(p99_latency),
            "throughput_items_per_second": float(throughput),
            "memory_usage_mb": float(memory_usage) if memory_usage is not None else None,
            "all_latencies_ms": [float(l) for l in latencies]
        }
        
        # Print results
        print(f"\nResults for batch size {batch_size}:")
        print(f"  Average latency: {avg_latency:.2f} ms")
        print(f"  P50 latency: {p50_latency:.2f} ms")
        print(f"  P95 latency: {p95_latency:.2f} ms")
        print(f"  Throughput: {throughput:.2f} items/second")
        if memory_usage is not None:
            print(f"  Memory usage: {memory_usage:.2f} MB")
    
    # Clean up
    cleanup_success = skill.cleanup()
    
    # Prepare final results
    final_results = {
        "model_name": "bert-base-uncased",
        "model_type": "text_embedding",
        "hardware": "cuda",
        "hardware_info": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        "benchmark_config": {
            "batch_sizes": batch_sizes,
            "iterations": iterations,
            "warmup_iterations": warmup
        },
        "results_by_batch": results_by_batch,
        "setup_time_seconds": setup_time,
        "success": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results to file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"benchmark_bert_base_uncased_cuda_{int(time.time())}.json")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark bert-base-uncased on cuda")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16",
                        help="Comma-separated list of batch sizes")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations for each batch size")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                        help="Directory to store results")
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    
    # Run benchmark
    benchmark_bert_base_uncased_on_cuda(
        batch_sizes=batch_sizes,
        iterations=args.iterations,
        warmup=args.warmup,
        output_dir=args.output_dir
    )
```

## Visualization Dashboard and Integrated Reports System

The Template System now integrates with the Visualization Dashboard and Integrated Reports System for exploring test results and performance metrics interactively:

### Template Result Visualization

The Visualization Dashboard provides special views for template-generated components:

- **Template Success Rate**: View success rates for components generated from different templates
- **Template Performance Comparison**: Compare performance metrics across different template versions
- **Template Evolution**: Track how template changes affect performance over time
- **Hardware-Specific Template Analysis**: Analyze how hardware-specific templates perform compared to generic templates

### Using the Dashboard with Templates

To use the Visualization Dashboard with template-generated components:

```bash
# Start the visualization dashboard
python visualization_dashboard.py

# Filter to view only template-generated components
# (Select "Template Source" filter in the dashboard UI)
```

### Using the Integrated System with Templates

The Integrated Visualization and Reports System combines the Visualization Dashboard with the Enhanced CI/CD Reports Generator, providing a unified interface for both interactive exploration and report generation:

```bash
# Start the dashboard and filter by template source
python integrated_visualization_reports.py --dashboard

# Generate template-specific reports
python integrated_visualization_reports.py --reports --template-analysis

# Generate comparison reports for different template versions
python integrated_visualization_reports.py --reports --template-evolution

# Do both: start dashboard and generate template reports
python integrated_visualization_reports.py --dashboard --reports --template-analysis

# Export template performance visualizations
python integrated_visualization_reports.py --dashboard-export

# Use a specific database and automatically open browser
python integrated_visualization_reports.py --dashboard --db-path ./test_template_db.duckdb --open-browser
```

The key benefits of the integrated system include:

- **Unified Interface**: Single command-line interface for both dashboard and reports
- **Consistent Database Access**: Same database connection used across all components
- **Process Management**: Handles dashboard process lifecycle automatically
- **Coordinated Analysis**: Generate reports based on the same data viewed in the dashboard
- **Multiple Report Types**: Support for simulation validation, cross-hardware comparison, and combined reports
- **Export Capabilities**: Export dashboard visualizations for offline viewing
- **Browser Integration**: Option to automatically open the dashboard in a web browser

### Template-Specific Features in the Integrated System

The integrated system provides specialized features for template-generated components:

```bash
# Generate comparative report for different template versions
python integrated_visualization_reports.py --reports --template-evolution

# Analyze how hardware-specific templates perform vs. generic ones
python integrated_visualization_reports.py --reports --template-hardware-analysis

# Generate comprehensive report on template system performance
python integrated_visualization_reports.py --reports --template-system-report

# Generate combined template and hardware report
python integrated_visualization_reports.py --reports --combined-report --template-analysis
```

These commands enable you to:
- Compare performance across different template versions
- Generate specialized reports for template-based components
- Export visualizations of template performance metrics
- Track template evolution over time with historical analysis
- Identify optimal template configurations for different hardware platforms
- Generate comprehensive documentation for the template system

### Dashboard Export for Templates

The dashboard export functionality creates static exports of template-specific visualizations:

```bash
# Export all template-related visualizations
python integrated_visualization_reports.py --dashboard-export --template-focus

# Export hardware-specific template analysis
python integrated_visualization_reports.py --dashboard-export --template-hardware-focus
```

For detailed instructions on all visualization and reporting capabilities, see the [VISUALIZATION_DASHBOARD_README.md](VISUALIZATION_DASHBOARD_README.md) for comprehensive documentation.

## Conclusion

The template system provides a powerful, flexible way to generate model components for the End-to-End Testing Framework. By centralizing templates and automating component generation, it enables the framework to focus on fixing generators rather than individual files, making maintenance more efficient and consistent across the entire test suite.

By implementing this DuckDB-based template database and renderer system, we've established a foundation for:

1. **Consistent Component Generation**: All model components follow the same structure and style
2. **Efficient Maintenance**: Updates to templates automatically propagate to all generated components
3. **Hardware-Specific Optimizations**: Templates can be tailored for specific hardware platforms
4. **Model Family Abstractions**: Common patterns for model families are codified in templates
5. **Systematic Documentation**: Documentation is generated alongside code for better maintainability
6. **Validation Integration**: Generated components are automatically validated to ensure correctness
7. **Interactive Visualization**: Performance metrics from template-generated components can be explored through the Visualization Dashboard

## Troubleshooting

For issues with the template system or the integrated visualization and reports components, please refer to the following resources:

1. **Template System Issues**:
   - For problems with template rendering or database operations, check the detailed logging:
     ```bash
     python template_database.py --validate-templates --verbose
     ```
   - For template validation issues, use:
     ```bash
     python template_validation.py --validate-all --verbose
     ```

2. **Visualization Dashboard and Reports Issues**:
   - For comprehensive troubleshooting solutions for the Integrated Visualization and Reports System, refer to [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md).
   - This guide provides detailed solutions for dashboard process management, database connectivity, report generation, visualization rendering, browser integration, and more.

The template system ultimately helps achieve the primary goal of the End-to-End Testing Framework: focusing on fixing generators rather than individual test files, making the entire testing process more efficient and maintainable.