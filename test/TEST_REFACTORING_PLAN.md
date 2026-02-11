# Test Refactoring Plan for HuggingFace Transformers Full Coverage

## 1. Current Status Analysis

### Current Systems
- **Template System**: Architecture-specific templates in `/skills/templates/`
    - 7 core templates: encoder-only, decoder-only, encoder-decoder, vision, vision-text, speech, multimodal
    - Several backup/variant templates for each architecture
    - Templates follow different patterns and have independent implementations
    
- **Generator System**: 
    - `generate_model_class_inventory.py`: Creates inventory of HF models with from_pretrained method
    - `generate_missing_model_tests.py`: Generates tests for missing models based on priority
    - `regenerate_tests_with_fixes.py`: Regenerates test files using architecture-aware templates
    
- **Validation System**:
    - Basic syntax validation
    - Template-specific validation
    - No unified validation against ModelTest pattern

### Current Limitations
1. **Architecture Detection**: Simple string matching is prone to errors
2. **No Base Class**: Templates don't inherit from a common base class
3. **Inconsistent Implementation**: Different templates implement same functionality differently
4. **Limited Mock Support**: Mock systems are implemented inconsistently
5. **No ModelTest Integration**: Templates don't conform to ModelTest pattern
6. **No Validation Pipeline**: No validation of generated files against ModelTest pattern
7. **Hyphenated Model Names**: Special handling needed for hyphenated model names
8. **Limited Hardware Support**: Hardware detection is inconsistent

## 2. Unified Architecture

### 2.1. Template System Refactoring

Create a unified template system with proper inheritance in the `refactored_test_suite` folder:

```
refactored_test_suite/
    ├── model_test_base.py
    ├── architecture_specific/
    │     ├── encoder_only_test.py
    │     ├── decoder_only_test.py
    │     ├── encoder_decoder_test.py
    │     ├── vision_test.py
    │     ├── speech_test.py
    │     ├── vision_text_test.py
    │     └── multimodal_test.py
    ├── templates/
    │     ├── encoder_only_template.py
    │     ├── decoder_only_template.py
    │     └── ...
    ├── scripts/generators/
    │     ├── test_generator.py
    │     └── architecture_detector.py
    └── validation/
                └── test_validator.py
```

Class hierarchy:

```
ModelTestBase (abstract base class)
    ├── EncoderOnlyModelTest
    │     └── BertModelTest, RoBERTaModelTest, etc.
    ├── DecoderOnlyModelTest
    │     └── GPT2ModelTest, LlamaModelTest, etc.
    ├── EncoderDecoderModelTest
    │     └── T5ModelTest, BartModelTest, etc.
    ├── VisionModelTest
    │     └── ViTModelTest, SwinModelTest, etc.
    ├── SpeechModelTest
    │     └── WhisperModelTest, Wav2Vec2ModelTest, etc.
    ├── VisionTextModelTest
    │     └── CLIPModelTest, BLIPModelTest, etc.
    └── MultimodalModelTest
                └── LLaVAModelTest, FlavaModelTest, etc.
```

### 2.2. Advanced Architecture Detection

Create a robust architecture detection system:
- Use model class introspection from HF inventory
- Implement model name and class name pattern matching
- Consider model capabilities (e.g., has_vision_inputs, has_audio_inputs)
- Create a comprehensive model registry with architecture types

### 2.3. Template Generator

Create a unified template generator that:
- Selects appropriate template based on architecture
- Handles model-specific variations
- Generates standardized test files with ModelTest pattern
- Addresses hyphenated model names automatically
- Implements consistent mocking system

### 2.4. Validation System

Create a validation system that:
- Verifies syntax of generated files
- Validates against ModelTest pattern
- Checks required methods implementation
- Ensures consistent hardware detection
- Validates indentation and style

## 3. Implementation Plan

### 3.1. Phase 1: Create ModelTest Base Classes

1. Create `refactored_test_suite/model_test_base.py` with abstract ModelTest base class:
     - Define required methods: test_model_loading, load_model, verify_model_output, detect_preferred_device
     - Implement common functionality: hardware detection, mock system, result collection
     - Add abstract methods for architecture-specific functionality

2. Create architecture-specific model test base classes in `refactored_test_suite/architecture_specific/`:
     - EncoderOnlyModelTest
     - DecoderOnlyModelTest
     - EncoderDecoderModelTest
     - VisionModelTest
     - SpeechModelTest
     - VisionTextModelTest
     - MultimodalModelTest

3. Implement shared helper methods:
     - Hardware detection functions
     - Mock system for dependencies
     - Test input generation
     - Result reporting functions

### 3.2. Phase 2: Create Enhanced Architecture Detection

1. Refactor `get_architecture_type` function in `refactored_test_suite/scripts/generators/architecture_detector.py`:
     - Model class detection (use HF API when available)
     - Model configuration detection
     - Fallback to pattern matching with enhanced rules
     - Handle edge cases like hyphenated names

2. Create comprehensive mapping between model types and architectures:
     - Build on existing ARCHITECTURE_TYPES mapping
     - Add model-specific details like default models, tasks
     - Include method overrides for special cases

3. Create a model registry system:
     - Map model types to architecture-specific configurations
     - Include default models, tasks, and class names
     - Provide architecture detection overrides

### 3.3. Phase 3: Refactor Generator System

1. Create a unified generator class in `refactored_test_suite/scripts/generators/test_generator.py`:
     - Factory methods for different architecture types
     - Template selection based on architecture detection
     - Special handling for hyphenated names
     - Consistent mocking system

2. Enhance template application with:
     - Model-specific configuration injection
     - Proper indentation handling
     - Syntax validation
     - ModelTest pattern compliance

3. Create a validation pipeline in `refactored_test_suite/validation/`:
     - Syntax validation
     - ModelTest pattern validation
     - Hardware detection validation
     - Mock system validation

### 3.4. Phase 4: Full Integration

1. Create a unified CLI for test generation:
     - Options for architecture type, model name, output directory
     - Batch generation capability
     - Validation flags
     - Reporting options

2. Create a reporting system:
     - Generate coverage reports
     - Track compliance with ModelTest pattern
     - Identify missing models
     - Create prioritized model lists

3. Create a documentation system:
     - Update documentation with new generator capabilities
     - Document ModelTest pattern requirements
     - Create model coverage matrix 
     - Document hardware compatibility

## 4. Implementation Details

### 4.1. ModelTest Base Class (refactored_test_suite/model_test_base.py)

```python
class ModelTest:
        """Base class for all model tests."""
        
        def __init__(self, model_id=None, device=None):
                """Initialize the model test."""
                self.model_id = model_id or self.get_default_model_id()
                self.device = device or self.detect_preferred_device()
        
        def get_default_model_id(self):
                """Get the default model ID for this model type."""
                raise NotImplementedError("Subclasses must implement get_default_model_id()")
        
        def detect_preferred_device(self):
                """Detect the best available device for inference."""
                # Common device detection logic
                pass
        
        def load_model(self, model_id=None):
                """Load a model with the given ID."""
                raise NotImplementedError("Subclasses must implement load_model()")
        
        def test_model_loading(self):
                """Test basic model loading functionality."""
                raise NotImplementedError("Subclasses must implement test_model_loading()")
        
        def verify_model_output(self, model, input_data, expected_output=None):
                """Verify model outputs against expected values or sanity checks."""
                raise NotImplementedError("Subclasses must implement verify_model_output()")
        
        def run_tests(self):
                """Run all tests for this model."""
                results = {}
                results["model_loading"] = self.test_model_loading()
                # Add more test results as needed
                return results
```

### 4.2. Architecture-Specific Model Tests

```python
# refactored_test_suite/architecture_specific/encoder_only_test.py
class EncoderOnlyModelTest(ModelTest):
        """Base class for encoder-only models like BERT, RoBERTa, etc."""
        
        def get_default_model_id(self):
                """Get the default model ID for this model type."""
                return f"{self.model_type}-base-uncased"
        
        def load_model(self, model_id=None):
                """Load an encoder-only model."""
                from transformers import AutoModel
                model_id = model_id or self.model_id
                return AutoModel.from_pretrained(model_id)
        
        def test_model_loading(self):
                """Test encoder-only model loading."""
                try:
                        model = self.load_model()
                        return {"success": True, "model": self.model_id}
                except Exception as e:
                        return {"success": False, "error": str(e)}
        
        def verify_model_output(self, model, input_data, expected_output=None):
                """Verify encoder outputs."""
                # Encoder-specific verification
                pass
```

### 4.3. Architecture Detection

```python
# refactored_test_suite/scripts/generators/architecture_detector.py
def get_architecture_type(model_name):
        """Determine architecture type based on model name."""
        # Try introspection if transformers is available
        try:
                import transformers
                if hasattr(transformers, "AutoConfig"):
                        config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                        # Check model architecture based on config properties
                        
                        # Check if it's a text-to-text model (encoder-decoder)
                        if hasattr(config, "is_encoder_decoder") and config.is_encoder_decoder:
                                return "encoder-decoder"
                        
                        # Check if it's a vision model
                        if hasattr(config, "num_channels") and config.num_channels > 0:
                                # Check if it also has text components
                                if hasattr(config, "vocab_size") and config.vocab_size > 0:
                                        return "vision-encoder-text-decoder"
                                return "vision"
                        
                        # Check if it's a speech model
                        if "speech" in type(config).__name__.lower() or "audio" in type(config).__name__.lower():
                                return "speech"
                        
                        # Check if it's a decoder-only model
                        if hasattr(config, "is_decoder") and config.is_decoder:
                                return "decoder-only"
                        
                        # Default to encoder-only for remaining text models
                        return "encoder-only"
        except:
                pass
        
        # Fallback to pattern matching if introspection fails
        model_name_lower = model_name.lower()
        
        # Match by model name patterns
        for arch_type, models in ARCHITECTURE_TYPES.items():
                if any(model in model_name_lower for model in models):
                        return arch_type
        
        # Default to encoder-only if unknown
        return "encoder-only"
```

### 4.4. Unified Generator

```python
# refactored_test_suite/scripts/generators/test_generator.py
class ModelTestGenerator:
        """Generator for model test files."""
        
        def __init__(self, output_dir="generated_tests"):
                """Initialize the generator."""
                self.output_dir = output_dir
                os.makedirs(output_dir, exist_ok=True)
        
        def generate_test_file(self, model_name, force=False, verify=True):
                """Generate a test file for the given model name."""
                # Determine architecture type
                arch_type = get_architecture_type(model_name)
                
                # Get template for architecture
                template_content = self.get_template_for_architecture(arch_type)
                
                # Fill in template with model-specific information
                content = self.fill_template(template_content, model_name, arch_type)
                
                # Create file path
                file_name = f"test_hf_{model_name.replace('-', '_')}.py"
                file_path = os.path.join(self.output_dir, file_name)
                
                # Write content to file
                with open(file_path, "w") as f:
                        f.write(content)
                
                # Verify if requested
                if verify:
                        self.verify_test_file(file_path)
                
                return file_path
        
        def get_template_for_architecture(self, arch_type):
                """Get the template content for an architecture type."""
                # Implementation would read from template files
                pass
        
        def fill_template(self, template_content, model_name, arch_type):
                """Fill in the template with model-specific information."""
                # Implementation would do string replacements
                pass
        
        def verify_test_file(self, file_path):
                """Verify a generated test file."""
                # Syntax check
                # ModelTest pattern check
                # Hardware detection check
                pass
```

## 5. Example Usage

```bash
# Generate a test for a specific model
python refactored_test_suite/generate_model_tests.py --model bert

# Generate tests for all high-priority models
python refactored_test_suite/generate_model_tests.py --priority high

# Generate tests and validate against ModelTest pattern
python refactored_test_suite/generate_model_tests.py --validate

# Generate a coverage report
python refactored_test_suite/generate_model_tests.py --report
```

## 6. Testing Strategy

1. **Unit Tests for Components**:
     - Test architecture detection functions
     - Test template selection logic
     - Test template filling functions
     
2. **Integration Tests**:
     - Test end-to-end generation pipeline
     - Test validation pipeline
     
3. **Validation Tests**:
     - Verify generated files against ModelTest pattern
     - Verify syntax of generated files
     
4. **Model Tests**:
     - Test with a sample of each architecture type
     - Test with edge cases (hyphenated names, unusual models)

## 7. Timeline

1. **Week 1**: Create ModelTest base class and architecture-specific subclasses
2. **Week 2**: Implement enhanced architecture detection system
3. **Week 3**: Refactor template system for proper inheritance
4. **Week 4**: Create unified generator system
5. **Week 5**: Implement validation pipeline
6. **Week 6**: Full integration and testing
7. **Week 7**: Documentation and deployment

## 8. Success Metrics

1. **Coverage**: 100% of HuggingFace Transformers models covered
2. **Compliance**: 100% of generated tests comply with ModelTest pattern
3. **Validation**: 100% of generated tests pass syntax validation
4. **Hardware Support**: All tests support multiple hardware platforms
5. **Mock Support**: All tests work in CI/CD with mocked dependencies

## 9. Future Enhancements

1. **Model Registry Updates**: Regular updates to model registry for new models
2. **Performance Tests**: Add performance testing to generated tests
3. **Hardware-Specific Tests**: Generate hardware-specific test variants
4. **Benchmark Integration**: Integrate with benchmarking system
5. **Advanced Mocking**: More sophisticated mock implementations for CI/CD