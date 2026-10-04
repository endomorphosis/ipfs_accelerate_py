# Template Database System

This directory contains a template database system for generating test files and models. The system now uses a JSON-based database (`template_db.json`) to store templates, which allows for easier template management, validation, and instantiation.

## Template Database Status (March 10, 2025)

- **Total Templates**: 26
- **Templates with Valid Syntax**: 14 (54%)
- **Templates with Syntax Errors**: 12 (46%)

### Fixed Templates
We have successfully fixed 14 templates that now have valid Python syntax while maintaining template placeholders:

1. text_embedding_test_template_text_generation.py (text_embedding/test)
2. vision_test_template_vision_language.py (vision/test)
3. llava_test_template_llava.py (llava/test)
4. whisper_test_template_whisper.py (whisper/test)
5. clap_test_template_clap.py (clap/test)
6. audio_test_template_audio.py (audio/test)
7. bert_test_template_bert.py (bert/test)
8. inheritance_test_template_inheritance_system_fixed.py (inheritance/test)
9. selection_test_template_selection.py (selection/test)
10. validator_test_template_validator.py (validator/test)
11. wav2vec2_test_template_wav2vec2.py (wav2vec2/test)
12. llava_test_template_llava_next.py (llava/test)
13. verifier_test_template_verifier.py (verifier/test)
14. hardware_test_template_hardware_detection.py (hardware/test)

### Templates Still Needing Fixes
These templates still have syntax errors that need to be fixed:

1. video_test_template_video.py (video/test) - Unexpected indent
2. cpu_test_template_cpu_embedding.py (cpu/test) - Invalid syntax
3. llama_test_template_llama.py (llama/test) - Missing indented block
4. text_embedding_test_template_text_embedding.py (text_embedding/test) - Mismatched brackets
5. t5_test_template_t5.py (t5/test) - Unexpected indent
6. xclip_test_template_xclip.py (xclip/test) - Unexpected indent
7. clip_test_template_clip.py (clip/test) - Mismatched brackets
8. test_test_template_test_generator.py (test/test) - Invalid syntax
9. vision_test_template_vision.py (vision/test) - Mismatched brackets
10. detr_test_template_detr.py (detr/test) - Mismatched brackets
11. qwen2_test_template_qwen2.py (qwen2/test) - Unexpected indent
12. vit_test_template_vit.py (vit/test) - Mismatched brackets

## Common Issues and Fixes
The most common syntax issues found and fixed are:

1. **Indentation Issues**: Misaligned code blocks and unexpected indentation
2. **Missing try/except Blocks**: Handlers lacking proper error handling
3. **Method Placement**: Methods defined outside classes that should be inside
4. **Mismatched Brackets**: In template strings with dictionary definitions
5. **Missing Platform Constants**: Undefined platform constants
6. **Missing Platform Support**: Incomplete hardware platform support

## Hardware Platform Support
All fixed templates now include comprehensive support for these platforms:

- CPU: Standard CPU implementation
- CUDA: NVIDIA GPU implementation
- OpenVINO: Intel hardware acceleration
- MPS: Apple Silicon GPU implementation
- ROCm: AMD GPU implementation
- Qualcomm: Qualcomm AI Engine/Hexagon DSP implementation (NEW - March 2025)
- WebNN: Web Neural Network API (browser)
- WebGPU: Web GPU API (browser)

## Using the Template System

To use these templates, use the template instantiation system:

```python
from template_database import TemplateDatabase

# Get a template
template_db = TemplateDatabase()
template = template_db.get_template("text_embedding", "test")

# Replace placeholders
instantiated_code = template.replace("{{model_name}}", "bert-base-uncased")
```

## Working with the Template Database

### Extracting Templates
Use the template_extractor.py tool to extract templates from the database:

```bash
python template_extractor.py --extract-template text_embedding/test --output my_template.py
```

### Listing Templates
List all templates in the database:

```bash
python template_extractor.py --list-templates
```

### Saving Fixed Templates
After fixing a template, save it back to the database:

```bash
python template_extractor.py --save-template template_id fixed_template.py
```

## Note About Template Syntax

Templates contain placeholders like `{{model_name}}` and code with curly braces like `{k: v for k, v in items}`. When fixing templates, it's important to:

1. Maintain valid Python syntax
2. Keep all placeholder variables intact
3. Ensure all class methods are properly indented
4. Follow our standard error handling approach
5. Support all hardware platforms consistently
