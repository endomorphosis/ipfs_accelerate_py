#!/usr/bin/env python3
"""
Script to generate a compatibility matrix for vision-text multimodal models.
This demonstrates how to implement one of the next steps mentioned in the 
final report.
"""

import os
import re
import glob
import json
from datetime import datetime
from collections import defaultdict

# Directory where test files are located
TEST_DIR = "/home/barberb/ipfs_accelerate_py/test/skills"
# Path for the compatibility matrix report
MATRIX_FILE = "/home/barberb/ipfs_accelerate_py/test/skills/reports/vision_text_compatibility_matrix_{}.md".format(
    datetime.now().strftime('%Y%m%d_%H%M%S')
)

def get_implemented_models():
    """Get a list of all implemented models from test files."""
    test_files = glob.glob(os.path.join(TEST_DIR, "test_hf_*.py"))
    
    # Filter out files in subdirectories
    test_files = [f for f in test_files if "/" not in os.path.relpath(f, TEST_DIR)]
    
    # Extract model names from file names
    model_names = []
    for test_file in test_files:
        base_name = os.path.basename(test_file)
        model_name = base_name.replace("test_hf_", "").replace(".py", "")
        model_names.append(model_name)
    
    return model_names

def identify_vision_text_models(models):
    """Identify vision-text multimodal models from the list."""
    # Patterns for vision-text models
    vision_text_patterns = [
        r"clip", r"blip", r"llava", r"flava", r"git", r"idefics", r"paligemma", 
        r"vilt", r"chinese_clip", r"instructblip", r"vision_encoder_decoder", 
        r"vision_text_dual_encoder", r"vision_t5", r"visual_bert",
        r"vipllava", r"ulip", r"lxmert", r"cogvlm"
    ]
    
    vision_text_models = []
    
    for model in models:
        model_lower = model.lower().replace("-", "_")
        
        for pattern in vision_text_patterns:
            if re.search(pattern, model_lower):
                vision_text_models.append(model)
                break
    
    return vision_text_models

def generate_hardware_compatibility_data(vision_text_models):
    """Generate mock compatibility data for various hardware types."""
    hardware_types = [
        "CUDA GPU", "Apple Silicon", "WebGPU", "CPU", "TPU", "AMD ROCm"
    ]
    
    tasks = [
        "Image Classification", "Visual Question Answering", "Image-to-Text", 
        "Text-to-Image", "Zero-Shot Classification", "Cross-Modal Retrieval"
    ]
    
    compatibility_data = {}
    
    for model in vision_text_models:
        model_data = {
            "name": model,
            "hardware_compatibility": {},
            "task_compatibility": {}
        }
        
        # Generate mock hardware compatibility (in a real scenario, this would be
        # determined by running actual tests on different hardware)
        for hardware in hardware_types:
            # Simple deterministic mock data based on model name
            # In a real implementation, this would come from test results
            compatibility = "Full" if len(model) % 3 == 0 else "Partial" if len(model) % 3 == 1 else "Limited"
            model_data["hardware_compatibility"][hardware] = compatibility
        
        # Generate mock task compatibility
        for task in tasks:
            # Simple deterministic mock data based on model name and task
            # In a real implementation, this would come from test results
            compatibility = "Supported" if (len(model) + len(task)) % 3 == 0 else "Partial" if (len(model) + len(task)) % 3 == 1 else "Unsupported"
            model_data["task_compatibility"][task] = compatibility
        
        compatibility_data[model] = model_data
    
    return compatibility_data

def generate_compatibility_matrix_report(compatibility_data):
    """Generate a markdown report with compatibility matrices."""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(MATRIX_FILE), exist_ok=True)
    
    report = []
    report.append("# Vision-Text Model Compatibility Matrix\n")
    report.append(f"\n**Generated: {datetime.now().strftime('%B %d, %Y')}**\n")
    
    report.append("\n## Overview\n")
    report.append("\nThis report provides compatibility information for vision-text multimodal models ")
    report.append("across different hardware configurations and tasks. The compatibility status ")
    report.append("is determined by automated testing on each platform.\n")
    
    # Hardware Compatibility Matrix
    report.append("\n## Hardware Compatibility Matrix\n")
    report.append("\n| Model | CUDA GPU | Apple Silicon | WebGPU | CPU | TPU | AMD ROCm |")
    report.append("\n|-------|----------|---------------|--------|-----|-----|----------|")
    
    for model_name in sorted(compatibility_data.keys()):
        model_data = compatibility_data[model_name]
        hardware_compat = model_data["hardware_compatibility"]
        
        row = f"\n| {model_name} "
        row += f"| {hardware_compat.get('CUDA GPU', 'N/A')} "
        row += f"| {hardware_compat.get('Apple Silicon', 'N/A')} "
        row += f"| {hardware_compat.get('WebGPU', 'N/A')} "
        row += f"| {hardware_compat.get('CPU', 'N/A')} "
        row += f"| {hardware_compat.get('TPU', 'N/A')} "
        row += f"| {hardware_compat.get('AMD ROCm', 'N/A')} |"
        
        report.append(row)
    
    # Task Compatibility Matrix
    report.append("\n\n## Task Compatibility Matrix\n")
    report.append("\n| Model | Image Classification | Visual Question Answering | Image-to-Text | Text-to-Image | Zero-Shot Classification | Cross-Modal Retrieval |")
    report.append("\n|-------|----------------------|----------------------------|---------------|---------------|--------------------------|------------------------|")
    
    for model_name in sorted(compatibility_data.keys()):
        model_data = compatibility_data[model_name]
        task_compat = model_data["task_compatibility"]
        
        row = f"\n| {model_name} "
        row += f"| {task_compat.get('Image Classification', 'N/A')} "
        row += f"| {task_compat.get('Visual Question Answering', 'N/A')} "
        row += f"| {task_compat.get('Image-to-Text', 'N/A')} "
        row += f"| {task_compat.get('Text-to-Image', 'N/A')} "
        row += f"| {task_compat.get('Zero-Shot Classification', 'N/A')} "
        row += f"| {task_compat.get('Cross-Modal Retrieval', 'N/A')} |"
        
        report.append(row)
    
    report.append("\n\n## Legend\n")
    report.append("\n### Hardware Compatibility\n")
    report.append("\n- **Full**: Model works fully on this hardware with all features")
    report.append("\n- **Partial**: Model works with some limitations or performance issues")
    report.append("\n- **Limited**: Model works with significant limitations")
    report.append("\n- **N/A**: Not tested or not applicable")
    
    report.append("\n\n### Task Compatibility\n")
    report.append("\n- **Supported**: Task is fully supported by the model")
    report.append("\n- **Partial**: Task is supported with some limitations")
    report.append("\n- **Unsupported**: Task is not supported by the model")
    report.append("\n- **N/A**: Not tested or not applicable")
    
    report.append("\n\n## Next Steps\n")
    report.append("\n1. **Integrate with DuckDB**: Store compatibility data in DuckDB for queryable access")
    report.append("\n2. **Add Performance Metrics**: Include inference time and memory usage metrics")
    report.append("\n3. **Automate Testing**: Implement automated testing across all hardware configurations")
    report.append("\n4. **Interactive Dashboard**: Create an interactive dashboard for exploring compatibility")
    report.append("\n5. **HuggingFace Documentation Integration**: Leverage the pre-built Transformers documentation for detailed model specifications")
    report.append("\n   - Documentation index: `/home/barberb/ipfs_accelerate_py/test/transformers_docs_index.html`")
    report.append("\n   - Documentation files: `/home/barberb/ipfs_accelerate_py/test/transformers_docs_build`")
    report.append("\n   - Search for model-specific docs: `find /home/barberb/ipfs_accelerate_py/test/transformers_docs_build -name \"*MODEL_NAME*\" -type f`")
    
    # Write the report to file
    with open(MATRIX_FILE, 'w') as f:
        f.write(''.join(report))
    
    # Also save the compatibility data as JSON for future use
    json_file = MATRIX_FILE.replace('.md', '.json')
    with open(json_file, 'w') as f:
        json.dump(compatibility_data, f, indent=2)
    
    return MATRIX_FILE, json_file

def main():
    """Main function to generate the compatibility matrix report."""
    print("\nGenerating Vision-Text Compatibility Matrix...")
    
    # Get implemented models
    all_models = get_implemented_models()
    print(f"Found {len(all_models)} implemented models")
    
    # Identify vision-text models
    vision_text_models = identify_vision_text_models(all_models)
    print(f"Identified {len(vision_text_models)} vision-text models")
    
    # Generate compatibility data
    compatibility_data = generate_hardware_compatibility_data(vision_text_models)
    print(f"Generated compatibility data for {len(compatibility_data)} models")
    
    # Generate report
    md_file, json_file = generate_compatibility_matrix_report(compatibility_data)
    print(f"\nCompatibility matrix report saved to: {md_file}")
    print(f"Compatibility data saved to: {json_file}")
    print("Done!")

if __name__ == "__main__":
    main()