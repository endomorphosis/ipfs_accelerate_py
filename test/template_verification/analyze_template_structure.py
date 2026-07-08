#!/usr/bin/env python3
"""
Analyze the template structure of HuggingFace test files.

This script:
1. Analyzes template files to understand the expected structure
2. Compares manually created tests against the template structure
3. Identifies missing components and structural issues
4. Reports findings in a detailed analysis

Usage:
    python analyze_template_structure.py [--verbose]
"""

import os
import sys
import ast
import argparse
import logging
from pathlib import Path
import difflib
import re
from typing import Dict, List, Set, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKILLS_DIR = REPO_ROOT / "skills"
TEMPLATES_DIR = SKILLS_DIR / "templates"
FINAL_MODELS_DIR = REPO_ROOT / "final_models"
FIXED_TESTS_DIR = SKILLS_DIR / "fixed_tests"
OUTPUT_DIR = REPO_ROOT / "template_verification"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the key components that should be present in all templates
REQUIRED_COMPONENTS = [
    "hardware_detection",          # Hardware detection imports and setup
    "dependency_mocking",          # Environment variable controls for mocking
    "model_registry",              # Model registry with configurations
    "test_class",                  # Test class implementation
    "test_pipeline",               # Pipeline API test method
    "test_from_pretrained",        # From_pretrained API test method
    "mock_objects",                # Mock objects for CI/CD testing
    "result_collection",           # Standardized result collection
    "runtime_checking",            # Runtime capability checking
    "main_function"                # Main function for CLI execution
]

# Define architecture-specific components
ARCHITECTURE_COMPONENTS = {
    "encoder-only": ["token_prediction", "mask_handling"],
    "decoder-only": ["autoregressive_generation", "causal_attention"],
    "encoder-decoder": ["sequence_to_sequence", "decoder_input_handling"],
    "vision": ["image_preprocessing", "pixel_normalization"],
    "vision-encoder-text-decoder": ["image_text_processing", "cross_modal_attention"],
    "speech": ["audio_processing", "feature_extraction"],
    "multimodal": ["multiple_modalities", "multimodal_fusion"]
}

# Define the mapping of manually created models to their expected architecture
MANUAL_MODEL_ARCHITECTURES = {
    "layoutlmv2": "vision-encoder-text-decoder",
    "layoutlmv3": "vision-encoder-text-decoder",
    "clvp": "speech",
    "bigbird": "encoder-decoder",
    "seamless_m4t_v2": "speech",
    "xlm_prophetnet": "encoder-decoder"
}

def parse_python_file(file_path: str) -> ast.Module:
    """Parse a Python file into an AST."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return ast.parse(content, filename=file_path)
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return None

def extract_imports(tree: ast.Module) -> Set[str]:
    """Extract import statements from an AST."""
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for name in node.names:
                imports.add(f"{module}.{name.name}")
    return imports

def extract_classes(tree: ast.Module) -> List[ast.ClassDef]:
    """Extract class definitions from an AST."""
    return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

def extract_functions(tree: ast.Module) -> List[ast.FunctionDef]:
    """Extract function definitions from an AST."""
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

def extract_class_methods(class_def: ast.ClassDef) -> List[ast.FunctionDef]:
    """Extract methods from a class definition."""
    return [node for node in class_def.body if isinstance(node, ast.FunctionDef)]

def extract_variables(tree: ast.Module) -> Dict[str, Any]:
    """Extract top-level variable assignments from an AST."""
    variables = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.targets:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        # Try to evaluate the value
                        if isinstance(node.value, ast.Constant):
                            variables[target.id] = node.value.value
                        elif isinstance(node.value, ast.Dict):
                            variables[target.id] = "dict"
                        elif isinstance(node.value, ast.List):
                            variables[target.id] = "list"
                        else:
                            variables[target.id] = str(ast.unparse(node.value))
                    except:
                        variables[target.id] = "unknown"
    return variables

def analyze_template_file(template_path: str) -> Dict[str, Any]:
    """Analyze a template file to extract its structure."""
    tree = parse_python_file(template_path)
    if not tree:
        return None
    
    analysis = {
        "imports": extract_imports(tree),
        "classes": [cls.name for cls in extract_classes(tree)],
        "functions": [func.name for func in extract_functions(tree)],
        "variables": extract_variables(tree),
        "components": detect_components(tree, template_path)
    }
    
    # Extract class methods for test classes
    class_methods = {}
    for cls in extract_classes(tree):
        methods = extract_class_methods(cls)
        class_methods[cls.name] = [method.name for method in methods]
    analysis["class_methods"] = class_methods
    
    return analysis

def detect_components(tree: ast.Module, file_path: str) -> Dict[str, bool]:
    """Detect the presence of required components in a template."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    components = {comp: False for comp in REQUIRED_COMPONENTS}
    
    # Check for hardware detection
    if "hardware_detection" in content or "HAS_CUDA" in content:
        components["hardware_detection"] = True
    
    # Check for dependency mocking
    if "MOCK_TORCH" in content or "MOCK_TRANSFORMERS" in content:
        components["dependency_mocking"] = True
    
    # Check for model registry
    if "_MODELS_REGISTRY" in content or "MODEL_REGISTRY" in content:
        components["model_registry"] = True
    
    # Check for test class
    classes = extract_classes(tree)
    if any("Test" in cls.name for cls in classes):
        components["test_class"] = True
    
    # Check for test pipeline method
    for cls in classes:
        methods = extract_class_methods(cls)
        if any("test_pipeline" in method.name for method in methods):
            components["test_pipeline"] = True
        if any("test_from_pretrained" in method.name for method in methods):
            components["test_from_pretrained"] = True
    
    # Check for mock objects
    if "class Mock" in content or "MagicMock" in content:
        components["mock_objects"] = True
    
    # Check for result collection
    if "results =" in content or "results[" in content or "save_results" in content:
        components["result_collection"] = True
    
    # Check for runtime checking
    if "torch.cuda.is_available()" in content or "select_device" in content:
        components["runtime_checking"] = True
    
    # Check for main function
    functions = extract_functions(tree)
    if any(func.name == "main" for func in functions):
        components["main_function"] = True
    
    return components

def analyze_test_file(test_path: str, architecture: str) -> Dict[str, Any]:
    """Analyze a test file to check its conformance to a template."""
    tree = parse_python_file(test_path)
    if not tree:
        return {
            "success": False,
            "error": "Syntax error in the file",
            "components": {comp: False for comp in REQUIRED_COMPONENTS}
        }
    
    components = detect_components(tree, test_path)
    architecture_components = ARCHITECTURE_COMPONENTS.get(architecture, [])
    
    # Add architecture-specific component checks
    for comp in architecture_components:
        components[comp] = False
    
    # Read file content for further analysis
    with open(test_path, 'r') as f:
        content = f.read()
    
    # Check architecture-specific components
    if architecture == "encoder-only":
        if "mask_token" in content or "Masked" in content:
            components["mask_handling"] = True
        if "predict" in content or "token_pred" in content:
            components["token_prediction"] = True
    elif architecture == "decoder-only":
        if "generate" in content or "generated_text" in content:
            components["autoregressive_generation"] = True
        if "causal" in content or "attention_mask" in content:
            components["causal_attention"] = True
    elif architecture == "encoder-decoder":
        if "sequence" in content or "seq2seq" in content:
            components["sequence_to_sequence"] = True
        if "decoder_input" in content or "encoder_outputs" in content:
            components["decoder_input_handling"] = True
    elif architecture == "vision":
        if "image" in content or "pixel" in content:
            components["image_preprocessing"] = True
        if "normalize" in content or "transform" in content:
            components["pixel_normalization"] = True
    elif architecture == "vision-encoder-text-decoder":
        if "image" in content and "text" in content:
            components["image_text_processing"] = True
        if "cross" in content or "cross_attention" in content:
            components["cross_modal_attention"] = True
    elif architecture == "speech":
        if "audio" in content or "wav" in content:
            components["audio_processing"] = True
        if "feature" in content or "spectrogram" in content:
            components["feature_extraction"] = True
    elif architecture == "multimodal":
        if "image" in content and "text" in content:
            components["multiple_modalities"] = True
        if "fusion" in content or "combine" in content:
            components["multimodal_fusion"] = True
    
    # Calculate conformance score
    total_components = len(REQUIRED_COMPONENTS) + len(architecture_components)
    present_components = sum(1 for comp, present in components.items() if present)
    conformance_score = (present_components / total_components) * 100 if total_components > 0 else 0
    
    return {
        "success": True,
        "components": components,
        "conformance_score": conformance_score,
        "architecture": architecture,
        "imports": extract_imports(tree),
        "classes": [cls.name for cls in extract_classes(tree)],
        "functions": [func.name for func in extract_functions(tree)]
    }

def get_reference_template(architecture: str) -> str:
    """Get the path to the reference template for an architecture."""
    template_map = {
        "encoder-only": "encoder_only_template.py",
        "decoder-only": "decoder_only_template.py",
        "encoder-decoder": "encoder_decoder_template.py",
        "vision": "vision_template.py",
        "vision-encoder-text-decoder": "vision_text_template.py",
        "speech": "speech_template.py",
        "multimodal": "multimodal_template.py"
    }
    
    template_file = template_map.get(architecture, "encoder_only_template.py")
    return os.path.join(TEMPLATES_DIR, template_file)

def compare_with_template(test_path: str, template_path: str) -> Dict[str, Any]:
    """Compare a test file with its reference template."""
    with open(test_path, 'r') as f:
        test_content = f.read()
    
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Calculate similarity
    diff = difflib.SequenceMatcher(None, test_content, template_content)
    similarity_score = diff.ratio() * 100
    
    # Identify missing sections
    missing_sections = []
    for section in ["hardware detection", "mock objects", "model registry", "test_pipeline", "test_from_pretrained", "main function"]:
        pattern = f"# {section}"
        if pattern.lower() in template_content.lower() and pattern.lower() not in test_content.lower():
            missing_sections.append(section)
    
    # Check for syntax errors
    syntax_errors = []
    try:
        ast.parse(test_content)
    except SyntaxError as e:
        syntax_errors.append(f"Line {e.lineno}: {e.msg}")
    
    return {
        "similarity_score": similarity_score,
        "missing_sections": missing_sections,
        "syntax_errors": syntax_errors
    }

def generate_template_fix_recommendation(test_path: str, template_path: str, architecture: str) -> str:
    """Generate recommendations for fixing a test file to conform to its template."""
    test_analysis = analyze_test_file(test_path, architecture)
    template_analysis = analyze_template_file(template_path)
    comparison = compare_with_template(test_path, template_path)
    
    recommendations = []
    
    # Add general recommendation
    recommendations.append(f"# Recommendations for fixing {os.path.basename(test_path)}")
    recommendations.append("")
    
    # Add conformance score
    conformance_score = test_analysis.get("conformance_score", 0)
    if conformance_score < 50:
        recommendations.append(f"## LOW CONFORMANCE SCORE: {conformance_score:.1f}%")
        recommendations.append("This file has significant structural differences from the template.")
        recommendations.append("Consider regenerating it completely using the template system.")
    elif conformance_score < 80:
        recommendations.append(f"## MODERATE CONFORMANCE SCORE: {conformance_score:.1f}%")
        recommendations.append("This file needs several fixes to match the template structure.")
    else:
        recommendations.append(f"## HIGH CONFORMANCE SCORE: {conformance_score:.1f}%")
        recommendations.append("This file is mostly conformant with the template structure.")
        recommendations.append("Only minor fixes are needed.")
    
    recommendations.append("")
    
    # Add missing components
    missing_components = [comp for comp, present in test_analysis.get("components", {}).items() if not present]
    if missing_components:
        recommendations.append("## Missing Components")
        for comp in missing_components:
            recommendations.append(f"- {comp}")
        recommendations.append("")
    
    # Add syntax errors
    syntax_errors = comparison.get("syntax_errors", [])
    if syntax_errors:
        recommendations.append("## Syntax Errors")
        for error in syntax_errors:
            recommendations.append(f"- {error}")
        recommendations.append("")
    
    # Add specific recommendations
    if "hardware_detection" in missing_components:
        recommendations.append("## Hardware Detection")
        recommendations.append("Add the following hardware detection code:")
        recommendations.append("```python")
        recommendations.append("try:")
        recommendations.append("    from scripts.generators.hardware.hardware_detection import (")
        recommendations.append("        HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,")
        recommendations.append("        detect_all_hardware")
        recommendations.append("    )")
        recommendations.append("    HAS_HARDWARE_DETECTION = True")
        recommendations.append("except ImportError:")
        recommendations.append("    HAS_HARDWARE_DETECTION = False")
        recommendations.append("    # We'll detect hardware manually as fallback")
        recommendations.append("```")
        recommendations.append("")
    
    if "dependency_mocking" in missing_components:
        recommendations.append("## Dependency Mocking")
        recommendations.append("Add environment variable controls for mocking dependencies:")
        recommendations.append("```python")
        recommendations.append("# Check if we should mock specific dependencies")
        recommendations.append("MOCK_TORCH = os.environ.get('MOCK_TORCH', 'False').lower() == 'true'")
        recommendations.append("MOCK_TRANSFORMERS = os.environ.get('MOCK_TRANSFORMERS', 'False').lower() == 'true'")
        recommendations.append("```")
        recommendations.append("")
    
    if "test_class" in missing_components:
        recommendations.append("## Test Class")
        recommendations.append("Add a properly structured test class with methods for pipeline and from_pretrained testing.")
        recommendations.append("")
    
    # Add regeneration command
    model_name = os.path.basename(test_path).replace("test_", "").replace(".py", "")
    recommendations.append("## Regeneration Command")
    recommendations.append("To regenerate this file with the correct template structure, run:")
    recommendations.append("```bash")
    recommendations.append(f"python fix_manual_models.py --model {model_name} --verify --apply")
    recommendations.append("```")
    
    return "\n".join(recommendations)

def analyze_all_manual_models():
    """Analyze all manually created model test files."""
    results = {}
    
    for model, architecture in MANUAL_MODEL_ARCHITECTURES.items():
        # Check if the model exists in the final_models directory
        model_path = os.path.join(FINAL_MODELS_DIR, f"test_{model}.py")
        if not os.path.exists(model_path):
            model_path = os.path.join(FINAL_MODELS_DIR, f"test_hf_{model}.py")
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found for {model}")
                continue
        
        # Get the reference template
        template_path = get_reference_template(architecture)
        if not os.path.exists(template_path):
            logger.warning(f"Template file not found for {architecture}")
            continue
        
        # Analyze the test file
        analysis = analyze_test_file(model_path, architecture)
        comparison = compare_with_template(model_path, template_path)
        
        # Generate recommendations
        recommendations = generate_template_fix_recommendation(model_path, template_path, architecture)
        
        # Save recommendations
        recommendation_path = os.path.join(OUTPUT_DIR, f"{model}_recommendations.md")
        with open(recommendation_path, 'w') as f:
            f.write(recommendations)
        
        # Combine results
        results[model] = {
            "analysis": analysis,
            "comparison": comparison,
            "recommendation_path": recommendation_path
        }
    
    return results

def generate_summary_report(results):
    """Generate a summary report of the template analysis."""
    summary = []
    
    summary.append("# Template Conformance Analysis Summary")
    summary.append("")
    
    # Add a table of model conformance scores
    summary.append("## Model Conformance Scores")
    summary.append("")
    summary.append("| Model | Architecture | Conformance Score | Syntax Errors | Missing Components |")
    summary.append("|-------|--------------|------------------|---------------|-------------------|")
    
    for model, result in results.items():
        analysis = result.get("analysis", {})
        comparison = result.get("comparison", {})
        
        conformance_score = analysis.get("conformance_score", 0)
        syntax_errors = len(comparison.get("syntax_errors", []))
        missing_components = ", ".join([comp for comp, present in analysis.get("components", {}).items() if not present][:3])
        if len([comp for comp, present in analysis.get("components", {}).items() if not present]) > 3:
            missing_components += ", ..."
        
        architecture = MANUAL_MODEL_ARCHITECTURES.get(model, "unknown")
        
        summary.append(f"| {model} | {architecture} | {conformance_score:.1f}% | {syntax_errors} | {missing_components} |")
    
    summary.append("")
    
    # Add overall recommendation
    summary.append("## Overall Recommendation")
    summary.append("")
    
    # Calculate average conformance score
    avg_score = sum(result.get("analysis", {}).get("conformance_score", 0) for result in results.values()) / len(results) if results else 0
    
    if avg_score < 50:
        summary.append("The manually created model tests have significant template conformance issues.")
        summary.append("It is recommended to regenerate all of these tests using the template system.")
    elif avg_score < 80:
        summary.append("The manually created model tests have moderate template conformance issues.")
        summary.append("Most tests should be regenerated, while some may be fixed manually.")
    else:
        summary.append("The manually created model tests have good template conformance overall.")
        summary.append("Only minor fixes are needed for some tests.")
    
    summary.append("")
    summary.append("To regenerate all tests with proper template conformance, run:")
    summary.append("```bash")
    summary.append("python fix_manual_models.py --verify --apply")
    summary.append("```")
    
    # Save summary report
    summary_path = os.path.join(OUTPUT_DIR, "template_analysis_summary.md")
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary))
    
    return summary_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze template conformance of test files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Analyze all manual models
    logger.info("Analyzing manually created model tests...")
    results = analyze_all_manual_models()
    
    # Generate summary report
    logger.info("Generating summary report...")
    summary_path = generate_summary_report(results)
    
    logger.info(f"Analysis complete. Summary report saved to {summary_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())