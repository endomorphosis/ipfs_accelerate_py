#!/usr/bin/env python3
"""
Master script to systematically improve HuggingFace test coverage over time.
This script:
1. Analyzes current test coverage
2. Identifies missing tests based on model type and pipeline tasks
3. Generates tests in priority order (critical, high, medium, low)
4. Updates implementation status documentation

Run this script periodically to incrementally add more model tests.
"""

import os
import sys
import json
import time
import datetime
import traceback
import argparse
import subprocess
from pathlib import Path

# Priority tiers of models
CRITICAL_MODELS = [
    # Models supporting unique/uncovered pipeline tasks
    "tapas",              # table-question-answering (only model)
    "esm",                # protein-folding (only model)
    "patchtst",           # time-series-prediction (most efficient)
    "informer",           # time-series-prediction (popular forecasting)
    "autoformer",         # time-series-prediction (alternative architecture)
    
    # Popular foundational models
    "resnet",             # image-classification (extremely widely used)
    "gpt_neox",           # text-generation (popular open-source LLM)
    "rwkv",               # text-generation (unique RNN-like architecture)
    
    # Specialized capabilities
    "visual_bert",        # visual-question-answering (popular foundation)
    "instructblip",       # image-to-text & visual-question-answering
    "markuplm",           # document-question-answering & token-classification
    "donut-swin",         # document-question-answering & image-to-text
    "canine",             # token-classification & text-classification
    "big_bird",           # fill-mask & question-answering (long-context)
    
    # Advanced multimodal models
    "video_llava",        # image-to-text & visual-question-answering (video)
    "qwen2_audio",        # automatic-speech-recognition & text-to-audio
    "seamless_m4t_v2",    # translation_XX_to_YY, ASR & text-to-audio
    "siglip",             # image-classification & feature-extraction
    "zoedepth",           # depth-estimation (state-of-the-art monocular)
]

HIGH_PRIORITY_MODELS = [
    # Popular language models
    "gpt_neox_japanese",  # Japanese language model
    "switch_transformers", # Mixture of experts transformer
    "longt5",             # Long document T5
    "rembert",            # Multilingual BERT
    "realm",              # Retrieval augmented language model
    
    # Vision models
    "swinv2",             # Hierarchical vision transformer v2
    "vitdet",             # Detection with vision transformer
    "vit_mae",            # Masked autoencoder ViT
    "vit_msn",            # Masked siamese network
    "vanillanet",         # Vanilla CNN
    
    # Audio models
    "univnet",            # Neural vocoder
    "speech_to_text",     # Popular ASR model
    "unispeech-sat",      # Self-supervised speech
    "audio-spectrogram-transformer", # Audio transformer
    
    # Specialized models
    "oneformer",          # Universal image segmentation
    "time_series_transformer", # Time series model
    "flava",              # Foundation language and vision alignment
    "deit",               # Distilled vision transformer
    "nllb-moe",           # No Language Left Behind
    "megatron-bert",      # Large-scale BERT
]

def load_pipeline_maps():
    """Load model-pipeline mappings from JSON files."""
    try:
        with open('huggingface_model_pipeline_map.json', 'r') as f:
            model_to_pipeline = json.load(f)
        
        with open('huggingface_pipeline_model_map.json', 'r') as f:
            pipeline_to_model = json.load(f)
            
        return model_to_pipeline, pipeline_to_model
    except Exception as e:
        print(f"Error loading pipeline maps: {e}")
        return {}, {}

def load_model_types():
    """Load all model types from JSON file."""
    try:
        with open('huggingface_model_types.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading model types: {e}")
        return []

def get_existing_tests():
    """Get list of existing test files in skills directory."""
    test_files = glob.glob('skills/test_hf_*.py')
    
    # Extract model names from test file names
    existing_tests = []
    for test_file in test_files:
        model_name = test_file.replace('skills/test_hf_', '').replace('.py', '')
        existing_tests.append(model_name)
    
    return existing_tests

def normalize_model_name(name):
    """Normalize model name to match file naming conventions."""
    return name.replace('-', '_').replace('.', '_').lower()

def denormalize_model_name(normalized_name):
    """Try to convert normalized name back to original format.
    This is a best effort approach and may not be perfect."""
    # Special cases
    if normalized_name == "gpt_neo":
        return "gpt_neo"  # Keep as is
    if normalized_name == "wav2vec2_bert":
        return "wav2vec2-bert"
    if normalized_name == "data2vec_audio":
        return "data2vec-audio"
    if normalized_name == "data2vec_vision":
        return "data2vec-vision"
    if normalized_name == "deberta_v2":
        return "deberta-v2"
    if normalized_name == "xlm_roberta":
        return "xlm-roberta"
    if normalized_name == "deepseek_r1":
        return "deepseek-r1"
    if normalized_name == "deepseek_distil":
        return "deepseek-distil"
    if normalized_name == "blenderbot_small":
        return "blenderbot-small"
    if normalized_name == "qwen2_vl":
        return "qwen2_vl"  # Keep as is
    
    # General case: replace underscores with dashes
    return normalized_name.replace('_', '-')

def get_missing_tests(all_models, existing_tests, model_to_pipeline):
    """Identify models missing test implementations and assign priorities."""
    missing_tests = []
    
    for model in all_models:
        normalized_name = normalize_model_name(model)
        
        # Skip if test already exists
        if normalized_name in existing_tests:
            continue
            
        # Get associated pipeline tasks
        pipeline_tasks = model_to_pipeline.get(model, [])
        
        # Determine priority
        if model in CRITICAL_MODELS:
            priority = "CRITICAL"
        elif model in HIGH_PRIORITY_MODELS:
            priority = "HIGH"
        else:
            priority = "MEDIUM"
        
        missing_tests.append({
            "model": model,
            "normalized_name": normalized_name,
            "pipeline_tasks": pipeline_tasks,
            "priority": priority
        })
        
    return missing_tests

def generate_test_template(model_info):
    """
    Generate test file template for a specific model.
    (Implementation details omitted for brevity)
    
    Args:
        model_info (dict): Model information including name and pipeline tasks
        
    Returns:
        str: Generated test file content
    """
    # For brevity, we'll call existing generation scripts
    try:
        if model_info["priority"] == "CRITICAL":
            subprocess.run(["python3", "generate_critical_tests.py", "--model", model_info["model"]])
        else:
            subprocess.run(["python3", "generate_missing_test_files.py", "--model", model_info["model"]])
        return True
    except Exception as e:
        print(f"Error generating test for {model_info['model']}: {e}")
        return False

def analyze_pipeline_coverage(model_to_pipeline, existing_tests):
    """
    Analyze current coverage of pipeline tasks.
    
    Args:
        model_to_pipeline (dict): Mapping of models to pipeline tasks
        existing_tests (list): List of models with implemented tests
        
    Returns:
        dict: Coverage statistics by pipeline task
    """
    pipeline_coverage = {}
    
    # First, identify all pipeline tasks
    all_pipeline_tasks = set()
    for model, tasks in model_to_pipeline.items():
        all_pipeline_tasks.update(tasks)
    
    # Initialize coverage counters for each task
    for task in all_pipeline_tasks:
        pipeline_coverage[task] = {
            "total_models": 0,
            "implemented_models": 0,
            "coverage_percentage": 0,
            "implemented": [],
            "missing": []
        }
    
    # Count models for each pipeline task
    for model, tasks in model_to_pipeline.items():
        normalized_name = normalize_model_name(model)
        is_implemented = normalized_name in existing_tests
        
        for task in tasks:
            pipeline_coverage[task]["total_models"] += 1
            
            if is_implemented:
                pipeline_coverage[task]["implemented_models"] += 1
                pipeline_coverage[task]["implemented"].append(model)
            else:
                pipeline_coverage[task]["missing"].append(model)
    
    # Calculate coverage percentages
    for task, stats in pipeline_coverage.items():
        if stats["total_models"] > 0:
            stats["coverage_percentage"] = (stats["implemented_models"] / stats["total_models"]) * 100
    
    return pipeline_coverage

def update_coverage_documentation(pipeline_coverage, missing_tests, implemented_count, total_count):
    """
    Update the Markdown documentation with current coverage status.
    
    Args:
        pipeline_coverage (dict): Coverage statistics by pipeline task
        missing_tests (list): List of missing test implementations
        implemented_count (int): Number of implemented tests
        total_count (int): Total number of model types
    """
    # Get today's date
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Sort tasks by coverage percentage
    sorted_tasks = sorted(
        pipeline_coverage.items(),
        key=lambda x: x[1]["coverage_percentage"]
    )
    
    # Create markdown content
    markdown = f"""# Hugging Face Model Test Coverage Report

*Generated on: {today}*

## Implementation Status

- **Total Model Types**: {total_count}
- **Implemented Tests**: {implemented_count} ({implemented_count/total_count*100:.1f}%)
- **Remaining to Implement**: {total_count - implemented_count} ({(total_count-implemented_count)/total_count*100:.1f}%)

## Pipeline Task Coverage

| Pipeline Task | Implemented | Total | Coverage | Key Missing Models |
|---------------|-------------|-------|----------|-------------------|
"""
    
    # Add rows for each pipeline task
    for task, stats in sorted_tasks:
        implemented = stats["implemented_models"]
        total = stats["total_models"]
        coverage = stats["coverage_percentage"]
        
        # Get top 3 missing models for this task by priority
        missing_by_priority = []
        for model in stats["missing"]:
            for test in missing_tests:
                if test["model"] == model:
                    missing_by_priority.append((model, test["priority"]))
                    break
        
        # Sort by priority (CRITICAL first, then HIGH, then MEDIUM)
        missing_by_priority.sort(key=lambda x: 0 if x[1] == "CRITICAL" else (1 if x[1] == "HIGH" else 2))
        
        # Take top 3
        top_missing = [m[0] for m in missing_by_priority[:3]]
        missing_str = ", ".join(top_missing) if top_missing else "None"
        
        markdown += f"| {task} | {implemented} | {total} | {coverage:.1f}% | {missing_str} |\n"
    
    # Add next steps section
    markdown += """
## Next Steps

### Critical Models to Implement

The following models should be implemented first as they provide unique capabilities or fill gaps in pipeline coverage:

"""
    
    # Add critical models section
    critical_models = [m for m in missing_tests if m["priority"] == "CRITICAL"]
    for model in critical_models[:10]:  # Show top 10
        tasks = ", ".join(model["pipeline_tasks"])
        markdown += f"- **{model['model']}**: {tasks}\n"
    
    if len(critical_models) > 10:
        markdown += f"- *...and {len(critical_models) - 10} more critical models*\n"
    
    # Add high priority models section
    markdown += """
### High Priority Models

These models are widely used and should be implemented after the critical models:

"""
    
    high_priority = [m for m in missing_tests if m["priority"] == "HIGH"]
    for model in high_priority[:10]:  # Show top 10
        tasks = ", ".join(model["pipeline_tasks"])
        markdown += f"- **{model['model']}**: {tasks}\n"
    
    if len(high_priority) > 10:
        markdown += f"- *...and {len(high_priority) - 10} more high priority models*\n"
    
    # Add implementation plan
    markdown += """
## Implementation Timeline

1. **Phase 1 (Critical Models)**: Focus on models that provide unique capabilities or fill gaps in pipeline coverage
2. **Phase 2 (High Priority Models)**: Implement widely used models across various tasks
3. **Phase 3 (Medium Priority Models)**: Complete coverage for remaining models

To generate tests for the next batch of models, run:

```bash
python3 complete_test_coverage.py --batch 5 --priority critical
```

## Recently Generated Tests

"""
    
    # Try to list recently created test files
    try:
        test_files = os.listdir("skills")
        test_files = [f for f in test_files if f.startswith("test_hf_") and f.endswith(".py")]
        test_files.sort(key=lambda x: os.path.getmtime(os.path.join("skills", x)), reverse=True)
        
        for test_file in test_files[:10]:
            model_name = test_file.replace("test_hf_", "").replace(".py", "")
            original_name = denormalize_model_name(model_name)
            
            # Try to get pipeline tasks for this model
            tasks = []
            for m, t in model_to_pipeline.items():
                if normalize_model_name(m) == model_name:
                    tasks = t
                    break
            
            tasks_str = ", ".join(tasks) if tasks else "N/A"
            modification_time = datetime.datetime.fromtimestamp(
                os.path.getmtime(os.path.join("skills", test_file))
            ).strftime("%Y-%m-%d %H:%M")
            
            markdown += f"- **{original_name}** ({modification_time}): {tasks_str}\n"
    except Exception as e:
        markdown += f"*Error listing recent tests: {e}*\n"
    
    # Write to file
    try:
        with open("huggingface_test_coverage_report.md", "w") as f:
            f.write(markdown)
        print(f"Updated coverage documentation at huggingface_test_coverage_report.md")
    except Exception as e:
        print(f"Error writing coverage report: {e}")

def generate_next_batch(missing_tests, batch_size=5, priority=None):
    """
    Generate test files for the next batch of models.
    
    Args:
        missing_tests (list): List of missing test implementations
        batch_size (int): Number of tests to generate
        priority (str, optional): Filter by priority level
    
    Returns:
        int: Number of tests generated
    """
    # Filter by priority if specified
    if priority:
        priority = priority.upper()
        filtered_tests = [t for t in missing_tests if t["priority"] == priority]
    else:
        # Otherwise, sort by priority
        filtered_tests = sorted(
            missing_tests,
            key=lambda x: 0 if x["priority"] == "CRITICAL" else (1 if x["priority"] == "HIGH" else 2)
        )
    
    print(f"Found {len(filtered_tests)} tests to potentially generate")
    
    # Create output directory if it doesn't exist
    skills_dir = Path("skills")
    if not skills_dir.exists():
        skills_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {skills_dir}")
    
    # Generate tests
    generated_count = 0
    failed_models = []
    
    for model_info in filtered_tests:
        if generated_count >= batch_size:
            break
            
        model = model_info["model"]
        normalized_name = model_info["normalized_name"]
        
        # Skip if test already exists
        test_file_path = skills_dir / f"test_hf_{normalized_name}.py"
        if test_file_path.exists():
            print(f"Test file already exists for {model}, skipping")
            continue
        
        print(f"Generating test for {model} ({model_info['priority']})...")
        
        # Generate test based on priority
        success = False
        try:
            if model_info["priority"] == "CRITICAL":
                # For critical models, use dedicated script with specialized templates
                result = subprocess.run(
                    ["python3", "generate_critical_tests.py", "--single-model", model],
                    capture_output=True,
                    text=True
                )
                success = result.returncode == 0 and test_file_path.exists()
            else:
                # For other models, use the standard generator
                result = subprocess.run(
                    ["python3", "generate_missing_test_files.py", "--single-model", model],
                    capture_output=True,
                    text=True
                )
                success = result.returncode == 0 and test_file_path.exists()
        except Exception as e:
            print(f"Error generating test for {model}: {e}")
            failed_models.append((model, str(e)))
            continue
        
        if success:
            generated_count += 1
            print(f"Successfully generated test for {model}")
        else:
            print(f"Failed to generate test for {model}")
            failed_models.append((model, "Unknown error"))
    
    # Print summary
    print(f"\nGenerated {generated_count} test files")
    
    if failed_models:
        print("\nFailed to generate tests for:")
        for model, error in failed_models:
            print(f"- {model}: {error}")
    
    return generated_count

def main():
    """Main function to analyze and improve test coverage"""
    parser = argparse.ArgumentParser(description="Analyze and improve HuggingFace test coverage")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze current coverage without generating tests")
    parser.add_argument("--batch", type=int, default=5, help="Number of tests to generate in this batch")
    parser.add_argument("--priority", choices=["critical", "high", "medium", "all"], default="all", 
                        help="Priority level of tests to generate")
    
    args = parser.parse_args()
    
    print(f"Starting coverage analysis at {datetime.datetime.now().isoformat()}")
    
    # Load data
    all_models = load_model_types()
    model_to_pipeline, pipeline_to_model = load_pipeline_maps()
    
    # Verify data loaded successfully
    if not all_models or not model_to_pipeline:
        print("Failed to load required data files")
        sys.exit(1)
    
    print(f"Loaded {len(all_models)} model types and {len(model_to_pipeline)} pipeline mappings")
    
    # Get existing tests
    import glob
    existing_tests = []
    test_files = glob.glob('skills/test_hf_*.py')
    for test_file in test_files:
        model_name = os.path.basename(test_file).replace('test_hf_', '').replace('.py', '')
        existing_tests.append(model_name)
    
    print(f"Found {len(existing_tests)} existing test implementations")
    
    # Identify missing tests
    missing_tests = get_missing_tests(all_models, existing_tests, model_to_pipeline)
    print(f"Identified {len(missing_tests)} missing test implementations")
    
    # Analyze pipeline coverage
    pipeline_coverage = analyze_pipeline_coverage(model_to_pipeline, existing_tests)
    
    # Print coverage summary
    print("\nPipeline Coverage Summary:")
    for task, stats in sorted(pipeline_coverage.items(), key=lambda x: x[1]["coverage_percentage"]):
        print(f"- {task}: {stats['implemented_models']}/{stats['total_models']} " +
              f"({stats['coverage_percentage']:.1f}%)")
    
    # Update documentation
    update_coverage_documentation(
        pipeline_coverage,
        missing_tests,
        len(existing_tests),
        len(all_models)
    )
    
    # Generate tests if not in analyze-only mode
    if not args.analyze_only:
        print(f"\nGenerating next batch of {args.batch} tests (priority: {args.priority})...")
        priority_filter = None if args.priority == "all" else args.priority.upper()
        generate_next_batch(missing_tests, args.batch, priority_filter)
    
    print("\nComplete!")

if __name__ == "__main__":
    main()