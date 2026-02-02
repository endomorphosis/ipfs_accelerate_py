#!/usr/bin/env python3
"""
Bulk conversion script for converting existing test_hf_*.py files to improved pytest format.

This script automates the conversion of legacy HuggingFace model tests to the
improved pytest-compatible format with performance monitoring and regression detection.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


class TestConverter:
    """Converts legacy test files to improved format."""
    
    def __init__(self, template_path: str = "test/common/test_template_improved.py"):
        """Initialize the converter.
        
        Args:
            template_path: Path to the test template file
        """
        self.template_path = Path(template_path)
        self.template_content = self._load_template()
    
    def _load_template(self) -> str:
        """Load the test template."""
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")
        
        with open(self.template_path, 'r') as f:
            return f.read()
    
    def extract_model_info(self, test_file: Path) -> Optional[Dict[str, str]]:
        """Extract model information from existing test file.
        
        Args:
            test_file: Path to existing test file
            
        Returns:
            Dictionary with model info or None if parsing fails
        """
        try:
            with open(test_file, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {test_file}: {e}")
            return None
        
        # Extract model ID from filename
        filename = test_file.stem  # e.g., "test_hf_bert"
        model_name = filename.replace("test_hf_", "")  # e.g., "bert"
        
        # Try to find model ID in the file content
        model_id_patterns = [
            r'MODEL_ID\s*=\s*["\']([^"\']+)["\']',
            r'model_id\s*=\s*["\']([^"\']+)["\']',
            r'from_pretrained\(["\']([^"\']+)["\']',
        ]
        
        model_id = None
        for pattern in model_id_patterns:
            match = re.search(pattern, content)
            if match:
                model_id = match.group(1)
                break
        
        # If no model ID found, construct a default one
        if not model_id:
            # Common patterns for model IDs
            common_models = {
                "bert": "bert-base-uncased",
                "gpt2": "gpt2",
                "roberta": "roberta-base",
                "distilbert": "distilbert-base-uncased",
                "albert": "albert-base-v2",
                "t5": "t5-base",
                "bart": "facebook/bart-base",
                "clip": "openai/clip-vit-base-patch32",
                "vit": "google/vit-base-patch16-224",
                "whisper": "openai/whisper-base",
                "llama": "meta-llama/Llama-2-7b-hf",
            }
            
            model_id = common_models.get(model_name, f"{model_name}-base")
        
        # Detect model type/task
        task_type = self._detect_task_type(content, model_name)
        
        # Detect model category
        category = self._detect_model_category(model_name, content)
        
        return {
            "model_name": model_name.upper(),
            "model_id": model_id,
            "task_type": task_type,
            "category": category,
            "filename": filename,
        }
    
    def _detect_task_type(self, content: str, model_name: str) -> str:
        """Detect the primary task type for the model."""
        # Check content for task mentions
        task_keywords = {
            "fill-mask": ["fill-mask", "masked", "mlm"],
            "text-generation": ["text-generation", "causal", "lm"],
            "text-classification": ["text-classification", "classification"],
            "token-classification": ["token-classification", "ner"],
            "question-answering": ["question-answering", "qa"],
            "translation": ["translation", "seq2seq"],
            "summarization": ["summarization"],
            "image-classification": ["image-classification"],
            "object-detection": ["object-detection"],
            "image-to-text": ["image-to-text", "captioning"],
            "automatic-speech-recognition": ["speech-recognition", "asr", "whisper"],
            "zero-shot-classification": ["zero-shot"],
        }
        
        content_lower = content.lower()
        for task, keywords in task_keywords.items():
            if any(kw in content_lower for kw in keywords):
                return task
        
        # Default based on model name
        default_tasks = {
            "bert": "fill-mask",
            "gpt": "text-generation",
            "t5": "translation",
            "bart": "summarization",
            "roberta": "fill-mask",
            "clip": "zero-shot-classification",
            "vit": "image-classification",
            "whisper": "automatic-speech-recognition",
        }
        
        for key, task in default_tasks.items():
            if key in model_name.lower():
                return task
        
        return "feature-extraction"
    
    def _detect_model_category(self, model_name: str, content: str) -> str:
        """Detect model category (text, vision, audio, multimodal)."""
        model_lower = model_name.lower()
        content_lower = content.lower()
        
        # Check for explicit category markers
        if "vision" in content_lower or "image" in content_lower:
            return "vision"
        if "audio" in content_lower or "speech" in content_lower:
            return "audio"
        if "multimodal" in content_lower or ("image" in content_lower and "text" in content_lower):
            return "multimodal"
        
        # Check model name patterns
        vision_keywords = ["vit", "resnet", "efficientnet", "swin", "deit", "convnext"]
        audio_keywords = ["whisper", "wav2vec", "hubert", "speecht5"]
        multimodal_keywords = ["clip", "blip", "llava", "fuyu"]
        
        if any(kw in model_lower for kw in vision_keywords):
            return "vision"
        if any(kw in model_lower for kw in audio_keywords):
            return "audio"
        if any(kw in model_lower for kw in multimodal_keywords):
            return "multimodal"
        
        return "text"
    
    def generate_test(self, model_info: Dict[str, str]) -> str:
        """Generate improved test from template.
        
        Args:
            model_info: Dictionary with model information
            
        Returns:
            Generated test content
        """
        # Start with template
        content = self.template_content
        
        # Replace placeholders
        content = content.replace("{MODEL_ID}", model_info["model_id"])
        content = content.replace("{MODEL_NAME}", model_info["model_name"])
        content = content.replace("{TASK_TYPE}", model_info["task_type"])
        
        # Add category marker
        category_marker = f"@pytest.mark.{model_info['category']}"
        
        # Add model_test marker to all test classes
        content = content.replace("@pytest.mark.model", f"@pytest.mark.model_test\n@pytest.mark.model")
        
        return content
    
    def convert_file(self, input_file: Path, output_dir: Path, overwrite: bool = False) -> bool:
        """Convert a single test file.
        
        Args:
            input_file: Path to input test file
            output_dir: Directory for output file
            overwrite: Whether to overwrite existing files
            
        Returns:
            True if conversion successful
        """
        # Extract model info
        model_info = self.extract_model_info(input_file)
        if not model_info:
            print(f"⚠️  Failed to extract model info from {input_file.name}")
            return False
        
        # Generate output filename
        output_file = output_dir / f"{model_info['filename']}_improved.py"
        
        # Check if output already exists
        if output_file.exists() and not overwrite:
            print(f"⏭️  Skipping {input_file.name} (output exists)")
            return False
        
        # Generate test content
        try:
            test_content = self.generate_test(model_info)
        except Exception as e:
            print(f"❌ Failed to generate test for {input_file.name}: {e}")
            return False
        
        # Write output file
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(test_content)
            print(f"✅ Converted {input_file.name} → {output_file.name}")
            return True
        except Exception as e:
            print(f"❌ Failed to write {output_file}: {e}")
            return False
    
    def convert_batch(self, input_dir: Path, output_dir: Path, 
                     pattern: str = "test_hf_*.py", 
                     limit: Optional[int] = None,
                     overwrite: bool = False) -> Tuple[int, int]:
        """Convert a batch of test files.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory for output files
            pattern: Glob pattern for input files
            limit: Maximum number of files to convert (None = all)
            overwrite: Whether to overwrite existing files
            
        Returns:
            Tuple of (successful_count, total_count)
        """
        # Find input files
        input_files = sorted(input_dir.glob(pattern))
        
        if limit:
            input_files = input_files[:limit]
        
        print(f"\n{'='*70}")
        print(f"Bulk Test Conversion")
        print(f"{'='*70}")
        print(f"Input directory:  {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Pattern:          {pattern}")
        print(f"Files found:      {len(input_files)}")
        if limit:
            print(f"Limit:            {limit}")
        print(f"{'='*70}\n")
        
        # Convert files
        successful = 0
        for i, input_file in enumerate(input_files, 1):
            print(f"[{i}/{len(input_files)}] ", end="")
            if self.convert_file(input_file, output_dir, overwrite):
                successful += 1
        
        print(f"\n{'='*70}")
        print(f"Conversion complete: {successful}/{len(input_files)} successful")
        print(f"{'='*70}\n")
        
        return successful, len(input_files)


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert legacy HuggingFace model tests to improved pytest format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="test",
        help="Input directory containing test files (default: test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test/improved",
        help="Output directory for converted tests (default: test/improved)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="test_hf_*.py",
        help="Glob pattern for input files (default: test_hf_*.py)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to convert (default: all)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="test/common/test_template_improved.py",
        help="Path to test template (default: test/common/test_template_improved.py)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    template_path = Path(args.template)
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    # Validate template
    if not template_path.exists():
        print(f"Error: Template not found: {template_path}")
        return 1
    
    # Create converter
    converter = TestConverter(template_path=str(template_path))
    
    # Convert batch
    successful, total = converter.convert_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=args.pattern,
        limit=args.limit,
        overwrite=args.overwrite
    )
    
    # Return appropriate exit code
    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())
