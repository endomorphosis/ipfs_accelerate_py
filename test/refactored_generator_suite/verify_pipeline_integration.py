#!/usr/bin/env python3
"""
Verification script for the pipeline templates integration.

This script verifies that the generated model implementations contain the correct
specialized pipeline code for vision-text and audio models.
"""

import os
import sys
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_file_exists(file_path):
    """Check if a file exists."""
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
    return True

def check_file_content(file_path, patterns, min_size=10000):
    """
    Check if a file contains the expected patterns.
    
    Args:
        file_path: Path to the file to check
        patterns: List of regex patterns to search for
        min_size: Minimum expected file size in bytes
        
    Returns:
        Dict with results
    """
    if not check_file_exists(file_path):
        return {"success": False, "file": file_path, "error": "File not found"}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Check file size
        if len(content) < min_size:
            return {"success": False, "file": file_path, "error": f"File too small: {len(content)} bytes"}
            
        # Check for patterns
        missing_patterns = []
        for pattern in patterns:
            if not re.search(pattern, content):
                missing_patterns.append(pattern)
                
        if missing_patterns:
            return {
                "success": False, 
                "file": file_path, 
                "error": f"Missing patterns: {missing_patterns}"
            }
            
        return {"success": True, "file": file_path, "size": len(content)}
        
    except Exception as e:
        return {"success": False, "file": file_path, "error": str(e)}

def verify_vision_text_implementation(file_path):
    """Verify a vision-text model implementation."""
    patterns = [
        r"vision-text\s+pipeline",
        r"Vision-Text\s+pipeline",
        r"image_text_matching",
        r"visual_question_answering",
        r"image_captioning",
        r"def\s+resize_image",
        r"def\s+encode_image_base64"
    ]
    
    result = check_file_content(file_path, patterns)
    if result["success"]:
        logger.info(f"✅ Vision-text implementation verified: {file_path}")
    else:
        logger.error(f"❌ Vision-text implementation verification failed: {result['error']}")
        
    return result

def verify_audio_implementation(file_path):
    """Verify an audio model implementation."""
    patterns = [
        r"audio\s+pipeline",
        r"Audio\s+pipeline",
        r"speech_recognition",
        r"audio_classification",
        r"text_to_speech",
        r"def\s+encode_audio_base64",
        r"def\s+save_audio_to_file"
    ]
    
    result = check_file_content(file_path, patterns)
    if result["success"]:
        logger.info(f"✅ Audio implementation verified: {file_path}")
    else:
        logger.error(f"❌ Audio implementation verification failed: {result['error']}")
        
    return result

def main():
    """Main function."""
    logger.info("Starting pipeline integration verification...")
    
    # Path to generated test models
    output_dir = "generated_test_models"
    if not os.path.exists(output_dir):
        logger.error(f"Output directory not found: {output_dir}")
        return
    
    # Verify CLIP implementation
    clip_file = os.path.join(output_dir, "hf_clip.py")
    clip_result = verify_vision_text_implementation(clip_file)
    
    # Verify Whisper implementation
    whisper_file = os.path.join(output_dir, "hf_whisper.py")
    whisper_result = verify_audio_implementation(whisper_file)
    
    # Summary
    if clip_result["success"] and whisper_result["success"]:
        logger.info("✅ All implementations verified successfully! The pipeline templates integration works correctly.")
    else:
        logger.error("❌ Some implementations failed verification. Check the logs for details.")
    
    # Write verification report
    report_path = os.path.join(output_dir, "verification_report.md")
    with open(report_path, 'w') as f:
        f.write("# Pipeline Integration Verification Report\n\n")
        f.write("## Vision-Text Implementation (CLIP)\n\n")
        if clip_result["success"]:
            f.write("✅ **SUCCESS**: Vision-text implementation verified\n")
            f.write(f"- File size: {clip_result['size']} bytes\n")
        else:
            f.write("❌ **FAILED**: Vision-text implementation verification failed\n")
            f.write(f"- Error: {clip_result.get('error', 'Unknown error')}\n")
            
        f.write("\n## Audio Implementation (Whisper)\n\n")
        if whisper_result["success"]:
            f.write("✅ **SUCCESS**: Audio implementation verified\n")
            f.write(f"- File size: {whisper_result['size']} bytes\n")
        else:
            f.write("❌ **FAILED**: Audio implementation verification failed\n")
            f.write(f"- Error: {whisper_result.get('error', 'Unknown error')}\n")
            
        f.write("\n## Conclusion\n\n")
        if clip_result["success"] and whisper_result["success"]:
            f.write("✅ **All implementations verified successfully!** The pipeline templates integration works correctly.\n")
        else:
            f.write("❌ **Some implementations failed verification.** The pipeline templates integration may have issues.\n")
            
    logger.info(f"Verification report written to {report_path}")
            
if __name__ == "__main__":
    main()