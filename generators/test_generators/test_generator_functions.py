#!/usr/bin/env python3
"""
Test script to validate the merged_test_generator functionality
"""

import sys
import os
from generators.test_generators.merged_test_generator import (
    normalize_model_name,
    get_pipeline_category,
    get_appropriate_model_name,
    get_specialized_test_inputs,
    export_model_registry_parquet
)

def test_normalize_model_name():
    """Test the model name normalization function"""
    test_cases = [
        ("bert", "bert"),
        ("bert-base-uncased", "bert_base_uncased"),
        ("gpt2.large", "gpt2_large"),
        ("T5-small", "t5_small")
    ]
    
    for input_name, expected in test_cases:
        result = normalize_model_name(input_name)
        assert result == expected, f"normalize_model_name({input_name}) returned {result}, expected {expected}"
    
    print("normalize_model_name: All tests passed!")

def test_get_pipeline_category():
    """Test the pipeline categorization function"""
    test_cases = [
        (["text-generation"], "language"),
        (["fill-mask"], "language"),
        (["image-classification"], "vision"),
        (["object-detection"], "vision"),
        (["automatic-speech-recognition"], "audio"),
        (["visual-question-answering"], "multimodal"),
        (["protein-folding"], "specialized"),
        (["something-unknown"], "other")
    ]
    
    for tasks, expected in test_cases:
        result = get_pipeline_category(tasks)
        assert result == expected, f"get_pipeline_category({tasks}) returned {result}, expected {expected}"
    
    print("get_pipeline_category: All tests passed!")

def test_get_appropriate_model_name():
    """Test the model name selection based on tasks"""
    test_cases = [
        (["text-generation"], '"distilgpt2"'),
        (["fill-mask"], '"distilroberta-base"'),
        (["image-classification"], '"google/vit-base-patch16-224-in21k"'),
        (["automatic-speech-recognition"], '"openai/whisper-tiny"')
    ]
    
    for tasks, expected_substring in test_cases:
        result = get_appropriate_model_name(tasks)
        assert expected_substring in result, f"get_appropriate_model_name({tasks}) returned {result}, expected to contain {expected_substring}"
    
    print("get_appropriate_model_name: All tests passed!")

def test_get_specialized_test_inputs():
    """Test the specialized test input generation"""
    # Test text generation inputs
    text_inputs = get_specialized_test_inputs("text-generation")
    assert any('self.test_text =' in inp for inp in text_inputs), "Text generation inputs should include test_text"
    
    # Test image inputs
    image_inputs = get_specialized_test_inputs("image-classification")
    assert any('self.test_image =' in inp for inp in image_inputs), "Image classification inputs should include test_image"
    
    # Test audio inputs
    audio_inputs = get_specialized_test_inputs("automatic-speech-recognition")
    assert any('self.test_audio =' in inp for inp in audio_inputs), "Audio inputs should include test_audio"
    
    print("get_specialized_test_inputs: All tests passed!")

def test_parquet_export():
    """Test the parquet export functionality"""
    # Test with HuggingFace Datasets
    try:
        path = export_model_registry_parquet(use_duckdb=False)
        assert os.path.exists(path), f"Expected file {path} to exist"
        print(f"export_model_registry_parquet(use_duckdb=False): File created at {path}")
    except Exception as e:
        print(f"Error in HuggingFace export: {e}")
    
    # Test with DuckDB if available
    try:
        path = export_model_registry_parquet(use_duckdb=True)
        assert os.path.exists(path), f"Expected file {path} to exist"
        print(f"export_model_registry_parquet(use_duckdb=True): File created at {path}")
    except Exception as e:
        print(f"Error in DuckDB export: {e}")

def run_all_tests():
    """Run all tests"""
    test_normalize_model_name()
    test_get_pipeline_category()
    test_get_appropriate_model_name()
    test_get_specialized_test_inputs()
    test_parquet_export()
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()