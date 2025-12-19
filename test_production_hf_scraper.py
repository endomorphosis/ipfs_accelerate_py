#!/usr/bin/env python3
"""
Comprehensive tests for Production HuggingFace Scraper.

Tests the ProductionHFScraper class and its integration with:
- EnhancedModelScraper base class
- HuggingFace Hub API (in mock mode)
- Async operations
- Metadata extraction
"""

import sys
import os
import tempfile
import asyncio
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))


def test_production_scraper_initialization():
    """Test ProductionHFScraper initialization."""
    print("Testing Production HF Scraper initialization...")
    
    try:
        from production_hf_scraper import ProductionHFScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = ProductionHFScraper(data_dir=tmp_dir, api_token="test_token")
            
            # Verify initialization
            assert scraper.data_dir.exists(), "Data directory not created"
            assert scraper.api_token == "test_token", "API token not set"
            assert scraper.base_url == "https://huggingface.co/api", "Base URL incorrect"
            assert scraper.rate_limit_delay == 0.1, "Rate limit delay not set"
            
            # Verify inheritance from EnhancedModelScraper
            assert hasattr(scraper, 'save_to_parquet'), "Missing base class method"
            assert hasattr(scraper, 'build_search_index'), "Missing base class method"
            assert hasattr(scraper, 'search_models'), "Missing base class method"
            
            print("✅ Production HF Scraper initialized successfully")
            return True
            
    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_extraction():
    """Test model metadata extraction."""
    print("Testing metadata extraction...")
    
    try:
        from production_hf_scraper import ProductionHFScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = ProductionHFScraper(data_dir=tmp_dir)
            
            # Mock HuggingFace API response
            mock_model_data = {
                "id": "gpt2",
                "author": "openai",
                "modelId": "gpt2",
                "downloads": 5000000,
                "likes": 1500,
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "tags": ["gpt", "text-generation", "pytorch"],
                "languages": ["en"],
                "datasets": ["openwebtext"],
                "lastModified": "2024-01-01T00:00:00Z",
                "createdAt": "2023-01-01T00:00:00Z",
                "private": False,
                "gated": False,
                "disabled": False,
                "siblings": [{"rfilename": "model.bin"}, {"rfilename": "config.json"}],
                "config": {
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                    "vocab_size": 50257
                }
            }
            
            # Extract metadata
            model_record = scraper.extract_model_metadata(mock_model_data)
            
            # Verify extracted fields
            assert model_record.model_id == "gpt2", "Model ID incorrect"
            assert model_record.author == "openai", "Author incorrect"
            assert model_record.downloads == 5000000, "Downloads incorrect"
            assert model_record.likes == 1500, "Likes incorrect"
            assert model_record.pipeline_tag == "text-generation", "Pipeline tag incorrect"
            assert model_record.library_name == "transformers", "Library name incorrect"
            assert len(model_record.tags) > 0, "No tags extracted"
            assert model_record.model_size_mb > 0, "Model size not calculated"
            assert len(model_record.embedding_vector) == 384, "Embedding dimension incorrect"
            
            # Verify computed fields
            assert model_record.inference_time_ms > 0, "Inference time not calculated"
            assert model_record.throughput_tokens_per_sec > 0, "Throughput not calculated"
            assert model_record.memory_usage_mb > 0, "Memory usage not calculated"
            assert model_record.popularity_score > 0, "Popularity score not calculated"
            
            print(f"✅ Metadata extraction successful for model: {model_record.model_id}")
            return True
            
    except Exception as e:
        print(f"❌ Metadata extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_model_fetching():
    """Test async model fetching (mock mode)."""
    print("Testing async model fetching...")
    
    try:
        from production_hf_scraper import ProductionHFScraper
        
        async def run_test():
            with tempfile.TemporaryDirectory() as tmp_dir:
                scraper = ProductionHFScraper(data_dir=tmp_dir)
                
                # Note: In real test, this would fetch from API
                # For now, we test that the method exists and is async
                assert hasattr(scraper, 'get_all_models'), "Missing get_all_models method"
                assert asyncio.iscoroutinefunction(scraper.get_all_models), "get_all_models is not async"
                
                return True
        
        result = asyncio.run(run_test())
        if result:
            print("✅ Async model fetching test successful")
        return result
            
    except Exception as e:
        print(f"❌ Async model fetching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_base_class():
    """Test integration with EnhancedModelScraper base class."""
    print("Testing integration with base class...")
    
    try:
        from production_hf_scraper import ProductionHFScraper
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = ProductionHFScraper(data_dir=tmp_dir)
            
            # Verify inheritance
            assert isinstance(scraper, EnhancedModelScraper), "Not inheriting from EnhancedModelScraper"
            
            # Verify base class methods are accessible
            assert callable(getattr(scraper, 'save_to_parquet', None)), "save_to_parquet not accessible"
            assert callable(getattr(scraper, 'load_from_parquet', None)), "load_from_parquet not accessible"
            assert callable(getattr(scraper, 'build_search_index', None)), "build_search_index not accessible"
            assert callable(getattr(scraper, 'search_models', None)), "search_models not accessible"
            assert callable(getattr(scraper, 'get_statistics', None)), "get_statistics not accessible"
            
            # Verify base class attributes are accessible
            assert hasattr(scraper, 'data_dir'), "Missing data_dir attribute"
            assert hasattr(scraper, 'parquet_path'), "Missing parquet_path attribute"
            assert hasattr(scraper, 'stats'), "Missing stats attribute"
            
            print("✅ Integration with base class successful")
            return True
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_production_scraper_without_token():
    """Test ProductionHFScraper without API token."""
    print("Testing Production HF Scraper without token...")
    
    try:
        from production_hf_scraper import ProductionHFScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize without token
            scraper = ProductionHFScraper(data_dir=tmp_dir, api_token=None)
            
            # Verify scraper still initializes
            assert scraper.api_token is None, "API token should be None"
            assert scraper.data_dir.exists(), "Data directory not created"
            
            print("✅ Production HF Scraper works without token")
            return True
            
    except Exception as e:
        print(f"❌ No-token test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_record_structure():
    """Test ModelRecord dataclass structure."""
    print("Testing ModelRecord structure...")
    
    try:
        from enhanced_model_scraper import ModelRecord
        
        # Create a test model record
        model = ModelRecord(
            model_id="test-model",
            model_name="Test Model",
            author="test-author",
            downloads=1000,
            likes=50,
            last_modified="2024-01-01T00:00:00Z",
            created_at="2023-01-01T00:00:00Z",
            library_name="transformers",
            pipeline_tag="text-generation",
            task_type="text-generation",
            architecture="GPT",
            model_size_mb=100.0,
            model_file_count=5,
            private=False,
            gated=False,
            disabled=False,
            tags=["test", "gpt"],
            languages=["en"],
            datasets=["test-dataset"],
            metrics={"accuracy": 0.95},
            hardware_requirements={"min_ram_gb": 4},
            performance_benchmarks={"tokens_per_second": 100.0},
            memory_usage_mb=150.0,
            inference_time_ms=50.0,
            throughput_tokens_per_sec=100.0,
            gpu_memory_mb=200.0,
            cpu_cores_recommended=4,
            supports_quantization=True,
            supports_onnx=True,
            supports_tensorrt=False,
            license="mit",
            description="Test model description",
            embedding_vector=[0.0] * 384,
            popularity_score=75.0,
            efficiency_score=80.0,
            compatibility_score=90.0,
            scraped_at="2024-01-01T00:00:00Z"
        )
        
        # Verify all fields are accessible
        assert model.model_id == "test-model"
        assert model.model_name == "Test Model"
        assert model.author == "test-author"
        assert model.model_size_mb == 100.0
        assert len(model.embedding_vector) == 384
        
        print("✅ ModelRecord structure test successful")
        return True
        
    except Exception as e:
        print(f"❌ ModelRecord structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Production HF Scraper tests."""
    print("🧪 Testing Production HuggingFace Scraper")
    print("=" * 60)
    
    tests = [
        test_production_scraper_initialization,
        test_metadata_extraction,
        test_async_model_fetching,
        test_integration_with_base_class,
        test_production_scraper_without_token,
        test_model_record_structure,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
            print()
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All Production HF Scraper tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
