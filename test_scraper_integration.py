#!/usr/bin/env python3
"""
Integration tests for Scraper Architecture.

Tests the integration between:
- EnhancedModelScraper and HuggingFaceHubScanner
- ProductionHFScraper and HuggingFaceHubScanner
- Scraper workflow scripts (run_complete_scraping.py)
- ModelManager integration
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))


def test_scraper_hub_scanner_integration():
    """Test integration between scrapers and HuggingFaceHubScanner."""
    print("Testing scraper and HuggingFaceHubScanner integration...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Verify HuggingFaceHubScanner is initialized (if available)
            if scraper.scanner is not None:
                # Scanner is available
                assert hasattr(scraper.scanner, 'scan_all_models'), "Scanner missing scan_all_models method"
                assert hasattr(scraper.scanner, 'model_cache'), "Scanner missing model_cache"
                print("✅ HuggingFaceHubScanner integration available")
            else:
                # Scanner not available (mock mode)
                print("⚠️  HuggingFaceHubScanner not available (using mock mode)")
            
            # Verify ModelManager is initialized (if available)
            if scraper.model_manager is not None:
                assert hasattr(scraper.model_manager, 'add_model'), "ModelManager missing add_model method"
                print("✅ ModelManager integration available")
            else:
                print("⚠️  ModelManager not available")
            
            return True
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_production_scraper_inheritance():
    """Test that ProductionHFScraper properly extends EnhancedModelScraper."""
    print("Testing ProductionHFScraper inheritance...")
    
    try:
        from production_hf_scraper import ProductionHFScraper
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            prod_scraper = ProductionHFScraper(data_dir=tmp_dir)
            
            # Verify inheritance
            assert isinstance(prod_scraper, EnhancedModelScraper), "ProductionHFScraper doesn't inherit from EnhancedModelScraper"
            
            # Verify base class methods work
            assert callable(prod_scraper.save_to_parquet), "save_to_parquet not callable"
            assert callable(prod_scraper.load_from_parquet), "load_from_parquet not callable"
            assert callable(prod_scraper.build_search_index), "build_search_index not callable"
            assert callable(prod_scraper.search_models), "search_models not callable"
            
            # Verify production-specific methods
            assert callable(prod_scraper.extract_model_metadata), "extract_model_metadata not callable"
            assert hasattr(prod_scraper, 'scrape_production_models'), "Missing scrape_production_models method"
            
            print("✅ ProductionHFScraper inheritance test successful")
            return True
            
    except Exception as e:
        print(f"❌ Inheritance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scraper_workflow_components():
    """Test that all scraper workflow components are accessible."""
    print("Testing scraper workflow components...")
    
    try:
        # Test imports
        from enhanced_model_scraper import EnhancedModelScraper, ModelRecord
        from production_hf_scraper import ProductionHFScraper
        
        # Verify classes are accessible
        assert EnhancedModelScraper is not None, "EnhancedModelScraper not accessible"
        assert ProductionHFScraper is not None, "ProductionHFScraper not accessible"
        assert ModelRecord is not None, "ModelRecord not accessible"
        
        print("✅ All scraper workflow components accessible")
        return True
        
    except Exception as e:
        print(f"❌ Workflow components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_mock_scraping():
    """Test end-to-end scraping workflow in mock mode."""
    print("Testing end-to-end mock scraping...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Run complete workflow
            results = scraper.scrape_all_models(limit=25, mock_mode=True)
            
            # Verify workflow completed
            assert results['total_models'] == 25, f"Expected 25 models, got {results['total_models']}"
            assert results['search_index_built'], "Search index not built"
            
            # Verify files created
            assert scraper.parquet_path.exists(), "Parquet file not created"
            assert scraper.metadata_path.exists(), "Metadata file not created"
            
            # Test search functionality
            search_results = scraper.search_models("GPT", top_k=5)
            assert len(search_results) > 0, "No search results found"
            
            # Get statistics
            stats = scraper.get_statistics()
            assert stats['total_models'] == 25, "Statistics don't match"
            
            print(f"✅ End-to-end mock scraping successful: {results['total_models']} models")
            return True
            
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scraper_data_consistency():
    """Test data consistency across scraper operations."""
    print("Testing data consistency...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Generate data
            models = scraper.create_mock_comprehensive_dataset(size=30)
            original_count = len(models)
            
            # Save to Parquet
            scraper.save_to_parquet(models)
            
            # Load back
            df = scraper.load_from_parquet()
            loaded_count = len(df)
            
            # Verify consistency
            assert original_count == loaded_count, f"Data count mismatch: {original_count} vs {loaded_count}"
            
            # Verify model IDs are preserved
            original_ids = set(model.model_id for model in models)
            loaded_ids = set(df['model_id'].values)
            assert original_ids == loaded_ids, "Model IDs not preserved"
            
            print(f"✅ Data consistency test successful: {loaded_count} models")
            return True
            
    except Exception as e:
        print(f"❌ Data consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scraper_search_relevance():
    """Test search relevance and ranking."""
    print("Testing search relevance...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Setup: Create and index data
            results = scraper.scrape_all_models(limit=50, mock_mode=True)
            
            # Test specific queries
            gpt_results = scraper.search_models("GPT text generation", top_k=5)
            bert_results = scraper.search_models("BERT classification", top_k=5)
            
            # Verify results have relevance scores
            for result in gpt_results:
                assert 'similarity_score' in result, "Missing similarity score"
                assert 0 <= result['similarity_score'] <= 1, "Invalid similarity score"
            
            # Verify results are ranked (descending similarity)
            if len(gpt_results) > 1:
                for i in range(len(gpt_results) - 1):
                    assert gpt_results[i]['similarity_score'] >= gpt_results[i+1]['similarity_score'], \
                        "Results not properly ranked"
            
            print(f"✅ Search relevance test successful")
            return True
            
    except Exception as e:
        print(f"❌ Search relevance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scraper_error_handling():
    """Test scraper error handling."""
    print("Testing error handling...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Test loading non-existent Parquet file
            df = scraper.load_from_parquet()
            assert df is None, "Should return None for non-existent file"
            
            # Test loading non-existent metadata
            metadata = scraper.load_metadata()
            assert metadata == {}, "Should return empty dict for non-existent metadata"
            
            # Test search without index
            results = scraper.search_models("test query", top_k=5)
            assert results == [] or results is not None, "Should handle missing index gracefully"
            
            print("✅ Error handling test successful")
            return True
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_record_serialization():
    """Test ModelRecord serialization to/from Parquet."""
    print("Testing ModelRecord serialization...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper, ModelRecord
        import json
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Create model records
            models = scraper.create_mock_comprehensive_dataset(size=10)
            
            # Save and load
            scraper.save_to_parquet(models)
            df = scraper.load_from_parquet()
            
            # Verify complex types are properly serialized/deserialized
            first_row = df.iloc[0]
            
            # Verify lists
            assert isinstance(first_row['tags'], list), "Tags not deserialized as list"
            assert isinstance(first_row['languages'], list), "Languages not deserialized as list"
            
            # Verify dicts
            assert isinstance(first_row['hardware_requirements'], dict), "Hardware requirements not deserialized as dict"
            assert isinstance(first_row['performance_benchmarks'], dict), "Performance benchmarks not deserialized as dict"
            
            # Verify embedding vector
            assert isinstance(first_row['embedding_vector'], list), "Embedding vector not deserialized as list"
            assert len(first_row['embedding_vector']) == 384, "Embedding vector dimension incorrect"
            
            print("✅ ModelRecord serialization test successful")
            return True
            
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all scraper integration tests."""
    print("🧪 Testing Scraper Architecture Integration")
    print("=" * 60)
    
    tests = [
        test_scraper_hub_scanner_integration,
        test_production_scraper_inheritance,
        test_scraper_workflow_components,
        test_end_to_end_mock_scraping,
        test_scraper_data_consistency,
        test_scraper_search_relevance,
        test_scraper_error_handling,
        test_model_record_serialization,
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
        print("🎉 All scraper integration tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
