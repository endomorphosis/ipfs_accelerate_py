#!/usr/bin/env python3
"""
Comprehensive tests for Enhanced Model Scraper architecture.

Tests the EnhancedModelScraper class and its core functionality including:
- Initialization and configuration
- Mock data generation
- Parquet storage operations
- K-NN search index building
- Model search functionality
- Statistics gathering
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import json

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))


def test_enhanced_scraper_initialization():
    """Test EnhancedModelScraper initialization."""
    print("Testing Enhanced Model Scraper initialization...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Verify initialization
            assert scraper.data_dir.exists(), "Data directory not created"
            assert scraper.parquet_path == scraper.data_dir / "hf_models.parquet"
            assert scraper.metadata_path == scraper.data_dir / "scraper_metadata.json"
            assert scraper.index_path == scraper.data_dir / "search_index.pkl"
            assert scraper.embeddings_path == scraper.data_dir / "model_embeddings.npy"
            
            # Verify statistics initialized
            assert 'total_models_scraped' in scraper.stats
            assert scraper.stats['total_models_scraped'] == 0
            
            print("✅ Enhanced Model Scraper initialized successfully")
            return True
            
    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_data_generation():
    """Test mock data generation."""
    print("Testing mock data generation...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper, ModelRecord
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Generate small dataset
            models = scraper.create_mock_comprehensive_dataset(size=50)
            
            # Verify data
            assert len(models) == 50, f"Expected 50 models, got {len(models)}"
            assert all(isinstance(model, ModelRecord) for model in models), "Not all items are ModelRecord"
            
            # Verify model fields
            model = models[0]
            assert hasattr(model, 'model_id'), "Missing model_id field"
            assert hasattr(model, 'model_name'), "Missing model_name field"
            assert hasattr(model, 'architecture'), "Missing architecture field"
            assert hasattr(model, 'task_type'), "Missing task_type field"
            assert hasattr(model, 'model_size_mb'), "Missing model_size_mb field"
            assert hasattr(model, 'embedding_vector'), "Missing embedding_vector field"
            
            # Verify embedding dimensions
            assert len(model.embedding_vector) == 384, f"Expected 384-dim embedding, got {len(model.embedding_vector)}"
            
            print(f"✅ Mock data generation successful: {len(models)} models created")
            return True
            
    except Exception as e:
        print(f"❌ Mock data generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parquet_storage():
    """Test Parquet storage operations."""
    print("Testing Parquet storage...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Generate and save data
            models = scraper.create_mock_comprehensive_dataset(size=20)
            scraper.save_to_parquet(models)
            
            # Verify file created
            assert scraper.parquet_path.exists(), "Parquet file not created"
            
            # Load and verify data
            df = scraper.load_from_parquet()
            assert df is not None, "Failed to load Parquet data"
            assert len(df) == 20, f"Expected 20 models, got {len(df)}"
            
            # Verify columns
            expected_columns = ['model_id', 'model_name', 'architecture', 'task_type', 
                              'model_size_mb', 'downloads', 'likes']
            for col in expected_columns:
                assert col in df.columns, f"Missing column: {col}"
            
            print(f"✅ Parquet storage test successful: {len(df)} models stored and loaded")
            return True
            
    except Exception as e:
        print(f"❌ Parquet storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_index_building():
    """Test K-NN search index building."""
    print("Testing K-NN search index building...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Generate, save, and load data
            models = scraper.create_mock_comprehensive_dataset(size=30)
            scraper.save_to_parquet(models)
            df = scraper.load_from_parquet()
            
            # Build search index
            scraper.build_search_index(df)
            
            # Verify index components
            assert scraper.vectorizer is not None, "Vectorizer not initialized"
            assert scraper.embeddings_matrix is not None, "Embeddings matrix not created"
            assert len(scraper.model_index) > 0, "Model index is empty"
            
            # Verify index files created
            assert scraper.index_path.exists(), "Index file not created"
            assert scraper.embeddings_path.exists(), "Embeddings file not created"
            
            print(f"✅ Search index built successfully: {len(scraper.model_index)} models indexed")
            return True
            
    except Exception as e:
        print(f"❌ Search index building test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_search():
    """Test model search functionality."""
    print("Testing model search...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Setup: Generate data and build index
            models = scraper.create_mock_comprehensive_dataset(size=50)
            scraper.save_to_parquet(models)
            df = scraper.load_from_parquet()
            scraper.build_search_index(df)
            
            # Test search queries
            test_queries = [
                ("GPT text generation", 5),
                ("BERT classification", 3),
                ("vision transformer", 5)
            ]
            
            for query, top_k in test_queries:
                results = scraper.search_models(query, top_k=top_k)
                
                assert results is not None, f"Search returned None for query: {query}"
                assert isinstance(results, list), f"Search results not a list for query: {query}"
                assert len(results) <= top_k, f"Too many results for query: {query}"
                
                # Verify result structure
                if results:
                    result = results[0]
                    assert 'model_id' in result, "Missing model_id in result"
                    assert 'model_name' in result, "Missing model_name in result"
                    assert 'similarity_score' in result, "Missing similarity_score in result"
                    assert 'architecture' in result, "Missing architecture in result"
            
            print(f"✅ Model search test successful: tested {len(test_queries)} queries")
            return True
            
    except Exception as e:
        print(f"❌ Model search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scrape_all_models():
    """Test complete scraping workflow in mock mode."""
    print("Testing complete scraping workflow...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Run scraping in mock mode
            results = scraper.scrape_all_models(limit=100, mock_mode=True)
            
            # Verify results
            assert 'total_models' in results, "Missing total_models in results"
            assert results['total_models'] == 100, f"Expected 100 models, got {results['total_models']}"
            assert 'scrape_duration' in results, "Missing scrape_duration in results"
            assert 'models_per_second' in results, "Missing models_per_second in results"
            assert 'search_index_built' in results, "Missing search_index_built in results"
            
            # Verify files created
            assert scraper.parquet_path.exists(), "Parquet file not created"
            assert scraper.metadata_path.exists(), "Metadata file not created"
            
            # Verify statistics updated
            assert scraper.stats['total_models_scraped'] == 100
            
            print(f"✅ Complete scraping workflow successful: {results['total_models']} models")
            return True
            
    except Exception as e:
        print(f"❌ Complete scraping workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistics():
    """Test statistics gathering."""
    print("Testing statistics...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Setup: Scrape some models
            results = scraper.scrape_all_models(limit=40, mock_mode=True)
            
            # Get statistics
            stats = scraper.get_statistics()
            
            # Verify statistics
            assert 'total_models' in stats, "Missing total_models in stats"
            assert 'total_authors' in stats, "Missing total_authors in stats"
            assert 'total_architectures' in stats, "Missing total_architectures in stats"
            assert 'total_tasks' in stats, "Missing total_tasks in stats"
            assert 'avg_model_size_mb' in stats, "Missing avg_model_size_mb in stats"
            
            assert stats['total_models'] == 40, f"Expected 40 models, got {stats['total_models']}"
            assert stats['total_authors'] > 0, "No authors found"
            assert stats['total_architectures'] > 0, "No architectures found"
            
            print(f"✅ Statistics test successful: {stats['total_models']} models analyzed")
            return True
            
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_persistence():
    """Test metadata saving and loading."""
    print("Testing metadata persistence...")
    
    try:
        from enhanced_model_scraper import EnhancedModelScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = EnhancedModelScraper(data_dir=tmp_dir)
            
            # Run scraping to generate metadata
            results = scraper.scrape_all_models(limit=10, mock_mode=True)
            
            # Verify metadata file exists
            assert scraper.metadata_path.exists(), "Metadata file not created"
            
            # Load metadata
            loaded_metadata = scraper.load_metadata()
            
            # Verify metadata content
            assert 'total_models_scraped' in loaded_metadata
            assert loaded_metadata['total_models_scraped'] == 10
            
            print(f"✅ Metadata persistence test successful")
            return True
            
    except Exception as e:
        print(f"❌ Metadata persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Enhanced Model Scraper tests."""
    print("🧪 Testing Enhanced Model Scraper Architecture")
    print("=" * 60)
    
    tests = [
        test_enhanced_scraper_initialization,
        test_mock_data_generation,
        test_parquet_storage,
        test_search_index_building,
        test_model_search,
        test_scrape_all_models,
        test_statistics,
        test_metadata_persistence,
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
        print("🎉 All Enhanced Model Scraper tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
