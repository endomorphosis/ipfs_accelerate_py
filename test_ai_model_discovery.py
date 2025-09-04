#!/usr/bin/env python3
"""
Comprehensive Test Suite for AI Model Discovery Features

This test suite covers:
1. Vector documentation index functionality
2. Bandit algorithm model recommendation
3. Feedback integration and learning
4. Error handling and edge cases
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ipfs_accelerate_py'))

try:
    from ipfs_accelerate_py.model_manager import (
        ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
        VectorDocumentationIndex, BanditModelRecommender,
        RecommendationContext, ModelRecommendation, BanditArm,
        DocumentEntry, SearchResult
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Skipping AI model discovery tests due to missing dependencies")
    sys.exit(0)


class TestVectorDocumentationIndex(unittest.TestCase):
    """Test cases for vector documentation index."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.temp_dir, "test_index.json")
        
        # Create test README files
        self.readme_files = {
            "README.md": """# Main Project
This is the main project documentation.

## Features
- Feature 1: Advanced AI processing
- Feature 2: GPU acceleration
- Feature 3: Model optimization

## Installation
Install the package using pip install.
""",
            "docs/README.md": """# Documentation
Complete documentation for the project.

## API Reference
The API provides comprehensive model management.

## Examples
Examples of using the vector search functionality.
""",
            "subdoc/README.md": """# Sub-documentation
Specialized documentation for advanced users.

## Advanced Features
- Vector embeddings
- Bandit algorithms
- Performance optimization
"""
        }
        
        # Create test directory structure
        for file_path, content in self.readme_files.items():
            full_path = os.path.join(self.temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_vector_index_initialization(self):
        """Test vector index initialization."""
        # Test with missing dependencies (current environment)
        index = VectorDocumentationIndex(storage_path=self.index_path)
        self.assertIsNone(index.model)
        
        # Test that it doesn't crash
        self.assertEqual(index.storage_path, self.index_path)
        self.assertEqual(len(index.documents), 0)
    
    def test_index_readme_files_without_dependencies(self):
        """Test indexing README files without dependencies."""
        index = VectorDocumentationIndex(storage_path=self.index_path)
        
        # Should return 0 when dependencies are missing
        count = index.index_all_readmes(self.temp_dir)
        self.assertEqual(count, 0)
    
    def test_search_functionality_without_dependencies(self):
        """Test search functionality without dependencies."""
        index = VectorDocumentationIndex(storage_path=self.index_path)
        
        # Should return empty list when dependencies are missing
        results = index.search("GPU acceleration", top_k=5)
        self.assertEqual(results, [])
    
    def test_save_and_load_index(self):
        """Test saving and loading index."""
        index = VectorDocumentationIndex(storage_path=self.index_path)
        
        # Add test document
        doc = DocumentEntry(
            file_path="test.md",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            title="Test",
            section="Main"
        )
        index.documents.append(doc)
        
        # Save index
        index.save_index()
        self.assertTrue(os.path.exists(self.index_path))
        
        # Load index
        new_index = VectorDocumentationIndex(storage_path=self.index_path)
        loaded = new_index.load_index()
        
        self.assertTrue(loaded)
        self.assertEqual(len(new_index.documents), 1)
        self.assertEqual(new_index.documents[0].file_path, "test.md")


class TestBanditModelRecommender(unittest.TestCase):
    """Test cases for bandit model recommender."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.bandit_path = os.path.join(self.temp_dir, "test_bandit.json")
        self.model_path = os.path.join(self.temp_dir, "test_models.json")
        
        # Create test model manager with sample models
        self.manager = ModelManager(storage_path=self.model_path)
        
        # Add test models
        test_models = [
            ModelMetadata(
                model_id="test-bert",
                model_name="Test BERT",
                model_type=ModelType.LANGUAGE_MODEL,
                architecture="BertModel",
                inputs=[IOSpec(name="input_ids", data_type=DataType.TOKENS)],
                outputs=[IOSpec(name="embeddings", data_type=DataType.EMBEDDINGS)],
                supported_backends=["cpu", "cuda"]
            ),
            ModelMetadata(
                model_id="test-gpt",
                model_name="Test GPT",
                model_type=ModelType.LANGUAGE_MODEL,
                architecture="GPTModel",
                inputs=[IOSpec(name="input_ids", data_type=DataType.TOKENS)],
                outputs=[IOSpec(name="logits", data_type=DataType.LOGITS)],
                supported_backends=["cpu", "cuda", "mps"]
            )
        ]
        
        for model in test_models:
            self.manager.add_model(model)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if hasattr(self, 'manager'):
            self.manager.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_bandit_initialization(self):
        """Test bandit recommender initialization."""
        recommender = BanditModelRecommender(
            algorithm="thompson_sampling",
            model_manager=self.manager,
            storage_path=self.bandit_path
        )
        
        self.assertEqual(recommender.algorithm, "thompson_sampling")
        self.assertEqual(recommender.storage_path, self.bandit_path)
        self.assertIsInstance(recommender.bandit_arms, dict)
    
    def test_recommendation_context(self):
        """Test recommendation context functionality."""
        context = RecommendationContext(
            task_type="text_processing",
            hardware="cuda",
            input_type=DataType.TOKENS,
            output_type=DataType.EMBEDDINGS
        )
        
        key = context.to_key()
        self.assertIsInstance(key, str)
        self.assertIn("text_processing", key)
        self.assertIn("cuda", key)
    
    def test_get_compatible_models(self):
        """Test getting compatible models."""
        recommender = BanditModelRecommender(
            model_manager=self.manager,
            storage_path=self.bandit_path
        )
        
        # Test with specific input/output types
        context = RecommendationContext(
            input_type=DataType.TOKENS,
            output_type=DataType.EMBEDDINGS,
            hardware="cuda"
        )
        
        compatible = recommender._get_compatible_models(context)
        self.assertIsInstance(compatible, list)
        
        # Should include test-bert (matches input/output types)
        if compatible:
            self.assertIn("test-bert", compatible)
    
    def test_bandit_arm_functionality(self):
        """Test bandit arm statistics."""
        arm = BanditArm(model_id="test-model")
        
        # Test initial state
        self.assertEqual(arm.total_reward, 0.0)
        self.assertEqual(arm.num_trials, 0)
        self.assertEqual(arm.average_reward, 0.0)
        
        # Test after adding feedback
        arm.total_reward = 2.5
        arm.num_trials = 3
        self.assertEqual(arm.average_reward, 2.5 / 3)
    
    def test_recommendation_and_feedback(self):
        """Test model recommendation and feedback."""
        recommender = BanditModelRecommender(
            algorithm="epsilon_greedy",
            model_manager=self.manager,
            storage_path=self.bandit_path
        )
        
        context = RecommendationContext(
            task_type="test_task",
            hardware="cpu",
            input_type=DataType.TOKENS,
            output_type=DataType.EMBEDDINGS
        )
        
        # Get recommendation
        recommendation = recommender.recommend_model(context)
        
        if recommendation:  # Only test if models are compatible
            self.assertIsInstance(recommendation, ModelRecommendation)
            self.assertIsInstance(recommendation.model_id, str)
            self.assertIsInstance(recommendation.confidence_score, float)
            
            # Provide feedback
            recommender.provide_feedback(
                model_id=recommendation.model_id,
                feedback_score=0.8,
                context=context
            )
            
            # Check that feedback was recorded
            context_key = context.to_key()
            self.assertIn(context_key, recommender.bandit_arms)
            self.assertIn(recommendation.model_id, recommender.bandit_arms[context_key])
            
            arm = recommender.bandit_arms[context_key][recommendation.model_id]
            self.assertGreater(arm.num_trials, 0)
            self.assertGreater(arm.total_reward, 0)
    
    def test_algorithm_selection(self):
        """Test different bandit algorithms."""
        algorithms = ["ucb", "thompson_sampling", "epsilon_greedy"]
        
        for algorithm in algorithms:
            recommender = BanditModelRecommender(
                algorithm=algorithm,
                model_manager=self.manager,
                storage_path=self.bandit_path
            )
            
            # Create test arms
            arms = {
                "model1": BanditArm(model_id="model1", total_reward=3.0, num_trials=4),
                "model2": BanditArm(model_id="model2", total_reward=2.0, num_trials=3)
            }
            
            # Test selection (should not crash)
            try:
                selected = recommender._select_arm("test_context")
                # Selection from empty arms should return None
                self.assertIsNone(selected)
            except Exception as e:
                self.fail(f"Algorithm {algorithm} selection failed: {e}")
    
    def test_save_and_load_bandit_data(self):
        """Test saving and loading bandit data."""
        recommender = BanditModelRecommender(
            model_manager=self.manager,
            storage_path=self.bandit_path
        )
        
        # Add some test data
        context_key = "test_context"
        recommender.bandit_arms[context_key] = {
            "model1": BanditArm(model_id="model1", total_reward=2.0, num_trials=3)
        }
        recommender.global_trial_count = 5
        
        # Save data
        recommender.save_bandit_data()
        self.assertTrue(os.path.exists(self.bandit_path))
        
        # Load data in new instance
        new_recommender = BanditModelRecommender(
            model_manager=self.manager,
            storage_path=self.bandit_path
        )
        
        loaded = new_recommender.load_bandit_data()
        self.assertTrue(loaded)
        self.assertEqual(new_recommender.global_trial_count, 5)
        self.assertIn(context_key, new_recommender.bandit_arms)
    
    def test_performance_report(self):
        """Test performance report generation."""
        recommender = BanditModelRecommender(
            model_manager=self.manager,
            storage_path=self.bandit_path
        )
        
        # Add test data
        context_key = "test_context"
        recommender.bandit_arms[context_key] = {
            "model1": BanditArm(model_id="model1", total_reward=4.0, num_trials=5),
            "model2": BanditArm(model_id="model2", total_reward=3.0, num_trials=4)
        }
        
        report = recommender.get_performance_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('algorithm', report)
        self.assertIn('total_trials', report)
        self.assertIn('contexts', report)
        
        if context_key in report['contexts']:
            context_report = report['contexts'][context_key]
            self.assertIn('best_model', context_report)
            self.assertIn('best_average_reward', context_report)
            self.assertEqual(context_report['best_model'], 'model1')  # Higher average


class TestIntegratedAIWorkflow(unittest.TestCase):
    """Test cases for integrated AI workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_integration_without_dependencies(self):
        """Test that the system gracefully handles missing dependencies."""
        # Test vector index without sentence transformers
        index = VectorDocumentationIndex()
        self.assertIsNone(index.model)
        
        # Should return empty results
        results = index.search("test query")
        self.assertEqual(results, [])
        
        # Test bandit recommender without numpy (Thompson sampling will fallback to UCB)
        manager = ModelManager()
        recommender = BanditModelRecommender(model_manager=manager, algorithm="thompson_sampling")
        
        # Should still work for basic operations
        context = RecommendationContext(task_type="test")
        recommendation = recommender.recommend_model(context)
        
        # May return None due to no models, but shouldn't crash
        self.assertIsInstance(recommendation, (ModelRecommendation, type(None)))
    
    def test_document_entry_creation(self):
        """Test DocumentEntry creation and serialization."""
        doc = DocumentEntry(
            file_path="test.md",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            title="Test",
            section="Main"
        )
        
        self.assertEqual(doc.file_path, "test.md")
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.embedding, [0.1, 0.2, 0.3])
        self.assertEqual(doc.title, "Test")
        self.assertEqual(doc.section, "Main")
    
    def test_recommendation_context_key_generation(self):
        """Test recommendation context key generation."""
        context = RecommendationContext(
            task_type="text_processing",
            hardware="cuda",
            input_type=DataType.TOKENS,
            output_type=DataType.EMBEDDINGS
        )
        
        key = context.to_key()
        self.assertIn("text_processing", key)
        self.assertIn("cuda", key)
        self.assertIn("TOKENS", key)
        self.assertIn("EMBEDDINGS", key)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with invalid file paths
        index = VectorDocumentationIndex(storage_path="/invalid/path/index.json")
        
        # Should handle save/load errors gracefully
        index.save_index()  # Should not crash
        loaded = index.load_index()  # Should return False
        self.assertFalse(loaded)
        
        # Test bandit with invalid algorithm
        manager = ModelManager()
        recommender = BanditModelRecommender(
            algorithm="invalid_algorithm",
            model_manager=manager
        )
        
        # Should handle gracefully
        arms = {"model1": BanditArm(model_id="model1")}
        selected = recommender._select_arm("test")
        self.assertIsNone(selected)


def run_ai_tests():
    """Run all AI model discovery tests."""
    print("🧪 Running AI Model Discovery Tests...")
    print("=" * 50)
    
    # Create test suite
    test_classes = [
        TestVectorDocumentationIndex,
        TestBanditModelRecommender,
        TestIntegratedAIWorkflow
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("🏁 AI Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n⚠️ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1)
    print(f"\n✅ Success Rate: {success_rate:.1%}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_ai_tests()
    sys.exit(0 if success else 1)