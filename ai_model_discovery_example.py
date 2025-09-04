#!/usr/bin/env python3
"""
AI-Powered Model Discovery and Recommendation Example

This example demonstrates the advanced AI features of the Model Manager:
1. Vector index for semantic documentation search
2. Bandit algorithm for intelligent model recommendations
3. Feedback-driven continuous improvement

Features showcased:
- Semantic search through all README files in the repository
- Context-aware model recommendations using multi-armed bandit algorithms
- User feedback integration for improving recommendations over time
- Performance analytics and recommendation explanations
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ipfs_accelerate_py'))

try:
    from ipfs_accelerate_py.model_manager import (
        ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
        VectorDocumentationIndex, BanditModelRecommender,
        RecommendationContext, ModelRecommendation
    )
    from model_manager_integration import ModelManagerIntegration
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed and paths are correct")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_vector_documentation_search():
    """Demonstrate vector-based documentation search."""
    print("\n" + "="*70)
    print("🔍 VECTOR DOCUMENTATION SEARCH DEMO")
    print("="*70)
    
    # Initialize vector documentation index
    doc_index = VectorDocumentationIndex(storage_path="demo_doc_index.json")
    
    # Load existing index or create new one
    if not doc_index.load_index():
        print("📚 Creating vector index of all README files...")
        indexed_count = doc_index.index_all_readmes()
        print(f"✅ Successfully indexed {indexed_count} document sections")
    else:
        print(f"📖 Loaded existing index with {len(doc_index.documents)} document sections")
    
    # Demonstrate semantic search
    search_queries = [
        "How to optimize CUDA performance?",
        "WebGPU acceleration setup",
        "model compatibility and hardware requirements",
        "installation and troubleshooting",
        "testing and validation procedures"
    ]
    
    for query in search_queries:
        print(f"\n🔎 Searching for: '{query}'")
        results = doc_index.search(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. 📄 {result.document.file_path}")
                print(f"     📂 Section: {result.document.section}")
                print(f"     🎯 Similarity: {result.similarity_score:.3f}")
                print(f"     📝 Preview: {result.document.content[:100]}...")
                print()
        else:
            print("     ❌ No relevant documentation found")
    
    return doc_index


def demo_bandit_model_recommendation():
    """Demonstrate bandit algorithm for model recommendation."""
    print("\n" + "="*70)
    print("🎰 BANDIT MODEL RECOMMENDATION DEMO")
    print("="*70)
    
    # Initialize model manager and populate with some models
    with ModelManager(storage_path="demo_models.json") as manager:
        # Add some example models if not already present
        if not manager.list_models():
            print("📝 Adding example models to the registry...")
            
            # Add various model types for testing
            models = [
                ModelMetadata(
                    model_id="bert-base-uncased",
                    model_name="BERT Base Uncased",
                    model_type=ModelType.LANGUAGE_MODEL,
                    architecture="BertModel",
                    inputs=[IOSpec(name="input_ids", data_type=DataType.TOKENS)],
                    outputs=[IOSpec(name="embeddings", data_type=DataType.EMBEDDINGS)],
                    supported_backends=["cpu", "cuda"],
                    tags=["nlp", "embeddings", "classification"]
                ),
                ModelMetadata(
                    model_id="gpt2-medium",
                    model_name="GPT-2 Medium",
                    model_type=ModelType.LANGUAGE_MODEL,
                    architecture="GPT2LMHeadModel",
                    inputs=[IOSpec(name="input_ids", data_type=DataType.TOKENS)],
                    outputs=[IOSpec(name="logits", data_type=DataType.LOGITS)],
                    supported_backends=["cpu", "cuda", "mps"],
                    tags=["nlp", "generation", "text"]
                ),
                ModelMetadata(
                    model_id="distilbert-base-uncased",
                    model_name="DistilBERT Base Uncased",
                    model_type=ModelType.LANGUAGE_MODEL,
                    architecture="DistilBertModel",
                    inputs=[IOSpec(name="input_ids", data_type=DataType.TOKENS)],
                    outputs=[IOSpec(name="embeddings", data_type=DataType.EMBEDDINGS)],
                    supported_backends=["cpu", "cuda", "webgpu"],
                    tags=["nlp", "lightweight", "embeddings"]
                ),
                ModelMetadata(
                    model_id="resnet50",
                    model_name="ResNet-50",
                    model_type=ModelType.VISION_MODEL,
                    architecture="ResNet",
                    inputs=[IOSpec(name="pixel_values", data_type=DataType.IMAGE)],
                    outputs=[IOSpec(name="logits", data_type=DataType.LOGITS)],
                    supported_backends=["cpu", "cuda", "mps"],
                    tags=["vision", "classification", "cnn"]
                )
            ]
            
            for model in models:
                manager.add_model(model)
            
            print(f"✅ Added {len(models)} example models")
        
        # Initialize bandit recommender
        recommender = BanditModelRecommender(
            algorithm="thompson_sampling",
            model_manager=manager,
            storage_path="demo_bandit.json"
        )
        
        print(f"🤖 Initialized bandit recommender using {recommender.algorithm}")
        
        # Demonstrate recommendations for different contexts
        contexts = [
            RecommendationContext(
                task_type="text_classification",
                hardware="cuda",
                input_type=DataType.TOKENS,
                output_type=DataType.LOGITS,
                user_id="demo_user_1"
            ),
            RecommendationContext(
                task_type="text_embeddings",
                hardware="cpu",
                input_type=DataType.TOKENS,
                output_type=DataType.EMBEDDINGS,
                user_id="demo_user_2"
            ),
            RecommendationContext(
                task_type="image_classification",
                hardware="mps",
                input_type=DataType.IMAGE,
                output_type=DataType.LOGITS,
                user_id="demo_user_3"
            )
        ]
        
        # Simulate recommendation and feedback loop
        print("\n🔄 Simulating recommendation and feedback loop...")
        
        for round_num in range(1, 4):
            print(f"\n--- Round {round_num} ---")
            
            for i, context in enumerate(contexts, 1):
                print(f"\n🎯 Context {i}: {context.task_type} on {context.hardware}")
                
                # Get recommendation
                recommendation = recommender.recommend_model(context)
                
                if recommendation:
                    print(f"  💡 Recommended: {recommendation.model_id}")
                    print(f"  🎯 Confidence: {recommendation.confidence_score:.3f}")
                    print(f"  📝 Reasoning: {recommendation.reasoning}")
                    
                    # Simulate user feedback (random for demo)
                    import random
                    feedback_score = random.uniform(0.4, 0.9)  # Simulate realistic feedback
                    
                    # Provide feedback
                    recommender.provide_feedback(
                        model_id=recommendation.model_id,
                        feedback_score=feedback_score,
                        context=context
                    )
                    
                    print(f"  📊 User feedback: {feedback_score:.3f}")
                else:
                    print("  ❌ No suitable model found")
        
        # Show performance report
        print("\n📈 BANDIT PERFORMANCE REPORT")
        print("-" * 40)
        
        report = recommender.get_performance_report()
        print(f"Algorithm: {report['algorithm']}")
        print(f"Total trials: {report['total_trials']}")
        
        for context_key, context_data in report['contexts'].items():
            print(f"\nContext: {context_key}")
            print(f"  Total arms: {context_data['total_arms']}")
            print(f"  Total trials: {context_data['total_trials']}")
            print(f"  Best model: {context_data['best_model']}")
            print(f"  Best avg reward: {context_data['best_average_reward']:.3f}")
            
            print("  Model performance:")
            for model_id, arm_data in context_data['arms'].items():
                print(f"    {model_id}: avg={arm_data['average_reward']:.3f}, "
                      f"trials={arm_data['num_trials']}")
        
        return recommender


def demo_integrated_ai_workflow():
    """Demonstrate integrated AI workflow combining documentation search and model recommendation."""
    print("\n" + "="*70)
    print("🧠 INTEGRATED AI WORKFLOW DEMO")
    print("="*70)
    
    # Scenario: User wants to find information about WebGPU and get model recommendations
    print("📋 Scenario: User needs WebGPU-compatible models for text processing")
    
    # 1. Use documentation search to find relevant information
    print("\n1️⃣ SEARCHING DOCUMENTATION")
    doc_index = VectorDocumentationIndex()
    if doc_index.load_index():
        results = doc_index.search("WebGPU text processing models", top_k=2)
        
        if results:
            print("📚 Found relevant documentation:")
            for result in results:
                print(f"  📄 {result.document.file_path} (similarity: {result.similarity_score:.3f})")
                print(f"  📝 {result.document.content[:150]}...")
        else:
            print("📚 No specific documentation found, using general knowledge")
    
    # 2. Get model recommendations based on requirements
    print("\n2️⃣ GETTING MODEL RECOMMENDATIONS")
    
    with ModelManager() as manager:
        recommender = BanditModelRecommender(model_manager=manager)
        
        # Context based on WebGPU requirement
        context = RecommendationContext(
            task_type="text_processing",
            hardware="webgpu",
            input_type=DataType.TOKENS,
            output_type=DataType.EMBEDDINGS,
            performance_requirements={"latency": "<200ms", "memory": "<2GB"}
        )
        
        recommendation = recommender.recommend_model(context)
        
        if recommendation:
            print(f"🎯 AI Recommendation: {recommendation.model_id}")
            print(f"📊 Confidence: {recommendation.confidence_score:.3f}")
            print(f"💭 Reasoning: {recommendation.reasoning}")
            
            # Get additional model details
            try:
                model_details = manager.get_model(recommendation.model_id)
                if model_details:
                    print(f"📋 Model Details:")
                    print(f"  Name: {model_details.model_name}")
                    print(f"  Type: {model_details.model_type.value}")
                    print(f"  Architecture: {model_details.architecture}")
                    print(f"  Supported backends: {model_details.supported_backends}")
                    print(f"  Tags: {', '.join(model_details.tags)}")
            except Exception as e:
                print(f"⚠️ Could not retrieve model details: {e}")
        else:
            print("❌ No suitable model recommendation available")
    
    # 3. Simulate feedback loop
    print("\n3️⃣ FEEDBACK COLLECTION")
    print("User tests the recommended model...")
    time.sleep(1)  # Simulate user testing time
    
    # Simulate positive feedback
    feedback_score = 0.85
    print(f"✅ User provides positive feedback: {feedback_score}")
    
    if recommendation:
        recommender.provide_feedback(
            model_id=recommendation.model_id,
            feedback_score=feedback_score,
            context=context
        )
        print("📈 Feedback integrated into recommendation system")


def main():
    """Run all AI model discovery demos."""
    print("🚀 AI-POWERED MODEL DISCOVERY AND RECOMMENDATION SYSTEM")
    print("=" * 70)
    print("This demo showcases advanced AI features for model discovery:")
    print("- 🔍 Vector-based documentation search")
    print("- 🎰 Multi-armed bandit model recommendations") 
    print("- 🔄 Feedback-driven continuous improvement")
    print("- 🧠 Integrated AI workflow")
    
    try:
        # Demo 1: Vector documentation search
        doc_index = demo_vector_documentation_search()
        
        # Demo 2: Bandit model recommendation
        recommender = demo_bandit_model_recommendation()
        
        # Demo 3: Integrated workflow
        demo_integrated_ai_workflow()
        
        print("\n" + "="*70)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📊 Summary:")
        print(f"- 📚 Indexed {len(doc_index.documents) if doc_index.documents else 0} documentation sections")
        print(f"- 🎯 Tested {recommender.global_trial_count} model recommendations")
        print(f"- 🧠 Demonstrated integrated AI workflow")
        print("\n💡 The AI system is learning from feedback and improving recommendations!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("\nTroubleshooting tips:")
        print("- Ensure all dependencies are installed (sentence-transformers, numpy)")
        print("- Check that model manager files are accessible")
        print("- Verify Python path configuration")


if __name__ == "__main__":
    main()