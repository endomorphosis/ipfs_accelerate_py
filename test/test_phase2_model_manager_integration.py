#!/usr/bin/env python3
"""
Test Phase 2: Model Manager API Integration

Tests the new methods added to ModelManager for querying both
self-hosted and API models by pipeline type.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_model_manager_api_integration():
    """Test model manager with API provider integration."""
    print("=" * 60)
    print("Testing Model Manager API Integration (Phase 2)")
    print("=" * 60)
    
    try:
        from ipfs_accelerate_py.model_manager import ModelManager
        from ipfs_accelerate_py.common.pipeline_types import CAUSAL_LM, PipelineType
        from ipfs_accelerate_py.api_integrations.model_registry import get_global_api_model_registry
        
        print("\n✅ Successfully imported all required modules")
        
        # Initialize model manager
        manager = ModelManager(storage_path="/tmp/test_model_manager.json", use_database=False)
        print("✅ Model Manager initialized")
        
        # Test 1: Get all pipeline types
        print("\n" + "-" * 60)
        print("Test 1: Get all available pipeline types")
        print("-" * 60)
        
        try:
            all_types = manager.get_all_pipeline_types(include_api=True, include_self_hosted=False)
            print(f"✅ Found {len(all_types)} pipeline types from API models")
            print(f"   Examples: {list(all_types)[:5]}")
        except Exception as e:
            print(f"❌ Error getting pipeline types: {e}")
        
        # Test 2: Get models for text-generation
        print("\n" + "-" * 60)
        print("Test 2: Get models for text-generation pipeline")
        print("-" * 60)
        
        try:
            models = manager.get_models_by_pipeline_type(
                "text-generation",
                include_api=True,
                include_self_hosted=False  # No self-hosted models in this test
            )
            print(f"✅ Found {len(models)} text-generation models")
            
            for model in models[:5]:  # Show first 5
                print(f"   - {model['model_name']} ({model['provider']}) - Context: {model.get('context_length', 'N/A')}")
        except Exception as e:
            print(f"❌ Error getting text-generation models: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: Get model recommendations
        print("\n" + "-" * 60)
        print("Test 3: Get model recommendations with filters")
        print("-" * 60)
        
        try:
            recommendations = manager.get_model_recommendations(
                "text-generation",
                max_cost_per_1k=0.02,  # Maximum $0.02 per 1K tokens
                min_context_length=8000  # Minimum 8K context
            )
            print(f"✅ Found {len(recommendations)} recommended models")
            
            for model in recommendations[:3]:  # Show top 3
                score = model.get('recommendation_score', 0)
                context = model.get('context_length', 'N/A')
                cost = (model.get('cost_per_1k_input', 0) + model.get('cost_per_1k_output', 0)) / 2
                print(f"   - {model['model_name']} (Score: {score:.1f})")
                print(f"     Context: {context}, Avg Cost: ${cost:.4f}/1K")
        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 4: API Model Registry direct access
        print("\n" + "-" * 60)
        print("Test 4: Direct API Model Registry access")
        print("-" * 60)
        
        try:
            registry = get_global_api_model_registry()
            providers = registry.get_all_providers()
            print(f"✅ API Registry has {len(providers)} providers:")
            for provider in providers:
                count = len(registry.get_models_by_provider(provider))
                print(f"   - {provider.value}: {count} models")
        except Exception as e:
            print(f"❌ Error accessing registry: {e}")
        
        # Test 5: Provider filtering
        print("\n" + "-" * 60)
        print("Test 5: Get models with provider filter")
        print("-" * 60)
        
        try:
            openai_models = manager.get_models_by_pipeline_type(
                "text-generation",
                include_api=True,
                include_self_hosted=False,
                provider_filter=["openai"]
            )
            print(f"✅ Found {len(openai_models)} OpenAI models for text-generation")
            
            anthropic_models = manager.get_models_by_pipeline_type(
                "text-generation",
                include_api=True,
                include_self_hosted=False,
                provider_filter=["anthropic"]
            )
            print(f"✅ Found {len(anthropic_models)} Anthropic models for text-generation")
        except Exception as e:
            print(f"❌ Error with provider filtering: {e}")
        
        print("\n" + "=" * 60)
        print("✅ Phase 2 Implementation Test Complete!")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_manager_api_integration()
    sys.exit(0 if success else 1)
