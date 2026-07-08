#!/usr/bin/env python
"""
Script to test the improved OpenAI API implementation.
"""

import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add the project root to path
sys.path = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))] + sys.path

try:
    from ipfs_accelerate_py.api_backends.openai_api import openai_api
    print("Successfully imported openai_api")
except ImportError as e:
    print(f"Error importing openai_api: {e}")
    sys.exit(1)

def run_simple_tests():
    """Run some basic tests on the new implementation."""
    print("\n=== Running simple OpenAI API Tests ===")
    
    # Create an instance with no actual API key
    api = openai_api(api_key="test_key", metrics_enabled=True)
    
    # Test model mappings
    chat_model = api._determine_model("chat", "default")
    embedding_model = api._determine_model("embedding", "default")
    speech_model = api._determine_model("speech", "default")
    
    print(f"Default chat model: {chat_model}")
    print(f"Default embedding model: {embedding_model}")
    print(f"Default speech model: {speech_model}")
    
    # Test endpoint handler creation
    endpoint_handler = api.create_openai_api_endpoint_handler()
    print(f"Endpoint handler created: {callable(endpoint_handler)}")
    
    # Test process_messages
    messages = [
        {"role": "user", "content": "Hello"}
    ]
    
    processed = api.process_messages(messages, system_message="You are a helpful assistant")
    print(f"Messages processed: {len(processed)}")
    
    # Test get_metrics with no actual API calls
    metrics = api.get_metrics()
    print("Initial metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Test voice agent creation
    agent_config = api.create_voice_agent(
        system_message="You are a helpful voice assistant",
        voice="nova"
    )
    
    print(f"Voice agent created: {agent_config['success']}")
    if agent_config['success']:
        print(f"Agent voice: {agent_config['agent']['voice']}")
        print(f"Agent model: {agent_config['agent']['model']}")
    
    return {
        "chat_model": chat_model,
        "embedding_model": embedding_model,
        "speech_model": speech_model,
        "endpoint_handler": callable(endpoint_handler),
        "processed_messages": len(processed),
        "metrics": metrics,
        "voice_agent": agent_config["success"]
    }

def run_mock_api_calls():
    """Run tests with mocked API calls."""
    print("\n=== Running Mock API Tests ===")
    
    # Create an instance with no actual API key
    api = openai_api(api_key="test_key", metrics_enabled=True)
    
    results = {}
    
    # Test embedding with mock
    with patch.object(api.client.embeddings, 'create') as mock_embed:
        # Create a mock that mimics the OpenAI client response
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4] 
        mock_response.data[0].index = 0
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 8
        mock_response.usage.total_tokens = 8
        mock_response.model = "text-embedding-3-small"
        mock_embed.return_value = mock_response
        
        embedding = api.embedding(model="text-embedding-3-small", text="Test embedding")
        
        results["embedding"] = {
            "success": isinstance(embedding, dict) and "text" in embedding,
            "dimensions": len(embedding["embedding"]) if isinstance(embedding, dict) and "embedding" in embedding else 0
        }
        
        print(f"Embedding test successful: {results['embedding']['success']}")
    
    # Test chat completion with mock
    with patch.object(api.client.chat.completions, 'create') as mock_chat:
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-123"
        mock_response.object = "chat.completion"
        mock_response.created = 1677825464
        mock_response.model = "gpt-4o"
        
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "This is a test response"
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_choice.index = 0
        
        mock_response.choices = [mock_choice]
        
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_response.usage = mock_usage
        
        mock_chat.return_value = mock_response
        
        completion = api.chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        results["chat_completion"] = {
            "success": isinstance(completion, dict) and "text" in completion,
            "text": completion.get("text", "") if isinstance(completion, dict) else ""
        }
        
        print(f"Chat completion test successful: {results['chat_completion']['success']}")
    
    # Test metrics collection
    metrics = api.get_metrics()
    print(f"Metrics after tests: {json.dumps(metrics, indent=2)}")
    
    return results

if __name__ == "__main__":
    print("Testing improved OpenAI API implementation")
    
    try:
        # Run simple tests
        simple_results = run_simple_tests()
        
        # Run mock API tests
        mock_results = run_mock_api_calls()
        
        # Combined results
        results = {
            "simple_tests": simple_results,
            "mock_api_tests": mock_results,
            "overall_success": all([
                simple_results["endpoint_handler"],
                mock_results["embedding"]["success"],
                mock_results["chat_completion"]["success"]
            ])
        }
        
        print("\n=== Test Summary ===")
        print(f"Overall success: {results['overall_success']}")
        
        # Save results to file
        results_file = os.path.join(os.path.dirname(__file__), "apis", "collected_results", "improved_openai_api_test_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {results_file}")
        
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nTests completed successfully")