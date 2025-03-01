#!/usr/bin/env python
import os
import sys
import json
import time
import itertools
from datetime import datetime
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ipfs_accelerate_py'))

# Import the groq implementation
from api_backends import groq as groq_module
from api_backends.groq import ALL_MODELS, CHAT_MODELS, VISION_MODELS, AUDIO_MODELS

class ExploreGroqModels:
    """Explore and discover potential unlisted Groq models"""
    
    def __init__(self):
        # Use provided API key
        self.api_key = "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"
            
        # Initialize Groq client
        self.groq_client = groq_module(resources={}, metadata={"groq_api_key": self.api_key})
        
        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "known_models": list(ALL_MODELS.keys()),
            "discovered_models": [],
            "model_details": {},
            "performance_metrics": {}
        }
        
        # Base model names and patterns to try
        self.base_models = [
            "llama3", "llama-3", "llama-3.1", "llama-3.2", "llama-3.3",
            "llama-70b", "llama-8b", "llama-guard",
            "mixtral", "gemma", "gemma-2", "gemma2",
            "qwen", "qwen-1.5", "qwen-2", "qwen-2.5",
            "mistral", "mistral-7b", "mistral-8x7b", "mistral-medium", "mistral-small", "mistral-large",
            "deepseek", "whisper"
        ]
        
        self.sizes = [
            "tiny", "mini", "small", "medium", "large", "xl", "xxl", 
            "1b", "2b", "3b", "7b", "8b", "11b", "13b", "32b", "70b", "90b", "128b"
        ]
        
        self.types = [
            "base", "instruct", "chat", "it", "vision", "code", "coder",
            "audio", "multilingual", "sft", "rlhf", "dpo"
        ]
        
        self.suffixes = [
            "", "-v1", "-v2", "-v3", "-latest", "-preview", "-fast", "-turbo",
            "-8k", "-16k", "-32k", "-64k", "-100k", "-128k", "-long",
            "-en", "-fr", "-de", "-es", "-zh", "-ja", "-ko",
            "-specdec", "-instant", "-versatile"
        ]
    
    def check_model(self, model_name):
        """Check if a model exists by making a minimal API call"""
        print(f"Checking model: {model_name}")
        
        try:
            # Make a minimal request
            response = self.groq_client.chat(
                model_name=model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=0.7
            )
            
            # If we get here, the model exists
            print(f"  ✓ Model exists: {model_name}")
            
            model_info = {
                "status": "available",
                "in_known_list": model_name in ALL_MODELS,
                "example_response": response.get("text", ""),
                "model_reported": response.get("model", model_name)
            }
            
            self.results["model_details"][model_name] = model_info
            
            if model_name not in ALL_MODELS:
                self.results["discovered_models"].append(model_name)
                
            return True
            
        except Exception as e:
            error = str(e)
            if "Model not found" in error or "Model 'nonexistent-model' not found" in error:
                print(f"  ✗ Model not found: {model_name}")
                return False
            elif "API key" in error or "Authentication" in error:
                print(f"  ⚠ Authentication error: {error}")
                return None  # Signal authentication issue
            else:
                print(f"  ⚠ Other error: {error}")
                return False
    
    def test_model_performance(self, model_name):
        """Test a specific model and return performance metrics"""
        print(f"\n--- Performance Testing {model_name} ---")
        
        # Basic prompt for testing
        messages = [{"role": "user", "content": "Summarize the key features of the James Webb Space Telescope in 3-5 bullet points."}]
        
        try:
            # Measure performance
            start_time = time.time()
            response = self.groq_client.chat(model_name, messages)
            end_time = time.time()
            
            # Calculate stats
            duration = end_time - start_time
            tokens = response.get("usage", {}).get("total_tokens", 0)
            
            # Extract queue and processing times if available
            queue_time = response.get("usage", {}).get("queue_time", 0)
            processing_time = duration - queue_time if queue_time else duration
            
            # Print results
            print(f"Response time: {duration:.2f}s (Queue: {queue_time:.2f}s, Processing: {processing_time:.2f}s)")
            print(f"Tokens: {tokens} tokens")
            print(f"Speed: {tokens/processing_time:.1f} tokens/second")
            print(f"Response: {response['text'][:100]}...")
            
            self.results["performance_metrics"][model_name] = {
                "response_time": duration,
                "queue_time": queue_time,
                "processing_time": processing_time,
                "tokens": tokens,
                "tokens_per_second": tokens/processing_time if processing_time > 0 else 0,
                "success": True
            }
            
            return True
            
        except Exception as e:
            print(f"Error testing performance: {str(e)}")
            self.results["performance_metrics"][model_name] = {
                "error": str(e),
                "success": False
            }
            return False
    
    def explore_systematic_variants(self, limit=100):
        """Systematically explore model variants"""
        print("\n=== Exploring Systematic Model Variants ===")
        
        checked = 0
        discovered = 0
        
        # Generate combinations up to the limit
        combinations = []
        for base in self.base_models:
            # Try base model alone
            combinations.append(base)
            
            # Try base + size
            for size in self.sizes:
                combinations.append(f"{base}-{size}")
                
                # Try base + size + type
                for type_ in self.types:
                    combinations.append(f"{base}-{size}-{type_}")
                    
                    # Try with suffixes for promising combinations
                    for suffix in self.suffixes:
                        combinations.append(f"{base}-{size}-{type_}{suffix}")
            
            # Try base + type
            for type_ in self.types:
                combinations.append(f"{base}-{type_}")
                
                # Try with suffixes
                for suffix in self.suffixes:
                    combinations.append(f"{base}-{type_}{suffix}")
        
        # Deduplicate
        combinations = sorted(list(set(combinations)))
        
        # Filter out known models to focus on discovery
        combinations = [model for model in combinations if model not in ALL_MODELS]
        
        # Limit the number of combinations to check
        combinations = combinations[:limit]
        
        print(f"Generated {len(combinations)} combinations to check (limited to {limit})")
        
        # Check each combination
        for model in combinations:
            checked += 1
            result = self.check_model(model)
            
            if result is None:  # Authentication issue
                print("Authentication error detected. Stopping exploration.")
                break
                
            if result:
                discovered += 1
                
            # Throttle requests to avoid rate limiting
            time.sleep(0.5)
            
            # Progress update
            if checked % 10 == 0:
                print(f"Progress: {checked}/{len(combinations)} checked, {discovered} discovered")
        
        print(f"Completed systematic exploration: {checked} models checked, {discovered} new models discovered")
    
    def explore_informed_variants(self):
        """Explore variants based on known model patterns"""
        print("\n=== Exploring Informed Model Variants ===")
        
        # Specific patterns that might exist
        patterns = [
            # LLaMA 3 variants
            "llama-3-70b-versatile", "llama-3-8b-instant", "llama-3.1-70b",
            "llama-3.2-1b", "llama-3.2-3b", "llama-3.2-11b", "llama-3.2-90b",
            "llama-3.2-11b-vision", "llama-3.2-90b-vision",
            "llama-3.3-70b", "llama-3.3-70b-versatile",
            
            # Mistral variants
            "mistral-saba-7b", "mistral-saba-24b", "mistral-medium", "mistral-small", "mistral-large",
            
            # Qwen variants
            "qwen-2.5-1.5b", "qwen-2.5-7b", "qwen-2.5-14b", "qwen-2.5-32b", 
            "qwen-2.5-72b", "qwen-2.5-coder-7b", "qwen-2.5-coder-32b",
            
            # Gemma variants
            "gemma2-2b", "gemma2-9b-it", "gemma2-27b", "gemma2-9b", "gemma2-9b-vision",
            
            # DeepSeek variants
            "deepseek-r1-instruct", "deepseek-r1-distill-llama-70b", "deepseek-coder", 
            "deepseek-r1-distill-qwen-32b", "deepseek-r1-distill-llama-70b-specdec",
            
            # Whisper variants
            "whisper-large-v3", "whisper-large-v3-turbo", "distil-whisper-large-v3-en",
            
            # Mixtral variants
            "mixtral-8x7b-32768", "mixtral-8x22b", "mixtral-8x22b-v0.1", "mixtral-large"
        ]
        
        # Check each pattern
        checked = 0
        discovered = 0
        
        for model in patterns:
            if model not in ALL_MODELS:
                checked += 1
                result = self.check_model(model)
                
                if result is None:  # Authentication issue
                    print("Authentication error detected. Stopping exploration.")
                    break
                    
                if result:
                    discovered += 1
                    
                # Throttle requests to avoid rate limiting
                time.sleep(0.5)
        
        print(f"Completed informed exploration: {checked} models checked, {discovered} new models discovered")
    
    def test_known_models(self):
        """Test performance of known models"""
        print("\n=== Testing Known Models ===")
        
        # Select models to test
        top_models = ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"]
        small_models = ["llama3-8b-8192", "gemma2-9b-it"]
        
        # Test top models
        for model in top_models:
            if model in ALL_MODELS:
                self.test_model_performance(model)
                
        # Test small models
        for model in small_models:
            if model in ALL_MODELS:
                self.test_model_performance(model)
                
        # Test discovered models
        for model in self.results["discovered_models"]:
            self.test_model_performance(model)
    
    def save_results(self):
        """Save exploration results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"groq_model_exploration_{timestamp}.json"
        
        # Add summary
        self.results["summary"] = {
            "known_model_count": len(self.results["known_models"]),
            "discovered_model_count": len(self.results["discovered_models"]),
            "discovered_models": self.results["discovered_models"]
        }
        
        # Format performance summary
        perf_summary = []
        for model, metrics in self.results["performance_metrics"].items():
            if metrics.get("success"):
                perf_summary.append({
                    "model": model,
                    "response_time": metrics["response_time"],
                    "tokens_per_second": metrics["tokens_per_second"]
                })
        
        self.results["performance_summary"] = perf_summary
        
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        return filename
    
    def print_performance_summary(self):
        """Print performance summary table"""
        if not self.results["performance_metrics"]:
            return
            
        print("\n=== Model Performance Summary ===")
        print(f"{'Model':30} | {'Time (s)':10} | {'Tokens/s':10} | {'Success':10}")
        print("-" * 65)
        
        for model, metrics in self.results["performance_metrics"].items():
            if metrics.get("success"):
                print(f"{model:30} | {metrics['response_time']:.2f}s | {metrics['tokens_per_second']:.1f} | {'✅':10}")
            else:
                print(f"{model:30} | {'N/A':10} | {'N/A':10} | {'❌':10}")
    
    def run(self, systematic_limit=50, performance_test=True):
        """Run model exploration"""
        print("=== Groq Model Explorer ===")
        print(f"Known models: {len(ALL_MODELS)}")
        
        # First, run targeted exploration
        self.explore_informed_variants()
        
        # Then run systematic exploration with limit
        self.explore_systematic_variants(limit=systematic_limit)
        
        # Test performance if requested
        if performance_test and self.results["discovered_models"]:
            self.test_known_models()
            
        # Save results
        results_file = self.save_results()
        
        # Print summaries
        print("\n=== Exploration Summary ===")
        print(f"Known models: {len(self.results['known_models'])}")
        print(f"Newly discovered models: {len(self.results['discovered_models'])}")
        
        if self.results["discovered_models"]:
            print("\nDiscovered models:")
            for model in self.results["discovered_models"]:
                print(f"  - {model}")
        
        # Print performance summary
        self.print_performance_summary()
        
        print(f"\nDetailed results saved to: {results_file}")


def test_model(groq_api, model_name):
    """Test a specific model and return performance metrics"""
    print(f"\n--- Testing {model_name} ---")
    
    # Basic prompt for testing
    messages = [{"role": "user", "content": "Summarize the key features of the James Webb Space Telescope in 3-5 bullet points."}]
    
    try:
        # Measure performance
        start_time = time.time()
        response = groq_api.chat(model_name, messages)
        end_time = time.time()
        
        # Calculate stats
        duration = end_time - start_time
        tokens = response.get("usage", {}).get("total_tokens", 0)
        
        # Extract queue and processing times if available
        queue_time = response.get("usage", {}).get("queue_time", 0)
        processing_time = duration - queue_time if queue_time else duration
        
        # Print results
        print(f"Response time: {duration:.2f}s (Queue: {queue_time:.2f}s, Processing: {processing_time:.2f}s)")
        print(f"Tokens: {tokens} tokens")
        print(f"Speed: {tokens/processing_time:.1f} tokens/second")
        print(f"Response: {response['text'][:100]}...")
        
        return {
            "model": model_name,
            "response_time": duration,
            "queue_time": queue_time,
            "processing_time": processing_time,
            "tokens": tokens,
            "tokens_per_second": tokens/processing_time if processing_time > 0 else 0,
            "success": True
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "model": model_name,
            "error": str(e),
            "success": False
        }

def main():
    # Get API key from environment
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: No GROQ_API_KEY found in environment")
        sys.exit(1)
    
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--explore":
        # Run the model explorer
        explorer = ExploreGroqModels()
        
        # Get limit from command line if provided
        systematic_limit = 50
        if len(sys.argv) > 2:
            try:
                systematic_limit = int(sys.argv[2])
            except ValueError:
                pass
                
        # Run with specified limit
        explorer.run(systematic_limit=systematic_limit)
        return
    
    # Otherwise run the original model performance test
    
    # Initialize API client
    groq_api = groq_module(resources={}, metadata={"groq_api_key": api_key})
    
    # Get list of production models from the module
    from api_backends.groq import PRODUCTION_MODELS
    production_models = list(PRODUCTION_MODELS.keys())
    
    # Filter to include only LLM models (exclude audio models)
    llm_models = [m for m in production_models if "whisper" not in m.lower()]
    
    # Get top 3 most capable models to test
    top_models = ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"]
    
    # Get a smaller model for comparison
    small_models = ["llama3-8b-8192", "gemma2-9b-it"]
    
    # Test models and collect results
    results = []
    print("\n=== Testing Groq Models ===")
    
    # Test top models
    for model in top_models:
        if model in production_models:
            result = test_model(groq_api, model)
            results.append(result)
    
    # Test small models
    for model in small_models:
        if model in production_models:
            result = test_model(groq_api, model)
            results.append(result)
    
    # Display summary
    print("\n=== Model Performance Summary ===")
    print(f"{'Model':30} | {'Time (s)':10} | {'Tokens/s':10} | {'Success':10}")
    print("-" * 65)
    
    for result in results:
        if result["success"]:
            print(f"{result['model']:30} | {result['response_time']:.2f}s | {result['tokens_per_second']:.1f} | {'✅':10}")
        else:
            print(f"{result['model']:30} | {'N/A':10} | {'N/A':10} | {'❌':10}")
    
    # Save results
    output_file = "groq_model_performance.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()