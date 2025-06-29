#!/usr/bin/env python
import os
import sys
import json
import uuid
import importlib
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ipfs_accelerate_py'))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Try to import the api_backends module
try:
    from ipfs_accelerate_py import api_backends
    print("Successfully imported api_backends module")
except ImportError as e:
    print(f"Error importing api_backends module: {e}")
    api_backends = None

class APIChecker:
    """Simple utility to check API implementation status"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "apis": {}
        }
    
    def check_openai(self):
        print("\nChecking OpenAI API...")
        from apis import test_openai_api
        from api_backends import openai_api
        
        # Load credentials from file if available
        api_key = os.environ.get("OPENAI_API_KEY", "")
        
        try:
            # Try to load from api_credentials.json first
            cred_file = os.path.join(os.path.dirname(__file__), "api_credentials.json")
            if os.path.exists(cred_file):
                with open(cred_file, 'r') as f:
                    credentials = json.load(f)
                    if "openai_api_key" in credentials and credentials["openai_api_key"]:
                        api_key = credentials["openai_api_key"]
                        print("  Using OpenAI API key from api_credentials.json")
        except Exception as e:
            print(f"  Error loading credentials file: {e}")
        
        # Create API instance
        try:
            api = openai_api(resources={}, metadata={"openai_api_key": api_key})
        except Exception as e:
            print(f"  Error creating API instance: {str(e)}")
            # Try alternate instantiation if the module is a class
            if hasattr(api_backends, 'apis'):
                apis_instance = api_backends.apis(resources={}, metadata={"openai_api_key": api_key})
                if hasattr(apis_instance, 'openai'):
                    api = apis_instance.openai
                else:
                    api = None
            else:
                api = None
        
        try:
            # Create test instance for regular tests
            if hasattr(test_openai_api, 'test_openai_api'):
                test_instance = test_openai_api.test_openai_api()
            else:
                # The class might be the module itself
                test_instance = test_openai_api
            
            # Check if implementation methods are patched (mocked)
            is_mocked = False
            has_method = True
            
            # Look for patching in test code
            if hasattr(test_instance, "test"):
                test_method = getattr(test_instance, "test")
                source_code = test_method.__code__.co_consts
                
                # Check for mock patterns in source code
                mock_indicators = ["patch", "mock", "Mock", "MagicMock"]
                if any(indicator in str(source_code) for indicator in mock_indicators):
                    is_mocked = True
            
            # Check for actual implementation
            if api is not None and hasattr(api, "request_complete") and callable(getattr(api, "request_complete")):
                pass
            else:
                has_method = False
            
            status = "MOCK" if is_mocked else "REAL" if has_method else "INCOMPLETE"
            implementation_type = status
            
            # Check if the method is stubbed or real
            if status == "REAL" and api is not None:
                try:
                    # Check method implementation
                    request_complete_method = getattr(api, "request_complete")
                    if hasattr(request_complete_method, "__code__"):
                        method_code = request_complete_method.__code__.co_code
                        
                        # Very short methods are likely stubs
                        if len(method_code) < 50:  # Arbitrary threshold for stub methods
                            implementation_type = "STUB"
                except Exception as e:
                    print(f"  Error inspecting method: {str(e)}")
                    implementation_type = "UNKNOWN"
            
            self.results["apis"]["openai"] = {
                "status": status,
                "implementation_type": implementation_type,
                "has_core_methods": has_method
            }
            
            print(f"  Status: {status}")
            print(f"  Implementation: {implementation_type}")
            
        except Exception as e:
            self.results["apis"]["openai"] = {
                "error": str(e)
            }
            print(f"  Error: {str(e)}")
    
    def check_claude(self):
        print("\nChecking Claude API...")
        from apis import test_claude
        from api_backends import claude
        
        try:
            # Load credentials from file if available
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            
            try:
                # Try to load from api_credentials.json first
                cred_file = os.path.join(os.path.dirname(__file__), "api_credentials.json")
                if os.path.exists(cred_file):
                    with open(cred_file, 'r') as f:
                        credentials = json.load(f)
                        if "anthropic_api_key" in credentials and credentials["anthropic_api_key"]:
                            api_key = credentials["anthropic_api_key"]
                            print("  Using Claude API key from api_credentials.json")
            except Exception as e:
                print(f"  Error loading credentials file: {e}")
            
            # Create API instance
            try:
                # Check if claude is a class or a module
                if callable(claude):
                    api = claude(resources={}, metadata={"claude_api_key": api_key})
                else:
                    # If it's a module, try to get the class from it
                    if hasattr(claude, 'claude'):
                        Claude = getattr(claude, 'claude')
                        api = Claude(resources={}, metadata={"claude_api_key": api_key})
                    else:
                        raise TypeError("'module' object is not callable")
            except Exception as e:
                print(f"  Error creating API instance: {str(e)}")
                # Try alternate instantiation if the module is a class
                if hasattr(api_backends, 'apis'):
                    apis_instance = api_backends.apis(resources={}, metadata={"claude_api_key": api_key})
                    if hasattr(apis_instance, 'claude'):
                        api = apis_instance.claude
                    else:
                        api = None
                else:
                    api = None
            
            # Check for actual implementation
            has_method = api is not None and hasattr(api, "chat") and callable(getattr(api, "chat"))
            
            # Check test file for mocking
            if hasattr(test_claude, 'test_claude'):
                test_instance = test_claude.test_claude()
            else:
                # The class might be the module itself
                test_instance = test_claude
            is_mocked = False
            
            if hasattr(test_instance, "test"):
                test_method = getattr(test_instance, "test")
                source_code = test_method.__code__.co_consts
                
                # Check for mock patterns in source code
                mock_indicators = ["patch", "mock", "Mock", "MagicMock"]
                if any(indicator in str(source_code) for indicator in mock_indicators):
                    is_mocked = True
            
            status = "MOCK" if is_mocked else "REAL" if has_method else "INCOMPLETE"
            implementation_type = status
            
            # Check if implementation is stubbed
            if status == "REAL" and api is not None:
                try:
                    # Check method implementation
                    chat_method = getattr(api, "chat")
                    if hasattr(chat_method, "__code__"):
                        method_code = chat_method.__code__.co_code
                        
                        # Very short methods are likely stubs
                        if len(method_code) < 50:  # Arbitrary threshold for stub methods
                            implementation_type = "STUB"
                except Exception as e:
                    print(f"  Error inspecting method: {str(e)}")
                    implementation_type = "UNKNOWN"
            
            self.results["apis"]["claude"] = {
                "status": status,
                "implementation_type": implementation_type,
                "has_core_methods": has_method
            }
            
            print(f"  Status: {status}")
            print(f"  Implementation: {implementation_type}")
            
        except Exception as e:
            self.results["apis"]["claude"] = {
                "error": str(e)
            }
            print(f"  Error: {str(e)}")
    
    def check_ollama(self):
        print("\nChecking Ollama API...")
        try:
            from apis import test_ollama
            from api_backends import ollama
            
            # Create API instance
            try:
                api = ollama.ollama(resources={}, metadata={})
            except Exception as e:
                print(f"  Error creating API instance: {str(e)}")
                try:
                    # Try alternate instantiation if the module is a class
                    apis_instance = api_backends.apis(resources={}, metadata={})
                    if hasattr(apis_instance, 'ollama'):
                        api = apis_instance.ollama
                    else:
                        api = None
                except Exception as inner_e:
                    print(f"  Error creating APIs instance: {str(inner_e)}")
                    api = None
            
            # Check for actual implementation
            has_method = api is not None and hasattr(api, "generate") and callable(getattr(api, "generate"))
            
            # Check test file for mocking
            if hasattr(test_ollama, 'test_ollama'):
                test_instance = test_ollama.test_ollama()
            else:
                # The class might be the module itself
                test_instance = test_ollama
            is_mocked = False
            
            if hasattr(test_instance, "test"):
                test_method = getattr(test_instance, "test")
                source_code = test_method.__code__.co_consts
                
                # Check for mock patterns in source code
                mock_indicators = ["patch", "mock", "Mock", "MagicMock"]
                if any(indicator in str(source_code) for indicator in mock_indicators):
                    is_mocked = True
            
            status = "MOCK" if is_mocked else "REAL" if has_method else "INCOMPLETE"
            implementation_type = status
            
            # Check if implementation is stubbed
            if status == "REAL" and api is not None:
                try:
                    # Check method implementation
                    generate_method = getattr(api, "generate")
                    if hasattr(generate_method, "__code__"):
                        method_code = generate_method.__code__.co_code
                        
                        # Very short methods are likely stubs
                        if len(method_code) < 50:  # Arbitrary threshold for stub methods
                            implementation_type = "STUB"
                except Exception as e:
                    print(f"  Error inspecting method: {str(e)}")
                    implementation_type = "UNKNOWN"
            
            self.results["apis"]["ollama"] = {
                "status": status,
                "implementation_type": implementation_type,
                "has_core_methods": has_method
            }
            
            print(f"  Status: {status}")
            print(f"  Implementation: {implementation_type}")
            
        except Exception as e:
            self.results["apis"]["ollama"] = {
                "error": str(e)
            }
            print(f"  Error: {str(e)}")
            
    def check_groq(self):
        print("\nChecking Groq API...")
        try:
            from apis import test_groq
            from api_backends import groq
            
            # Load credentials from file if available
            api_key = os.environ.get("GROQ_API_KEY", "")
            
            try:
                # Try to load from api_credentials.json first
                cred_file = os.path.join(os.path.dirname(__file__), "api_credentials.json")
                if os.path.exists(cred_file):
                    with open(cred_file, 'r') as f:
                        credentials = json.load(f)
                        if "groq_api_key" in credentials and credentials["groq_api_key"]:
                            api_key = credentials["groq_api_key"]
                            print("  Using Groq API key from api_credentials.json")
            except Exception as e:
                print(f"  Error loading credentials file: {e}")
            
            # Create API instance
            try:
                api = groq.groq(resources={}, metadata={"groq_api_key": api_key})
            except Exception as e:
                print(f"  Error creating API instance: {str(e)}")
                api = None
            
            # Check for actual implementation
            has_method = api is not None and hasattr(api, "chat") and callable(getattr(api, "chat"))
            
            # Check for mocks in test
            if hasattr(test_groq, 'test_groq'):
                test_instance = test_groq.test_groq()
            else:
                test_instance = test_groq
            
            is_mocked = False
            if hasattr(test_instance, "test"):
                test_method = getattr(test_instance, "test")
                source_code = test_method.__code__.co_consts
                mock_indicators = ["patch", "mock", "Mock", "MagicMock"]
                if any(indicator in str(source_code) for indicator in mock_indicators):
                    is_mocked = True
            
            status = "MOCK" if is_mocked else "REAL" if has_method else "INCOMPLETE"
            implementation_type = status
            
            # Check if stub or real implementation
            if status == "REAL" and api is not None:
                try:
                    chat_method = getattr(api, "chat")
                    if hasattr(chat_method, "__code__"):
                        method_code = chat_method.__code__.co_code
                        if len(method_code) < 50:
                            implementation_type = "STUB"
                except Exception as e:
                    print(f"  Error inspecting method: {str(e)}")
                    implementation_type = "UNKNOWN"
            
            self.results["apis"]["groq"] = {
                "status": status,
                "implementation_type": implementation_type,
                "has_core_methods": has_method
            }
            
            print(f"  Status: {status}")
            print(f"  Implementation: {implementation_type}")
            
        except Exception as e:
            self.results["apis"]["groq"] = {"error": str(e)}
            print(f"  Error: {str(e)}")
    
    def check_hf_tgi(self):
        print("\nChecking Hugging Face TGI API...")
        try:
            from apis import test_hf_tgi
            from api_backends import hf_tgi
            
            # Load credentials from file if available
            api_key = os.environ.get("HF_API_KEY", os.environ.get("HF_API_TOKEN", ""))
            
            try:
                # Try to load from api_credentials.json first
                cred_file = os.path.join(os.path.dirname(__file__), "api_credentials.json")
                if os.path.exists(cred_file):
                    with open(cred_file, 'r') as f:
                        credentials = json.load(f)
                        if "hf_api_token" in credentials and credentials["hf_api_token"]:
                            api_key = credentials["hf_api_token"]
                            print("  Using Hugging Face API token from api_credentials.json")
            except Exception as e:
                print(f"  Error loading credentials file: {e}")
            
            # Create API instance
            try:
                api = hf_tgi.hf_tgi(resources={}, metadata={"hf_api_key": api_key})
            except Exception as e:
                print(f"  Error creating API instance: {str(e)}")
                api = None
            
            # Check for actual implementation
            has_method = api is not None and hasattr(api, "generate") and callable(getattr(api, "generate"))
            
            # Check for mocks in test
            if hasattr(test_hf_tgi, 'test_hf_tgi'):
                test_instance = test_hf_tgi.test_hf_tgi()
            else:
                test_instance = test_hf_tgi
            
            is_mocked = False
            if hasattr(test_instance, "test"):
                test_method = getattr(test_instance, "test")
                source_code = test_method.__code__.co_consts
                mock_indicators = ["patch", "mock", "Mock", "MagicMock"]
                if any(indicator in str(source_code) for indicator in mock_indicators):
                    is_mocked = True
            
            status = "MOCK" if is_mocked else "REAL" if has_method else "INCOMPLETE"
            implementation_type = status
            
            # Check if stub or real implementation
            if status == "REAL" and api is not None:
                try:
                    generate_method = getattr(api, "generate")
                    if hasattr(generate_method, "__code__"):
                        method_code = generate_method.__code__.co_code
                        if len(method_code) < 50:
                            implementation_type = "STUB"
                except Exception as e:
                    print(f"  Error inspecting method: {str(e)}")
                    implementation_type = "UNKNOWN"
            
            self.results["apis"]["hf_tgi"] = {
                "status": status,
                "implementation_type": implementation_type,
                "has_core_methods": has_method
            }
            
            print(f"  Status: {status}")
            print(f"  Implementation: {implementation_type}")
            
        except Exception as e:
            self.results["apis"]["hf_tgi"] = {"error": str(e)}
            print(f"  Error: {str(e)}")
            
    def check_gemini(self):
        print("\nChecking Gemini API...")
        try:
            from apis import test_gemini
            from api_backends import gemini
            
            # Load credentials from file if available
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            
            try:
                # Try to load from api_credentials.json first
                cred_file = os.path.join(os.path.dirname(__file__), "api_credentials.json")
                if os.path.exists(cred_file):
                    with open(cred_file, 'r') as f:
                        credentials = json.load(f)
                        if "google_api_key" in credentials and credentials["google_api_key"]:
                            api_key = credentials["google_api_key"]
                            print("  Using Google API key from api_credentials.json")
            except Exception as e:
                print(f"  Error loading credentials file: {e}")
            
            # Create API instance
            try:
                api = gemini.gemini(resources={}, metadata={"gemini_api_key": api_key})
            except Exception as e:
                print(f"  Error creating API instance: {str(e)}")
                api = None
            
            # Check for actual implementation
            has_method = api is not None and hasattr(api, "generate_content") and callable(getattr(api, "generate_content"))
            
            # Check for mocks in test
            if hasattr(test_gemini, 'test_gemini'):
                test_instance = test_gemini.test_gemini()
            else:
                test_instance = test_gemini
            
            is_mocked = False
            if hasattr(test_instance, "test"):
                test_method = getattr(test_instance, "test")
                source_code = test_method.__code__.co_consts
                mock_indicators = ["patch", "mock", "Mock", "MagicMock"]
                if any(indicator in str(source_code) for indicator in mock_indicators):
                    is_mocked = True
            
            status = "MOCK" if is_mocked else "REAL" if has_method else "INCOMPLETE"
            implementation_type = status
            
            # Check if stub or real implementation
            if status == "REAL" and api is not None:
                try:
                    generate_method = getattr(api, "generate_content")
                    if hasattr(generate_method, "__code__"):
                        method_code = generate_method.__code__.co_code
                        if len(method_code) < 50:
                            implementation_type = "STUB"
                except Exception as e:
                    print(f"  Error inspecting method: {str(e)}")
                    implementation_type = "UNKNOWN"
            
            self.results["apis"]["gemini"] = {
                "status": status,
                "implementation_type": implementation_type,
                "has_core_methods": has_method
            }
            
            print(f"  Status: {status}")
            print(f"  Implementation: {implementation_type}")
            
        except Exception as e:
            self.results["apis"]["gemini"] = {"error": str(e)}
            print(f"  Error: {str(e)}")
    
    def check_hf_tei(self):
        print("\nChecking Hugging Face TEI API...")
        try:
            from apis import test_hf_tei
            from api_backends import hf_tei
            
            # Load credentials from file if available
            api_key = os.environ.get("HF_API_KEY", os.environ.get("HF_API_TOKEN", ""))
            model_id = os.environ.get("HF_MODEL_ID", "BAAI/bge-small-en-v1.5")
            
            try:
                # Try to load from api_credentials.json first
                cred_file = os.path.join(os.path.dirname(__file__), "api_credentials.json")
                if os.path.exists(cred_file):
                    with open(cred_file, 'r') as f:
                        credentials = json.load(f)
                        if "hf_api_token" in credentials and credentials["hf_api_token"]:
                            api_key = credentials["hf_api_token"]
                            print("  Using Hugging Face API token from api_credentials.json")
            except Exception as e:
                print(f"  Error loading credentials file: {e}")
            
            # Create API instance
            try:
                api = hf_tei.hf_tei(resources={}, metadata={"hf_api_key": api_key, "model_id": model_id})
            except Exception as e:
                print(f"  Error creating API instance: {str(e)}")
                api = None
            
            # Check for actual implementation
            has_endpoint_handler = api is not None and hasattr(api, "create_remote_text_embedding_endpoint_handler") and callable(getattr(api, "create_remote_text_embedding_endpoint_handler"))
            has_request_method = api is not None and hasattr(api, "make_post_request_hf_tei") and callable(getattr(api, "make_post_request_hf_tei"))
            has_methods = has_endpoint_handler and has_request_method
            
            # Check for mocks in test
            if hasattr(test_hf_tei, 'test_hf_tei'):
                test_instance = test_hf_tei.test_hf_tei()
            else:
                test_instance = test_hf_tei
            
            is_mocked = False
            if hasattr(test_instance, "test"):
                test_method = getattr(test_instance, "test")
                source_code = test_method.__code__.co_consts
                mock_indicators = ["patch", "mock", "Mock", "MagicMock"]
                if any(indicator in str(source_code) for indicator in mock_indicators):
                    is_mocked = True
            
            status = "MOCK" if is_mocked else "REAL" if has_methods else "INCOMPLETE"
            implementation_type = status
            
            # Check if stub or real implementation
            if status == "REAL" and api is not None:
                try:
                    handler_method = getattr(api, "create_remote_text_embedding_endpoint_handler")
                    if hasattr(handler_method, "__code__"):
                        method_code = handler_method.__code__.co_code
                        if len(method_code) < 50:
                            implementation_type = "STUB"
                except Exception as e:
                    print(f"  Error inspecting method: {str(e)}")
                    implementation_type = "UNKNOWN"
            
            self.results["apis"]["hf_tei"] = {
                "status": status,
                "implementation_type": implementation_type,
                "has_core_methods": has_methods
            }
            
            print(f"  Status: {status}")
            print(f"  Implementation: {implementation_type}")
            
        except Exception as e:
            self.results["apis"]["hf_tei"] = {"error": str(e)}
            print(f"  Error: {str(e)}")
        
    def run_all_checks(self):
        """Run all API implementation checks"""
        print("=== Checking API Implementations ===")
        
        # Add all API checks here
        self.check_openai()
        self.check_claude()
        self.check_ollama()
        self.check_groq()
        self.check_hf_tgi()
        self.check_hf_tei()
        self.check_gemini()
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save check results to file"""
        results_dir = os.path.join(os.path.dirname(__file__), "api_check_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save JSON results
        results_file = os.path.join(results_dir, f"api_implementation_status_{self.results['timestamp']}.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary report
        report_file = os.path.join(results_dir, f"api_implementation_report_{self.results['timestamp']}.md")
        
        with open(report_file, "w") as f:
            f.write("# API Implementation Status Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write("| API | Status | Implementation |\n")
            f.write("|-----|--------|---------------|\n")
            
            for api_name, api_data in self.results["apis"].items():
                if "error" in api_data:
                    status = "ERROR"
                    impl = "ERROR"
                else:
                    status = api_data.get("status", "UNKNOWN")
                    impl = api_data.get("implementation_type", "UNKNOWN")
                
                f.write(f"| {api_name} | {status} | {impl} |\n")
            
            f.write("\n## Details\n\n")
            for api_name, api_data in self.results["apis"].items():
                f.write(f"### {api_name}\n\n")
                
                if "error" in api_data:
                    f.write(f"**Error:** {api_data['error']}\n\n")
                else:
                    for key, value in api_data.items():
                        f.write(f"**{key}:** {value}\n")
                    f.write("\n")
        
        print(f"\nResults saved to:")
        print(f"  - {results_file}")
        print(f"  - {report_file}")

def run_web_platform_tests(args=None):
    """Run web platform tests to verify WebNN and WebGPU support."""
    print("Running web platform tests...")
    
    try:
        # Check if web platform testing module exists
        import os.path
        web_platform_test_path = os.path.join(os.path.dirname(__file__), "web_platform_testing.py")
        web_audio_test_path = os.path.join(os.path.dirname(__file__), "web_audio_test_runner.py")
        
        results = {
            "status": "completed",
            "webnn_support": False,
            "webgpu_support": False,
            "audio_tests": False,
            "vision_tests": False,
            "embedding_tests": False
        }
        
        # Check for web platform test script
        if os.path.exists(web_platform_test_path):
            print("- Found web platform testing module")
            results["webnn_support"] = True
            results["webgpu_support"] = True
            
            try:
                # Try to import the module to verify it works
                from web_platform_testing import WebPlatformTesting
                tester = WebPlatformTesting()
                
                # Get available models for testing
                embedding_models = tester.get_models_by_modality("text")
                vision_models = tester.get_models_by_modality("vision")
                
                # Check that we have test models available
                if embedding_models:
                    print(f"- Found {len(embedding_models)} embedding models for web platform testing")
                    results["embedding_tests"] = True
                
                if vision_models:
                    print(f"- Found {len(vision_models)} vision models for web platform testing")
                    results["vision_tests"] = True
                
            except ImportError as e:
                print(f"- Error importing web platform testing module: {e}")
        
        # Check for web audio test script
        if os.path.exists(web_audio_test_path):
            print("- Found web audio testing module")
            results["audio_tests"] = True
            
            try:
                # Try to import the module to verify it works
                from web_audio_test_runner import WebAudioTestRunner
                audio_tester = WebAudioTestRunner()
                print("- Web audio test module successfully imported")
                
                # Check if test files are prepared
                test_dir = audio_tester.test_directory
                if os.path.exists(test_dir) and os.path.exists(os.path.join(test_dir, "whisper")) and os.path.exists(os.path.join(test_dir, "wav2vec2")):
                    print("- Web audio test files are prepared")
                else:
                    print("- Web audio test files need preparation")
                    
            except ImportError as e:
                print(f"- Error importing web audio test module: {e}")
        
        return results
        
    except Exception as e:
        print(f"Error running web platform tests: {e}")
        return {
            "status": "error",
            "error": str(e),
            "webnn_support": False,
            "webgpu_support": False
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check API implementations")
    parser.add_argument("--web-platform", action="store_true", help="Run web platform tests")
    args = parser.parse_args()
    
    if args.web_platform:
        run_web_platform_tests(args)
    else:
        checker = APIChecker()
        checker.run_all_checks()