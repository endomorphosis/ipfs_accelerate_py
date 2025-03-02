#!/usr/bin/env python
import os
import sys
import json
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ipfs_accelerate_py'))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class APIChecker:
    """Enhanced utility to check API implementation status for all backends including the new ones.
    
    Checks Claude, OpenAI, Gemini, Groq, HF TGI, HF TEI, Ollama, LLVM, and OVMS APIs
    for implementation status, looking for core methods and checking if they are mocked.
    """
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "apis": {}
        }
        
        # Map of API types to their core method names
        self.core_methods = {
            "claude": "chat",
            "openai": "request_complete",
            "gemini": "generate_content",
            "groq": "chat",
            "hf_tgi": "generate",
            "hf_tei": "create_remote_text_embedding_endpoint_handler",
            "ollama": "generate",
            "llvm": "generate",
            "ovms": "predict"
        }
        
        # Map of API types to test module names
        self.test_modules = {
            "claude": "test_claude",
            "openai": "test_openai_api",
            "gemini": "test_gemini",
            "groq": "test_groq",
            "hf_tgi": "test_hf_tgi",
            "hf_tei": "test_hf_tei",
            "ollama": "test_ollama",
            "llvm": "test_llvm",
            "ovms": "test_ovms"
        }
        
        # Map of API types to their unified test modules (if available)
        self.unified_test_modules = {
            "hf_tgi": "test_hf_tgi_unified",
            "hf_tei": "test_hf_tei_unified",
            "ollama": "test_ollama_unified",
            "llvm": "test_llvm_unified",
            "ovms": "test_ovms_unified"
        }
    
    def check_openai(self):
        print("\nChecking OpenAI API...")
        from apis import test_openai_api
        from api_backends import openai_api
        
        # Create API instance
        try:
            api = openai_api(resources={}, metadata={"openai_api_key": os.environ.get("OPENAI_API_KEY", "")})
        except Exception as e:
            print(f"  Error creating API instance: {str(e)}")
            # Try alternate instantiation if the module is a class
            if hasattr(api_backends, 'apis'):
                apis_instance = api_backends.apis(resources={}, metadata={"openai_api_key": os.environ.get("OPENAI_API_KEY", "")})
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
            # Create API instance
            try:
                api = claude.claude(resources={}, metadata={"claude_api_key": os.environ.get("ANTHROPIC_API_KEY", "")})
            except Exception as e:
                print(f"  Error creating API instance: {str(e)}")
                # Try alternate instantiation if the module is a class
                if hasattr(api_backends, 'apis'):
                    apis_instance = api_backends.apis(resources={}, metadata={"claude_api_key": os.environ.get("ANTHROPIC_API_KEY", "")})
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
            
            # Create API instance
            try:
                api = groq.groq(resources={}, metadata={"groq_api_key": os.environ.get("GROQ_API_KEY", "")})
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
            
            # Create API instance
            try:
                api_key = os.environ.get("HF_API_KEY", os.environ.get("HF_API_TOKEN", ""))
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
            
            # Create API instance
            try:
                api = gemini.gemini(resources={}, metadata={"gemini_api_key": os.environ.get("GOOGLE_API_KEY", "")})
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
            
            # Create API instance
            try:
                api_key = os.environ.get("HF_API_KEY", os.environ.get("HF_API_TOKEN", ""))
                model_id = os.environ.get("HF_MODEL_ID", "BAAI/bge-small-en-v1.5")
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
        
    def check_generic_api(self, api_type):
        """Generic method to check any API implementation"""
        print(f"\nChecking {api_type.upper()} API...")
        
        # Import test module
        test_module_name = self.test_modules[api_type]
        try:
            test_module = __import__(f"apis.{test_module_name}", fromlist=["apis"])
        except ImportError as e:
            print(f"  Error importing test module {test_module_name}: {str(e)}")
            self.results["apis"][api_type] = {"error": f"Missing test module: {str(e)}"}
            return
        
        # Import API backend module
        try:
            api_backend = __import__(f"api_backends.{api_type}", fromlist=["api_backends"])
        except ImportError as e:
            print(f"  Error importing API backend {api_type}: {str(e)}")
            self.results["apis"][api_type] = {"error": f"Missing backend module: {str(e)}"}
            return
        
        # Create API instance
        try:
            # Get the class from the module - assume it's the same name as the module
            api_class = getattr(api_backend, api_type)
            
            # Create metadata with API key if environment variable is available
            env_var_name = f"{api_type.upper()}_API_KEY"
            api_key_name = f"{api_type}_api_key"
            api_key = os.environ.get(env_var_name, "")
            
            # Special case for HF
            if api_type.startswith("hf_"):
                api_key = os.environ.get("HF_API_KEY", os.environ.get("HF_API_TOKEN", ""))
                api_key_name = "hf_api_key"
            
            metadata = {api_key_name: api_key}
            
            # Create instance
            api = api_class(resources={}, metadata=metadata)
        except Exception as e:
            print(f"  Error creating API instance: {str(e)}")
            api = None
        
        # Check for core method
        core_method_name = self.core_methods.get(api_type)
        has_method = api is not None and hasattr(api, core_method_name) and callable(getattr(api, core_method_name))
        
        # Get test class from test module
        test_class_name = test_module_name
        try:
            if hasattr(test_module, test_class_name):
                test_instance = getattr(test_module, test_class_name)()
            else:
                # The class might be the module itself
                test_instance = test_module
        except Exception as e:
            print(f"  Error creating test instance: {str(e)}")
            test_instance = None
        
        # Check for mocking in test code
        is_mocked = False
        if test_instance and hasattr(test_instance, "test"):
            test_method = getattr(test_instance, "test")
            source_code = test_method.__code__.co_consts
            
            # Check for mock patterns in source code
            mock_indicators = ["patch", "mock", "Mock", "MagicMock"]
            if any(indicator in str(source_code) for indicator in mock_indicators):
                is_mocked = True
        
        # Check for unified test class
        has_unified_test = api_type in self.unified_test_modules
        if has_unified_test:
            unified_test_module_name = self.unified_test_modules[api_type]
            try:
                unified_test_module = __import__(f"apis.{unified_test_module_name}", fromlist=["apis"])
                has_unified_test = True
            except ImportError:
                has_unified_test = False
        
        # Determine implementation status
        if has_method:
            if is_mocked:
                status = "MOCK"
            else:
                status = "REAL"
        else:
            status = "INCOMPLETE"
        
        implementation_type = status
        
        # Check for endpoint implementation (current API improvements)
        has_endpoint_methods = all(hasattr(api, method) for method in ["create_endpoint", "get_endpoint", "update_endpoint"])
        has_queue = hasattr(api, "_process_queue")
        has_stats = hasattr(api, "get_stats") and hasattr(api, "reset_stats")
        
        # Check if the method is stubbed or real
        if status == "REAL" and api is not None:
            try:
                # Check method implementation
                core_method = getattr(api, core_method_name)
                if hasattr(core_method, "__code__"):
                    method_code = core_method.__code__.co_code
                    
                    # Very short methods are likely stubs
                    if len(method_code) < 50:  # Arbitrary threshold for stub methods
                        implementation_type = "STUB"
            except Exception as e:
                print(f"  Error inspecting method: {str(e)}")
                implementation_type = "UNKNOWN"
        
        # Store results
        self.results["apis"][api_type] = {
            "status": status,
            "implementation_type": implementation_type,
            "has_core_methods": has_method,
            "has_endpoint_methods": has_endpoint_methods,
            "has_queue": has_queue,
            "has_stats": has_stats,
            "has_unified_test": has_unified_test
        }
        
        print(f"  Status: {status}")
        print(f"  Implementation: {implementation_type}")
        if has_endpoint_methods:
            print(f"  Endpoint Implementation: ✓")
        else:
            print(f"  Endpoint Implementation: ✗")
        if has_queue:
            print(f"  Queue Implementation: ✓")
        else:
            print(f"  Queue Implementation: ✗")
        if has_stats:
            print(f"  Stats Implementation: ✓")
        else:
            print(f"  Stats Implementation: ✗")
    
    def run_all_checks(self):
        """Run all API implementation checks"""
        print("=== Checking API Implementations ===")
        
        for api_type in self.core_methods.keys():
            # Use specific check method if it exists, otherwise use generic
            specific_check_method = getattr(self, f"check_{api_type}", None)
            if specific_check_method and callable(specific_check_method):
                specific_check_method()
            else:
                self.check_generic_api(api_type)
        
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

if __name__ == "__main__":
    checker = APIChecker()
    checker.run_all_checks()