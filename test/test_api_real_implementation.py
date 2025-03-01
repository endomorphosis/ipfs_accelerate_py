#!/usr/bin/env python
import os
import sys
import json
import time
import uuid
import getpass
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ipfs_accelerate_py'))
import api_backends

# Import test modules directly with proper path resolution
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from apis import (
    test_claude, test_groq, test_hf_tgi, test_hf_tei, test_llvm,
    test_openai_api, test_ovms, test_ollama, test_s3_kit,
    test_gemini, test_opea
)

class CredentialManager:
    """Simple credential storage for API testing"""
    
    def __init__(self):
        self.cred_file = os.path.join(os.path.expanduser("~"), ".ipfs_api_credentials")
        self.credentials = self._load_credentials()
        
    def _load_credentials(self):
        """Load credentials from file"""
        if os.path.exists(self.cred_file):
            try:
                with open(self.cred_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
            
    def _save_credentials(self):
        """Save credentials to file"""
        with open(self.cred_file, 'w') as f:
            json.dump(self.credentials, f)
        os.chmod(self.cred_file, 0o600)  # Secure the file
            
    def get(self, key):
        """Get credential by key"""
        return self.credentials.get(key, "")
        
    def set(self, key, value):
        """Set credential by key"""
        self.credentials[key] = value
        self._save_credentials()


class APIImplementationTester:
    """Tests API backends to verify real implementations vs mocks"""
    
    def __init__(self):
        self.cred_manager = CredentialManager()
        self.results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "apis": {}
        }
        
    def _get_api_key(self, name, env_var, prompt_msg):
        """Get API key from environment, credential store, or user input"""
        # Try environment variable first
        api_key = os.environ.get(env_var, "")
        
        # If not in environment, try credential store
        if not api_key:
            api_key = self.cred_manager.get(name)
            
        # If still not found, prompt user
        if not api_key:
            print(f"\n{prompt_msg}")
            api_key = getpass.getpass(f"Enter {name} API key: ")
            
            if api_key:
                save = input("Save this API key for future use? (y/n): ").lower() == 'y'
                if save:
                    self.cred_manager.set(name, api_key)
                
        return api_key
    
    def setup_metadata(self):
        """Set up test metadata with API credentials"""
        metadata = {
            "tokens_per_word": 4,
            "max_tokens": 2048
        }
        
        # OpenAI
        metadata["openai_api_key"] = self._get_api_key(
            "openai", "OPENAI_API_KEY", 
            "OpenAI API test requires an API key."
        )
        
        # Claude (Anthropic)
        metadata["claude_api_key"] = self._get_api_key(
            "anthropic", "ANTHROPIC_API_KEY",
            "Claude API test requires an API key."
        )
        
        # Gemini
        metadata["gemini_api_key"] = self._get_api_key(
            "gemini", "GOOGLE_API_KEY",
            "Gemini API test requires a Google API key."
        )
        
        # Groq
        metadata["groq_api_key"] = self._get_api_key(
            "groq", "GROQ_API_KEY",
            "Groq API test requires an API key."
        )
        
        # Hugging Face
        metadata["hf_api_token"] = self._get_api_key(
            "huggingface", "HF_API_TOKEN",
            "Hugging Face API tests require an API token."
        )
        
        return metadata
    
    def _test_implementation(self, api_instance, api_name):
        """Test if API implementation is real or mock"""
        try:
            # Generate unique test input
            test_id = str(uuid.uuid4())[:8]
            prompt = f"Test input [{test_id}] to verify real implementation."
            
            # Get the base API object
            api_obj = None
            if hasattr(api_instance, api_name.replace('test_', '')):
                api_obj = getattr(api_instance, api_name.replace('test_', ''))
            
            if not api_obj:
                return {"status": "ERROR", "error": "Could not access API object"}
                
            # Try to make a real API call
            response = None
            is_real = False
            
            # Different APIs have different methods
            try:
                if hasattr(api_obj, 'chat'):
                    messages = [{"role": "user", "content": prompt}]
                    response = api_obj.chat(messages)
                elif hasattr(api_obj, 'request_complete'):
                    api_obj.messages = [{"role": "user", "content": prompt}]
                    api_obj.model = None  # Use default model
                    api_obj.method = "chat"
                    response = api_obj.request_complete()
                elif hasattr(api_obj, 'text_complete'):
                    response = api_obj.text_complete(prompt)
                
                # Analyze response to determine if it's real or mock
                if response:
                    # Mocks often have static responses or specific patterns
                    mock_patterns = [
                        "This is a test response",
                        "test chat response"
                    ]
                    
                    response_str = str(response)
                    
                    # Real APIs typically return substantial responses
                    if len(response_str) > 100:
                        # Check if response matches known mock patterns
                        if not any(pattern in response_str for pattern in mock_patterns):
                            is_real = True
            except Exception as e:
                error_msg = str(e).lower()
                
                # Authentication errors often indicate real APIs
                auth_terms = ["auth", "key", "token", "credential", "permission"]
                rate_terms = ["rate", "limit", "quota"]
                
                if any(term in error_msg for term in auth_terms + rate_terms):
                    is_real = True  # Error suggests real API connection attempt
                    
                return {
                    "status": "ERROR", 
                    "error": str(e),
                    "is_real": is_real
                }
                    
            return {
                "status": "SUCCESS",
                "is_real": is_real,
                "response_length": len(str(response)) if response else 0
            }
            
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
    def run_tests(self):
        """Run implementation tests for all API backends"""
        resources = {}
        metadata = self.setup_metadata()
        
        # Initialize APIs
        if "apis" not in resources:
            resources["apis"] = api_backends.apis(resources=resources, metadata=metadata)
            
        print("\n=== Testing API Backend Implementations ===\n")
        
        # Test each API
        api_tests = [
            ("openai_api", test_openai_api.test_openai_api),
            ("claude", test_claude.test_claude),
            ("gemini", test_gemini.test_gemini),
            ("groq", test_groq.test_groq),
            ("hf_tgi", test_hf_tgi.test_hf_tgi),
            ("hf_tei", test_hf_tei.test_hf_tei),
            ("llvm", test_llvm.test_llvm),
            ("ovms", test_ovms.test_ovms),
            ("ollama", test_ollama.test_ollama),
            ("s3_kit", test_s3_kit.test_s3_kit),
            ("opea", test_opea.test_opea)
        ]
        
        for api_name, test_class in api_tests:
            print(f"Testing {api_name}...")
            try:
                # Run standard tests
                api_instance = test_class(resources=resources, metadata=metadata)
                std_results = api_instance.test()
                
                # Test if implementation is real
                impl_results = self._test_implementation(api_instance, api_name)
                
                self.results["apis"][api_name] = {
                    "standard_tests": std_results,
                    "implementation_check": impl_results
                }
                
                # Show real/mock status
                status = "REAL" if impl_results.get("is_real", False) else "MOCK"
                if impl_results.get("status") == "ERROR":
                    if impl_results.get("is_real", False):
                        status = "REAL (with errors)"
                    else:
                        status = "ERROR"
                        
                print(f"  Implementation: {status}")
                
            except Exception as e:
                self.results["apis"][api_name] = {
                    "error": str(e)
                }
                print(f"  Error: {str(e)}")
            
        # Save results
        self.save_results()
        self.generate_report()
        
    def save_results(self):
        """Save test results to file"""
        results_dir = os.path.join(os.path.dirname(__file__), "api_test_results")
        os.makedirs(results_dir, exist_ok=True)
        
        result_file = os.path.join(results_dir, f"api_implementation_results_{self.results['timestamp']}.json")
        with open(result_file, "w") as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\nResults saved to {result_file}")
        
    def generate_report(self):
        """Generate summary report of API implementation status"""
        report_dir = os.path.join(os.path.dirname(__file__), "api_test_results")
        report_file = os.path.join(report_dir, f"api_implementation_report_{self.results['timestamp']}.md")
        
        summary = {
            "real": 0,
            "mock": 0,
            "error": 0,
            "total": len(self.results["apis"])
        }
        
        with open(report_file, "w") as f:
            f.write("# API Implementation Test Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Implementation Status\n\n")
            f.write("| API | Implementation | Status |\n")
            f.write("|-----|---------------|--------|\n")
            
            for api_name, results in self.results["apis"].items():
                impl_check = results.get("implementation_check", {})
                is_real = impl_check.get("is_real", False)
                status = impl_check.get("status", "ERROR")
                
                if is_real:
                    impl_type = "REAL"
                    summary["real"] += 1
                elif status == "ERROR" and not is_real:
                    impl_type = "ERROR"
                    summary["error"] += 1
                else:
                    impl_type = "MOCK"
                    summary["mock"] += 1
                    
                success = "✅" if status == "SUCCESS" else "❌"
                f.write(f"| {api_name} | {impl_type} | {success} |\n")
                
            f.write("\n## Summary\n\n")
            f.write(f"- Total APIs tested: {summary['total']}\n")
            f.write(f"- Real implementations: {summary['real']}\n")
            f.write(f"- Mock implementations: {summary['mock']}\n")
            f.write(f"- Error/undetermined: {summary['error']}\n")
            
            f.write("\n## Details\n\n")
            for api_name, results in self.results["apis"].items():
                f.write(f"### {api_name}\n\n")
                
                impl_check = results.get("implementation_check", {})
                f.write(f"- **Implementation:** {'REAL' if impl_check.get('is_real', False) else 'MOCK'}\n")
                f.write(f"- **Status:** {impl_check.get('status', 'ERROR')}\n")
                
                if "error" in impl_check:
                    f.write(f"- **Error:** {impl_check['error']}\n")
                    
                f.write("\n")
                
        print(f"Report generated: {report_file}")
                

if __name__ == "__main__":
    try:
        tester = APIImplementationTester()
        tester.run_tests()
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()