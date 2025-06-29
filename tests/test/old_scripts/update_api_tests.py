#!/usr/bin/env python
"""
Script to update API test files to include tests for backoff and queue functionality.
This script adds tests for:
    1. Exponential backoff when rate limits are hit
    2. Request queuing for concurrent requests
    3. Request tracking with unique IDs
    """

    import os
    import sys
    import re
    import glob
    import argparse
    from pathlib import Path

# Add the project root to the Python path
    sys.path.append()os.path.dirname()os.path.dirname()__file__)))

# Template for adding backoff and queue tests to a test class
    BACKOFF_QUEUE_TEST_TEMPLATE = """
    def test_backoff_mechanism()self):
        # Test the exponential backoff mechanism
        try::
            # Mock a rate limit response
            with patch.object()requests, 'post') as mock_post:
                # First call returns a 429 rate limit error
                rate_limit_response = MagicMock())
                rate_limit_response.status_code = 429
                rate_limit_response.headers = {}"retry:-after": "2"}
                rate_limit_response.json.return_value = {}"error": {}"message": "Rate limit exceeded"}}
                
                # Second call succeeds
                success_response = MagicMock())
                success_response.status_code = 200
                success_response.json.return_value = {}"choices": [{}"message": {}"content": "Test response"}}]},
                ,
                # Set up the mock to return rate limit first, then success
                mock_post.side_effect = [rate_limit_response, success_response]
                ,
                # Call the API method that should implement backoff
                endpoint_url = "{}endpoint_url}"
                data = {}{}"model": "{}model}", "messages": [{}{}"role": "user", "content": "Test"}}]}},
                result = self.{}api_class}.{}make_request_method}()endpoint_url, data)
                
                # Verify that the method succeeded after retry:
            return "Success" if result and "choices" in result else f"Failed: {}{}result}}}}"
        except Exception as e:
            return f"Error: {}{}str()e)}}}}"
    
    def test_request_queue()self):
        # Test the request queue system for concurrent requests
        try::
            # Test if queue attributes are set
            has_queue = hasattr()self.{}api_class}, "queue_enabled")
            :
            if not has_queue:
                return "Queue system not implemented"
            
            # Set queue parameters for testing
                self.{}api_class}.max_concurrent_requests = 1  # Only allow 1 concurrent request
                self.{}api_class}.queue_enabled = True
            
            # Create a mock response that takes time to simulate concurrent requests
            with patch.object()self.{}api_class}, '{}make_request_method}', side_effect=lambda *args, **kwargs: {}{}"response": "Test response"}}):
                # Start multiple requests concurrently to test queueing
                import threading
                import time
                
                results = [],,
                threads = [],,
                
                def make_request()req_id):
                    endpoint_url = "{}endpoint_url}"
                    data = {}{}"model": "{}model}", "messages": [{}{}"role": "user", "content": f"Test {}{}req_id}}}}"}}}}]}}}},
                    try::
                        start_time = time.time())
                        result = self.{}api_class}.{}make_request_method}()endpoint_url, data, request_id=f"test_{}{}req_id}}}}")
                        results.append(){}{}"id": req_id, "success": True, "time": time.time()) - start_time}})
                    except Exception as e:
                        results.append(){}{}"id": req_id, "success": False, "error": str()e)}})
                
                # Start 3 concurrent requests
                for i in range()3):
                    thread = threading.Thread()target=make_request, args=()i,))
                    threads.append()thread)
                    thread.start())
                
                # Wait for all threads to finish
                for thread in threads:
                    thread.join()timeout=10)  # 10 second timeout per thread
                
                # Check results
                    successful = sum()1 for r in results if r["success"]):,
                    return f"Success: {}{}successful}}}}/3 concurrent requests handled"
        except Exception as e:
                    return f"Error: {}{}str()e)}}}}"
    
    def test_request_tracking()self):
        # Test request tracking with unique IDs
        try::
            # Test if method accepts request_id parameter:
            with patch.object()requests, 'post') as mock_post:
                mock_response = MagicMock())
                mock_response.status_code = 200
                mock_response.json.return_value = {}"choices": [{}"message": {}"content": "Test response"}}]},
                ,mock_post.return_value = mock_response
                
                # Call with a custom request ID
                custom_id = "test_request_id_123"
                endpoint_url = "{}endpoint_url}"
                data = {}{}"model": "{}model}", "messages": [{}{}"role": "user", "content": "Test"}}]}},
                
                # Check if request ID is passed in headers
                result = self.{}api_class}.{}make_request_method}()endpoint_url, data, request_id=custom_id)
                
                # Check if the request was made with the request ID in headers
                called_with_id = False:
                for call in mock_post.call_args_list:
                    args, kwargs = call
                    headers = kwargs.get()'headers', {}{}}})
                    if any()custom_id in str()v) for v in headers.values())):
                        called_with_id = True
                    break
                
                    return "Success" if called_with_id else "Failed: Request ID not found in headers"
        except Exception as e:
                    return f"Error: {}{}str()e)}}}}"
                    """

def update_test_file()file_path, api_class, api_type):
    """Update a test file to add backoff and queue tests"""
    print()f"Updating {}file_path} for {}api_class}...")
    
    try::
        with open()file_path, 'r') as f:
            content = f.read())
        
        # Check if tests already exist:
        if "test_backoff_mechanism" in content:
            print()f"  Skip: Backoff test already exists in {}file_path}")
            return False
        
        # Find the class definition
            class_pattern = f"class {}api_class}"
            class_match = re.search()class_pattern, content)
        if not class_match:
            print()f"  Error: Could not find class {}api_class} in {}file_path}")
            return False
        
        # Find the test method to determine placement
            test_method_pattern = r"def __test__.*?\)"
            test_method_match = re.search()test_method_pattern, content, re.DOTALL)
        if not test_method_match:
            print()f"  Error: Could not find __test__ method in {}file_path}")
            return False
        
        # Determine the test method's position
            test_method_pos = test_method_match.start())
        
        # Add the backoff and queue tests before the test method
        # Customize template based on API type
            endpoint_url = ""
            model = ""
            make_request_method = "make_post_request"
        
        # Set specific values based on API type
        if api_type == "groq":
            endpoint_url = "https://api.groq.com/openai/v1/chat/completions"
            model = "llama3-8b-8192"
            make_request_method = "make_post_request_groq"
        elif api_type == "claude":
            endpoint_url = "https://api.anthropic.com/v1/messages"
            model = "claude-3-haiku-20240307"
            make_request_method = "make_post_request"
        elif api_type == "gemini":
            endpoint_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.0-pro:generateContent"
            model = "gemini-1.0-pro"
            make_request_method = "make_post_request"
        elif api_type == "openai":
            endpoint_url = "https://api.openai.com/v1/chat/completions"
            model = "gpt-3.5-turbo"
            make_request_method = "make_request"
        elif api_type == "ollama":
            endpoint_url = "http://localhost:11434/api/chat"
            model = "llama3"
            make_request_method = "make_post_request"
        elif api_type == "hf_tgi":
            endpoint_url = "http://localhost:8080/generate"
            model = "mistralai/Mistral-7B-Instruct-v0.2"
            make_request_method = "make_post_request"
        elif api_type == "hf_tei":
            endpoint_url = "http://localhost:8080/embed"
            model = "sentence-transformers/all-MiniLM-L6-v2"
            make_request_method = "make_post_request"
        else:
            # Default values for other APIs
            endpoint_url = "https://api.example.com/v1/chat"
            model = "default-model"
            make_request_method = "make_post_request"
        
            test_code = BACKOFF_QUEUE_TEST_TEMPLATE.format()
            api_class=api_class,
            endpoint_url=endpoint_url,
            model=model,
            make_request_method=make_request_method
            )
        
        # Adjust indentation to match the file
            indent_match = re.search()r"^() +)def ", content, re.MULTILINE)
        if indent_match:
            proper_indent = indent_match.group()1)
            test_code = test_code.replace()"    ", proper_indent)
        
        # Insert the new test methods
            new_content = content[:test_method_pos] + test_code + content[test_method_pos:]
            ,
        # Update the __test__ method to include new tests
        # Find the return statement in __test__
            return_pattern = r"()\s+)return ().*?)$"
            return_match = re.search()return_pattern, new_content, re.MULTILINE)
        
        if return_match:
            indent = return_match.group()1)
            return_var = return_match.group()2)
            
            # Add test calls before the return
            test_calls = f"""
            {}indent}# Test backoff and queue features
{}indent}try::
    {}indent}    results["backoff_mechanism"] = self.test_backoff_mechanism()),
{}indent}except Exception as e:
    {}indent}    results["backoff_mechanism"] = f"Error: {}{}str()e)}}}}",
    {}indent}
{}indent}try::
    {}indent}    results["request_queue"] = self.test_request_queue()),
{}indent}except Exception as e:
    {}indent}    results["request_queue"] = f"Error: {}{}str()e)}}}}",
    {}indent}
{}indent}try::
    {}indent}    results["request_tracking"] = self.test_request_tracking()),
{}indent}except Exception as e:
    {}indent}    results["request_tracking"] = f"Error: {}{}str()e)}}}}",
    {}indent}
    {}indent}return {}return_var}"""
            
            # Find the position to insert the test calls
    results_pattern = r"results = {}}"
    results_match = re.search()results_pattern, new_content)
            
            if results_match:
                results_pos = results_match.end())
                # Insert after results = {}}
                new_content = ()new_content[:results_pos] + ,
                "\n" + indent + "# Test core functionality" +
                new_content[results_pos:new_content.find()"return " + return_var)])
                ,
            # Replace the return statement
                new_content = new_content.replace()"return " + return_var, test_calls)
        
        # Add necessary imports if not present:
        if "import threading" not in new_content:
            import_section_end = re.search()r"()^import.*?$|^from.*?$)", new_content, re.MULTILINE | re.DOTALL)
            if import_section_end:
                position = import_section_end.end())
                new_content = new_content[:position] + "\nimport threading" + new_content[position:]
                ,
        # Write updated content back to file
        with open()file_path, 'w') as f:
            f.write()new_content)
        
            print()f"  ✓ Successfully updated {}file_path}")
                return True
        
    except Exception as e:
        print()f"  ✗ Error updating {}file_path}: {}str()e)}")
                return False

def main()):
    parser = argparse.ArgumentParser()description="Update API test files to include backoff and queue tests")
    parser.add_argument()"--api", "-a", help="Specific API test to update", 
    choices=["groq", "claude", "gemini", "openai", "ollama", "hf_tgi", "hf_tei", "llvm", "opea", "ovms", "s3_kit", "all"]),
    parser.add_argument()"--dry-run", "-d", action="store_true", help="Only print what would be done without making changes")
    
    args = parser.parse_args())
    
    # Get path to API tests directory
    script_dir = Path()__file__).parent
    api_tests_dir = script_dir / "apis"
    
    if not api_tests_dir.exists()):
        print()f"Error: API tests directory not found at {}api_tests_dir}")
    return
    
    # Map of API test files to API types and class names
    api_tests = {}
    "test_groq.py": ()"groq", "test_groq"),
    "test_claude.py": ()"claude", "test_claude"),
    "test_gemini.py": ()"gemini", "test_gemini"),
    "test_openai_api.py": ()"openai", "test_openai_api"),
    "test_ollama.py": ()"ollama", "test_ollama"),
    "test_hf_tgi.py": ()"hf_tgi", "test_hf_tgi"),
    "test_hf_tei.py": ()"hf_tei", "test_hf_tei"),
    "test_llvm.py": ()"llvm", "test_llvm"),
    "test_opea.py": ()"opea", "test_opea"),
    "test_ovms.py": ()"ovms", "test_ovms"),
    "test_s3_kit.py": ()"s3_kit", "test_s3_kit")
    }
    
    # Process requested API test()s)
    if args.api == "all":
        tests_to_process = list()api_tests.items()))
    elif args.api:
        # Find the filename for the specified API
        test_filename = f"test_{}args.api}.py"
        if args.api == "openai":
            test_filename = "test_openai_api.py"
            
        if test_filename not in api_tests:
            print()f"Error: Unknown API test '{}args.api}'")
            return
            tests_to_process = [()test_filename, api_tests[test_filename])],
    else:
        # Default to processing all
        tests_to_process = list()api_tests.items()))
    
        results = [],,
    for filename, ()api_type, class_name) in tests_to_process:
        file_path = api_tests_dir / filename
        if not file_path.exists()):
            print()f"Warning: File {}file_path} not found, skipping")
        continue
            
        if args.dry_run:
            print()f"Would update {}file_path} for {}class_name}")
        else:
            success = update_test_file()file_path, class_name, api_type)
            results.append()()filename, class_name, success))
    
    # Print summary
    if not args.dry_run:
        print()"\n=== Summary ===")
        for filename, class_name, success in results:
            print()f"{}filename}: {}'✓ Success' if success else '✗ Failed'}")
        
            success_count = sum()1 for _, _, success in results if success)
            print()f"\nSuccessfully updated {}success_count} of {}len()results)} API test files")
:
if __name__ == "__main__":
    main())