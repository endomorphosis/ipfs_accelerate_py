#!/usr/bin/env python
"""
Example script demonstrating the queue and backoff features of the IPFS Accelerate API backends.
This script shows:
    1. Basic queue configuration and usage
    2. Exponential backoff behavior
    3. Circuit breaker pattern
    4. Concurrent request handling with queues
    5. Performance comparison with/without queue
    """

    import os
    import sys
    import time
    import threading
    import concurrent.futures
    from datetime import datetime

# Add the project root to the Python path
    sys.path.append()))))))))))))))))))os.path.join()))))))))))))))))))os.path.dirname()))))))))))))))))))os.path.dirname()))))))))))))))))))os.path.dirname()))))))))))))))))))__file__))), "ipfs_accelerate_py"))

# Import API backends
    from ipfs_accelerate_py.api_backends import openai_api, groq, claude, ollama

def print_section()))))))))))))))))))title):
    """Print a section title with formatting"""
    print()))))))))))))))))))"\n" + "=" * 60)
    print()))))))))))))))))))f" {}}}}title} ".center()))))))))))))))))))60, "="))
    print()))))))))))))))))))"=" * 60 + "\n")

def print_subsection()))))))))))))))))))title):
    """Print a subsection title with formatting"""
    print()))))))))))))))))))"\n" + "-" * 50)
    print()))))))))))))))))))f" {}}}}title} ".center()))))))))))))))))))50, "-"))
    print()))))))))))))))))))"-" * 50 + "\n")

def print_api_status()))))))))))))))))))client, api_name):
    """Print the queue and backoff configuration status for an API client"""
    print()))))))))))))))))))f"API: {}}}}api_name}")
    
    # Queue configuration
    queue_size = getattr()))))))))))))))))))client, "queue_size", "Unknown")
    max_concurrent = getattr()))))))))))))))))))client, "max_concurrent_requests", "Unknown")
    queue_enabled = getattr()))))))))))))))))))client, "queue_enabled", "Unknown")
    
    print()))))))))))))))))))f"  Queue Enabled: {}}}}queue_enabled}")
    print()))))))))))))))))))f"  Queue Size: {}}}}queue_size}")
    print()))))))))))))))))))f"  Max Concurrent Requests: {}}}}max_concurrent}")
    
    # Backoff configuration
    max_retries = getattr()))))))))))))))))))client, "max_retries", "Unknown")
    initial_delay = getattr()))))))))))))))))))client, "initial_retry_delay", "Unknown")
    backoff_factor = getattr()))))))))))))))))))client, "backoff_factor", "Unknown")
    max_delay = getattr()))))))))))))))))))client, "max_retry_delay", "Unknown")
    
    print()))))))))))))))))))f"  Max Retries: {}}}}max_retries}")
    print()))))))))))))))))))f"  Initial Retry Delay: {}}}}initial_delay}s")
    print()))))))))))))))))))f"  Backoff Factor: {}}}}backoff_factor}")
    print()))))))))))))))))))f"  Max Retry Delay: {}}}}max_delay}s")
    
    # Circuit breaker status
    if hasattr()))))))))))))))))))client, "circuit_state"):
        print()))))))))))))))))))f"  Circuit Breaker State: {}}}}client.circuit_state}")

def run_concurrent_requests()))))))))))))))))))client, api_name, model, num_requests=5, delay=0):
    """Run multiple concurrent requests to demonstrate queue behavior"""
    print_subsection()))))))))))))))))))f"Running {}}}}num_requests} concurrent requests to {}}}}api_name}")
    
    # Generate prompts
    prompts = [],
    f"What is {}}}}i+1} + {}}}}i+2}? Answer with just the number."
        for i in range()))))))))))))))))))num_requests):
            ]
    
    # Track start time
            start_time = time.time())))))))))))))))))))
            results = [],]
    
    with concurrent.futures.ThreadPoolExecutor()))))))))))))))))))max_workers=num_requests) as executor:
        # Submit all requests
        futures = [],]
        for i, prompt in enumerate()))))))))))))))))))prompts):
            # Small delay between submissions if specified:
            if delay > 0 and i > 0:
                time.sleep()))))))))))))))))))delay)
                
            # Create unique request ID
                request_id = f"example_{}}}}api_name}_{}}}}i+1}_{}}}}int()))))))))))))))))))time.time()))))))))))))))))))))}"
            
            # For OpenAI API
            if api_name == "openai":
                future = executor.submit()))))))))))))))))))
                client.chat,
                model,
                [],{}}"role": "user", "content": prompt}],
                max_tokens=10,
                request_id=request_id
                )
            # For Claude API
            elif api_name == "claude":
                future = executor.submit()))))))))))))))))))
                client.chat,
                [],{}}"role": "user", "content": prompt}],
                model=model,
                max_tokens=10,
                request_id=request_id
                )
            # For Groq API
            elif api_name == "groq":
                future = executor.submit()))))))))))))))))))
                client.chat,
                model,
                [],{}}"role": "user", "content": prompt}],
                max_tokens=10,
                request_id=request_id
                )
            # For Ollama API
            elif api_name == "ollama":
                future = executor.submit()))))))))))))))))))
                client.chat,
                model,
                [],{}}"role": "user", "content": prompt}],
                max_tokens=10,
                request_id=request_id
                )
                futures.append()))))))))))))))))))()))))))))))))))))))i+1, future))
        
        # Process results as they complete
        for idx, future in futures:
            try:
                result = future.result())))))))))))))))))))
                # Add timing information
                completion_time = time.time()))))))))))))))))))) - start_time
                result_info = {}}
                "request_num": idx,
                "time_taken": completion_time,
                "response": result.get()))))))))))))))))))"text", str()))))))))))))))))))result)),
                "status": "success"
                }
                results.append()))))))))))))))))))result_info)
                
                print()))))))))))))))))))f"Request {}}}}idx}: Completed in {}}}}completion_time:.2f}s")
                print()))))))))))))))))))f"  Response: {}}}}result_info[],'response'][],:50]}...")
            except Exception as e:
                completion_time = time.time()))))))))))))))))))) - start_time
                result_info = {}}
                "request_num": idx,
                "time_taken": completion_time,
                "error": str()))))))))))))))))))e),
                "status": "error"
                }
                results.append()))))))))))))))))))result_info)
                
                print()))))))))))))))))))f"Request {}}}}idx}: Failed in {}}}}completion_time:.2f}s")
                print()))))))))))))))))))f"  Error: {}}}}str()))))))))))))))))))e)}")
    
    # Sort results by request number
                results.sort()))))))))))))))))))key=lambda x: x[],"request_num"])
    
    # Calculate statistics
                total_time = time.time()))))))))))))))))))) - start_time
    successful = sum()))))))))))))))))))1 for r in results if r[],"status"] == "success"):
    :
        print()))))))))))))))))))f"\nCompleted {}}}}successful}/{}}}}num_requests} requests in {}}}}total_time:.2f} seconds")
        print()))))))))))))))))))f"Average time per request: {}}}}total_time/num_requests:.2f}s")
    
        return results, total_time

def test_with_without_queue()))))))))))))))))))client, api_name, model, num_requests=3):
    """Compare performance with and without queue enabled"""
    print_section()))))))))))))))))))f"Testing {}}}}api_name} API With and Without Queue")
    
    # Check if queue can be enabled/disabled:
    if not hasattr()))))))))))))))))))client, "queue_enabled"):
        print()))))))))))))))))))f"Warning: {}}}}api_name} API does not support enabling/disabling queue")
    return
    
    # Save original settings
    original_queue_enabled = client.queue_enabled
    original_max_concurrent = client.max_concurrent_requests
    
    try:
        # Test WITH queue ()))))))))))))))))))limited concurrency)
        print_subsection()))))))))))))))))))"Testing WITH queue")
        client.queue_enabled = True
        client.max_concurrent_requests = 1  # Limit to force queueing
        
        with_queue_results, with_queue_time = run_concurrent_requests()))))))))))))))))))
        client, api_name, model, num_requests
        )
        
        # Wait a bit before next test
        time.sleep()))))))))))))))))))2)
        
        # Test WITHOUT queue
        print_subsection()))))))))))))))))))"Testing WITHOUT queue")
        client.queue_enabled = False
        
        without_queue_results, without_queue_time = run_concurrent_requests()))))))))))))))))))
        client, api_name, model, num_requests
        )
        
        # Compare results
        print_subsection()))))))))))))))))))"Comparison")
        print()))))))))))))))))))f"WITH queue ()))))))))))))))))))max_concurrent=1): {}}}}with_queue_time:.2f}s total")
        print()))))))))))))))))))f"WITHOUT queue: {}}}}without_queue_time:.2f}s total")
        
        successful_with = sum()))))))))))))))))))1 for r in with_queue_results if r[],"status"] == "success"):
    :    successful_without = sum()))))))))))))))))))1 for r in without_queue_results if r[],"status"] == "success"):
    :    
        print()))))))))))))))))))f"Success rate WITH queue: {}}}}successful_with}/{}}}}num_requests}")
        print()))))))))))))))))))f"Success rate WITHOUT queue: {}}}}successful_without}/{}}}}num_requests}")
        
        # Timing comparison
        if with_queue_time < without_queue_time:
            print()))))))))))))))))))f"Queue was {}}}}()))))))))))))))))))without_queue_time/with_queue_time - 1)*100:.1f}% faster")
        else:
            print()))))))))))))))))))f"Direct requests were {}}}}()))))))))))))))))))with_queue_time/without_queue_time - 1)*100:.1f}% faster")
    
    finally:
        # Restore original settings
        client.queue_enabled = original_queue_enabled
        client.max_concurrent_requests = original_max_concurrent

def test_backoff_behavior()))))))))))))))))))client, api_name, model):
    """Test the exponential backoff behavior"""
    print_section()))))))))))))))))))f"Testing {}}}}api_name} API Backoff Behavior")
    
    # Save original backoff settings
    original_max_retries = getattr()))))))))))))))))))client, "max_retries", 3)
    original_delay = getattr()))))))))))))))))))client, "initial_retry_delay", 1)
    original_factor = getattr()))))))))))))))))))client, "backoff_factor", 2)
    
    try:
        # Configure backoff for demonstration
        if hasattr()))))))))))))))))))client, "max_retries"):
            client.max_retries = 3
        if hasattr()))))))))))))))))))client, "initial_retry_delay"):
            client.initial_retry_delay = 1
        if hasattr()))))))))))))))))))client, "backoff_factor"):
            client.backoff_factor = 2
        
            print()))))))))))))))))))f"Configured backoff: max_retries={}}}}getattr()))))))))))))))))))client, 'max_retries', 3)}, " +
            f"initial_delay={}}}}getattr()))))))))))))))))))client, 'initial_retry_delay', 1)}s, " +
            f"factor={}}}}getattr()))))))))))))))))))client, 'backoff_factor', 2)}")
        
        # Create an intentionally problematic request ()))))))))))))))))))very large input)
            print()))))))))))))))))))"\nSending an intentionally problematic request to trigger backoff...")
        
        # Track start time
            start_time = time.time())))))))))))))))))))
        
        try:
            # For OpenAI API
            if api_name == "openai":
                # Very long input to potentially trigger rate limits
                long_prompt = "Explain quantum physics " * 200  # Very long prompt
                response = client.chat()))))))))))))))))))
                model,
                [],{}}"role": "user", "content": long_prompt}],
                request_id=f"backoff_test_{}}}}int()))))))))))))))))))time.time()))))))))))))))))))))}"
                )
            # For Claude API
            elif api_name == "claude":
                long_prompt = "Explain quantum physics " * 200
                response = client.chat()))))))))))))))))))
                [],{}}"role": "user", "content": long_prompt}],
                model=model,
                request_id=f"backoff_test_{}}}}int()))))))))))))))))))time.time()))))))))))))))))))))}"
                )
            # For Groq API
            elif api_name == "groq":
                long_prompt = "Explain quantum physics " * 200
                response = client.chat()))))))))))))))))))
                model,
                [],{}}"role": "user", "content": long_prompt}],
                request_id=f"backoff_test_{}}}}int()))))))))))))))))))time.time()))))))))))))))))))))}"
                )
            # For Ollama API
            elif api_name == "ollama":
                long_prompt = "Explain quantum physics " * 200
                response = client.chat()))))))))))))))))))
                model,
                [],{}}"role": "user", "content": long_prompt}],
                request_id=f"backoff_test_{}}}}int()))))))))))))))))))time.time()))))))))))))))))))))}"
                )
            
            # Request succeeded
                total_time = time.time()))))))))))))))))))) - start_time
                print()))))))))))))))))))f"Request completed successfully in {}}}}total_time:.2f}s")
                print()))))))))))))))))))f"Response: {}}}}str()))))))))))))))))))response.get()))))))))))))))))))'text', ''))[],:50]}...")
            
        except Exception as e:
            # Request failed
            total_time = time.time()))))))))))))))))))) - start_time
            print()))))))))))))))))))f"Request failed after {}}}}total_time:.2f}s")
            print()))))))))))))))))))f"Error: {}}}}str()))))))))))))))))))e)}")
            
            # Check if backoff was mentioned in the error:
            if "retry" in str()))))))))))))))))))e).lower()))))))))))))))))))) or "backoff" in str()))))))))))))))))))e).lower()))))))))))))))))))):
                print()))))))))))))))))))"Backoff mechanism was triggered as expected")
    
    finally:
        # Restore original settings
        if hasattr()))))))))))))))))))client, "max_retries"):
            client.max_retries = original_max_retries
        if hasattr()))))))))))))))))))client, "initial_retry_delay"):
            client.initial_retry_delay = original_delay
        if hasattr()))))))))))))))))))client, "backoff_factor"):
            client.backoff_factor = original_factor

def test_circuit_breaker()))))))))))))))))))client, api_name):
    """Test the circuit breaker pattern if implemented"""
    print_section()))))))))))))))))))f"Testing {}}}}api_name} API Circuit Breaker")
    
    # Check if circuit breaker is implemented:
    if not hasattr()))))))))))))))))))client, "circuit_state"):
        print()))))))))))))))))))f"Circuit breaker not implemented in {}}}}api_name} API")
    return
    
    # Save original state
    original_state = client.circuit_state
    original_failure_count = getattr()))))))))))))))))))client, "failure_count", 0)
    
    try:
        # Reset circuit state to CLOSED
        client.circuit_state = "CLOSED"
        client.failure_count = 0
        
        print()))))))))))))))))))f"Initial circuit state: {}}}}client.circuit_state}")
        
        # Simulate failures to open the circuit
        failure_threshold = getattr()))))))))))))))))))client, "circuit_failure_threshold", 5)
        print()))))))))))))))))))f"Simulating {}}}}failure_threshold} failures to open circuit...")
        
        # Manually increment failure count
        client.failure_count = failure_threshold
        client.circuit_state = "OPEN"
        
        print()))))))))))))))))))f"Circuit state after failures: {}}}}client.circuit_state}")
        
        # Try to make a request with circuit OPEN
        print()))))))))))))))))))"\nAttempting request with circuit OPEN...")
        try:
            # Create a dummy request that should fail immediately
            if api_name == "openai":
                response = client.chat()))))))))))))))))))
                "gpt-3.5-turbo",
                [],{}}"role": "user", "content": "Hello"}],
                request_id=f"circuit_test_{}}}}int()))))))))))))))))))time.time()))))))))))))))))))))}"
                )
            elif api_name == "claude":
                response = client.chat()))))))))))))))))))
                [],{}}"role": "user", "content": "Hello"}],
                request_id=f"circuit_test_{}}}}int()))))))))))))))))))time.time()))))))))))))))))))))}"
                )
            elif api_name == "groq":
                response = client.chat()))))))))))))))))))
                "llama3-8b-8192",
                [],{}}"role": "user", "content": "Hello"}],
                request_id=f"circuit_test_{}}}}int()))))))))))))))))))time.time()))))))))))))))))))))}"
                )
            elif api_name == "ollama":
                response = client.chat()))))))))))))))))))
                "llama3",
                [],{}}"role": "user", "content": "Hello"}],
                request_id=f"circuit_test_{}}}}int()))))))))))))))))))time.time()))))))))))))))))))))}"
                )
                
                print()))))))))))))))))))"Request unexpectedly succeeded")
        except Exception as e:
            print()))))))))))))))))))f"Request failed as expected: {}}}}str()))))))))))))))))))e)}")
            
            # Check if circuit breaker is mentioned in the error:
            if "circuit" in str()))))))))))))))))))e).lower()))))))))))))))))))) or "open" in str()))))))))))))))))))e).lower()))))))))))))))))))):
                print()))))))))))))))))))"Circuit breaker pattern working correctly")
        
        # Simulate circuit resetting to HALF-OPEN
                print()))))))))))))))))))"\nSimulating timeout and resetting to HALF-OPEN...")
                client.circuit_state = "HALF-OPEN"
                print()))))))))))))))))))f"Circuit state: {}}}}client.circuit_state}")
        
        # Simulate successful request in HALF-OPEN state
                print()))))))))))))))))))"\nSimulating successful request in HALF-OPEN state...")
                client.success_count = getattr()))))))))))))))))))client, "circuit_success_threshold", 3)
                client.circuit_state = "CLOSED"
        
                print()))))))))))))))))))f"Final circuit state: {}}}}client.circuit_state}")
        
    finally:
        # Restore original state
        client.circuit_state = original_state
        if hasattr()))))))))))))))))))client, "failure_count"):
            client.failure_count = original_failure_count

def test_different_queue_sizes()))))))))))))))))))client, api_name, model):
    """Test with different queue size configurations"""
    print_section()))))))))))))))))))f"Testing {}}}}api_name} API with Different Queue Sizes")
    
    # Check if queue size can be configured:
    if not hasattr()))))))))))))))))))client, "queue_size"):
        print()))))))))))))))))))f"Queue size not configurable in {}}}}api_name} API")
    return
    
    # Save original settings
    original_queue_size = client.queue_size
    original_max_concurrent = client.max_concurrent_requests
    
    try:
        # Set max_concurrent_requests to 1 to force queueing
        client.max_concurrent_requests = 1
        
        # Test with different queue sizes
        queue_sizes = [],3, 10]
        for size in queue_sizes:
            client.queue_size = size
            print_subsection()))))))))))))))))))f"Testing with queue_size = {}}}}size}")
            
            # Run concurrent requests ()))))))))))))))))))one more than queue size to potentially overflow)
            num_requests = size + 1
            try:
                results, total_time = run_concurrent_requests()))))))))))))))))))
                client, api_name, model, num_requests, delay=0.1
                )
                
                successful = sum()))))))))))))))))))1 for r in results if r[],"status"] == "success"):
                    :            print()))))))))))))))))))f"\nQueue size {}}}}size}: {}}}}successful}/{}}}}num_requests} successful")
                    print()))))))))))))))))))f"Total time: {}}}}total_time:.2f}s")
                
            except Exception as e:
                print()))))))))))))))))))f"Test failed: {}}}}str()))))))))))))))))))e)}")
    
    finally:
        # Restore original settings
        client.queue_size = original_queue_size
        client.max_concurrent_requests = original_max_concurrent

def test_request_tracking()))))))))))))))))))client, api_name, model):
    """Test request tracking with specific request IDs"""
    print_section()))))))))))))))))))f"Testing {}}}}api_name} API Request Tracking")
    
    # Generate unique request ID
    request_id = f"test_{}}}}api_name}_{}}}}int()))))))))))))))))))time.time()))))))))))))))))))))}"
    print()))))))))))))))))))f"Using request ID: {}}}}request_id}")
    
    try:
        # Make request with specified ID
        start_time = time.time())))))))))))))))))))
        
        if api_name == "openai":
            response = client.chat()))))))))))))))))))
            model,
            [],{}}"role": "user", "content": "Hello, how are you?"}],
            request_id=request_id
            )
        elif api_name == "claude":
            response = client.chat()))))))))))))))))))
            [],{}}"role": "user", "content": "Hello, how are you?"}],
            model=model,
            request_id=request_id
            )
        elif api_name == "groq":
            response = client.chat()))))))))))))))))))
            model,
            [],{}}"role": "user", "content": "Hello, how are you?"}],
            request_id=request_id
            )
        elif api_name == "ollama":
            response = client.chat()))))))))))))))))))
            model,
            [],{}}"role": "user", "content": "Hello, how are you?"}],
            request_id=request_id
            )
        
            completion_time = time.time()))))))))))))))))))) - start_time
            print()))))))))))))))))))f"Request completed in {}}}}completion_time:.2f}s")
            print()))))))))))))))))))f"Response: {}}}}str()))))))))))))))))))response.get()))))))))))))))))))'text', ''))[],:50]}...")
        
        # Check if request ID is included in response:
        if isinstance()))))))))))))))))))response, dict) and "request_id" in response:
            print()))))))))))))))))))f"Request ID in response: {}}}}response[],'request_id']}")
            
            if response[],"request_id"] == request_id:
                print()))))))))))))))))))"Request tracking working correctly - ID in response matches request")
        else:
            print()))))))))))))))))))"Request ID not included in response ()))))))))))))))))))this is normal for some APIs)")
    
    except Exception as e:
        print()))))))))))))))))))f"Request failed: {}}}}str()))))))))))))))))))e)}")

def main()))))))))))))))))))):
    print_section()))))))))))))))))))"IPFS Accelerate API Queue and Backoff Example")
    print()))))))))))))))))))f"Time: {}}}}datetime.now()))))))))))))))))))).isoformat())))))))))))))))))))}")
    
    # Initialize API clients
    apis = {}}}
    models = {}}}
    
    # Try to initialize OpenAI client
    try:
        api_key = os.environ.get()))))))))))))))))))"OPENAI_API_KEY")
        if api_key:
            apis[],"openai"] = openai_api()))))))))))))))))))metadata={}}"openai_api_key": api_key})
            models[],"openai"] = "gpt-3.5-turbo"
            print()))))))))))))))))))"✓ OpenAI API initialized")
        else:
            print()))))))))))))))))))"✗ OpenAI API key not found in environment")
    except Exception as e:
        print()))))))))))))))))))f"✗ Failed to initialize OpenAI API: {}}}}str()))))))))))))))))))e)}")
    
    # Try to initialize Claude client
    try:
        api_key = os.environ.get()))))))))))))))))))"ANTHROPIC_API_KEY")
        if api_key:
            apis[],"claude"] = claude()))))))))))))))))))metadata={}}"anthropic_api_key": api_key})
            models[],"claude"] = "claude-3-haiku-20240307"
            print()))))))))))))))))))"✓ Claude API initialized")
        else:
            print()))))))))))))))))))"✗ Claude API key not found in environment")
    except Exception as e:
        print()))))))))))))))))))f"✗ Failed to initialize Claude API: {}}}}str()))))))))))))))))))e)}")
    
    # Try to initialize Groq client
    try:
        api_key = os.environ.get()))))))))))))))))))"GROQ_API_KEY")
        if api_key:
            apis[],"groq"] = groq()))))))))))))))))))metadata={}}"groq_api_key": api_key})
            models[],"groq"] = "llama3-8b-8192"
            print()))))))))))))))))))"✓ Groq API initialized")
        else:
            print()))))))))))))))))))"✗ Groq API key not found in environment")
    except Exception as e:
        print()))))))))))))))))))f"✗ Failed to initialize Groq API: {}}}}str()))))))))))))))))))e)}")
    
    # Try to initialize Ollama client
    try:
        host = os.environ.get()))))))))))))))))))"OLLAMA_HOST", "http://localhost:11434")
        apis[],"ollama"] = ollama()))))))))))))))))))metadata={}}"ollama_host": host})
        models[],"ollama"] = "llama3"
        print()))))))))))))))))))"✓ Ollama API initialized")
    except Exception as e:
        print()))))))))))))))))))f"✗ Failed to initialize Ollama API: {}}}}str()))))))))))))))))))e)}")
    
    # Print configuration status for each API
        print_section()))))))))))))))))))"API Configurations")
    for api_name, client in apis.items()))))))))))))))))))):
        print_api_status()))))))))))))))))))client, api_name)
    
    # Run tests for each available API
    for api_name, client in apis.items()))))))))))))))))))):
        if api_name not in models:
            print()))))))))))))))))))f"Skipping tests for {}}}}api_name} - no model specified")
        continue
            
        model = models[],api_name]
        
        # Test with and without queue
        test_with_without_queue()))))))))))))))))))client, api_name, model, num_requests=3)
        
        # Test backoff behavior
        test_backoff_behavior()))))))))))))))))))client, api_name, model)
        
        # Test circuit breaker ()))))))))))))))))))if implemented)
        test_circuit_breaker()))))))))))))))))))client, api_name)
        
        # Test different queue sizes
        test_different_queue_sizes()))))))))))))))))))client, api_name, model)
        
        # Test request tracking
        test_request_tracking()))))))))))))))))))client, api_name, model)
    
        print_section()))))))))))))))))))"Example Complete")
        print()))))))))))))))))))"This example demonstrated the queue and backoff features of the IPFS Accelerate API backends.")
        print()))))))))))))))))))"For more information, see QUEUE_BACKOFF_GUIDE.md")
:
if __name__ == "__main__":
    main())))))))))))))))))))