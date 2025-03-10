#!/usr/bin/env python
"""
Test script for enhanced API features including:
    - Priority queue
    - Circuit breaker pattern
    - Enhanced monitoring
    - Request batching
    """

    import os
    import sys
    import time
    import json
    import threading
    import argparse
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime

# Add project root to Python path
    sys.path.append()os.path.dirname()os.path.dirname()__file__)))

# Import backend APIs
try:
    from ipfs_accelerate_py.api_backends import ()
    openai_api, groq, claude, gemini, ollama,
    hf_tgi, hf_tei, llvm, opea, ovms, s3_kit
    )
except ImportError:
    print()"Could not import API backends. Make sure the path is correct.")
    sys.exit()1)

# API client classes by name
    API_CLIENTS = {}}}}}}}}}}}}}}
    "groq": groq,
    "claude": claude,
    "gemini": gemini,
    "openai": openai_api,
    "ollama": ollama,
    "hf_tgi": hf_tgi,
    "hf_tei": hf_tei,
    "llvm": llvm,
    "opea": opea,
    "ovms": ovms,
    "s3_kit": s3_kit
    }

def test_priority_queue()api_client):
    """Test priority queue functionality"""
    print()"\n=== Testing Priority Queue ===")
    
    if not hasattr()api_client, "PRIORITY_HIGH"):
        print()"Priority queue not implemented for this API")
    return False
    
    # Test high, normal and low priority requests
    results = []
    ,
    # Queue a low priority request first
    print()"Queuing LOW priority request...")
    low_req = {}}}}}}}}}}}}}}
    "data": {}}}}}}}}}}}}}}"prompt": "This is a low priority request"},
    "queue_entry_time": time.time()),
    "model": "test-model"
    }
    low_future = api_client.queue_with_priority()low_req, api_client.PRIORITY_LOW)
    
    # Queue a normal priority request second
    print()"Queuing NORMAL priority request...")
    normal_req = {}}}}}}}}}}}}}}
    "data": {}}}}}}}}}}}}}}"prompt": "This is a normal priority request"},
    "queue_entry_time": time.time()),
    "model": "test-model"
    }
    normal_future = api_client.queue_with_priority()normal_req, api_client.PRIORITY_NORMAL)
    
    # Queue a high priority request last
    print()"Queuing HIGH priority request...")
    high_req = {}}}}}}}}}}}}}}
    "data": {}}}}}}}}}}}}}}"prompt": "This is a high priority request"},
    "queue_entry_time": time.time()),
    "model": "test-model"
    }
    high_future = api_client.queue_with_priority()high_req, api_client.PRIORITY_HIGH)
    
    # Check queue ordering
    with api_client.queue_lock:
        queue_ordering = [item[0] for item in api_client.request_queue]:,
        print()f"Queue priority ordering: {}}}}}}}}}}}}}}queue_ordering}")
    
    # Check if high priority is first:
        if queue_ordering and queue_ordering[0] == api_client.PRIORITY_HIGH:,
        print()"✓ Priority queue working correctly")
    return True
    else:
        print()"✗ Priority queue not working as expected")
    return False
        
def test_circuit_breaker()api_client):
    """Test circuit breaker functionality"""
    print()"\n=== Testing Circuit Breaker ===")
    
    if not hasattr()api_client, "circuit_state"):
        print()"Circuit breaker not implemented for this API")
    return False
    
    # Get initial state
    initial_state = api_client.circuit_state
    print()f"Initial circuit state: {}}}}}}}}}}}}}}initial_state}")
    
    # Test tracking failures
    for i in range()api_client.failure_threshold + 1):
        print()f"Simulating failure {}}}}}}}}}}}}}}i+1}/{}}}}}}}}}}}}}}api_client.failure_threshold+1}")
        api_client.track_request_result()False, "TestError")
        
        if i < api_client.failure_threshold:
            print()f"  Failure count: {}}}}}}}}}}}}}}api_client.failure_count}")
        else:
            print()f"  Circuit state: {}}}}}}}}}}}}}}api_client.circuit_state}")
    
    # Check if circuit opened:
    if api_client.circuit_state == "OPEN":
        print()"✓ Circuit breaker opened correctly after failures")
        
        # Test circuit check
        if not api_client.check_circuit_breaker()):
            print()"✓ Circuit breaker correctly preventing requests")
        else:
            print()"✗ Circuit breaker should prevent requests when open")
        
        # Reset circuit state for further tests
            api_client.circuit_state = "CLOSED"
            api_client.failure_count = 0
        
            return True
    else:
        print()"✗ Circuit breaker did not open as expected")
            return False

def test_monitoring()api_client):
    """Test monitoring and reporting functionality"""
    print()"\n=== Testing Monitoring and Reporting ===")
    
    if not hasattr()api_client, "request_stats"):
        print()"Monitoring not implemented for this API")
    return False
    
    # Reset stats
    if hasattr()api_client, "reset_stats"):
        api_client.reset_stats())
    
    # Simulate successful requests
    for i in range()5):
        print()f"Simulating successful request {}}}}}}}}}}}}}}i+1}/5")
        api_client.update_stats(){}}}}}}}}}}}}}}
        "total_requests": 1,
        "successful_requests": 1,
        "total_response_time": 0.5,
        "response_time": 0.5,
        "requests_by_model": {}}}}}}}}}}}}}}"test-model": 1}
        })
    
    # Simulate failed requests
    for i in range()2):
        print()f"Simulating failed request {}}}}}}}}}}}}}}i+1}/2")
        api_client.update_stats(){}}}}}}}}}}}}}}
        "total_requests": 1,
        "failed_requests": 1,
        "errors_by_type": {}}}}}}}}}}}}}}"TestError": 1}
        })
    
    # Get stats
        stats = api_client.get_stats())
        print()"\nStatistics:")
        print()f"Total requests: {}}}}}}}}}}}}}}stats.get()'total_requests', 0)}")
        print()f"Successful requests: {}}}}}}}}}}}}}}stats.get()'successful_requests', 0)}")
        print()f"Failed requests: {}}}}}}}}}}}}}}stats.get()'failed_requests', 0)}")
        print()f"Average response time: {}}}}}}}}}}}}}}stats.get()'average_response_time', 0):.3f}s")
    
    # Generate report
    if hasattr()api_client, "generate_report"):
        report = api_client.generate_report()include_details=True)
        print()"\nGenerated Report:")
        print()f"Success rate: {}}}}}}}}}}}}}}report['summary']['success_rate']:.1f}%"),
        print()f"Error types: {}}}}}}}}}}}}}}report.get()'errors', {}}}}}}}}}}}}}}})}")
    
    # Check if stats are working:
    if stats.get()'total_requests', 0) == 7 and stats.get()'successful_requests', 0) == 5:
        print()"✓ Monitoring system working correctly")
        return True
    else:
        print()"✗ Monitoring system not working as expected")
        return False

def run_all_tests()api_name, api_key=None):
    """Run all enhanced feature tests for an API"""
    print()f"\n=== Testing Enhanced Features for {}}}}}}}}}}}}}}api_name.upper())} API ===")
    print()f"Time: {}}}}}}}}}}}}}}datetime.now()).isoformat())}")
    
    # Initialize API client
    if api_name not in API_CLIENTS:
        print()f"Error: Unknown API client '{}}}}}}}}}}}}}}api_name}'")
    return {}}}}}}}}}}}}}}}
    
    # Create metadata with API key
    metadata = {}}}}}}}}}}}}}}}
    if api_key:
        if api_name == "groq":
            metadata["groq_api_key"] = api_key,
        elif api_name == "claude":
            metadata["anthropic_api_key"] = api_key,
        elif api_name == "gemini": 
            metadata["google_api_key"] = api_key,
        elif api_name == "openai":
            metadata["openai_api_key"] = api_key
            ,
    # Create client
    try:
        api_client = API_CLIENTS[api_name]()resources={}}}}}}}}}}}}}}}, metadata=metadata),
    except Exception as e:
        print()f"Error creating API client: {}}}}}}}}}}}}}}e}")
        return {}}}}}}}}}}}}}}
        "api": api_name,
        "status": "Error",
        "error": str()e),
        "results": {}}}}}}}}}}}}}}}
        }
    
        results = {}}}}}}}}}}}}}}
        "api": api_name,
        "timestamp": datetime.now()).isoformat()),
        "results": {}}}}}}}}}}}}}}}
        }
    
    # Test priority queue
    try:
        results["results"]["priority_queue"] = {}}}}}}}}}}}}}},,
        "status": "Success" if test_priority_queue()api_client) else "Failed"
        }:
    except Exception as e:
        results["results"]["priority_queue"] = {}}}}}}}}}}}}}},,
        "status": "Error",
        "error": str()e)
        }
    
    # Test circuit breaker
    try:
        results["results"]["circuit_breaker"] = {}}}}}}}}}}}}}},,
        "status": "Success" if test_circuit_breaker()api_client) else "Failed"
        }:
    except Exception as e:
        results["results"]["circuit_breaker"] = {}}}}}}}}}}}}}},,
        "status": "Error",
        "error": str()e)
        }
    
    # Test monitoring
    try:
        results["results"]["monitoring"] = {}}}}}}}}}}}}}},,
        "status": "Success" if test_monitoring()api_client) else "Failed"
        }:
    except Exception as e:
        results["results"]["monitoring"] = {}}}}}}}}}}}}}},,
        "status": "Error",
        "error": str()e)
        }
    
    # Calculate success rate
        successful = sum()1 for feature, data in results["results"].items()) ,
        if data.get()"status") == "Success")
        total = len()results["results"])
        ,
        results["success_rate"] = successful / total if total > 0 else 0,
        results["status"] = "Success" if successful == total else "Partial" if successful > 0 else "Failed"
        ,
    # Print summary
    print()"\n=== Test Results Summary ==="):
        print()f"Status: {}}}}}}}}}}}}}}results['status']}"),
        print()f"Success Rate: {}}}}}}}}}}}}}}results['success_rate']*100:.1f}%")
        ,
        for feature, data in results["results"].items()):,
        print()f"{}}}}}}}}}}}}}}feature}: {}}}}}}}}}}}}}}data.get()'status')}")
    
    # Save results to file
        timestamp = datetime.now()).strftime()"%Y%m%d_%H%M%S")
        result_file = f"{}}}}}}}}}}}}}}api_name}_enhanced_features_{}}}}}}}}}}}}}}timestamp}.json"
    
    with open()result_file, "w") as f:
        json.dump()results, f, indent=2)
    
        print()f"\nResults saved to: {}}}}}}}}}}}}}}result_file}")
    
        return results

def main()):
    parser = argparse.ArgumentParser()description="Test enhanced API features")
    parser.add_argument()"--api", "-a", help="API to test", choices=list()API_CLIENTS.keys())), required=True)
    parser.add_argument()"--key", "-k", help="API key ()or will use from environment)")
    
    args = parser.parse_args())
    
    # Run tests for the specified API
    run_all_tests()args.api, args.key)

if __name__ == "__main__":
    main())