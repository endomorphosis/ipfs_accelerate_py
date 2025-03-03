# Advanced API Features Test Commands

This document provides test commands and examples for verifying the advanced features implemented in the API backends.

## Testing Priority Queue System

Test the priority queue system with varying priority levels:

```bash
# Test all features of a specific API backend
python test_enhanced_api_features.py --api openai

# Test with your own API key
python test_enhanced_api_features.py --api groq --key your-api-key
```

## Testing Circuit Breaker Pattern

Verify the circuit breaker pattern functionality:

```bash
# Test circuit breaker with forced failures
python test_circuit_breaker.py --api openai --failure-count 6

# Test circuit breaker self-healing
python test_circuit_breaker.py --api claude --half-open
```

## Testing Enhanced Monitoring and Reporting

Check the monitoring and reporting capabilities:

```bash
# Generate a detailed report for an API
python test_enhanced_api_features.py --api openai --report-only

# Test monitoring with concurrent requests
python test_enhanced_api_features.py --api groq --concurrent 10
```

## Testing Request Batching

Test request batching optimization:

```bash
# Test embedding batching capabilities
python test_enhanced_api_features.py --api openai --batch-test

# Compare performance with and without batching
python test_enhanced_api_features.py --api openai --compare-batching
```

## Testing API Key Multiplexing

Test API key multiplexing capabilities:

```bash
# Test with basic round-robin strategy
python test_api_multiplexing_enhanced.py --api openai --strategy round-robin

# Test with least-loaded strategy
python test_api_multiplexing_enhanced.py --api openai --strategy least-loaded

# Test with high concurrency
python test_api_multiplexing_enhanced.py --api openai --concurrent 20
```

## Running Comprehensive Tests

Run all advanced feature tests:

```bash
# Run all tests for all API backends
python run_queue_backoff_tests.py

# Test specific APIs only
python run_queue_backoff_tests.py --apis openai groq claude

# Skip specific APIs
python run_queue_backoff_tests.py --skip-apis vllm opea ovms

# Run comprehensive Ollama tests
python test_ollama_backoff_comprehensive.py --model llama3
```

## Testing Specific API Backends

Commands for testing individual API backends:

```bash
# Test OpenAI API
python test_enhanced_api_features.py --api openai

# Test Claude API
python test_enhanced_api_features.py --api claude

# Test Groq API
python test_enhanced_api_features.py --api groq

# Test Ollama API
python test_ollama_backoff_comprehensive.py

# Test HF TGI API
python test_enhanced_api_features.py --api hf_tgi

# Test HF TEI API
python test_enhanced_api_features.py --api hf_tei
```

## Verifying Implementation Status

Test the implementation status of advanced features:

```bash
# Check implementation status for all APIs
python check_api_implementation.py

# Generate a detailed report
python check_api_implementation.py --report-file advanced_features_status.json
```

## Troubleshooting Tests

If you encounter issues with the tests:

```bash
# Enable detailed logging
python test_enhanced_api_features.py --api openai --verbose

# Run with debug mode
python test_enhanced_api_features.py --api openai --debug

# Mock API responses for testing
python test_enhanced_api_features.py --api openai --mock
```