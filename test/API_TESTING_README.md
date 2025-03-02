# API Backend Implementation Testing

This directory contains tools to test the IPFS Accelerate API backends and verify whether they are using real API implementations or mock objects.

## Getting Started

To test the API backends, follow these steps:

1. **Set up API credentials**:
   - Copy the `api_credentials.json.template` file to `~/.ipfs_api_credentials`
   - Fill in your API keys for the services you want to test
   - Alternatively, set the appropriate environment variables:
     - `OPENAI_API_KEY` for OpenAI
     - `ANTHROPIC_API_KEY` for Claude
     - `GOOGLE_API_KEY` for Gemini
     - `GROQ_API_KEY` for Groq
     - `HF_API_KEY` or `HF_API_TOKEN` for Hugging Face

2. **Run the test script**:
   ```bash
   python3 test_api_real_implementation.py
   ```

3. **Review the results**:
   - The script will create two files in the `api_test_results` directory:
     - A JSON file with detailed test results
     - A Markdown report summarizing the implementation status of each API

## Understanding the Results

The test checks each API by:
1. Running standard tests (checking function calls and basic structure)
2. Attempting to make real API calls to detect real vs. mock implementations
3. Analyzing responses and errors to determine implementation type

Each API will be classified as:
- **REAL**: Successfully verified as a real implementation
- **MOCK**: Determined to be a mock/stub implementation
- **ERROR**: Could not determine implementation type due to errors

## Available API Tests

The testing framework supports the following API backends:

| API | Basic Test | Unified Test | Configuration |
|-----|-------------|--------------|---------------|
| OpenAI | `test_openai_api.py` | ✅ Integrated | `OPENAI_API_KEY` |
| Claude | `test_claude.py` | ✅ Integrated | `ANTHROPIC_API_KEY` |
| Gemini | `test_gemini.py` | ✅ Integrated | `GOOGLE_API_KEY` |
| Groq | `test_groq.py` | ✅ Integrated | `GROQ_API_KEY` |
| Hugging Face TGI | `test_hf_tgi.py` | `test_hf_tgi_unified.py` | `HF_API_KEY` |
| Hugging Face TEI | `test_hf_tei.py` | `test_hf_tei_unified.py` | `HF_API_KEY` |
| Ollama | `test_ollama.py` | `test_ollama_unified.py` | Local endpoint |
| OVMS | `test_ovms.py` | `test_ovms_unified.py` | Local endpoint |
| LLVM | `test_llvm.py` | `test_llvm_unified.py` | Local endpoint |
| OPEA | `test_opea.py` | ⚠️ Pending | Local endpoint |
| S3 Kit | `test_s3_kit.py` | ⚠️ Pending | AWS credentials |

### Implementation Status and Advanced Test Tools

#### Basic Implementation Status

To check the overall implementation status of all APIs, run:

```bash
python3 check_all_api_implementation.py
```

This generates a comprehensive report showing:
- Implementation type (REAL, MOCK, PARTIAL)
- Availability of key features like endpoint management
- API structure conformance 
- Authentication handling

#### Advanced Queue and Backoff Testing

New test tools are available to validate our enhanced queue and backoff implementations:

```bash
# Test a specific API's queue and backoff systems
python test_api_backoff_queue.py --api claude

# Run comprehensive tests for all API backends
python run_queue_backoff_tests.py

# Run a specific set of API tests
python run_queue_backoff_tests.py --apis openai claude groq

# Skip specific APIs in the test suite
python run_queue_backoff_tests.py --skip-apis llvm s3_kit

# Run comprehensive Ollama-specific tests
python test_ollama_backoff_comprehensive.py
```

These tests verify:
- **Thread-safe request queues** - Concurrent request handling with proper locking
- **Exponential backoff** - Rate limit handling with progressive retry delays
- **Circuit breaker pattern** - Service outage detection and recovery
- **Priority queueing** - Processing of high-priority requests before lower priority ones
- **Request tracking** - Generation and handling of unique request IDs

## Specialized API Tests

Some API backends have specialized test suites:

### Hugging Face TGI (Text Generation Inference)

For testing the HuggingFace Text Generation Inference API and containers:

```bash
# Run standard API tests
python3 apis/test_hf_tgi_unified.py --standard

# Run container tests
python3 apis/test_hf_tgi_unified.py --container

# Run all tests
python3 apis/test_hf_tgi_unified.py --all
```

See `apis/HF_TGI_TESTING_README.md` for more details.

### Hugging Face TEI (Text Embedding Inference)

For testing the HuggingFace Text Embedding Inference API and containers:

```bash
# Run standard API tests
python3 apis/test_hf_tei_unified.py --standard

# Run container tests
python3 apis/test_hf_tei_unified.py --container

# Run performance tests
python3 apis/test_hf_tei_unified.py --performance

# Run all tests
python3 apis/test_hf_tei_unified.py --all
```

See `apis/HF_TEI_TESTING_README.md` for more details.

### Ollama

For testing the Ollama API with local LLM deployments:

```bash
# Run standard API tests
python3 apis/test_ollama_unified.py --standard

# Run performance tests
python3 apis/test_ollama_unified.py --performance

# Run real connection tests to a local Ollama server
python3 apis/test_ollama_unified.py --real

# Run all tests
python3 apis/test_ollama_unified.py --all
```

See `apis/OLLAMA_TESTING_README.md` for more details.

### OpenVINO Model Server (OVMS)

For testing the OpenVINO Model Server API for optimized inference:

```bash
# Run standard API tests
python3 apis/test_ovms_unified.py --standard

# Run performance tests
python3 apis/test_ovms_unified.py --performance

# Run real connection tests to a running OVMS server
python3 apis/test_ovms_unified.py --real

# Run all tests
python3 apis/test_ovms_unified.py --all
```

See `apis/OVMS_TESTING_README.md` for more details.

### LLVM API

For testing the LLVM API for optimized model execution:

```bash
# Run standard API tests
python3 apis/test_llvm_unified.py --standard

# Run performance tests
python3 apis/test_llvm_unified.py --performance

# Run real connection tests to a running LLVM server
python3 apis/test_llvm_unified.py --real

# Run all tests
python3 apis/test_llvm_unified.py --all
```

See `apis/LLVM_TESTING_README.md` for more details.

## Notes

- For APIs that require specific configuration beyond API keys (like local endpoints), you may need to modify the test script.
- Some APIs might show as "MOCK" if they can't be tested automatically due to specific requirements.
- Authentication and rate limit errors often indicate a real implementation is being used, even if the API call fails.
- The test attempts to make minimal API calls to conserve your API usage/credits.
- Container tests require Docker to be installed and running.

## Adding New API Tests

To add tests for new API backends:
1. Create a corresponding test class in the `test.apis` package
2. Add the credential handling in the `setup_metadata()` method
3. Add the new API to the `api_tests` list in the `run_tests()` method
4. Create any specialized test documentation in an appropriately named README file

## Troubleshooting

- **Missing API Keys**: Double-check your environment variables or credentials file
- **Container Issues**: Ensure Docker is running if testing containers
- **Test Timeouts**: For large models or slow connections, increase the timeout parameter
- **Port Conflicts**: If testing containers, ensure the necessary ports are available
- **GPU Access**: For GPU-dependent tests, verify CUDA is properly configured