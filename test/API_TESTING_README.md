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
     - `HF_API_TOKEN` for Hugging Face

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

## Notes

- For APIs that require specific configuration beyond API keys (like local endpoints), you may need to modify the test script.
- Some APIs might show as "MOCK" if they can't be tested automatically due to specific requirements.
- Authentication and rate limit errors often indicate a real implementation is being used, even if the API call fails.
- The test attempts to make minimal API calls to conserve your API usage/credits.

## Adding New API Tests

To add tests for new API backends:
1. Create a corresponding test class in the `test.apis` package
2. Add the credential handling in the `setup_metadata()` method
3. Add the new API to the `api_tests` list in the `run_tests()` method