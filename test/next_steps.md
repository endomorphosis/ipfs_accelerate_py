# Template Migration Project - Next Steps

## Progress Summary (March 10, 2025)

We've made significant progress with the template system migration:

- ✅ Analyzed the current template database structure
- ✅ Identified 26 templates in the database, with 16/26 (61.5%) having valid syntax
- ✅ Successfully used the existing template system to generate a working BERT test file
- ✅ Validated that the generated test file runs correctly on both CPU and CUDA hardware
- ✅ Created a basic DuckDB migration script (waiting on DuckDB installation)
- ✅ Evaluated and attempted to fix the templates with syntax errors

## Remaining Work

1. **Fix Remaining Templates with Syntax Errors**:
   - 10/26 templates still have syntax errors that need to be fixed manually
   - Key templates to fix:
     - `text_embedding_test_template_text_embedding.py`
     - `vision_test_template_vision.py`
     - `detr_test_template_detr.py`
     - `video_test_template_video.py`

2. **Complete DuckDB Migration**:
   - Install DuckDB in a Python virtual environment
   - Run the migration script to convert JSON to DuckDB
   - Update the test generator to use DuckDB instead of JSON

3. **Update Hardware Platform Support**:
   - Ensure all templates support all 8 hardware platforms
   - Key platforms to verify: ROCm, MPS, OpenVINO, Qualcomm, WebNN, WebGPU

4. **Enhance Template Placeholder Handling**:
   - Fix template placeholder issues
   - Add proper validation for template variables
   
5. **Implement Template Inheritance System**:
   - Enhance the template system to support inheritance
   - Create base templates for common functionality

6. **Comprehensive Testing**:
   - Create tests for each template model type
   - Verify hardware compatibility across platforms

## Action Plan for Next Session

1. Manually fix the 10 templates with syntax errors:
   - Prioritize `text_embedding_test_template_text_embedding.py` (key template)
   - Focus on bracket mismatches and indentation issues

2. Create a Python virtual environment for DuckDB:
   ```bash
   python3 -m venv ../template_venv
   source ../template_venv/bin/activate
   pip install duckdb
   cd ..
   python generators/duckdb/create_template_db.py
   ```

3. Update the test generator to use the DuckDB database:
   - Add DuckDB support to `create_template_based_test_generator.py`
   - Add a `--use-duckdb` flag to prefer DuckDB over JSON

4. Expand hardware support in the template validator:
   - Ensure all templates support all hardware platforms
   - Add validation functions specific to each platform

## DuckDB Migration Benefits

Moving to DuckDB will provide:

- Faster template queries with SQL support
- Better integrity constraints to prevent invalid templates
- Easier template management and querying
- Improved scalability for hundreds of templates
- Better concurrent access for CI/CD pipelines

## Hardware Support Status (Current)

| Hardware Platform | Templates Supporting | % Coverage |
|-------------------|----------------------|------------|
| CPU               | 26/26                | 100%       |
| CUDA              | 26/26                | 100%       |
| ROCm              | 16/26                | 61.5%      |
| MPS               | 16/26                | 61.5%      |
| OpenVINO          | 16/26                | 61.5%      |
| Qualcomm          | 16/26                | 61.5%      |
| WebNN             | 16/26                | 61.5%      |
| WebGPU            | 16/26                | 61.5%      |

## Template Type Status

| Model Type        | Valid Templates | Total | % Valid |
|-------------------|----------------|-------|---------|
| bert              | 1/1            | 1     | 100%    |
| vision            | 1/2            | 2     | 50%     |
| text_embedding    | 1/2            | 2     | 50%     |
| llama             | 0/1            | 1     | 0%      |
| t5                | 0/1            | 1     | 0%      |
| vit               | 1/1            | 1     | 100%    |
| others...         | -              | -     | -       |

The team should focus next on fixing high-priority templates that currently have syntax errors, particularly those related to common model types.
