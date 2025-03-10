# Template Database System

This document describes the DuckDB-based template database system that replaces static file templates with database-driven templates. This implementation completes the "Migrate generators to use database templates instead of static files" task that was marked at 95% complete.

## Overview

The template database system provides a centralized, queryable storage solution for model test, benchmark, and skill templates. It addresses several limitations of the previous static file approach:

1. **Reduced File Redundancy**: Instead of hundreds of static template files for different model types, templates are stored in a single, efficient database
2. **Hardware-Specific Templates**: Specialized templates for different hardware platforms (CUDA, ROCm, WebNN, etc.)
3. **Template Inheritance**: Fallback mechanisms to use default templates when specific ones aren't available
4. **Efficient Querying**: Fast retrieval of templates based on model family, template type, and hardware platform 
5. **Centralized Management**: Easy administration and template sharing across team members

## Getting Started

### Prerequisites

- **DuckDB**: The system uses DuckDB for storing templates. It's already installed in the project's virtual environment.
- **Python 3.9+**: The system uses f-strings and other modern Python features.

### Template Database Management

```bash
# Create a new template database (will overwrite if exists)
python create_template_database.py --create

# List available templates in the database
python create_template_database.py --list

# Export templates to JSON file (useful for backup or sharing)
python create_template_database.py --export --json-path templates.json

# Import templates from JSON file (useful for restoration or migration)
python create_template_database.py --import --json-path templates.json

# Validate the template database
python validate_db_templates.py
```

The validation script checks if:
1. The database file exists
2. Templates can be listed successfully
3. Templates can be retrieved using the fallback logic

## Using Database Templates with Generators

The test generator has been updated to use database templates instead of static files, completing the migration goal. When the `--use-db-templates` flag is specified, the generator will:

1. Attempt to load the appropriate template from the database
2. Apply the fallback logic if needed
3. Render the template with the provided context variables

### Test Generator with Database Templates

```bash
# Generate a test using database templates
python test_generator_with_resource_pool.py --model bert-base-uncased --use-db-templates

# Generate a hardware-specific test with database templates
python test_generator_with_resource_pool.py --model bert-base-uncased --use-db-templates --device cuda

# Specify a custom database path
python test_generator_with_resource_pool.py --model bert-base-uncased --use-db-templates --db-path ./custom_templates.duckdb
```

### Performance Improvements

The database-driven approach offers performance benefits:
- **Faster generation**: Only the needed template is loaded from the database
- **Reduced memory usage**: Templates are loaded on-demand rather than all at once
- **Efficient updates**: Templates can be updated in a single location

## Database Schema and Implementation

### DuckDB Schema

The template database uses a simple, efficient schema optimized for template storage and retrieval:

```sql
CREATE TABLE templates (
    model_type VARCHAR,        -- Model family (bert, t5, vit, etc.)
    template_type VARCHAR,     -- Type of template (test, benchmark, skill)
    template TEXT,             -- Actual template content
    hardware_platform VARCHAR  -- Hardware platform (NULL for generic templates)
)
```

### Implementation Details

The template database is implemented in DuckDB, a high-performance analytical database system that:
- Supports SQL for template queries
- Offers excellent performance for this use case
- Has a small footprint
- Allows direct access to templates without file I/O operations

The schema is intentionally simple to minimize overhead and maximize query performance. Templates can be several KB in size, so storing them efficiently is important.

## Intelligent Template Selection and Rendering

### Fallback Logic

The template selection system uses an intelligent fallback mechanism to find the most appropriate template:

1. **Hardware-Specific Template**: First attempts to find a template specific to the requested model type, template type, and hardware platform
   ```sql
   SELECT template FROM templates
   WHERE model_type = ? AND template_type = ? AND hardware_platform = ?
   ```

2. **Generic Model Template**: If no hardware-specific template exists, falls back to a generic template for the model type
   ```sql
   SELECT template FROM templates
   WHERE model_type = ? AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
   ```

3. **Default Template**: If no model-specific template exists, falls back to a default template for the template type
   ```sql
   SELECT template FROM templates
   WHERE model_type = 'default' AND template_type = ? AND (hardware_platform IS NULL OR hardware_platform = '')
   ```

This multi-level fallback system ensures that generators can always find an appropriate template while allowing for specialization when needed.

### Template Variables

Templates use a standard variable substitution system with the format `{variable_name}`. During rendering, these variables are replaced with context-specific values.

#### Core Variables
- `{model_name}`: Full model name (e.g., "bert-base-uncased")
- `{normalized_name}`: Normalized model name for class names (e.g., "bert_base_uncased")
- `{generated_at}`: Generation timestamp

#### Hardware-Related Variables
- `{best_hardware}`: Best available hardware for the model
- `{torch_device}`: PyTorch device to use ("cuda", "mps", "cpu")
- `{has_cuda}`: Boolean indicating CUDA availability
- `{has_rocm}`: Boolean indicating ROCm availability
- `{has_mps}`: Boolean indicating MPS (Apple Silicon) availability
- `{has_openvino}`: Boolean indicating OpenVINO availability
- `{has_webnn}`: Boolean indicating WebNN availability
- `{has_webgpu}`: Boolean indicating WebGPU availability

#### Model-Related Variables
- `{model_family}`: Model family classification
- `{model_subfamily}`: Model subfamily classification

## Managing Templates

### Adding New Templates

To add a new template to the database system:

1. Edit `create_template_database.py` and add the template to the `TEMPLATES` dictionary:
   ```python
   TEMPLATES = {
       "model_family": {
           "template_type": """Template content here with {variable} placeholders"""
       }
   }
   ```

2. Run `python create_template_database.py --create` to recreate the database with your new template

3. Validate the template is available with `python validate_db_templates.py`

### Creating Hardware-Specific Templates

For specialized hardware templates, define them in the `TEMPLATES` dictionary and then insert them with the hardware platform specified:

```python
# In create_template_database.py
conn.execute("""
INSERT INTO templates 
(model_type, template_type, template, hardware_platform)
VALUES (?, ?, ?, ?)
""", ["bert", "test", cuda_optimized_template, "cuda"])
```

## Completion of Migration

This implementation completes the task "Migrate generators to use database templates instead of static files" which was marked as 95% complete in CLAUDE.md.

### Migration Benefits

1. **Code Maintainability**: Centralized template storage reduces duplication and simplifies updates
2. **Storage Efficiency**: Reduced from hundreds of template files to a single database
3. **Specialization Support**: More flexible hardware-specific template variations
4. **Lookup Performance**: Efficient SQL-based queries for template retrieval
5. **Consistency**: Templates follow a standard pattern with unified variable naming

### Files Changed/Created

- `create_template_database.py` - Core database management functionality
- `validate_db_templates.py` - Validation utility
- `test_generator_with_resource_pool.py` - Updated to use DB templates
- `TEMPLATE_DATABASE_README.md` - This documentation

## Future Enhancements

- **Template Validation System**: Enhanced validation of template correctness before insertion
- **Template Versioning**: Track template changes over time
- **More Template Types**: Add documentation, examples, and other template types
- **Hardware Coverage**: Add more hardware-specific templates
- **API Layer**: Create a RESTful API for template management
- **Dependency System**: Formalize template dependencies and inheritance