# Database Template Integration Guide for Skillset Generators

## Overview

This guide explains how the new database template integration works in the Integrated Skillset Generator and provides detailed information about the implementation, benefits, and usage patterns.

## Key Components

### 1. DuckDB Storage System

The template storage system uses DuckDB, a high-performance analytical database, to store and retrieve templates. Benefits include:

- **Efficient Storage**: Significantly more space-efficient than individual JSON files
- **Advanced Querying**: SQL capabilities for complex template selection
- **Schema Enforcement**: Structured schema ensures data consistency
- **Transactional Safety**: ACID-compliant database operations

The database is organized with the following primary tables:

- `model_templates`: Store templates with metadata
- `template_helpers`: Store helper functions shared across templates
- `hardware_platforms`: Store information about hardware platform support

### 2. Template Loading System

Templates are loaded from the database through a sophisticated retrieval mechanism:

```python
def load_template_from_db(model_type, template_type='base', platform=None):
    if not HAS_DUCKDB or not os.path.exists(TEMPLATE_DB_PATH):
        return None
    
    try:
        conn = duckdb.connect(TEMPLATE_DB_PATH)
        
        # Build query based on parameters
        query = "SELECT template_content FROM model_templates WHERE model_id = ?"
        params = [model_type]
        
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        
        # Try exact match first
        result = conn.execute(query, params).fetchone()
        
        # Fallback mechanisms if no direct match
        # ... [fallback logic here]
        
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error loading template from database: {e}")
        return None
```

The fallback mechanisms include:
- First, try exact model match
- Then, try model family match
- Finally, try models in the same category

### 3. Result Storage System

Test results are stored in the database instead of JSON files when `DEPRECATE_JSON_OUTPUT=1` (default now):

```python
def save_test_result(result, model_name, test_type="skillset_generation"):
    if not DEPRECATE_JSON_OUTPUT:
        return True  # Will be saved as JSON by caller
    
    if not HAS_DUCKDB:
        logger.warning("DuckDB not available, cannot save to database")
        return False
    
    try:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        conn = duckdb.connect(db_path)
        
        # Store results in database tables
        # ... [storage logic here]
        
        return True
    except Exception as e:
        logger.error(f"Error saving test result to database: {e}")
        return False
```

The data is stored in two tables:
- `skillset_tests`: Basic test metadata
- `skillset_test_details`: Full JSON result data

## Usage Patterns

### 1. Basic Template Usage

To generate a model implementation using database templates:

```bash
python integrated_skillset_generator.py --model bert --use-db-templates
```

### 2. Database Path Configuration

To specify a custom database path:

```bash
export TEMPLATE_DB_PATH=/path/to/template_db.duckdb
export BENCHMARK_DB_PATH=/path/to/benchmark_db.duckdb
python integrated_skillset_generator.py --model bert
```

### 3. Cross-Platform Generation

To generate implementations with all hardware platforms:

```bash
python integrated_skillset_generator.py --model bert --hardware all --cross-platform
```

### 4. Bulk Generation

To generate implementations for all models in a family:

```bash
python integrated_skillset_generator.py --family text-embedding --use-db-templates
```

### 5. Task-Based Generation

To generate implementations for all models with a specific task:

```bash
python integrated_skillset_generator.py --task text_generation --use-db-templates
```

## Implementation Details

### 1. Environment Variables

The system respects the following environment variables:

- `TEMPLATE_DB_PATH`: Path to the template database
- `BENCHMARK_DB_PATH`: Path to the benchmark results database
- `DEPRECATE_JSON_OUTPUT`: Set to "1" to use database storage (default)

### 2. Template Fallback Mechanism

When a specific template isn't found, the system tries:

1. Exact model match (`model_id = 'bert'`)
2. Model family match (`model_family = 'bert'`)
3. Model category match (e.g., all text embedding models)
4. Default template as last resort

### 3. Hardware Customization

Templates can be customized for specific hardware platforms:

```bash
python integrated_skillset_generator.py --model bert --hardware cuda,rocm
```

This loads templates specific to the specified hardware if available.

### 4. Template Validation

The template validation system ensures templates are valid:

```bash
python hardware_test_templates/template_database.py --validate-templates
```

## Migration Guide

### 1. From JSON to Database Templates

If you have custom JSON templates:

```bash
python hardware_test_templates/template_database.py --import-json /path/to/templates.json
```

### 2. From JSON to Database Results

To migrate existing JSON results:

```bash
python duckdb_api/migrate_all_json_files.py --db-path ./benchmark_db.duckdb
```

### 3. Legacy Support

The system maintains backward compatibility through:

- Fallback to in-memory templates if database is unavailable
- Support for both database and JSON outputs during migration

## Best Practices

1. **Always use database templates** when possible.
2. **Set database paths via environment variables** for consistency.
3. **Use cross-platform mode** for comprehensive testing.
4. **Specify required hardware platforms** explicitly.
5. **Validate implementations** after generation.

## Troubleshooting

### 1. Template Not Found

If templates aren't loading:

```bash
python hardware_test_templates/template_database.py --list-templates
```

### 2. Database Connectivity Issues

Check database connectivity:

```bash
python -c "import duckdb; print(duckdb.connect('template_db.duckdb').execute('SELECT 1').fetchone())"
```

### 3. Missing Hardware Support

If hardware platforms aren't detected:

```bash
python integrated_skillset_generator.py --model bert --hardware cpu --verbose
```

## FAQ

**Q: Do I need to install DuckDB separately?**
A: Yes, install with `pip install duckdb pandas`.

**Q: What happens if the database is missing?**
A: The system falls back to in-memory templates.

**Q: Can I still use JSON output?**
A: Yes, by setting `DEPRECATE_JSON_OUTPUT=0`, but this is not recommended.

**Q: How do I add a new template to the database?**
A: Use `hardware_test_templates/template_database.py --store-template --model-type bert --template-file template.py`.

**Q: How can I query the database directly?**
A: Use `python -c "import duckdb; print(duckdb.connect('benchmark_db.duckdb').execute('SELECT * FROM skillset_tests').fetchdf())"`.

## Conclusion

The database template integration significantly improves the skillset generation system through more efficient storage, better organization, and enhanced query capabilities. By following this guide, you can take full advantage of these improvements in your workflow.