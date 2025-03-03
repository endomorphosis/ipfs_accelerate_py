#!/usr/bin/env python
"""
Create Python ORM models for the benchmark database.

This script generates Python ORM models for the benchmark database tables,
providing a programmatic interface for accessing and manipulating the data.
The models can be used by test runners and other components to interact with 
the database in a type-safe manner.
"""

import os
import sys
import inspect
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Add parent directory to path for importing modules
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import create_benchmark_schema to get the table definitions
schema_path = os.path.join(os.path.dirname(__file__), "create_benchmark_schema.py")
spec = importlib.util.spec_from_file_location("create_benchmark_schema", schema_path)
schema_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schema_module)

class TableDefinition:
    """Represents a database table definition"""
    def __init__(self, name: str, columns: List[Dict[str, str]], primary_key: str,
                 foreign_keys: Optional[List[Dict[str, str]]] = None):
        self.name = name
        self.columns = columns
        self.primary_key = primary_key
        self.foreign_keys = foreign_keys or []

class CodeGenerator:
    """Generates Python ORM models for the database tables"""
    
    def __init__(self, output_file: str = "benchmark_db_models.py"):
        self.output_file = output_file
        self.type_mapping = {
            "INTEGER": "int",
            "VARCHAR": "str",
            "FLOAT": "float",
            "TIMESTAMP": "datetime.datetime",
            "BOOLEAN": "bool",
            "JSON": "Dict[str, Any]"
        }
    
    def extract_table_definitions(self) -> List[TableDefinition]:
        """Extract table definitions from the create_benchmark_schema module"""
        tables = []
        
        # Extract common tables
        common_tables_func = getattr(schema_module, "create_common_tables", None)
        if common_tables_func:
            source = inspect.getsource(common_tables_func)
            tables.extend(self._parse_create_table_statements(source))
        
        # Extract performance tables
        performance_tables_func = getattr(schema_module, "create_performance_tables", None)
        if performance_tables_func:
            source = inspect.getsource(performance_tables_func)
            tables.extend(self._parse_create_table_statements(source))
        
        # Extract hardware compatibility tables
        hardware_compat_func = getattr(schema_module, "create_hardware_compatibility_tables", None)
        if hardware_compat_func:
            source = inspect.getsource(hardware_compat_func)
            tables.extend(self._parse_create_table_statements(source))
        
        # Extract integration test tables
        integration_func = getattr(schema_module, "create_integration_test_tables", None)
        if integration_func:
            source = inspect.getsource(integration_func)
            tables.extend(self._parse_create_table_statements(source))
        
        return tables
    
    def _parse_create_table_statements(self, source: str) -> List[TableDefinition]:
        """Parse CREATE TABLE statements from source code"""
        import re
        tables = []
        
        # Find all CREATE TABLE statements
        table_matches = re.finditer(r'CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\)', 
                                   source, re.DOTALL)
        
        for match in table_matches:
            table_name = match.group(1)
            columns_text = match.group(2)
            
            # Parse columns
            columns = []
            primary_key = None
            foreign_keys = []
            
            for line in columns_text.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('--'):
                    continue
                
                # Check for foreign key
                if line.startswith('FOREIGN KEY'):
                    fk_match = re.search(r'FOREIGN KEY \((\w+)\) REFERENCES (\w+)\((\w+)\)', line)
                    if fk_match:
                        foreign_keys.append({
                            'column': fk_match.group(1),
                            'references_table': fk_match.group(2),
                            'references_column': fk_match.group(3)
                        })
                    continue
                
                # Parse column definition
                col_match = re.search(r'(\w+) ([A-Z]+(\([^)]*\))?)', line)
                if not col_match:
                    continue
                
                col_name = col_match.group(1)
                col_type = col_match.group(2)
                
                # Check for primary key
                if 'PRIMARY KEY' in line:
                    primary_key = col_name
                
                # Check for not null
                not_null = 'NOT NULL' in line
                
                # Extract type without modifiers
                base_type = col_type.split('(')[0].strip()
                
                columns.append({
                    'name': col_name,
                    'type': base_type,
                    'not_null': not_null
                })
            
            # Add table definition
            tables.append(TableDefinition(
                name=table_name,
                columns=columns,
                primary_key=primary_key,
                foreign_keys=foreign_keys
            ))
        
        return tables
    
    def generate_models(self, tables: List[TableDefinition]) -> str:
        """Generate Python ORM models for the tables"""
        imports = """#!/usr/bin/env python
\"\"\"
Generated Python ORM models for the benchmark database.

This module contains Python ORM models for the benchmark database tables,
providing a programmatic interface for accessing and manipulating the data.
\"\"\"

import os
import sys
import json
import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

try:
    import duckdb
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb")
    sys.exit(1)
"""
        
        # Add base model class
        base_model = """
@dataclass
class BaseModel:
    \"\"\"Base class for all database models\"\"\"
    
    @classmethod
    def from_row(cls, row: tuple) -> 'BaseModel':
        \"\"\"Create a model instance from a database row\"\"\"
        # Get field names from the dataclass
        import inspect
        signature = inspect.signature(cls.__init__)
        field_names = list(signature.parameters.keys())[1:]  # Skip 'self'
        
        # Create a dictionary of field values
        field_values = {}
        for i, name in enumerate(field_names):
            if i < len(row):
                field_values[name] = row[i]
        
        # Create and return the instance
        return cls(**field_values)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseModel':
        \"\"\"Create a model instance from a dictionary\"\"\"
        # Get field names from the dataclass
        import inspect
        signature = inspect.signature(cls.__init__)
        field_names = list(signature.parameters.keys())[1:]  # Skip 'self'
        
        # Create a dictionary of field values
        field_values = {}
        for name in field_names:
            if name in data:
                # Convert JSON strings to dictionaries
                if name.endswith('_json') and isinstance(data[name], str):
                    field_values[name] = json.loads(data[name])
                else:
                    field_values[name] = data[name]
        
        # Create and return the instance
        return cls(**field_values)
    
    def to_dict(self) -> Dict[str, Any]:
        \"\"\"Convert the model to a dictionary\"\"\"
        result = {}
        for key, value in self.__dict__.items():
            # Skip special attributes
            if key.startswith('_'):
                continue
            
            # Convert JSON fields to strings
            if key.endswith('_json') and value is not None:
                result[key] = json.dumps(value)
            else:
                result[key] = value
        
        return result
"""
        
        # Add database connection class
        db_connection = """
class BenchmarkDB:
    \"\"\"Database connection and operations\"\"\"
    
    def __init__(self, db_path: str = "./benchmark_db.duckdb", read_only: bool = False):
        \"\"\"
        Initialize the database connection.
        
        Args:
            db_path: Path to the DuckDB database
            read_only: Open the database in read-only mode
        \"\"\"
        self.db_path = db_path
        self.read_only = read_only
        
        # Connect to the database
        self._connect()
    
    def _connect(self):
        \"\"\"Connect to the DuckDB database\"\"\"
        try:
            # Check if database exists
            db_exists = os.path.exists(self.db_path)
            
            if not db_exists and self.read_only:
                raise FileNotFoundError(f"Database not found: {self.db_path}")
            
            # Connect to the database
            self.conn = duckdb.connect(self.db_path, read_only=self.read_only)
            
        except Exception as e:
            raise ConnectionError(f"Error connecting to database: {e}")
    
    def close(self):
        \"\"\"Close the database connection\"\"\"
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    # Model-specific query methods are added below
"""
        
        # Generate model classes
        model_classes = []
        repo_methods = []
        
        for table in tables:
            # Convert snake_case to CamelCase for class name
            class_name = ''.join(word.capitalize() for word in table.name.split('_'))
            
            # Generate field declarations
            fields = []
            for column in table.columns:
                field_type = self.type_mapping.get(column['type'], 'Any')
                if not column['not_null']:
                    field_type = f"Optional[{field_type}]"
                
                # Check if this is a primary key with auto-increment
                if column['name'] == table.primary_key:
                    fields.append(f"    {column['name']}: {field_type} = None  # Primary key")
                else:
                    default = "= None" if not column['not_null'] else ""
                    fields.append(f"    {column['name']}: {field_type} {default}")
            
            # Generate model class
            model_class = f"""
@dataclass
class {class_name}(BaseModel):
    \"\"\"Model for the {table.name} table\"\"\"
{os.linesep.join(fields)}
"""
            model_classes.append(model_class)
            
            # Generate repository methods
            repo_methods.extend(self._generate_repo_methods(table, class_name))
        
        # Combine everything
        return imports + base_model + db_connection + os.linesep.join(model_classes) + os.linesep.join(repo_methods)
    
    def _generate_repo_methods(self, table: TableDefinition, class_name: str) -> List[str]:
        """Generate repository methods for the table"""
        methods = []
        
        # Get all method
        get_all_method = f"""
    def get_all_{table.name}(self, limit: int = 1000, order_by: str = None) -> List[{class_name}]:
        \"\"\"
        Get all {table.name} records.
        
        Args:
            limit: Maximum number of records to return
            order_by: Optional column to order by (e.g., "created_at DESC")
            
        Returns:
            List of {class_name} objects
        \"\"\"
        query = f"SELECT * FROM {table.name}"
        
        if order_by:
            query += f" ORDER BY {{order_by}}"
            
        query += f" LIMIT {{limit}}"
        
        try:
            result = self.conn.execute(query).fetchall()
            return [{class_name}.from_row(row) for row in result]
        except Exception as e:
            raise Exception(f"Error getting {table.name}: {{e}}")
"""
        methods.append(get_all_method)
        
        # Get by primary key method
        get_by_pk_method = f"""
    def get_{table.name}_by_{table.primary_key}(self, {table.primary_key}: int) -> Optional[{class_name}]:
        \"\"\"
        Get a {table.name} record by its primary key.
        
        Args:
            {table.primary_key}: The primary key value
            
        Returns:
            {class_name} object if found, None otherwise
        \"\"\"
        query = f"SELECT * FROM {table.name} WHERE {table.primary_key} = ?"
        
        try:
            result = self.conn.execute(query, [{table.primary_key}]).fetchone()
            if result:
                return {class_name}.from_row(result)
            return None
        except Exception as e:
            raise Exception(f"Error getting {table.name} by {table.primary_key}: {{e}}")
"""
        methods.append(get_by_pk_method)
        
        # Insert method
        insert_method = f"""
    def insert_{table.name}(self, model: {class_name}) -> int:
        \"\"\"
        Insert a new {table.name} record.
        
        Args:
            model: The {class_name} object to insert
            
        Returns:
            The new {table.primary_key} value
        \"\"\"
        # Get model data as dict
        data = model.to_dict()
        
        # Remove primary key if not set
        if data['{table.primary_key}'] is None:
            del data['{table.primary_key}']
            
            # Get the next ID
            try:
                max_id = self.conn.execute(f"SELECT MAX({table.primary_key}) FROM {table.name}").fetchone()[0]
                next_id = max_id + 1 if max_id is not None else 1
            except Exception:
                next_id = 1
                
            data['{table.primary_key}'] = next_id
        
        # Build the SQL query
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        
        query = f"INSERT INTO {table.name} ({{columns}}) VALUES ({{placeholders}})"
        
        try:
            self.conn.execute(query, list(data.values()))
            return data['{table.primary_key}']
        except Exception as e:
            raise Exception(f"Error inserting {table.name}: {{e}}")
"""
        methods.append(insert_method)
        
        # Generate special query methods based on table name
        if table.name == 'performance_results':
            perf_query_method = f"""
    def get_performance_by_model_hardware(self, model_name: str, hardware_type: str) -> List[{class_name}]:
        \"\"\"
        Get performance results for a specific model and hardware.
        
        Args:
            model_name: The name of the model
            hardware_type: The type of hardware
            
        Returns:
            List of {class_name} objects
        \"\"\"
        query = \"\"\"
        SELECT pr.* 
        FROM performance_results pr
        JOIN models m ON pr.model_id = m.model_id
        JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        WHERE m.model_name LIKE ? AND hp.hardware_type = ?
        ORDER BY pr.created_at DESC
        \"\"\"
        
        try:
            result = self.conn.execute(query, [f"%{{model_name}}%", hardware_type]).fetchall()
            return [{class_name}.from_row(row) for row in result]
        except Exception as e:
            raise Exception(f"Error getting performance results: {{e}}")
"""
            methods.append(perf_query_method)
        
        elif table.name == 'hardware_compatibility':
            compat_query_method = f"""
    def get_compatibility_matrix(self) -> Dict[str, Dict[str, bool]]:
        \"\"\"
        Get a compatibility matrix of models vs hardware.
        
        Returns:
            Dictionary mapping model names to dictionaries mapping hardware types to compatibility
        \"\"\"
        query = \"\"\"
        SELECT 
            m.model_name,
            hp.hardware_type,
            hc.is_compatible
        FROM 
            hardware_compatibility hc
        JOIN 
            models m ON hc.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON hc.hardware_id = hp.hardware_id
        WHERE 
            (m.model_name, hp.hardware_type, hc.created_at) IN (
                SELECT 
                    m2.model_name, 
                    hp2.hardware_type, 
                    MAX(hc2.created_at)
                FROM 
                    hardware_compatibility hc2
                JOIN 
                    models m2 ON hc2.model_id = m2.model_id
                JOIN 
                    hardware_platforms hp2 ON hc2.hardware_id = hp2.hardware_id
                GROUP BY 
                    m2.model_name, hp2.hardware_type
            )
        \"\"\"
        
        try:
            result = self.conn.execute(query).fetchall()
            
            matrix = {{}}
            for row in result:
                model_name, hardware_type, is_compatible = row
                
                if model_name not in matrix:
                    matrix[model_name] = {{}}
                    
                matrix[model_name][hardware_type] = bool(is_compatible)
                
            return matrix
        except Exception as e:
            raise Exception(f"Error getting compatibility matrix: {{e}}")
"""
            methods.append(compat_query_method)
        
        return methods
    
    def generate_and_save(self):
        """Generate the ORM models and save to file"""
        # Extract table definitions
        tables = self.extract_table_definitions()
        
        # Generate code
        code = self.generate_models(tables)
        
        # Save to file
        with open(self.output_file, 'w') as f:
            f.write(code)
        
        print(f"Generated ORM models for {len(tables)} tables in {self.output_file}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Python ORM models for the benchmark database")
    parser.add_argument("--output", default="benchmark_db_models.py",
                       help="Output file for the generated models")
    args = parser.parse_args()
    
    generator = CodeGenerator(output_file=args.output)
    generator.generate_and_save()

if __name__ == "__main__":
    main()