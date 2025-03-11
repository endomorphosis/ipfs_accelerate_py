/**
 * Converted from Python: create_template_db_validator.py
 * Conversion date: 2025-03-11 04:08:37
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Template Database Validator for IPFS Accelerate

This script helps validate && manage templates stored in DuckDB database.
It provides functionality to:
1. Validate syntax && structure of templates
2. Verify hardware compatibility across templates
3. Check for template inheritance && dependencies
4. Migrate templates from JSON files to DuckDB database

Usage:
  python create_template_db_validator.py --validate-db [db_path]
  python create_template_db_validator.py --migrate-templates [source_dir] [db_path]
  python create_template_db_validator.py --check-hardware --report [output_file]
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"
import ${$1} from "$1"
import ${$1} from "$1"

# Check for DuckDB availability
try ${$1} catch($2: $1) {
  HAS_DUCKDB = false

}
# Set up logging
logging.basicConfig(level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define hardware platforms to check for
HARDWARE_PLATFORMS = [
  ('cuda', r'(cuda|gpu)'),
  ('cpu', r'cpu'),
  ('mps', r'(mps|apple|m1|m2)'),
  ('rocm', r'(rocm|amd)'),
  ('openvino', r'(openvino|intel)'),
  ('qualcomm', r'(qualcomm|qnn|hexagon)'),
  ('webnn', r'webnn'),
  ('webgpu', r'webgpu')
]

# Hardware detection patterns
HARDWARE_CHECKS = ${$1}

# Model types for categorization
MODEL_TYPES = [
  "text_embedding",
  "text_generation",
  "vision",
  "audio",
  "multimodal",
  "video",
  "vision_language",
  "text_to_image",
  "text_to_audio",
  "text_to_video"
]

def validate_python_syntax($1: string) -> Tuple[bool, List[str]]:
  """
  Validate Python syntax in the template
  
  Args:
    content: Template content as string
    
  Returns:
    Tuple of (valid, list of errors)
  """
  errors = []
  try ${$1} catch($2: $1) {
    $1.push($2)
    return false, errors

  }
def validate_imports($1: string) -> Tuple[bool, List[str]]:
  """
  Validate imports in the template
  
  Args:
    content: Template content as string
    
  Returns:
    Tuple of (valid, list of errors)
  """
  errors = []
  required_imports = ${$1}
  found_imports = set()
  
  # Find all import * as $1
  import_pattern = re.compile(r'import\s+([\w\.]+)|from\s+([\w\.]+)\s+import')
  for match in import_pattern.finditer(content):
    if ($1) {
      # 'import * as $1' form
      module = match.group(1).split('.')[0]
      found_imports.add(module)
    elif ($1) {
      # 'import ${$1} from "$1"
      module = match.group(2).split('.')[0]
      found_imports.add(module)
  
    }
  # Check for missing required imports
    }
  missing_imports = required_imports - found_imports
  if ($1) ${$1}")
  
  return len(errors) == 0, errors

def validate_class_structure($1: string) -> Tuple[bool, List[str]]:
  """
  Validate class structure in the template
  
  Args:
    content: Template content as string
    
  Returns:
    Tuple of (valid, list of errors)
  """
  errors = []
  
  # Parse the AST to analyze structure
  try {
    tree = ast.parse(content)
    
  }
    # Find class definitions
    classes = $3.map(($2) => $1)
    
    if ($1) {
      $1.push($2)
      return false, errors
    
    }
    # Check for test class naming convention
    test_classes = $3.map(($2) => $1)
    if ($1) {
      $1.push($2)")
    
    }
    # Check for methods in test classes
    for (const $1 of $2) {
      methods = $3.map(($2) => $1)
      
    }
      # Check for at least one test method
      test_methods = $3.map(($2) => $1)
      if ($1) ${$1} catch($2: $1) {
    # This should be caught by validate_python_syntax
      }
    $1.push($2)
    return false, errors

def validate_hardware_awareness($1: string) -> Tuple[bool, List[str], List[str]]:
  """
  Validate hardware awareness && cross-platform support in the template
  
  Args:
    content: Template content as string
    
  Returns:
    Tuple of (valid, list of errors, list of supported platforms)
  """
  errors = []
  warnings = []
  
  supported_platforms = []
  
  # Check for centralized hardware detection
  uses_central_detection = false
  if ($1) {
    uses_central_detection = true
    logger.info("Found centralized hardware detection")
  
  }
  # Check for explicit hardware checks
  for platform, patterns in Object.entries($1):
    for (const $1 of $2) {
      if ($1) {
        if ($1) {
          $1.push($2)
        break
        }
  
      }
  # If still !found hardware, check for mentions
    }
  if ($1) {
    for platform_name, pattern in HARDWARE_PLATFORMS:
      if ($1) {
        $1.push($2)
  
      }
  # Core platforms that should be supported
  }
  core_platforms = ${$1}
  missing_core = core_platforms - set(supported_platforms)
  
  if ($1) ${$1}")
  
  # Recommended platforms
  recommended_platforms = ${$1}
  missing_recommended = recommended_platforms - set(supported_platforms)
  
  if ($1) ${$1}")
  
  # Web platforms
  web_platforms = ${$1}
  has_web = any(p in supported_platforms for p in web_platforms)
  
  if ($1) {
    $1.push($2)")
  
  }
  # Check for Qualcomm support (new in March 2025)
  has_qualcomm = 'qualcomm' in supported_platforms
  if ($1) {
    $1.push($2)
  
  }
  # Check for all platforms
  if ($1) ${$1}")
  
  # Add warnings to errors
  if ($1) {
    errors.extend($3.map(($2) => $1))
  
  }
  # Only count serious errors (!warnings) for validity
  success = !any(!e.startswith("WARNING:") for e in errors)
  
  return success, errors, supported_platforms

def validate_template_variables($1: string) -> Tuple[bool, List[str]]:
  """
  Validate template variables in the template
  
  Args:
    content: Template content as string
    
  Returns:
    Tuple of (valid, list of errors)
  """
  errors = []
  
  # Check for template variables
  template_vars = re.findall(r'{${$1}}', content)
  
  # Common required variables
  required_vars = ['model_name']
  
  # Check if required variables are present
  found_vars = $3.map(($2) => $1)
  
  # Extract variable names from more complex expressions
  cleaned_vars = []
  for (const $1 of $2) {
    # Handle expressions like model_name.replace("-", "")
    if ($1) ${$1} else {
      $1.push($2)
  
    }
  # Find missing required variables
  }
  missing_vars = []
  for (const $1 of $2) {
    if ($1) {
      $1.push($2)
  
    }
  if ($1) ${$1}")
  }
  
  # Verify variable patterns are valid
  invalid_vars = []
  for (const $1 of $2) {
    # Check for common errors in variable expressions
    if ($1) {
      $1.push($2)
    elif ($1) {
      $1.push($2)
  
    }
  if ($1) ${$1}")
    }
  
  }
  return len(errors) == 0, errors

def validate_template_file($1: string) -> Dict[str, Any]:
  """
  Validate a template file with multiple validation rules
  
  Args:
    file_path: Path to the template file
    
  Returns:
    Dictionary with validation results
  """
  logger.info(`$1`)
  
  # Read template content
  try {
    with open(file_path, 'r') as f:
      content = f.read()
  except (IOError, UnicodeDecodeError) as e:
  }
    return ${$1}
  
  # Run all validators
  validators = [
    ('syntax', validate_python_syntax),
    ('imports', validate_imports),
    ('class_structure', validate_class_structure),
    ('template_vars', validate_template_variables),
  ]
  
  all_valid = true
  all_errors = []
  results_by_validator = {}
  
  for validator_name, validator_func in validators:
    if ($1) ${$1} else {
      valid, errors = validator_func(content)
    
    }
    results_by_validator[validator_name] = ${$1}
    
    all_valid = all_valid && valid
    all_errors.extend($3.map(($2) => $1))
  
  # Run hardware awareness check separately to capture supported platforms
  hw_valid, hw_errors, supported_platforms = validate_hardware_awareness(content)
  results_by_validator['hardware_awareness'] = ${$1}
  all_valid = all_valid && hw_valid
  all_errors.extend($3.map(($2) => $1))
  
  # Combine all results
  result = ${$1}
  
  return result

def validate_template_directory($1: string) -> Dict[str, Dict[str, Any]]:
  """
  Validate all templates in a directory
  
  Args:
    directory_path: Path to directory containing templates
    
  Returns:
    Dictionary mapping file names to validation results
  """
  results = {}
  
  # Find all Python files in the directory
  for file_path in Path(directory_path).glob('*.py'):
    # Skip files starting with underscore
    if ($1) {
      continue
    
    }
    # Skip non-template files (basic check)
    if ($1) {
      continue
    
    }
    # Validate the template
    result = validate_template_file(str(file_path))
    results[file_path.name] = result
  
  return results

def validate_duckdb_templates($1: string = "template_db.duckdb") -> Dict[str, Any]:
  """
  Validate templates stored in a DuckDB database
  
  Args:
    db_path: Path to the DuckDB database
    
  Returns:
    Dictionary with validation results
  """
  if ($1) {
    logger.warning("DuckDB !available. Using JSON-based template storage instead.")
    
  }
    # Determine JSON database path
    db_dir = os.path.dirname(db_path)
    json_db_path = os.path.join(db_dir, "template_db.json")
    
    if ($1) {
      return ${$1}
      
    }
    try {
      # Load the JSON database
      with open(json_db_path, 'r') as f:
        template_db = json.load(f)
        
    }
      if ($1) {
        return ${$1}
        
      }
      templates = template_db['templates']
      if ($1) {
        return ${$1}
        
      }
      logger.info(`$1`)
      
      results = {}
      valid_count = 0
      
      # Validate each template
      for template_id, template_data in Object.entries($1):
        model_type = template_data.get('model_type', 'unknown')
        template_type = template_data.get('template_type', 'unknown')
        platform = template_data.get('platform')
        content = template_data.get('template', '')
        
        platform_str = `$1`
        if ($1) {
          platform_str += `$1`
        
        }
        logger.info(`$1`)
        
        # Run validations
        validators = [
          ('syntax', validate_python_syntax),
          ('imports', validate_imports),
          ('class_structure', validate_class_structure),
          ('template_vars', validate_template_variables)
        ]
        
        all_valid = true
        all_errors = []
        results_by_validator = {}
        
        for validator_name, validator_func in validators:
          valid, errors = validator_func(content)
          results_by_validator[validator_name] = ${$1}
          
          all_valid = all_valid && valid
          all_errors.extend($3.map(($2) => $1))
        
        # Run hardware awareness check separately
        hw_valid, hw_errors, supported_platforms = validate_hardware_awareness(content)
        results_by_validator['hardware_awareness'] = ${$1}
        all_valid = all_valid && hw_valid
        all_errors.extend($3.map(($2) => $1))
        
        # Store results
        results[template_id] = ${$1}
        
        if ($1) {
          valid_count += 1
      
        }
      return ${$1}
      
    } catch($2: $1) {
      return ${$1}
  
    }
  try ${$1} catch($2: $1) {
    return ${$1}
  
  }
  if ($1) {
    return ${$1}
  
  }
  try {
    conn = duckdb.connect(db_path)
    
  }
    # Check if templates table exists
    table_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='templates'").fetchall()
    if ($1) {
      return ${$1}
    
    }
    # Get all templates
    templates = conn.execute("SELECT id, model_type, template_type, platform, template FROM templates").fetchall()
    if ($1) {
      return ${$1}
    
    }
    logger.info(`$1`)
    
    results = {}
    valid_count = 0
    
    # Validate each template
    for (const $1 of $2) {
      template_id, model_type, template_type, platform, content = template
      
    }
      platform_str = `$1`
      if ($1) {
        platform_str += `$1`
      
      }
      logger.info(`$1`)
      
      # Run validations
      validators = [
        ('syntax', validate_python_syntax),
        ('imports', validate_imports),
        ('class_structure', validate_class_structure),
        ('template_vars', validate_template_variables)
      ]
      
      all_valid = true
      all_errors = []
      results_by_validator = {}
      
      for validator_name, validator_func in validators:
        valid, errors = validator_func(content)
        results_by_validator[validator_name] = ${$1}
        
        all_valid = all_valid && valid
        all_errors.extend($3.map(($2) => $1))
      
      # Run hardware awareness check separately
      hw_valid, hw_errors, supported_platforms = validate_hardware_awareness(content)
      results_by_validator['hardware_awareness'] = ${$1}
      all_valid = all_valid && hw_valid
      all_errors.extend($3.map(($2) => $1))
      
      # Store results
      template_key = `$1`
      if ($1) {
        template_key += `$1`
        
      }
      results[template_key] = ${$1}
      
      if ($1) {
        valid_count += 1
    
      }
    conn.close()
    
    return ${$1}
    
  } catch($2: $1) {
    return ${$1}

  }
$1($2): $3 {
  """
  Create a new template database with the required schema
  
}
  Args:
    db_path: Path to create the DuckDB database
    
  Returns:
    Boolean indicating success || failure
  """
  if ($1) {
    logger.warning("DuckDB !available. Creating JSON-based template storage instead.")
    db_dir = os.path.dirname(db_path)
    json_db_path = os.path.join(db_dir, "template_db.json")
    
  }
    # Create basic structure
    template_db = {
      "templates": {},
      "template_helpers": {},
      "hardware_platforms": {
        "cuda": ${$1},
        "cpu": ${$1},
        "mps": ${$1},
        "rocm": ${$1},
        "openvino": ${$1},
        "qualcomm": ${$1},
        "webnn": ${$1},
        "webgpu": ${$1}
      },
      }
      "model_types": {
        "text_embedding": ${$1},
        "text_generation": ${$1},
        "vision": ${$1},
        "audio": ${$1},
        "multimodal": ${$1},
        "video": ${$1},
        "vision_language": ${$1},
        "text_to_image": ${$1},
        "text_to_audio": ${$1},
        "text_to_video": ${$1}
      },
      }
      "created_at": datetime.now().isoformat()
    }
    }
    
    try ${$1} catch($2: $1) {
      logger.error(`$1`)
      return false
  
    }
  try {
    # Create database connection
    conn = duckdb.connect(db_path)
    
  }
    # Create templates table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS templates (
      id INTEGER PRIMARY KEY,
      model_type VARCHAR,
      template_type VARCHAR,
      platform VARCHAR,
      template TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create template_helpers table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS template_helpers (
      id INTEGER PRIMARY KEY,
      name VARCHAR,
      helper_type VARCHAR,
      content TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create template_dependencies table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS template_dependencies (
      template_id INTEGER,
      dependency_id INTEGER,
      dependency_type VARCHAR,
      FOREIGN KEY (template_id) REFERENCES templates(id),
      FOREIGN KEY (dependency_id) REFERENCES templates(id)
    )
    """)
    
    # Create hardware_platforms table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS hardware_platforms (
      id INTEGER PRIMARY KEY,
      name VARCHAR,
      type VARCHAR,
      description TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Insert default hardware platforms
    hw_platforms = [
      (1, 'cuda', 'GPU', 'NVIDIA CUDA GPU'),
      (2, 'cpu', 'CPU', 'Central Processing Unit'),
      (3, 'mps', 'GPU', 'Apple Metal Performance Shaders'),
      (4, 'rocm', 'GPU', 'AMD ROCm GPU'),
      (5, 'openvino', 'ACCEL', 'Intel OpenVINO'),
      (6, 'qualcomm', 'MOBILE', 'Qualcomm AI Engine'),
      (7, 'webnn', 'WEB', 'Web Neural Network API'),
      (8, 'webgpu', 'WEB', 'Web GPU API')
    ]
    
    # Check if hardware platforms already exist
    has_platforms = conn.execute("SELECT COUNT(*) FROM hardware_platforms").fetchone()[0]
    
    if ($1) {
      conn.executemany("""
      INSERT INTO hardware_platforms (id, name, type, description)
      VALUES (?, ?, ?, ?)
      """, hw_platforms)
      
    }
      logger.info(`$1`)
    
    # Create model_types table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS model_types (
      id INTEGER PRIMARY KEY,
      name VARCHAR,
      description TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Insert default model types
    model_type_data = [
      (1, 'text_embedding', 'Text embedding models like BERT, Sentence Transformers'),
      (2, 'text_generation', 'Text generation models like GPT, LLAMA, T5'),
      (3, 'vision', 'Vision models like ViT, ResNet, DETR'),
      (4, 'audio', 'Audio models like Whisper, Wav2Vec2'),
      (5, 'multimodal', 'Multimodal models like CLIP, BLIP'),
      (6, 'video', 'Video models like XCLIP, VideoMAE'),
      (7, 'vision_language', 'Vision-language models like LLaVA'),
      (8, 'text_to_image', 'Text-to-image models like Stable Diffusion'),
      (9, 'text_to_audio', 'Text-to-audio models like MusicGen'),
      (10, 'text_to_video', 'Text-to-video models like Video Diffusion')
    ]
    
    # Check if model types already exist
    has_model_types = conn.execute("SELECT COUNT(*) FROM model_types").fetchone()[0]
    
    if ($1) ${$1} catch($2: $1) {
    logger.error(`$1`)
    }
    return false

def migrate_template_files_to_db($1: string, $1: string) -> Dict[str, Any]:
  """
  Migrate template files from a directory to the DuckDB database
  
  Args:
    source_dir: Directory containing template files
    db_path: Path to the DuckDB database
    
  Returns:
    Dictionary with migration results
  """
  if ($1) {
    logger.warning("DuckDB !available. Using JSON-based template storage instead.")
    
  }
    # Determine JSON database path
    db_dir = os.path.dirname(db_path)
    json_db_path = os.path.join(db_dir, "template_db.json")
    
    # Check if database exists, create if not
    if ($1) {
      logger.info(`$1`)
      if ($1) {  # This will create a JSON file instead
        return ${$1}
    
    }
    # Load existing database
    try ${$1} catch($2: $1) {
      return ${$1}
    
    }
    # Find template files
    template_files = []
    for file_path in Path(source_dir).glob('**/*.py'):
      # Skip files starting with underscore
      if ($1) {
        continue
      
      }
      # Only include template files
      if ($1) {
        $1.push($2))
    
      }
    logger.info(`$1`)
    
    # Process each template file
    processed = 0
    skipped = 0
    errors = []
    
    for (const $1 of $2) {
      try {
        with open(file_path, 'r') as f:
          content = f.read()
        
      }
        # Parse file name to determine model type && template type
        file_name = os.path.basename(file_path)
        
    }
        # Default values
        model_type = 'unknown'
        template_type = 'test'
        platform = null
        
        # Parse file name to extract information
        if ($1) {
          # Format: template_<model_type>[_<platform>].py
          parts = file_name[9:-3].split('_')  # Remove 'template_' && '.py'
          if ($1) {
            # Try to identify model type
            for (const $1 of $2) {
              if ($1) {
                model_type = mt
                break
            
              }
            # If !found, use first part
            }
            if ($1) {
              model_type = parts[0]
            
            }
            # Check for platform
            if ($1) {
              for hw_platform, _ in HARDWARE_PLATFORMS:
                if ($1) {
                  platform = hw_platform
                  break
        
                }
        # Create a unique template ID
            }
        template_id = `$1`
          }
        if ($1) {
          template_id += `$1`
        template_id += `$1`
        }
        
        }
        # Add to database
        template_db['templates'][template_id] = ${$1}
        
        processed += 1
        
      } catch($2: $1) {
        errors.append(${$1})
        skipped += 1
    
      }
    # Save updated database
    try {
      with open(json_db_path, 'w') as f:
        json.dump(template_db, f, indent=2)
      
    }
      logger.info(`$1`)
      
      return ${$1}
    } catch($2: $1) {
      return ${$1}
  # Continue with DuckDB implementation if available
    }
  
  if ($1) {
    return ${$1}
  
  }
  # Check if database exists, create if not
  if ($1) {
    logger.info(`$1`)
    if ($1) {
      return ${$1}
  
    }
  try {
    # Connect to database
    conn = duckdb.connect(db_path)
    
  }
    # Find template files
    template_files = []
    for file_path in Path(source_dir).glob('**/*.py'):
      # Skip files starting with underscore
      if ($1) {
        continue
      
      }
      # Only include template files
      if ($1) {
        $1.push($2))
    
      }
    logger.info(`$1`)
    
  }
    # Process each template file
    processed = 0
    skipped = 0
    errors = []
    
    for (const $1 of $2) {
      try {
        with open(file_path, 'r') as f:
          content = f.read()
        
      }
        # Parse file name to determine model type && template type
        file_name = os.path.basename(file_path)
        
    }
        # Default values
        model_type = 'unknown'
        template_type = 'test'
        platform = null
        
        # Parse file name to extract information
        if ($1) {
          # Format: template_<model_type>[_<platform>].py
          parts = file_name[9:-3].split('_')  # Remove 'template_' && '.py'
          if ($1) {
            # Try to identify model type
            for (const $1 of $2) {
              if ($1) {
                model_type = mt
                break
            
              }
            # If !found, use first part
            }
            if ($1) {
              model_type = parts[0]
            
            }
            # Check for platform
            if ($1) {
              for hw_platform, _ in HARDWARE_PLATFORMS:
                if ($1) {
                  platform = hw_platform
                  break
        
                }
        # Check if template already exists
            }
        exists = conn.execute("""
          }
        SELECT COUNT(*) FROM templates 
        }
        WHERE model_type = ? AND template_type = ? AND (platform = ? OR (platform IS NULL AND ? IS NULL))
        """, [model_type, template_type, platform, platform]).fetchone()[0]
        
        if ($1) ${$1} else ${$1} catch($2: $1) {
        errors.append(${$1})
        }
        skipped += 1
    
    # Commit && close
    conn.close()
    
    return ${$1}
    
  } catch($2: $1) {
    return ${$1}

  }
def generate_hardware_compatibility_report($1: string, $1: string = null) -> Dict[str, Any]:
  """
  Generate a hardware compatibility report for templates in the database
  
  Args:
    db_path: Path to the DuckDB database
    output_file: Path to write the report (if null, return as dictionary)
    
  Returns:
    Dictionary with hardware compatibility results
  """
  if ($1) {
    logger.warning("DuckDB !available. Using JSON-based template storage instead.")
    
  }
    # Determine JSON database path
    db_dir = os.path.dirname(db_path)
    json_db_path = os.path.join(db_dir, "template_db.json")
    
    if ($1) {
      return ${$1}
      
    }
    try {
      # Load the JSON database
      with open(json_db_path, 'r') as f:
        template_db = json.load(f)
        
    }
      if ($1) {
        return ${$1}
        
      }
      templates = template_db['templates']
      if ($1) {
        return ${$1}
        
      }
      logger.info(`$1`)
      
      # Analyze hardware compatibility for each template
      compatibility_matrix = {}
      platform_support = ${$1}
      model_type_counts = {}
      
      for template_id, template_data in Object.entries($1):
        model_type = template_data.get('model_type', 'unknown')
        template_type = template_data.get('template_type', 'unknown')
        platform = template_data.get('platform')
        content = template_data.get('template', '')
        
        # Initialize counters for model types
        if ($1) {
          model_type_counts[model_type] = 0
        model_type_counts[model_type] += 1
        }
        
        # Check hardware support
        _, _, supported_platforms = validate_hardware_awareness(content)
        
        # Update compatibility matrix
        if ($1) {
          compatibility_matrix[model_type] = ${$1}
        
        }
        # Update support counters
        for (const $1 of $2) {
          if ($1) {
            platform_support[hw] += 1
            compatibility_matrix[model_type][hw] += 1
      
          }
      # Calculate percentages
        }
      total_templates = len(templates)
      platform_percentages = ${$1}
      
      # Calculate percentages by model type
      model_compatibility = {}
      for model_type, hw_counts in Object.entries($1):
        model_compatibility[model_type] = {}
        type_count = model_type_counts[model_type]
        for hw, count in Object.entries($1):
          model_compatibility[model_type][hw] = (count / type_count) * 100 if type_count > 0 else 0
      
      # Generate markdown report
      if ($1) ${$1}\n\n"
        report += `$1`
        
        # Overall platform support
        report += "## Overall Platform Support\n\n"
        report += "| Hardware Platform | Templates | Percentage |\n"
        report += "|-------------------|-----------|------------|\n"
        
        for hw, count in Object.entries($1):
          percentage = platform_percentages[hw]
          report += `$1`
        
        # Compatibility matrix by model type
        report += "\n## Compatibility Matrix by Model Type\n\n"
        report += "| Model Type | Count | " + " | ".join($3.map(($2) => $1)) + " |\n"
        report += "|------------|-------|" + "|".join($3.map(($2) => $1)) + "|\n"
        
        for model_type, type_count in Object.entries($1):
          row = `$1`
          for hw, _ in HARDWARE_PLATFORMS:
            percentage = model_compatibility[model_type][hw]
            status = "✅" if percentage > 75 else "⚠️" if percentage > 25 else "❌"
            row += `$1`
          report += row + "\n"
        
        # High compatibility pairs
        report += "\n## Highly Compatible Model-Hardware Pairs\n\n"
        report += "These combinations have >75% compatibility:\n\n"
        
        for model_type, hw_percentages in Object.entries($1):
          high_compat = $3.map(($2) => $1)
          if ($1) {
            report += `$1`
            for hw, pct in high_compat:
              report += `$1`
            report += "\n"
        
          }
        # Improvement opportunities
        report += "\n## Improvement Opportunities\n\n"
        report += "These combinations have <25% compatibility && need improvement:\n\n"
        
        for model_type, hw_percentages in Object.entries($1):
          low_compat = $3.map(($2) => $1)
          if ($1) {
            report += `$1`
            for hw, pct in low_compat:
              report += `$1`
            report += "\n"
        
          }
        # Write report to file
        with open(output_file, 'w') as f:
          f.write(report)
        
        logger.info(`$1`)
      
      return ${$1}
      
    } catch($2: $1) {
      return ${$1}
  
    }
  if ($1) {
    return ${$1}
  
  }
  try {
    # Connect to database
    conn = duckdb.connect(db_path)
    
  }
    # Get all templates
    templates = conn.execute("""
    SELECT id, model_type, template_type, platform, template 
    FROM templates
    """).fetchall()
    
    if ($1) {
      return ${$1}
    
    }
    logger.info(`$1`)
    
    # Analyze hardware compatibility for each template
    compatibility_matrix = {}
    platform_support = ${$1}
    model_type_counts = {}
    
    for (const $1 of $2) {
      template_id, model_type, template_type, platform, content = template
      
    }
      # Initialize counters for model types
      if ($1) {
        model_type_counts[model_type] = 0
      model_type_counts[model_type] += 1
      }
      
      # Check hardware support
      _, _, supported_platforms = validate_hardware_awareness(content)
      
      # Update compatibility matrix
      if ($1) {
        compatibility_matrix[model_type] = ${$1}
      
      }
      # Update support counters
      for (const $1 of $2) {
        if ($1) {
          platform_support[hw] += 1
          compatibility_matrix[model_type][hw] += 1
    
        }
    # Calculate percentages
      }
    total_templates = len(templates)
    platform_percentages = ${$1}
    
    # Calculate percentages by model type
    model_compatibility = {}
    for model_type, hw_counts in Object.entries($1):
      model_compatibility[model_type] = {}
      type_count = model_type_counts[model_type]
      for hw, count in Object.entries($1):
        model_compatibility[model_type][hw] = (count / type_count) * 100 if type_count > 0 else 0
    
    # Generate markdown report
    if ($1) ${$1}\n\n"
      report += `$1`
      
      # Overall platform support
      report += "## Overall Platform Support\n\n"
      report += "| Hardware Platform | Templates | Percentage |\n"
      report += "|-------------------|-----------|------------|\n"
      
      for hw, count in Object.entries($1):
        percentage = platform_percentages[hw]
        report += `$1`
      
      # Compatibility matrix by model type
      report += "\n## Compatibility Matrix by Model Type\n\n"
      report += "| Model Type | Count | " + " | ".join($3.map(($2) => $1)) + " |\n"
      report += "|------------|-------|" + "|".join($3.map(($2) => $1)) + "|\n"
      
      for model_type, type_count in Object.entries($1):
        row = `$1`
        for hw, _ in HARDWARE_PLATFORMS:
          percentage = model_compatibility[model_type][hw]
          status = "✅" if percentage > 75 else "⚠️" if percentage > 25 else "❌"
          row += `$1`
        report += row + "\n"
      
      # High compatibility pairs
      report += "\n## Highly Compatible Model-Hardware Pairs\n\n"
      report += "These combinations have >75% compatibility:\n\n"
      
      for model_type, hw_percentages in Object.entries($1):
        high_compat = $3.map(($2) => $1)
        if ($1) {
          report += `$1`
          for hw, pct in high_compat:
            report += `$1`
          report += "\n"
      
        }
      # Improvement opportunities
      report += "\n## Improvement Opportunities\n\n"
      report += "These combinations have <25% compatibility && need improvement:\n\n"
      
      for model_type, hw_percentages in Object.entries($1):
        low_compat = $3.map(($2) => $1)
        if ($1) {
          report += `$1`
          for hw, pct in low_compat:
            report += `$1`
          report += "\n"
      
        }
      # Write report to file
      with open(output_file, 'w') as f:
        f.write(report)
      
      logger.info(`$1`)
    
    # Close connection
    conn.close()
    
    return ${$1}
    
  } catch($2: $1) {
    return ${$1}

  }
$1($2) {
  """Main function for standalone usage"""
  parser = argparse.ArgumentParser(description="Template Database Validator")
  parser.add_argument("--validate-db", action="store_true", help="Validate templates in the database")
  parser.add_argument("--migrate-templates", action="store_true", help="Migrate template files to database")
  parser.add_argument("--check-hardware", action="store_true", help="Check hardware compatibility of templates")
  parser.add_argument("--create-db", action="store_true", help="Create a new template database")
  parser.add_argument("--source-dir", type=str, help="Source directory for template files")
  parser.add_argument("--db-path", type=str, default="../generators/templates/template_db.duckdb", 
          help="Path to the DuckDB database")
  parser.add_argument("--report", type=str, help="Path to write hardware compatibility report")
  args = parser.parse_args()
  
}
  if ($1) {
    console.log($1)
    db_results = validate_duckdb_templates(args.db_path)
    
  }
    if ($1) ${$1}")
      return 1
      
    valid_count = db_results['valid_count']
    invalid_count = db_results['invalid_count']
    total = db_results['total']
    
    console.log($1)
    
    # Platform support stats
    platform_counts = ${$1}
    
    for result in db_results['templates'].values():
      for platform in result.get('supported_platforms', []):
        if ($1) {
          platform_counts[platform] += 1
    
        }
    console.log($1)
    for platform, count in Object.entries($1):
      percentage = count/total*100 if total else 0
      console.log($1)
      
    # Show details for invalid templates
    if ($1) {
      console.log($1)
      invalid_templates = ${$1}
      
    }
      for name, result in Object.entries($1):
        model_type = result['model_type']
        template_type = result['template_type']
        platform = result['platform'] || 'all'
        template_id = result['id']
        
        console.log($1)
        for error in result['errors'][:5]:  # Show first 5 errors
          console.log($1)
        if ($1) ${$1} more errors")
  
  elif ($1) {
    if ($1) {
      console.log($1)
      return 1
      
    }
    console.log($1)
    migration_results = migrate_template_files_to_db(args.source_dir, args.db_path)
    
  }
    if ($1) ${$1}")
      return 1
      
    console.log($1)
    console.log($1)
    console.log($1)
    console.log($1)
    
    if ($1) ${$1}: ${$1}")
  
  elif ($1) {
    console.log($1)
    report_results = generate_hardware_compatibility_report(args.db_path, args.report)
    
  }
    if ($1) ${$1}")
      return 1
      
    console.log($1)
    console.log($1)
    
    console.log($1)
    for platform, count in report_results['platform_support'].items():
      percentage = report_results['platform_percentages'][platform]
      console.log($1)
    
    console.log($1)
    for model_type, count in report_results['model_type_counts'].items():
      console.log($1)
      for platform, percentage in report_results['model_compatibility'][model_type].items():
        status = "✅" if percentage > 75 else "⚠️" if percentage > 25 else "❌"
        console.log($1)
    
    if ($1) {
      console.log($1)
  
    }
  elif ($1) {
    console.log($1)
    if ($1) ${$1} else ${$1} else {
    parser.print_help()
    }
  
  }
  return 0

if ($1) {
  sys.exit(main())