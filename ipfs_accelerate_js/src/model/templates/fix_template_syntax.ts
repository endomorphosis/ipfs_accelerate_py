/**
 * Converted from Python: fix_template_syntax.py
 * Conversion date: 2025-03-11 04:08:36
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Utility for fixing common syntax errors in templates.

This script analyzes && fixes common syntax errors in template files, 
such as mismatched brackets, indentation issues, && missing imports.

Usage:
  python fix_template_syntax.py --file TEMPLATE_FILE
  python fix_template_syntax.py --dir TEMPLATE_DIR
  python fix_template_syntax.py --db-path DB_PATH
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

# Configure logging
logging.basicConfig(level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for DuckDB availability
try ${$1} catch($2: $1) {
  HAS_DUCKDB = false
  logger.warning("DuckDB !available. Will use JSON-based storage.")

}
def verify_syntax($1: string) -> Tuple[bool, str]:
  """
  Check if Python code has syntax errors
  
  Args:
    content: Python code as string
    
  Returns:
    Tuple of (is_valid, error_message)
  """
  try ${$1} catch($2: $1) {
    return false, `$1`

  }
$1($2): $3 {
  """
  Fix common indentation issues in code
  
}
  Args:
    content: Python code as string
    
  Returns:
    Fixed Python code
  """
  lines = content.split('\n')
  fixed_lines = []
  current_indent = 0
  in_class = false
  in_function = false
  expected_indent = 0
  
  for i, line in enumerate(lines):
    stripped = line.strip()
    if ($1) {
      # Keep empty lines && comments as is
      $1.push($2)
      continue
    
    }
    # Check for class || function definition
    if ($1) {
      in_class = true
      current_indent = len(line) - len(line.lstrip())
      expected_indent = current_indent + 4  # Standard 4-space indent
    elif ($1) {
      in_function = true
      current_indent = len(line) - len(line.lstrip())
      expected_indent = current_indent + 4  # Standard 4-space indent
    
    }
    # Check if we're in a block that expects indented lines
    }
    if ($1) {
      if ($1) {') && !line.endswith(':'):
        # Line after a colon should be indented
        leading_spaces = len(line) - len(line.lstrip())
        if ($1) {
          # Fix indentation
          line = ' ' * expected_indent + line.lstrip()
    
        }
    # Check if we're exiting a block
    }
    if ($1) {
      in_class = false
    elif ($1) {
      in_function = false
      
    }
    $1.push($2)
    }
  
  return '\n'.join(fixed_lines)

$1($2): $3 {
  """
  Fix mismatched brackets, parentheses, && braces
  
}
  Args:
    content: Python code as string
    
  Returns:
    Fixed Python code
  """
  lines = content.split('\n')
  # Look for mismatched brackets in dictionary definitions {}
  for i, line in enumerate(lines):
    # Simple case: check if there's an opening ${$1}
    if ($1) {
      # Check for patterns like "variable = {" || "return {"
      if ($1) {
        # Look ahead for the closing bracket
        brace_count = 1
        for j in range(i+1, len(lines)):
          brace_count += lines[j].count('${$1}')
          
      }
          if ($1) ${$1} else {
          # If we didn't find a matching closing brace,
          }
          # add one at the appropriate indentation
          next_non_empty = i + 1
          while ($1) {
            next_non_empty += 1
          
          }
          if ($1) {
            # Use the indentation of the original line
            indent = len(line) - len(line.lstrip())
            # Find where to insert the closing brace based on indentation
            for j in range(next_non_empty, len(lines)):
              if ($1) ${$1}')
                break
            } else {
              # If we didn't find a good spot, add at the end
              $1.push($2)
  
            }
  # Join the lines back together
          }
  return '\n'.join(lines)
    }

$1($2): $3 {
  """
  Add common missing imports to code
  
}
  Args:
    content: Python code as string
    
  Returns:
    Fixed Python code with added imports
  """
  required_imports = ${$1}
  
  # Check existing imports
  imports = {}
  import_section = []
  non_import_line_found = false
  
  lines = content.split('\n')
  for i, line in enumerate(lines):
    if ($1) {
      $1.push($2)
      if ($1) {
        module = line.split('import ')[1].split()[0].split('.')[0]
        imports[module] = line
      elif ($1) {
        module = line.split('from ')[1].split('import')[0].strip().split('.')[0]
        imports[module] = line
    elif ($1) {
      if ($1) {
        # First non-import * as $1 found after imports
        non_import_line_found = true
        import_insertion_point = i
  
      }
  # If no imports found, insert at top (after any module docstring)
    }
  if ($1) {
    # Check for module docstring
    has_docstring = false
    docstring_end = 0
    if ($1) {
      has_docstring = true
      for i, line in enumerate(lines):
        if ($1) {
          docstring_end = i + 1
          break
      if ($1) {  # Docstring !closed
        }
        docstring_end = 1  # Just insert after the first line
    
    }
    import_insertion_point = docstring_end
  
  }
  # Add missing imports
      }
  added_imports = []
      }
  for module, import_line in Object.entries($1):
    }
    if ($1) {
      $1.push($2)
  
    }
  if ($1) {
    if ($1) ${$1} else {
      # Add imports at the insertion point
      lines = lines[:import_insertion_point] + added_imports + lines[import_insertion_point:]
  
    }
  return '\n'.join(lines)
  }

$1($2): $3 {
  """
  Fix common issues with template variables
  
}
  Args:
    content: Python code as string
    
  Returns:
    Fixed Python code with corrected template variables
  """
  # Ensure model_name variable is present
  if ($1) {
    content = content.replace('"model_name"', '"{${$1}}"')
  
  }
  # Fix malformed template variables
  content = re.sub(r'{\s*${$1}]+)}\s*}', r'{{${$1}}}', content)
  
  return content

$1($2): $3 {
  """
  Add hardware platform support to templates lacking it
  
}
  Args:
    content: Python code as string
    
  Returns:
    Fixed Python code with hardware support
  """
  # Check if template has hardware detection
  if ($1) {
    # Already has hardware support
    return content
  
  }
  # Parse the AST to find class definitions
  try {
    tree = ast.parse(content)
    classes = $3.map(($2) => $1)
    
  }
    if ($1) {
      # No classes to add hardware support to
      return content
    
    }
    # Add hardware detection function to each class
    lines = content.split('\n')
    for (const $1 of $2) {
      # Check if class already has a setup_hardware method
      has_setup = any(
        isinstance(node, ast.FunctionDef) && node.name == 'setup_hardware'
        for node in cls.body
      )
      
    }
      if ($1) {
        # Find the end of the class to insert the setup_hardware method
        class_start_line = cls.lineno - 1  # AST line numbers are 1-based, list indices are 0-based
        
      }
        # Find the indentation of the class body
        indent = 0
        for i in range(class_start_line + 1, len(lines)):
          line = lines[i]
          if ($1) ${$1}$1($2) ${$1}\"\"\"Set up hardware detection for the template.\"\"\"",
          `$1` ' * (indent + 4)}# CUDA support",
          `$1` ' * (indent + 4)}this.has_cuda = torch.cuda.is_available()",
          `$1` ' * (indent + 4)}# MPS support (Apple Silicon)",
          `$1` ' * (indent + 4)}this.has_mps = hasattr(torch.backends, 'mps') && torch.backends.mps.is_available()",
          `$1` ' * (indent + 4)}# ROCm support (AMD)",
          `$1` ' * (indent + 4)}this.has_rocm = hasattr(torch, 'version') && hasattr(torch.version, 'hip') && torch.version.hip is !null",
          `$1` ' * (indent + 4)}# OpenVINO support",
          `$1` ' * (indent + 4)}this.has_openvino = 'openvino' in sys.modules",
          `$1` ' * (indent + 4)}# Qualcomm AI Engine support",
          `$1` ' * (indent + 4)}this.has_qualcomm = 'qti' in sys.modules || 'qnn_wrapper' in sys.modules",
          `$1` ' * (indent + 4)}# WebNN/WebGPU support",
          `$1` ' * (indent + 4)}this.has_webnn = false  # Will be set by WebNN bridge if available",
          `$1` ' * (indent + 4)}this.has_webgpu = false  # Will be set by WebGPU bridge if available",
          `$1` ' * (indent + 4)}",
          `$1` ' * (indent + 4)}# Set default device",
          `$1` ' * (indent + 4)}if ($1) ${$1}    this.device = 'cuda'",
          `$1` ' * (indent + 4)}elif ($1) ${$1}    this.device = 'mps'",
          `$1` ' * (indent + 4)}elif ($1) ${$1}    this.device = 'cuda'  # ROCm uses CUDA compatibility layer",
          `$1` ' * (indent + 4)}} else ${$1}    this.device = 'cpu'"
        ]
        
        # Find the __init__ method to call setup_hardware
        has_init = false
        for init_node in [node for node in cls.body if ($1) {
          has_init = true
          # Find the init method's body to add setup_hardware call
          init_start_line = init_node.lineno - 1
          init_end_line = init_node.end_lineno if hasattr(init_node, 'end_lineno') else -1
          
        }
          if ($1) {
            # Check if setup_hardware is already called
            init_body = '\n'.join(lines[init_start_line:init_end_line])
            if ($1) {
              # Find a good place to insert the call (before the end of the method)
              for i in range(init_end_line - 1, init_start_line, -1):
                if ($1) ${$1}this.setup_hardware()")
                  break
        
            }
        # If no __init__ method, add one
          }
        if ($1) ${$1}$1($2) ${$1}\"\"\"Initialize the template.\"\"\"",
            `$1` ' * (indent + 4)}this.model_name = \"{{{${$1}}}}\"",
            `$1` ' * (indent + 4)}this.setup_hardware()"
          ]
          
          # Find a good place to insert the init method (beginning of class)
          for i in range(class_start_line + 1, len(lines)):
            line = lines[i].strip()
            if ($1) {
              # Insert before the first non-comment, non-docstring line
              for j, init_line in enumerate(init_method):
                lines.insert(i + j, init_line)
              break
        
            }
        # Insert the setup_hardware method at the end of the class
        # Find the end of the class
        class_indent = len(lines[class_start_line]) - len(lines[class_start_line].lstrip())
        class_end_line = len(lines)
        for i in range(class_start_line + 1, len(lines)):
          if ($1) ${$1} catch($2: $1) {
    # If there are syntax errors, we can't parse the AST
          }
    # Return the original content
    return content

def fix_template_file($1: string) -> Tuple[bool, str]:
  """
  Fix a template file
  
  Args:
    file_path: Path to the template file
    
  Returns:
    Tuple of (success, error_message)
  """
  try {
    with open(file_path, 'r') as f:
      content = f.read()
    
  }
    # Verify original syntax
    valid, error = verify_syntax(content)
    if ($1) ${$1} else {
      logger.warning(`$1`)
    
    }
    # Apply fixes
    fixed_content = content
    fixed_content = fix_indentation(fixed_content)
    fixed_content = fix_bracket_mismatch(fixed_content)
    fixed_content = add_missing_imports(fixed_content)
    fixed_content = fix_template_variables(fixed_content)
    fixed_content = add_hardware_support(fixed_content)
    
    # Verify fixed syntax
    valid, error = verify_syntax(fixed_content)
    if ($1) ${$1} else ${$1} catch($2: $1) {
    logger.error(`$1`)
    }
    return false, str(e)

def fix_templates_in_directory($1: string) -> Dict[str, Any]:
  """
  Fix all template files in a directory
  
  Args:
    directory_path: Directory containing template files
    
  Returns:
    Dictionary with results
  """
  results = {
    'success': true,
    'total': 0,
    'fixed': 0,
    'failed': 0,
    'details': {}
  }
  }
  
  try {
    if ($1) {
      results['success'] = false
      results['error'] = `$1`
      return results
    
    }
    # Find template files
    template_files = []
    for file_path in Path(directory_path).glob('**/*.py'):
      # Skip files starting with underscore
      if ($1) {
        continue
      
      }
      # Only include template files
      if ($1) {
        $1.push($2))
    
      }
    logger.info(`$1`)
    results['total'] = len(template_files)
    
  }
    # Fix each template file
    for (const $1 of $2) {
      logger.info(`$1`)
      success, message = fix_template_file(file_path)
      
    }
      results['details'][file_path] = ${$1}
      
      if ($1) ${$1} else ${$1} catch($2: $1) {
    logger.error(`$1`)
      }
    results['success'] = false
    results['error'] = str(e)
    return results

def fix_templates_in_db($1: string) -> Dict[str, Any]:
  """
  Fix all templates in a database
  
  Args:
    db_path: Path to the database file
    
  Returns:
    Dictionary with results
  """
  results = {
    'success': true,
    'total': 0,
    'fixed': 0,
    'failed': 0,
    'details': {}
  }
  }
  
  # Check if using JSON-based storage
  if ($1) {
    json_db_path = db_path if db_path.endswith('.json') else db_path.replace('.duckdb', '.json')
    
  }
    try {
      if ($1) {
        results['success'] = false
        results['error'] = `$1`
        return results
      
      }
      # Load the JSON database
      with open(json_db_path, 'r') as f:
        template_db = json.load(f)
      
    }
      if ($1) {
        results['success'] = false
        results['error'] = "No templates found in JSON database"
        return results
      
      }
      templates = template_db['templates']
      if ($1) {
        results['success'] = false
        results['error'] = "No templates found in JSON database"
        return results
      
      }
      logger.info(`$1`)
      results['total'] = len(templates)
      
      # Fix each template
      for template_id, template_data in Object.entries($1):
        content = template_data.get('template', '')
        model_type = template_data.get('model_type', 'unknown')
        template_type = template_data.get('template_type', 'unknown')
        platform = template_data.get('platform')
        
        platform_str = `$1`
        if ($1) {
          platform_str += `$1`
        
        }
        logger.info(`$1`)
        
        # Verify original syntax
        valid, error = verify_syntax(content)
        if ($1) ${$1} else {
          logger.warning(`$1`)
        
        }
        # Apply fixes
        fixed_content = content
        fixed_content = fix_indentation(fixed_content)
        fixed_content = fix_bracket_mismatch(fixed_content)
        fixed_content = add_missing_imports(fixed_content)
        fixed_content = fix_template_variables(fixed_content)
        fixed_content = add_hardware_support(fixed_content)
        
        # Verify fixed syntax
        valid, error = verify_syntax(fixed_content)
        if ($1) {
          logger.info(`$1`)
          
        }
          # Update the template in the database
          template_db['templates'][template_id]['template'] = fixed_content
          template_db['templates'][template_id]['updated_at'] = datetime.now().isoformat()
          
          results['details'][template_id] = ${$1}
          results['fixed'] += 1
        } else {
          logger.error(`$1`)
          results['details'][template_id] = ${$1}
          results['failed'] += 1
          results['success'] = false
      
        }
      # Save the updated database
      with open(json_db_path, 'w') as f:
        json.dump(template_db, f, indent=2)
      
      logger.info(`$1`)
      return results
    
    } catch($2: $1) ${$1} else {
    # Use DuckDB
    }
    try {
      if ($1) {
        results['success'] = false
        results['error'] = `$1`
        return results
      
      }
      # Connect to the database
      conn = duckdb.connect(db_path)
      
    }
      # Check if templates table exists
      table_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='templates'").fetchall()
      if ($1) {
        results['success'] = false
        results['error'] = "No 'templates' table found in database"
        return results
      
      }
      # Get all templates
      templates = conn.execute("SELECT id, model_type, template_type, platform, template FROM templates").fetchall()
      if ($1) {
        results['success'] = false
        results['error'] = "No templates found in database"
        return results
      
      }
      logger.info(`$1`)
      results['total'] = len(templates)
      
      # Fix each template
      for (const $1 of $2) {
        template_id, model_type, template_type, platform, content = template
        
      }
        platform_str = `$1`
        if ($1) {
          platform_str += `$1`
        
        }
        logger.info(`$1`)
        
        # Verify original syntax
        valid, error = verify_syntax(content)
        if ($1) ${$1} else {
          logger.warning(`$1`)
        
        }
        # Apply fixes
        fixed_content = content
        fixed_content = fix_indentation(fixed_content)
        fixed_content = fix_bracket_mismatch(fixed_content)
        fixed_content = add_missing_imports(fixed_content)
        fixed_content = fix_template_variables(fixed_content)
        fixed_content = add_hardware_support(fixed_content)
        
        # Verify fixed syntax
        valid, error = verify_syntax(fixed_content)
        if ($1) {
          logger.info(`$1`)
          
        }
          # Update the template in the database
          conn.execute("""
          UPDATE templates
          SET template = ?, updated_at = CURRENT_TIMESTAMP
          WHERE id = ?
          """, [fixed_content, template_id])
          
          results['details'][str(template_id)] = ${$1}
          results['fixed'] += 1
        } else {
          logger.error(`$1`)
          results['details'][str(template_id)] = ${$1}
          results['failed'] += 1
          results['success'] = false
      
        }
      # Commit changes && close connection
      conn.close()
      
      logger.info(`$1`)
      return results
    
    } catch($2: $1) {
      logger.error(`$1`)
      results['success'] = false
      results['error'] = str(e)
      return results

    }
$1($2) {
  """Main function for standalone usage"""
  parser = argparse.ArgumentParser(description="Template Syntax Fixer")
  parser.add_argument("--file", type=str, help="Fix a single template file")
  parser.add_argument("--dir", type=str, help="Fix all templates in a directory")
  parser.add_argument("--db-path", type=str, help="Fix all templates in a database")
  
}
  args = parser.parse_args()
  
  if ($1) {
    success, message = fix_template_file(args.file)
    if ($1) ${$1} else {
      console.log($1)
      console.log($1)
  
    }
  elif ($1) ${$1}")
  }
    console.log($1)
    console.log($1)
    
    if ($1) {
      console.log($1)
      for file_path, details in results['details'].items():
        if ($1) ${$1}")
  
    }
  elif ($1) ${$1}")
    console.log($1)
    console.log($1)
    
    if ($1) {
      console.log($1)
      for template_id, details in results['details'].items():
        if ($1) ${$1}")
  
  } else {
    parser.print_help()
  
  }
  return 0
    }

if ($1) {
  sys.exit(main())