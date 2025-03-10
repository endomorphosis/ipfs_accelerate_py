#!/usr/bin/env python
"""
Fix attribute errors in the Hugging Face TGI and TEI backend implementations.

This script identifies and fixes attribute errors related to queue_processing and other 
missing attributes in the HF TGI and TEI API implementations.
"""

import os
import sys
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fix_hf_backends")

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

def find_hf_backend_files() -> Dict[],str, str]:,
"""
Find the HF TGI and TEI backend implementation files by searching common locations.
    
    Returns:
        Dict mapping backend names to file paths
        """
        backends = {}}}}}
        'hf_tgi': None,
        'hf_tei': None
        }
    
    # Common directory patterns
        common_dirs = [],
        os.path.join(parent_dir, "api_backends"),
        os.path.join(parent_dir, "ipfs_accelerate", "api_backends"),
        os.path.join(parent_dir, "src", "api_backends"),
        os.path.join(parent_dir, "backends"),
        os.path.join(parent_dir, "ipfs_accelerate_py", "api_backends")
        ]
    
    # Check common locations first
    for backend in backends.keys():
        filename = f"{}}}}}backend}.py"
        for directory in common_dirs:
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                backends[],backend] = filepath
                logger.info(f"Found {}}}}}backend} implementation at: {}}}}}filepath}")
    
    # Broader search for any files not found
    missing_backends = [],b for b, p in backends.items() if p is None]:
    if missing_backends:
        logger.info(f"Searching for backends: {}}}}}missing_backends}...")
        for root, dirs, files in os.walk(parent_dir):
            for backend in missing_backends:
                filename = f"{}}}}}backend}.py"
                if filename in files:
                    filepath = os.path.join(root, filename)
                    with open(filepath, 'r') as f:
                        content = f.read()
                        class_name = backend.replace('_', ' ').title().replace(' ', '') + 'Client'
                        if f"class {}}}}}class_name}" in content:
                            backends[],backend] = filepath
                            logger.info(f"Found {}}}}}backend} implementation at: {}}}}}filepath}")
                            missing_backends.remove(backend)
    
    # Log any backends not found
    for backend in missing_backends:
        logger.warning(f"Could not find {}}}}}backend} implementation file")
    
                            return backends

def analyze_hf_backend(filepath: str) -> Dict[],str, Any]:
    """
    Analyze a Hugging Face backend implementation to identify attribute errors.
    
    Args:
        filepath: Path to the backend implementation file
        
    Returns:
        Dict containing analysis results
        """
    with open(filepath, 'r') as f:
        content = f.read()
    
        backend_name = os.path.basename(filepath).replace('.py', '')
        issues = [],]
    
    # Extract class name from file
        class_match = re.search(r'class\s+(\w+)', content)
        class_name = class_match.group(1) if class_match else None
    
    # 1. Check for queue_processing attribute:
    if "queue_processing" in content and "self.queue_processing" in content:
        # Check if queue_processing is initialized:
        if not re.search(r'self\.queue_processing\s*=', content):
            init_match = re.search(r'def\s+__init__\s*\([],^)]*\):\s*\n', content)
            if init_match:
                init_end = init_match.end()
                
                # Find the end of the __init__ method
                method_body_start = content[],init_end:].find('\n')
                if method_body_start != -1:
                    method_body_start += init_end
                    
                    # Get the indentation level
                    next_line = content[],method_body_start+1:].lstrip('\n')
                    indent_match = re.match(r'^(\s+)', next_line)
                    indent = indent_match.group(1) if indent_match else '    '
                    
                    # Find where to insert the queue_processing initialization
                    issues.append({}}}}}:
                        'type': 'attribute_error',
                        'line': content[],:method_body_start].count('\n') + 1,
                        'description': "queue_processing attribute is used but not initialized in __init__",
                        'fix_type': 'insert',
                        'insert_point': method_body_start,
                        'insert_text': f"\n{}}}}}indent}self.queue_processing = False\n{}}}}}indent}self.queue_processor = None\n"
                        })
    
    # 2. Check for queue_processor attribute
    if "queue_processor" in content and "self.queue_processor" in content:
        # Check if queue_processor is initialized:
        if not re.search(r'self\.queue_processor\s*=', content):
            # Already added with queue_processing fix, no separate fix needed
        pass
    
    # 3. Check for _process_queue method
    if "self._process_queue" in content:
        # Check if the method is defined:
        if not re.search(r'def\s+_process_queue\s*\(', content):
            # Find where to add the method - at the end of the class
            class_end = re.search(r'class\s+\w+[],^{}}}}}]*:', content)
            if class_end:
                class_end_pos = class_end.end()
                
                # Look for the end of the class by finding the next top-level definition
                next_class_match = re.search(r'\nclass\s+\w+', content[],class_end_pos:])
                next_def_match = re.search(r'\ndef\s+\w+', content[],class_end_pos:])
                
                end_pos = None
                if next_class_match and next_def_match:
                    end_pos = min(next_class_match.start() + class_end_pos, next_def_match.start() + class_end_pos)
                elif next_class_match:
                    end_pos = next_class_match.start() + class_end_pos
                elif next_def_match:
                    end_pos = next_def_match.start() + class_end_pos
                else:
                    end_pos = len(content)
                
                # Get the indentation level
                    class_body_start = content[],class_end_pos:].find('\n')
                if class_body_start != -1:
                    class_body_start += class_end_pos
                    next_line = content[],class_body_start+1:].lstrip('\n')
                    indent_match = re.match(r'^(\s+)', next_line)
                    indent = indent_match.group(1) if indent_match else '    '
                    
                    # Add the _process_queue method:
                    process_queue_method = f"\n{}}}}}indent}def _process_queue(self):\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}\"\"\"Process requests in the queue with proper concurrency management.\"\"\"\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}while True:\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}try:\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}# Get request from queue\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}future, func, args, kwargs = self.request_queue.get()\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}# Update counters\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}with self.queue_lock:\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}self.active_requests += 1\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}# Process with retry logic\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}retry_count = 0\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}while retry_count <= self.max_retries:\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}try:\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}result = func(*args, **kwargs)\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}future.set_result(result)\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}break\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}except Exception as e:\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}retry_count += 1\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}if retry_count > self.max_retries:\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}future.set_exception(e)\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}break\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}# Calculate backoff delay\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}delay = min(\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}self.initial_retry_delay * (self.backoff_factor ** (retry_count - 1)),\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}self.max_retry_delay\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent})\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}# Sleep with backoff\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}time.sleep(delay)\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}# Update counters and mark task done\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}with self.queue_lock:\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}self.active_requests -= 1\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}self.request_queue.task_done()\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}except Exception as e:\n"
                    process_queue_method += f"{}}}}}indent}{}}}}}indent}{}}}}}indent}{}}}}}indent}logger.error(f\"Error in queue processor: {}}}}}{}}}}}e}}\")\n"
                    
                    issues.append({}}}}}
                    'type': 'method_missing',
                    'description': "Missing _process_queue method",
                    'fix_type': 'insert',
                    'insert_point': end_pos - 1,  # Insert before the end of the class
                    'insert_text': process_queue_method
                    })
    
    # 4. Check for missing imports
                    needed_imports = [],]
    
    # Check for queue import
    if "Queue" in content and not re.search(r'from\s+queue\s+import\s+Queue', content):
        needed_imports.append("from queue import Queue")
    
    # Check for threading import
    if "threading" in content and not re.search(r'import\s+threading', content):
        needed_imports.append("import threading")
    
    # Check for time import
    if ("time.sleep" in content or "time.time" in content) and not re.search(r'import\s+time', content):
        needed_imports.append("import time")
    
    # Check for logging import
    if "logger" in content and not re.search(r'import\s+logging', content):
        needed_imports.append("import logging")
    
    if needed_imports:
        # Find where to insert imports
        import_section_end = 0
        for line in content.split('\n'):
            if line.startswith('import ') or line.startswith('from '):
                import_section_end = content.find(line) + len(line)
        
                issues.append({}}}}}
                'type': 'missing_imports',
                'description': "Missing required imports",
                'fix_type': 'insert',
                'insert_point': import_section_end + 1,  # After the last import
                'insert_text': '\n' + '\n'.join(needed_imports) + '\n'
                })
    
            return {}}}}}
            'filepath': filepath,
            'backend': backend_name,
            'class_name': class_name,
            'issues': issues,
            'total_issues': len(issues)
            }

def fix_hf_backend(filepath: str, analysis: Dict[],str, Any], dry_run: bool = False) -> Tuple[],int, int]:
    """
    Fix the identified issues in a Hugging Face backend implementation.
    
    Args:
        filepath: Path to the backend implementation file
        analysis: Analysis results containing issues to fix
        dry_run: If True, only show what would be fixed without making changes
        
    Returns:
        Tuple of (total issues, fixed issues)
        """
    if not analysis[],'issues']:
        logger.info(f"No issues found in {}}}}}analysis[],'backend']} implementation")
        return 0, 0
    
    with open(filepath, 'r') as f:
        content = f.read()
    
        fixed_content = content
        fixed_issues = 0
    
    # Sort issues by insert point in descending order to avoid shifting positions
        sorted_issues = sorted(analysis[],'issues'], key=lambda x: x.get('insert_point', 0), reverse=True)
    
    # Process insertions
    for issue in sorted_issues:
        if issue.get('fix_type') == 'insert' and 'insert_point' in issue:
            if dry_run:
                logger.info(f"Would fix {}}}}}issue[],'type']} by inserting at position {}}}}}issue[],'insert_point']}")
                logger.debug(f"Insert text: {}}}}}issue[],'insert_text']}")
            else:
                # Insert content at the specified point
                fixed_content = fixed_content[],:issue[],'insert_point']] + issue[],'insert_text'] + fixed_content[],issue[],'insert_point']:]
                fixed_issues += 1
                logger.info(f"Fixed {}}}}}issue[],'type']}: {}}}}}issue[],'description']}")
    
    if not dry_run and fixed_issues > 0:
        # Backup the original file
        backup_path = filepath + '.bak'
        logger.info(f"Creating backup of original file at: {}}}}}backup_path}")
        with open(backup_path, 'w') as f:
            f.write(content)
        
        # Write the fixed content
        with open(filepath, 'w') as f:
            f.write(fixed_content)
        
            logger.info(f"Fixed {}}}}}fixed_issues} out of {}}}}}analysis[],'total_issues']} issues in {}}}}}filepath}")
    
            return analysis[],'total_issues'], fixed_issues

def main():
    """Main function to run the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix attribute errors in HF TGI and TEI backend implementations")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be fixed without making changes")
    parser.add_argument('--backend', choices=[],'hf_tgi', 'hf_tei', 'all'], default='all', 
    help="Which backend to fix (default: all)")
    parser.add_argument('--tgi-file', type=str, help="Path to the HF TGI implementation file (optional)")
    parser.add_argument('--tei-file', type=str, help="Path to the HF TEI implementation file (optional)")
    
    args = parser.parse_args()
    
    try:
        # Find backend files
        backend_files = {}}}}}}
        if args.backend == 'all' or args.backend == 'hf_tgi':
            if args.tgi_file:
                backend_files[],'hf_tgi'] = args.tgi_file
        if args.backend == 'all' or args.backend == 'hf_tei':
            if args.tei_file:
                backend_files[],'hf_tei'] = args.tei_file
        
        # If files not explicitly provided, search for them
        if not backend_files or len(backend_files) < (2 if args.backend == 'all' else 1):
            found_files = find_hf_backend_files()
            for backend, filepath in found_files.items():
                if filepath and (args.backend == 'all' or args.backend == backend):
                    if backend not in backend_files:
                        backend_files[],backend] = filepath
        
        if not backend_files:
            logger.error("No backend files found to fix")
            sys.exit(1)
        
        # Process each backend
            total_fixed = 0
            total_issues = 0
        
        for backend, filepath in backend_files.items():
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {}}}}}filepath}")
            continue
                
            # Analyze the file
            logger.info(f"Analyzing {}}}}}backend} implementation at {}}}}}filepath}...")
            analysis = analyze_hf_backend(filepath)
            
            # Fix the issues
            backend_issues, backend_fixed = fix_hf_backend(filepath, analysis, args.dry_run)
            total_issues += backend_issues
            total_fixed += backend_fixed
        
        # Print summary
        if args.dry_run:
            logger.info(f"Dry run completed. Found {}}}}}total_issues} issues that would be fixed.")
        else:
            logger.info(f"Fixed {}}}}}total_fixed} out of {}}}}}total_issues} issues across all backends.")
            
            if total_fixed == total_issues and total_issues > 0:
                logger.info("All issues in HF backend implementations have been fixed!")
            elif total_issues > 0:
                logger.warning(f"Could only fix {}}}}}total_fixed} out of {}}}}}total_issues} issues. Manual intervention may be required.")
            else:
                logger.info("No issues found in the HF backend implementations")
    
    except Exception as e:
        logger.error(f"Error: {}}}}}e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()