#!/usr/bin/env python3
"""
Execute the test generator integration plan.

This script:
1. Integrates fixes from test_generator_fixed.py into test_generator.py
2. Validates the changes by generating test files for different architectures
3. Reports on the success of the integration
"""
import os
import sys
import shutil
import subprocess
import tempfile
import argparse
import re
from datetime import datetime

# Set script constants
ORIGINAL_GENERATOR = "test_generator.py"
FIXED_GENERATOR = "test_generator_fixed.py"
BACKUP_SUFFIX = f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TEST_ARCHITECTURES = ["bert", "gpt2", "t5", "vit"]
TEMP_OUTPUT_DIR = "integration_test_output"

def create_backup():
    """Create a backup of the original generator file."""
    backup_path = f"{ORIGINAL_GENERATOR}{BACKUP_SUFFIX}"
    shutil.copy2(ORIGINAL_GENERATOR, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def extract_function(file_path, function_name):
    """Extract a function from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    pattern = re.compile(rf"def {function_name}\([^)]*\):(.*?)(?=\ndef|\Z)", re.DOTALL)
    match = pattern.search(content)
    
    if match:
        return match.group(0)
    else:
        print(f"Warning: Could not find function {function_name}")
        return None

def integrate_function(source_file, target_file, function_name):
    """Integrate a function from source_file into target_file."""
    function_code = extract_function(source_file, function_name)
    if not function_code:
        return False
    
    with open(target_file, 'r') as f:
        content = f.read()
    
    # Check if function already exists
    pattern = re.compile(rf"def {function_name}\([^)]*\):(.*?)(?=\ndef|\Z)", re.DOTALL)
    match = pattern.search(content)
    
    if match:
        # Replace existing function
        updated_content = content.replace(match.group(0), function_code)
    else:
        # Add function at the end before main
        if "if __name__ == \"__main__\":" in content:
            parts = content.split("if __name__ == \"__main__\":")
            updated_content = parts[0] + function_code + "\n\n" + "if __name__ == \"__main__\":" + parts[1]
        else:
            updated_content = content + "\n\n" + function_code
    
    with open(target_file, 'w') as f:
        f.write(updated_content)
    
    print(f"Integrated function {function_name}")
    return True

def update_model_families():
    """Update MODEL_FAMILIES to include architecture_type."""
    with open(FIXED_GENERATOR, 'r') as f:
        fixed_content = f.read()
    
    # Extract MODEL_FAMILIES from fixed generator
    fixed_pattern = re.compile(r"MODEL_FAMILIES\s*=\s*{(.*?)}", re.DOTALL)
    fixed_match = fixed_pattern.search(fixed_content)
    
    if not fixed_match:
        print("Could not find MODEL_FAMILIES in fixed generator")
        return False
    
    # Find MODEL_FAMILIES in original generator
    with open(ORIGINAL_GENERATOR, 'r') as f:
        content = f.read()
    
    pattern = re.compile(r"MODEL_FAMILIES\s*=\s*{(.*?)}", re.DOTALL)
    match = pattern.search(content)
    
    if not match:
        print("Could not find MODEL_FAMILIES in original generator")
        return False
    
    # Replace original with fixed version
    updated_content = content.replace(match.group(0), "MODEL_FAMILIES = {" + fixed_match.group(1) + "}")
    
    with open(ORIGINAL_GENERATOR, 'w') as f:
        f.write(updated_content)
    
    print("Updated MODEL_FAMILIES with architecture_type information")
    return True

def add_syntax_validation():
    """Add syntax validation function to the generator."""
    validation_function = """
def verify_python_syntax(file_path):
    \"\"\"
    Verify that the generated Python file has valid syntax.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple of (success, error_message)
    \"\"\"
    try:
        # Compile the file to check syntax without executing it
        with open(file_path, 'r') as f:
            code = f.read()
        compile(code, file_path, 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)
"""
    
    with open(ORIGINAL_GENERATOR, 'r') as f:
        content = f.read()
    
    # Add the function before main
    if "if __name__ == \"__main__\":" in content:
        parts = content.split("if __name__ == \"__main__\":")
        updated_content = parts[0] + validation_function + "\n" + "if __name__ == \"__main__\":" + parts[1]
    else:
        updated_content = content + "\n\n" + validation_function
    
    with open(ORIGINAL_GENERATOR, 'w') as f:
        f.write(updated_content)
    
    print("Added syntax validation function")
    return True

def update_main_function():
    """Update main function to include optional syntax validation."""
    # Find the main function in the original file
    with open(ORIGINAL_GENERATOR, 'r') as f:
        content = f.read()
    
    # Add syntax validation to the file writing section
    validation_code = """
    # Validate syntax if requested
    if args.validate_syntax:
        success, error = verify_python_syntax(output_path)
        if not success:
            print(f"Warning: Generated file contains syntax errors: {error}")
            print("You may need to fix indentation issues manually")
        else:
            print(f"Syntax validation passed for {output_path}")
"""
    
    # Find where the file is written and add validation after
    write_pattern = re.compile(r"with open\(output_path, \"w\"\) as f:\s+f\.write\(file_content\)\s+print\(f\"Test file generated: {output_path}\"\)")
    updated_content = re.sub(write_pattern, r"\g<0>" + validation_code, content)
    
    # Add validation option to argument parser
    parser_pattern = re.compile(r"parser = argparse\.ArgumentParser\(description=\"Generate model test files\"\)")
    if parser_pattern.search(updated_content):
        add_arg_pattern = re.compile(r"parser\.add_argument\(\"--output\", type=str, help=\"Output directory\"\)")
        if add_arg_pattern.search(updated_content):
            updated_content = add_arg_pattern.sub(r"\g<0>\n    parser.add_argument(\"--validate-syntax\", action=\"store_true\", help=\"Validate syntax of generated files\")", updated_content)
    
    with open(ORIGINAL_GENERATOR, 'w') as f:
        f.write(updated_content)
    
    print("Updated main function with syntax validation option")
    return True

def test_generation(architecture, output_dir=TEMP_OUTPUT_DIR):
    """Test generating a file for a specific architecture and validate it."""
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [sys.executable, ORIGINAL_GENERATOR, "--family", architecture, "--output", output_dir, "--validate-syntax"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error generating test file for {architecture}:")
            print(result.stderr)
            return False
        
        output_path = os.path.join(output_dir, f"test_hf_{architecture}.py")
        
        if not os.path.exists(output_path):
            print(f"Output file not created for {architecture}")
            return False
        
        # Manually validate syntax
        with open(output_path, 'r') as f:
            code = f.read()
        
        try:
            compile(code, output_path, 'exec')
            print(f"✅ Syntax validation passed for {architecture}")
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error in {architecture}: {e}")
            return False
    
    except Exception as e:
        print(f"Error running generator: {e}")
        return False

def execute_integration(dry_run=False):
    """Execute the integration plan."""
    # Check files exist
    if not os.path.exists(ORIGINAL_GENERATOR):
        print(f"Error: Original generator {ORIGINAL_GENERATOR} not found")
        return False
    
    if not os.path.exists(FIXED_GENERATOR):
        print(f"Error: Fixed generator {FIXED_GENERATOR} not found")
        return False
    
    if dry_run:
        print("=== DRY RUN MODE: No changes will be made ===")
        print(f"\nWould backup {ORIGINAL_GENERATOR} to {ORIGINAL_GENERATOR}{BACKUP_SUFFIX}")
        print("\nWould integrate these functions:")
        print("- apply_indentation")
        print("- fix_method_boundaries")
        print("- generate_pipeline_input_preparation")
        print("- generate_tokenizer_initialization")
        print("- generate_from_pretrained_input_preparation")
        print("- generate_from_pretrained_output_processing")
        print("\nWould update MODEL_FAMILIES with architecture_type information")
        print("\nWould add syntax validation function")
        print("\nWould update main function to include validation option")
        print("\nWould test generating files for these architectures:")
        for arch in TEST_ARCHITECTURES:
            print(f"- {arch}")
        
        return True
    
    # Create backup
    backup_path = create_backup()
    
    # Integrate helper functions
    helper_functions = [
        "apply_indentation",
        "fix_method_boundaries",
    ]
    
    generation_functions = [
        "generate_pipeline_input_preparation",
        "generate_tokenizer_initialization",
        "generate_from_pretrained_input_preparation",
        "generate_from_pretrained_output_processing",
    ]
    
    # Integrate helper functions
    for func in helper_functions:
        integrate_function(FIXED_GENERATOR, ORIGINAL_GENERATOR, func)
    
    # Integrate generation functions
    for func in generation_functions:
        integrate_function(FIXED_GENERATOR, ORIGINAL_GENERATOR, func)
    
    # Update MODEL_FAMILIES dictionary
    update_model_families()
    
    # Add syntax validation function
    add_syntax_validation()
    
    # Update main function
    update_main_function()
    
    # Test generation for each architecture
    print("\nTesting generation for each architecture:")
    success_count = 0
    
    for arch in TEST_ARCHITECTURES:
        if test_generation(arch):
            success_count += 1
    
    print(f"\nIntegration testing complete: {success_count}/{len(TEST_ARCHITECTURES)} architectures generated successfully")
    
    if success_count == len(TEST_ARCHITECTURES):
        print("\n✅ Integration successful!")
        print(f"Original generator backed up to: {backup_path}")
        print(f"Test files generated in: {TEMP_OUTPUT_DIR}/")
    else:
        print("\n⚠️ Integration partially successful")
        print(f"Original generator backed up to: {backup_path}")
        print("Some architectures failed generation - check logs for details")
    
    print("\nNext steps:")
    print("1. Review the generated test files")
    print("2. Run the tests to verify they execute correctly")
    print("3. If any issues remain, consider using post-processing scripts")
    
    return success_count == len(TEST_ARCHITECTURES)

def main():
    """Main entry point."""
    global TEMP_OUTPUT_DIR
    
    parser = argparse.ArgumentParser(description="Execute test generator integration plan")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--output-dir", type=str, default=TEMP_OUTPUT_DIR, help="Directory for test output files")
    
    args = parser.parse_args()
    TEMP_OUTPUT_DIR = args.output_dir
    
    execute_integration(dry_run=args.dry_run)

if __name__ == "__main__":
    main()