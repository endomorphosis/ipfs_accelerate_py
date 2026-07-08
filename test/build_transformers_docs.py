#!/usr/bin/env python
import sys
import os
import argparse
import re
import glob
import shutil
from pathlib import Path

# Add doc-builder source to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'huggingface_doc_builder/src'))

# Add transformers library to the path
transformers_src_path = os.path.join(os.path.dirname(__file__), 'transformers/src')
sys.path.insert(0, transformers_src_path)

# Import the necessary modules from doc-builder
from doc_builder import build_doc

def create_simplified_toc(temp_dir, language):
    """Create a simplified _toctree.yml file that only includes essential sections."""
    md_files = []
    language_dir = os.path.join(temp_dir, language)
    
    # Get all markdown files
    for root, _, files in os.walk(language_dir):
        for file in files:
            if file.endswith('.md'):
                # Extract relative path from language directory
                rel_path = os.path.relpath(os.path.join(root, file), language_dir)
                # Remove .md extension
                rel_path = os.path.splitext(rel_path)[0]
                md_files.append(f"{language}/{rel_path}")
    
    # Create simple TOC structure
    toc_content = f"""- sections:
  - local: {language}/index
    title: HuggingFace Transformers
    sections:
"""
    
    # Add each file to the TOC
    for md_file in md_files:
        file_name = os.path.basename(md_file)
        if file_name != "index":  # Skip index.md since we already added it
            title = file_name.replace("_", " ").replace("-", " ").title()
            toc_content += f"    - local: {md_file}\n      title: {title}\n"
    
    # Write TOC file
    with open(os.path.join(temp_dir, '_toctree.yml'), 'w') as f:
        f.write(toc_content)

def remove_autodoc_from_md_files(source_dir, temp_dir, language):
    """Create a copy of the source docs with autodoc markers removed."""
    # Copy entire source directory to temp dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Create the base directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy _config.py from the top-level source directory
    for item in ['_config.py']:
        source_item = os.path.join(source_dir, item)
        if os.path.exists(source_item):
            shutil.copy2(source_item, os.path.join(temp_dir, item))
    
    # Create language directory
    language_source_dir = os.path.join(source_dir, language)
    language_temp_dir = os.path.join(temp_dir, language)
    
    if os.path.exists(language_source_dir):
        # Copy the language directory
        shutil.copytree(language_source_dir, language_temp_dir)
        
        # Find all .md files in the language temp directory
        md_files = glob.glob(f"{language_temp_dir}/**/*.md", recursive=True)
        
        # Pattern to match autodoc markers
        autodoc_pattern = r'\[\[autodoc\]\].*?(?=\n\n|\Z)'
        
        # Process each file
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace autodoc markers with placeholder text
                modified_content = re.sub(autodoc_pattern, '[API documentation placeholder]', content, flags=re.DOTALL)
                
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
            except Exception as e:
                print(f"Error processing file {md_file}: {str(e)}")
    else:
        print(f"Warning: Language directory {language_source_dir} not found!")
    
    # Create a simplified TOC file
    create_simplified_toc(temp_dir, language)
    
    # Verify files were copied properly
    print(f"Files in temp dir: {os.listdir(temp_dir)}")
    
    return temp_dir

def create_subset_docs(source_dir, temp_dir, language, max_files=20):
    """Create a smaller subset of documentation files for easier testing."""
    # Copy essential files first
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy _config.py from the top-level source directory
    for item in ['_config.py']:
        source_item = os.path.join(source_dir, item)
        if os.path.exists(source_item):
            shutil.copy2(source_item, os.path.join(temp_dir, item))
    
    # Create language directory
    language_source_dir = os.path.join(source_dir, language)
    language_temp_dir = os.path.join(temp_dir, language)
    os.makedirs(language_temp_dir, exist_ok=True)
    
    # Always include index.md
    index_source = os.path.join(language_source_dir, 'index.md')
    if os.path.exists(index_source):
        shutil.copy2(index_source, os.path.join(language_temp_dir, 'index.md'))
    
    # Get all markdown files and select a subset
    md_files = glob.glob(f"{language_source_dir}/*.md")
    
    # Select a subset of files
    if len(md_files) > max_files:
        md_files = [index_source] + md_files[:max_files]
    
    # Copy selected files
    for md_file in md_files:
        if os.path.exists(md_file) and md_file != index_source:  # Skip index.md as we already copied it
            dest_file = os.path.join(language_temp_dir, os.path.basename(md_file))
            shutil.copy2(md_file, dest_file)
            
            # Process the file to remove autodoc markers
            try:
                with open(dest_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace autodoc markers with placeholder text
                autodoc_pattern = r'\[\[autodoc\]\].*?(?=\n\n|\Z)'
                modified_content = re.sub(autodoc_pattern, '[API documentation placeholder]', content, flags=re.DOTALL)
                
                with open(dest_file, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
            except Exception as e:
                print(f"Error processing file {dest_file}: {str(e)}")
    
    # Create a simplified TOC file
    create_simplified_toc(temp_dir, language)
    
    return temp_dir

def main():
    # Set up arguments
    parser = argparse.ArgumentParser(description='Build documentation for Transformers')
    parser.add_argument('--output-dir', default='./transformers_docs_build',
                        help='Where to output the built documentation')
    parser.add_argument('--language', default='en',
                        help='Language of the documentation to generate')
    parser.add_argument('--temp-dir', default='./temp_docs',
                        help='Directory for temporary processed docs')
    parser.add_argument('--subset', action='store_true',
                        help='Build only a subset of the documentation (for testing)')
    parser.add_argument('--max-files', type=int, default=20,
                        help='Maximum number of files to include in subset mode')
    args = parser.parse_args()

    # Set parameters
    library_name = 'transformers'
    path_to_docs = './transformers/docs/source'
    output_path = Path(args.output_dir) / args.language
    version = 'main'
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Building documentation for {library_name}")
    print(f"Source: {path_to_docs}")
    print(f"Output: {output_path}")
    print(f"Language: {args.language}")
    
    # Create documentation files
    if args.subset:
        print(f"Building subset documentation (max {args.max_files} files)...")
        temp_docs_path = create_subset_docs(path_to_docs, args.temp_dir, args.language, args.max_files)
    else:
        print("Creating temporary docs with autodoc markers removed...")
        temp_docs_path = remove_autodoc_from_md_files(path_to_docs, args.temp_dir, args.language)
    
    # Build the documentation from the modified source
    print("Building documentation from processed source...")
    try:
        build_doc(
            library_name,
            temp_docs_path,
            output_path,
            clean=True,
            version=version,
            version_tag=version,
            language=args.language,
            notebook_dir=None,
            is_python_module=False
        )
        print(f"Documentation built successfully to {output_path}")
    except Exception as e:
        print(f"Error building documentation: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()