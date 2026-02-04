#!/usr/bin/env python3
"""
Archive and organize WebNN and WebGPU documentation.

This comprehensive script:
    1. Archives outdated WebNN/WebGPU documentation to the archived_md_files directory
    2. Updates references in other documentation files to point to the new documentation
    3. Creates a comprehensive index of WebNN/WebGPU documentation
    4. Adds archive notices to old documentation files

    The goal is to clearly indicate that we now have REAL (not simulated) WebNN/WebGPU
    implementation with browser testing at various precision levels.

Usage:
    python archive_webnn_webgpu_docs.py --archive-only  # Just archive files
    python archive_webnn_webgpu_docs.py --update-refs   # Update references in other docs
    python archive_webnn_webgpu_docs.py                 # Do both
    """

    import argparse
    import os
    import re
    import shutil
    import sys
    import datetime
    from pathlib import Path
    import logging

# Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

# Files to be archived (potentially outdated WebNN/WebGPU docs)
    FILES_TO_ARCHIVE = []]]]],,,,,
    "WEB_PLATFORM_SUPPORT_COMPLETED.md",
    "WEB_PLATFORM_INTEGRATION_GUIDE_UPDATED.md",
    "WEB_PLATFORM_OPTIMIZATION_GUIDE_JUNE2025.md",
    "WEBGPU_IMPLEMENTATION_STATUS.md",
    "WEBNN_IMPLEMENTATION_STATUS.md",
    "WEB_PLATFORM_WEBGPU_GUIDE.md", 
    "MOCK_WEBGPU_IMPLEMENTATION.md",
    "MOCK_WEBNN_IMPLEMENTATION.md",
    "WEBGPU_SIMULATION_GUIDE.md",
    "WEBNN_SIMULATION_GUIDE.md",
    "WEBNN_WEBGPU_QUANTIZATION_README.md",
    "WEBGPU_WEBNN_QUANTIZATION_SUMMARY.md",
    ]

# New reference files that should be used instead
    NEW_REFERENCE_FILES = []]]]],,,,,
    "REAL_WEBNN_WEBGPU_TESTING.md",
    "WEBNN_WEBGPU_GUIDE.md",
    "WEBGPU_4BIT_INFERENCE_README.md",
    "WEBNN_VERIFICATION_GUIDE.md",
    ]

# Files to update references in
    FILES_TO_UPDATE_REFS = []]]]],,,,,
    "README.md",
    "CLAUDE.md",
    "WEB_PLATFORM_INTEGRATION_SUMMARY.md",
    "WEBGPU_STREAMING_DOCUMENTATION.md",
    ]

def archive_files(files_to_archive, archive_dir):
    """
    Archive files to the specified directory with timestamped filenames.
    
    Args:
        files_to_archive: List of files to archive
        archive_dir: Directory to archive files to
    
    Returns:
        List of successfully archived files
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir_path = Path(archive_dir)
        archive_dir_path.mkdir(exist_ok=True)
    
        archived_files = []]]]],,,,,]
    
    for file_name in files_to_archive:
        file_path = Path(file_name)
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist, skipping")
        continue
        
        # Create archive filename with timestamp
        archive_name = f"\1{file_path.suffix}\3"
        archive_path = archive_dir_path / archive_name
        
        # Copy file to archive
        try:
            shutil.copy2(file_path, archive_path)
            logger.info(f"\1{archive_path}\3")
            archived_files.append(file_path)
        except Exception as e:
            logger.error(f"\1{str(e)}\3")
    
            return archived_files

def update_references(files_to_update, old_refs, new_refs):
    """
    Update references in files to point to new documentation.
    
    Args:
        files_to_update: List of files to update references in
        old_refs: List of old reference files
        new_refs: List of new reference files
    
    Returns:
        List of files that were updated
        """
        updated_files = []]]]],,,,,]
    
    for file_name in files_to_update:
        file_path = Path(file_name)
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist, skipping")
        continue
        
        # Read file content
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"\1{str(e)}\3")
                continue
        
                original_content = content
                modified = False
        
        # Update references
        for old_ref in old_refs:
            # Only use the filename, not the path
            old_ref_name = Path(old_ref).name
            
            # Create patterns to match references to the old file
            patterns = []]]]],,,,,
            f"\\[]]]]],,,,,([]]]]],,,,,^\\]]+)\\]\\({old_ref_name}\\)",  # []]]]],,,,,Text](file.md)
            f"\\[]]]]],,,,,([]]]]],,,,,^\\]]+)\\]\\(\\./({old_ref_name})\\)",  # []]]]],,,,,Text](./file.md)
            f"See ({old_ref_name}) for",  # See file.md for
            f"in ({old_ref_name})\\.",  # in file.md.
            f"check ({old_ref_name})",  # check file.md
            ]
            
            # Replace with reference to new files
            new_refs_md = ", ".join([]]]]],,,,,f"[]]]]],,,,,{Path(ref).stem.replace('_', ' ')}]({Path(ref).name})" for ref in new_refs]):
                replacement = f"[]]]]],,,,,\\1](REAL_WEBNN_WEBGPU_TESTING.md) (See also: {new_refs_md})"
            
            for pattern in patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    modified = True
            
            # Also check for direct inclusion/requirement statements
                    include_patterns = []]]]],,,,,
                    f"require\\(\"({old_ref_name})\"\\)",  # require("file.md")
                    f"include\\(\"({old_ref_name})\"\\)",  # include("file.md")
                    f"import\\s+.*\\s+from\\s+\"({old_ref_name})\"",  # import ... from "file.md"
                    ]
            
            for pattern in include_patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, f"require(\"REAL_WEBNN_WEBGPU_TESTING.md\")", content)
                    modified = True
        
        # Add note about archived documentation
        if modified::
            archive_note = f"\n\n<!-- Note: Some WebNN/WebGPU documentation has been archived and replaced with comprehensive real implementation testing documentation. See REAL_WEBNN_WEBGPU_TESTING.md for details. -->\n"
            if "<!-- Note: Some WebNN/WebGPU documentation has been archived" not in content:
                content += archive_note
        
        # Write updated content if modified:
        if modified: and content != original_content:
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                    logger.info(f"\1{file_path}\3")
                    updated_files.append(file_path)
            except Exception as e:
                logger.error(f"\1{str(e)}\3")
    
                    return updated_files

def create_archive_notice_files(archived_files):
    """
    Create notice files in place of archived files.
    
    Args:
        archived_files: List of files that were archived
        """
    for file_path in archived_files:
        notice_content = f"""# {file_path.stem.replace('_', ' ')}

        **NOTICE: This documentation has been archived.**

The content from this file has been moved to the comprehensive WebNN/WebGPU real implementation testing documentation:

    - []]]]],,,,,Real WebNN/WebGPU Implementation Testing](REAL_WEBNN_WEBGPU_TESTING.md)
    - []]]]],,,,,WebNN/WebGPU Guide](WEBNN_WEBGPU_GUIDE.md)
    - []]]]],,,,,WebGPU 4-bit Inference](WEBGPU_4BIT_INFERENCE_README.md)
    - []]]]],,,,,WebNN Verification Guide](WEBNN_VERIFICATION_GUIDE.md)

    This file is kept as a redirect to maintain compatibility with existing links.

    Archived on: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """
        try:
            with open(file_path, 'w') as f:
                f.write(notice_content)
                logger.info(f"\1{file_path}\3")
        except Exception as e:
            logger.error(f"\1{str(e)}\3")

def create_documentation_index(archived_files, new_docs, archive_dir):
    """
    Create a comprehensive documentation index file.
    
    Args:
        archived_files: List of archived files
        new_docs: List of new documentation files
        archive_dir: Directory where files are archived
        """
        docs_index = "# WebNN and WebGPU Documentation Index\n\n"
        docs_index += f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add new documentation section
        docs_index += "## Current Documentation\n\n"
        docs_index += "### Real Implementation Testing\n\n"
    for doc in sorted(new_docs):
        if Path(doc).exists():
            docs_index += f"- []]]]],,,,,{Path(doc).stem.replace('_', ' ')}]({doc}): Comprehensive guide to testing real WebNN/WebGPU implementations\n"
    
    # Add archived docs section
            docs_index += "\n## Archived Documentation\n\n"
            docs_index += "The following documentation has been archived and replaced with the comprehensive real implementation testing guides above:\n\n"
    
    for file_path in sorted(archived_files):
        archive_name = f"\1{file_path.suffix}\3"
        archive_path = Path(archive_dir) / archive_name
        if archive_path.exists():
            docs_index += f"- []]]]],,,,,{file_path.stem.replace('_', ' ')}]({archive_dir}/{archive_name}): Archived on {datetime.datetime.now().strftime('%Y-%m-%d')}\n"
    
    # Add implementation status section
            docs_index += "\n## Implementation Status\n\n"
            docs_index += "### Browser Support Status\n\n"
            docs_index += """
            | Browser | WebNN Support | WebGPU Support | Best Use Case |
            |---------|--------------|----------------|---------------|
            | Chrome | ⚠️ Limited | ✅ Good | General WebGPU |
            | Edge | ✅ Excellent | ✅ Good | WebNN acceleration |
            | Firefox | ❌ Poor | ✅ Excellent | Audio models with WebGPU |
            | Safari | ⚠️ Limited | ⚠️ Limited | Metal API integration |
            """

            docs_index += "\n### Precision Support\n\n"
            docs_index += """
            | Precision | WebNN | WebGPU | Memory Reduction | Use Case |
            |-----------|-------|--------|------------------|----------|
            | 2-bit | ❌ Not Supported | ✅ Supported | ~87.5% | Ultra memory constrained |
            | 3-bit | ❌ Not Supported | ✅ Supported | ~81.25% | Very memory constrained |
            | 4-bit | ⚠️ Experimental | ✅ Supported | ~75% | Memory constrained |
            | 8-bit | ✅ Supported | ✅ Supported | ~50% | General purpose |
            | 16-bit | ✅ Supported | ✅ Supported | ~0% | High accuracy |
            | 32-bit | ✅ Supported | ✅ Supported | 0% | Maximum accuracy |
            """
    
    # Write the index file
    with open("WEBNN_WEBGPU_DOCS_INDEX.md", "w") as f:
        f.write(docs_index)
    
        logger.info("Documentation index created: WEBNN_WEBGPU_DOCS_INDEX.md")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Archive WebNN/WebGPU documentation")
    parser.add_argument("--archive-only", action="store_true", help="Only archive files, don't update references")
    parser.add_argument("--update-refs", action="store_true", help="Only update references, don't archive files")
    parser.add_argument("--archive-dir", default="archived_md_files", help="Directory to archive files to")
    
    args = parser.parse_args()
    
    # Convert file lists to Path objects
    files_to_archive_paths = []]]]],,,,,Path(f) for f in FILES_TO_ARCHIVE]:
    files_to_update_paths = []]]]],,,,,Path(f) for f in FILES_TO_UPDATE_REFS]:
        archived_files = []]]]],,,,,]
    
    # Archive files if requested::
    if not args.update_refs or (not args.archive_only and not args.update_refs):
        logger.info("Archiving files...")
        archived_files = archive_files(files_to_archive_paths, args.archive_dir)
        
        # Create archive notice files
        create_archive_notice_files(archived_files)
    
    # Update references if requested::
    if not args.archive_only or (not args.archive_only and not args.update_refs):
        logger.info("Updating references...")
        updated_files = update_references(files_to_update_paths, files_to_archive_paths, NEW_REFERENCE_FILES)
        
        logger.info(f"Updated references in {len(updated_files)} files")
    
    # Create documentation index
        create_documentation_index(archived_files, NEW_REFERENCE_FILES, args.archive_dir)
    
    # Print summary
        print("\nDocumentation Organization Complete!\n")
        print(f"\1{args.archive_dir}\3")
        print("New documentation index: WEBNN_WEBGPU_DOCS_INDEX.md")
        print("\nSummary of changes:")
        print("- Added real (not simulated) WebNN/WebGPU implementation testing")
        print("- Created comprehensive documentation for all precision levels (2-bit to 32-bit)")
        print("- Added browser-specific optimizations (especially Firefox for audio)")
        print("- Implemented Selenium bridge for real browser testing")
        print("- Archived previous documentation for reference")
    
        logger.info("Done")
        return 0

if __name__ == "__main__":
    sys.exit(main())