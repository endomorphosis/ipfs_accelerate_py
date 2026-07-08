#!/usr/bin/env python3
"""
Test script for optimization export ZIP archive functionality.

This script tests the ZIP archive export functionality in the OptimizationExporter.
"""

import os
import sys
import json
import argparse
import zipfile
from pathlib import Path
from io import BytesIO

# Add parent directory to path for importing
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import optimization exporter
from test.optimization_recommendation.optimization_exporter import OptimizationExporter

def test_export_with_archive(model_name, hardware_platform, output_dir, create_archive=True):
    """Test exporting optimizations with archive creation."""
    print(f"Testing export with archive for {model_name} on {hardware_platform}")
    
    # Create exporter
    exporter = OptimizationExporter(output_dir=output_dir)
    
    try:
        # Export optimization
        result = exporter.export_optimization(
            model_name=model_name,
            hardware_platform=hardware_platform,
            output_format="all"
        )
        
        if "error" in result:
            print(f"Error exporting optimization: {result['error']}")
            return False
        
        print(f"Exported {result['recommendations_exported']} optimization(s) to {result['base_directory']}")
        print(f"Files created: {len(result['exported_files'])}")
        
        # Create archive if requested
        if create_archive:
            archive_data = exporter.create_archive(result)
            
            if not archive_data:
                print("Failed to create archive")
                return False
            
            # Get filename for archive
            archive_filename = exporter.get_archive_filename(result)
            archive_path = output_dir
            full_path = os.path.join(archive_path, archive_filename)
            
            # Save archive
            with open(full_path, "wb") as f:
                f.write(archive_data.getvalue())
            
            print(f"Created archive: {full_path}")
            
            # Verify archive
            with zipfile.ZipFile(full_path, 'r') as zip_ref:
                # List contents
                print("\nArchive contents:")
                for file in zip_ref.namelist():
                    info = zip_ref.getinfo(file)
                    print(f"  {file} ({info.file_size} bytes)")
                
                # Validate archive
                validate_result = zip_ref.testzip()
                if validate_result:
                    print(f"Archive validation failed for: {validate_result}")
                    return False
                else:
                    print("Archive validation successful")
        
        return True
    
    finally:
        exporter.close()

def test_batch_export_with_archive(output_dir, create_archive=True):
    """Test batch export with archive creation."""
    print("Testing batch export with archive")
    
    # Create exporter
    exporter = OptimizationExporter(output_dir=output_dir)
    
    try:
        # Create sample recommendations report
        report = {
            "top_recommendations": [
                {
                    "model_name": "bert-base-uncased",
                    "hardware_platform": "cuda",
                    "recommendation": {
                        "name": "Mixed Precision Training",
                        "description": "Use mixed precision training for faster inference",
                        "confidence": 0.9,
                        "expected_improvements": {
                            "throughput_improvement": 0.5,
                            "latency_reduction": 0.3,
                            "memory_reduction": 0.4
                        }
                    }
                },
                {
                    "model_name": "gpt2",
                    "hardware_platform": "cpu",
                    "recommendation": {
                        "name": "Quantization",
                        "description": "Use INT8 quantization for CPU inference",
                        "confidence": 0.8,
                        "expected_improvements": {
                            "throughput_improvement": 0.3,
                            "latency_reduction": 0.2,
                            "memory_reduction": 0.5
                        }
                    }
                }
            ],
            "generated_at": "2025-05-01T12:00:00Z",
            "models": 2,
            "hardware_platforms": 2
        }
        
        # Export batch optimizations
        result = exporter.export_batch_optimizations(
            recommendations_report=report,
            output_dir=output_dir,
            output_format="all"
        )
        
        if "error" in result:
            print(f"Error exporting batch optimizations: {result['error']}")
            return False
        
        print(f"Exported {result['exported_count']} optimizations to {result['output_directory']}")
        
        # Create archive if requested
        if create_archive:
            archive_data = exporter.create_archive(result)
            
            if not archive_data:
                print("Failed to create archive")
                return False
            
            # Get filename for archive
            archive_filename = exporter.get_archive_filename(result)
            archive_path = output_dir
            full_path = os.path.join(archive_path, archive_filename)
            
            # Save archive
            with open(full_path, "wb") as f:
                f.write(archive_data.getvalue())
            
            print(f"Created batch archive: {full_path}")
            
            # Verify archive
            with zipfile.ZipFile(full_path, 'r') as zip_ref:
                # List contents
                print("\nBatch archive contents:")
                for file in zip_ref.namelist():
                    info = zip_ref.getinfo(file)
                    print(f"  {file} ({info.file_size} bytes)")
                
                # Validate archive
                validate_result = zip_ref.testzip()
                if validate_result:
                    print(f"Batch archive validation failed for: {validate_result}")
                    return False
                else:
                    print("Batch archive validation successful")
        
        return True
    
    finally:
        exporter.close()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Optimization Export ZIP Archive")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model name")
    parser.add_argument("--hardware", type=str, default="cuda", help="Hardware platform")
    parser.add_argument("--output-dir", type=str, default="./test_optimization_exports", help="Output directory")
    parser.add_argument("--no-archive", action="store_true", help="Skip archive creation")
    parser.add_argument("--test-batch", action="store_true", help="Test batch export")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.test_batch:
        # Test batch export
        success = test_batch_export_with_archive(
            output_dir=args.output_dir,
            create_archive=not args.no_archive
        )
    else:
        # Test single export
        success = test_export_with_archive(
            model_name=args.model,
            hardware_platform=args.hardware,
            output_dir=args.output_dir,
            create_archive=not args.no_archive
        )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())