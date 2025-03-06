#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Model Registry Integration

This script implements a basic test for the model registry integration system.
It creates the necessary database schema, adds model versions, calculates
suitability scores, and generates hardware recommendations.

Date: March 2025
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import model registry integration components
try:
    from model_registry_integration import (
        ModelRegistrySchema,
        ModelRegistryIntegration,
        HardwareRecommender,
        VersionControlSystem
    )
except ImportError:
    logger.error("Failed to import model_registry_integration module")
    sys.exit(1)

def create_schema(db_path: Optional[str] = None) -> bool:
    """Create schema extensions for model registry."""
    logger.info("Creating schema extensions...")
    schema = ModelRegistrySchema(db_path)
    return schema.create_schema_extensions()

def calculate_suitability_scores(db_path: Optional[str] = None) -> bool:
    """Calculate suitability scores for all models."""
    logger.info("Calculating suitability scores...")
    integration = ModelRegistryIntegration(db_path)
    scores = integration.calculate_suitability_scores(update_db=True)
    
    if not scores:
        logger.warning("No scores calculated, check database for test results")
        return False
    
    # Print summary of scores
    logger.info(f"Calculated scores for {len(scores)} models")
    for model_name, hardware_data in scores.items():
        logger.info(f"  {model_name}: {len(hardware_data)} hardware platforms")
        for hardware_type, score_data in hardware_data.items():
            logger.info(f"    {hardware_type}: compatibility={score_data['compatibility_score']:.4f}, suitability={score_data['suitability_score']:.4f}")
    
    return True

def add_model_versions(db_path: Optional[str] = None) -> bool:
    """Add model versions for testing."""
    logger.info("Adding model versions...")
    version_control = VersionControlSystem(db_path)
    
    # Test models to add versions for
    test_models = [
        ("bert-base-uncased", "v1.0.0", "abc123", {"author": "test", "description": "Initial version"}),
        ("bert-base-uncased", "v1.1.0", "def456", {"author": "test", "description": "Updated version"}),
        ("t5-small", "v1.0.0", "ghi789", {"author": "test", "description": "Initial version"}),
        ("clip-vit-base-patch32", "v1.0.0", "jkl012", {"author": "test", "description": "Initial version"}),
    ]
    
    success = True
    for model_name, version_tag, version_hash, metadata in test_models:
        version_id = version_control.add_model_version(
            model_name=model_name,
            version_tag=version_tag,
            version_hash=version_hash,
            metadata=metadata
        )
        
        if version_id:
            logger.info(f"Added version {version_tag} for {model_name} (ID: {version_id})")
        else:
            logger.warning(f"Failed to add version {version_tag} for {model_name}")
            success = False
    
    return success

def create_compatibility_snapshots(db_path: Optional[str] = None) -> bool:
    """Create compatibility snapshots for model versions."""
    logger.info("Creating compatibility snapshots...")
    version_control = VersionControlSystem(db_path)
    
    # Test models to create snapshots for
    test_models = [
        ("bert-base-uncased", "v1.0.0"),
        ("bert-base-uncased", "v1.1.0"),
        ("t5-small", "v1.0.0"),
        ("clip-vit-base-patch32", "v1.0.0"),
    ]
    
    success = True
    for model_name, version_tag in test_models:
        result = version_control.create_compatibility_snapshot(
            model_name=model_name,
            version_tag=version_tag
        )
        
        if result:
            logger.info(f"Created compatibility snapshot for {model_name} version {version_tag}")
        else:
            logger.warning(f"Failed to create compatibility snapshot for {model_name} version {version_tag}")
            success = False
    
    return success

def compare_versions(db_path: Optional[str] = None) -> bool:
    """Compare compatibility between model versions."""
    logger.info("Comparing model versions...")
    version_control = VersionControlSystem(db_path)
    
    # Test models to compare
    test_comparisons = [
        ("bert-base-uncased", "v1.0.0", "v1.1.0"),
    ]
    
    success = True
    for model_name, version_tag1, version_tag2 in test_comparisons:
        changes = version_control.compare_compatibility_versions(
            model_name=model_name,
            version_tag1=version_tag1,
            version_tag2=version_tag2
        )
        
        if changes:
            logger.info(f"Compatibility changes for {model_name} between {version_tag1} and {version_tag2}:")
            for i, change in enumerate(changes):
                logger.info(f"  {i+1}. {change['hardware_type']}:")
                logger.info(f"     Compatibility: {change['compatibility_v1']:.4f} -> {change['compatibility_v2']:.4f} ({change['compatibility_change']:+.2f}%)")
                logger.info(f"     Suitability: {change['suitability_v1']:.4f} -> {change['suitability_v2']:.4f} ({change['suitability_change']:+.2f}%)")
        else:
            logger.warning(f"No compatibility changes found for {model_name} between {version_tag1} and {version_tag2}")
            success = False
    
    return success

def recommend_hardware(db_path: Optional[str] = None) -> bool:
    """Generate hardware recommendations for models."""
    logger.info("Generating hardware recommendations...")
    recommender = HardwareRecommender(db_path)
    
    # Test models to generate recommendations for
    test_models = [
        ("bert-base-uncased", "inference", False, False),
        ("bert-base-uncased", "training", False, True),
        ("t5-small", "inference", True, False),
        ("clip-vit-base-patch32", "inference", False, False),
    ]
    
    success = True
    for model_name, task_type, latency_sensitive, memory_constrained in test_models:
        recommendations = recommender.recommend_hardware(
            model_name=model_name,
            task_type=task_type,
            latency_sensitive=latency_sensitive,
            memory_constrained=memory_constrained,
            top_k=3,
            update_db=True
        )
        
        if recommendations:
            logger.info(f"Hardware recommendations for {model_name} ({task_type}):")
            for i, rec in enumerate(recommendations):
                logger.info(f"  {i+1}. {rec['hardware_type']} (score: {rec['weighted_score']:.4f})")
        else:
            logger.warning(f"No recommendations found for {model_name} ({task_type})")
            success = False
    
    return success

def generate_report(db_path: Optional[str] = None, output_path: Optional[str] = None) -> bool:
    """Generate comprehensive model registry report."""
    logger.info("Generating model registry report...")
    
    integration = ModelRegistryIntegration(db_path)
    version_control = VersionControlSystem(db_path)
    recommender = HardwareRecommender(db_path)
    
    # Attempt to connect to DB
    try:
        conn = integration._get_connection()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return False
    
    try:
        # Get models
        models = conn.execute(
            "SELECT model_id, model_name FROM models"
        ).fetchall()
        
        # Get hardware platforms
        hardware = conn.execute(
            "SELECT hardware_id, hardware_type FROM hardware_platforms"
        ).fetchall()
        
        # Create report
        report = "# Model Registry Integration Report\n\n"
        report += f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Models section
        report += "## Models in Registry\n\n"
        report += "| Model ID | Model Name | Versions | Compatible Hardware |\n"
        report += "|----------|------------|----------|--------------------|\n"
        
        for model_id, model_name in models:
            # Get versions
            versions = version_control.get_version_history(model_name)
            version_count = len(versions)
            
            # Get compatible hardware
            compatible_hw = conn.execute(
                """
                SELECT DISTINCT hp.hardware_type
                FROM hardware_model_compatibility hmc
                JOIN hardware_platforms hp ON hmc.hardware_id = hp.hardware_id
                WHERE hmc.model_id = ?
                """,
                [model_id]
            ).fetchall()
            
            compatible_hw_str = ", ".join([hw[0] for hw in compatible_hw]) if compatible_hw else "None"
            
            report += f"| {model_id} | {model_name} | {version_count} | {compatible_hw_str} |\n"
        
        # Compatibility section
        report += "\n## Hardware Compatibility Matrix\n\n"
        report += "| Model | " + " | ".join([hw[1] for hw in hardware]) + " |\n"
        report += "|-------|" + "|".join(["----" for _ in hardware]) + "|\n"
        
        for model_id, model_name in models:
            row = f"| {model_name} |"
            
            for hw_id, hw_type in hardware:
                # Get compatibility score
                result = conn.execute(
                    """
                    SELECT compatibility_score
                    FROM hardware_model_compatibility
                    WHERE model_id = ? AND hardware_id = ?
                    """,
                    [model_id, hw_id]
                ).fetchone()
                
                if result:
                    score = f"{result[0]:.2f}"
                else:
                    score = "N/A"
                
                row += f" {score} |"
            
            report += row + "\n"
        
        # Recommendations section
        report += "\n## Hardware Recommendations\n\n"
        
        for model_id, model_name in models:
            report += f"### {model_name}\n\n"
            
            # Get inference recommendations
            inference_recs = recommender.recommend_hardware(
                model_name=model_name,
                task_type="inference",
                top_k=3,
                update_db=False
            )
            
            report += "**Inference:**\n\n"
            if inference_recs:
                for i, rec in enumerate(inference_recs):
                    report += f"{i+1}. {rec['hardware_type']} (score: {rec['weighted_score']:.4f})\n"
            else:
                report += "No recommendations available\n"
            
            # Get training recommendations
            training_recs = recommender.recommend_hardware(
                model_name=model_name,
                task_type="training",
                top_k=3,
                update_db=False
            )
            
            report += "\n**Training:**\n\n"
            if training_recs:
                for i, rec in enumerate(training_recs):
                    report += f"{i+1}. {rec['hardware_type']} (score: {rec['weighted_score']:.4f})\n"
            else:
                report += "No recommendations available\n"
            
            report += "\n"
        
        # Save or print report
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        else:
            print(report)
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return False
    
    finally:
        conn.close()

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Test Model Registry Integration')
    
    # General arguments
    parser.add_argument('--db-path', help='Database path (default: ./benchmark_db.duckdb)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup schema and calculate initial scores')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run all tests')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate model registry report')
    report_parser.add_argument('--output', help='Output path for report')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get database path
    db_path = args.db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
    
    # Execute command
    if args.command == 'setup':
        if create_schema(db_path):
            logger.info("Schema created successfully")
        else:
            logger.error("Failed to create schema")
            return 1
        
        if calculate_suitability_scores(db_path):
            logger.info("Suitability scores calculated successfully")
        else:
            logger.warning("Suitability score calculation had issues")
        
        logger.info("Setup completed")
    
    elif args.command == 'test':
        # Run all tests
        if not create_schema(db_path):
            logger.error("Failed to create schema")
            return 1
        
        if not add_model_versions(db_path):
            logger.warning("Issues adding model versions")
        
        if not calculate_suitability_scores(db_path):
            logger.warning("Issues calculating suitability scores")
        
        if not create_compatibility_snapshots(db_path):
            logger.warning("Issues creating compatibility snapshots")
        
        if not compare_versions(db_path):
            logger.warning("Issues comparing versions")
        
        if not recommend_hardware(db_path):
            logger.warning("Issues generating hardware recommendations")
        
        logger.info("All tests completed")
    
    elif args.command == 'report':
        if generate_report(db_path, args.output):
            logger.info("Report generated successfully")
        else:
            logger.error("Failed to generate report")
            return 1
    
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    # Import datetime here to avoid circular import in generate_report
    import datetime
    sys.exit(main())