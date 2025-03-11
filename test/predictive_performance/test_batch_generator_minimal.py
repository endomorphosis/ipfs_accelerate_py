#!/usr/bin/env python3
"""
Test Batch Generator - Minimal Test Version.

This script demonstrates the test batch generator functionality with a simplified implementation.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from pprint import pprint
from typing import List, Dict, Any, Optional, Tuple
import random
from scipy.spatial.distance import euclidean
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_batch_generator")

class TestBatchGenerator:
    """Test Batch Generator implementation for testing."""
    
    def __init__(self):
        """Initialize the batch generator."""
        # Model types and hardware platforms for testing
        self.model_types = ["text_embedding", "text_generation", "vision", "audio", "multimodal"]
        self.hardware_platforms = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        # Generate test configurations
        self.generate_test_configs()
    
    def generate_test_configs(self):
        """Generate test configurations."""
        # Generate all possible configurations
        self.all_configs = []
        for model_type in self.model_types:
            for hardware in self.hardware_platforms:
                for batch_size in self.batch_sizes[:3]:  # Use only 1, 2, 4 for testing
                    # Create a basic configuration
                    config = {
                        "model_name": f"example_{model_type}_model",
                        "model_type": model_type,
                        "hardware": hardware,
                        "batch_size": batch_size,
                        "expected_information_gain": random.uniform(0.1, 0.9),
                        "uncertainty": random.uniform(0.1, 0.9),
                        "diversity": random.uniform(0.1, 0.9),
                    }
                    self.all_configs.append(config)
        
        # Convert to DataFrame for easier handling
        self.configs_df = pd.DataFrame(self.all_configs)
        
        # Take a subset for testing
        self.configs_df = self.configs_df.sample(n=min(50, len(self.configs_df)))
        
        # Sort by expected information gain
        self.configs_df = self.configs_df.sort_values(by="expected_information_gain", ascending=False)
    
    def suggest_test_batch(self, configurations, batch_size=10, ensure_diversity=True, 
                           hardware_constraints=None, hardware_availability=None,
                           diversity_weight=0.5):
        """
        Generate an optimized batch of test configurations for benchmarking.
        
        Args:
            configurations: DataFrame or list of configuration dictionaries
            batch_size: Maximum number of configurations to include in the batch
            ensure_diversity: Whether to ensure diversity in the selected batch
            hardware_constraints: Dictionary mapping hardware types to maximum count in batch
            hardware_availability: Dictionary mapping hardware types to availability factor (0-1)
            diversity_weight: Weight to give diversity vs. information gain (0-1)
            
        Returns:
            DataFrame of selected configurations for the test batch
        """
        logger.info(f"Generating test batch with size {batch_size}, diversity={ensure_diversity}")
        
        # Convert to DataFrame if needed
        if isinstance(configurations, list):
            configs_df = pd.DataFrame(configurations)
        else:
            configs_df = configurations.copy()
            
        # Check if we have enough configurations
        if len(configs_df) <= batch_size:
            logger.info(f"Not enough configurations, returning all {len(configs_df)} available")
            return configs_df
            
        # Use different columns depending on which scoring system we're dealing with
        if "combined_score" in configs_df.columns:
            score_column = "combined_score"
        elif "adjusted_score" in configs_df.columns:
            score_column = "adjusted_score"
        elif "expected_information_gain" in configs_df.columns:
            score_column = "expected_information_gain"
        else:
            # If no score column exists, add a default one
            logger.warning("No score column found, using equal weights for all configurations")
            configs_df["score"] = 1.0
            score_column = "score"
            
        # Apply hardware availability constraints if provided
        if hardware_availability is not None:
            logger.info("Applying hardware availability constraints")
            configs_df = self._apply_hardware_availability(configs_df, 
                                                          hardware_availability, 
                                                          score_column)
            
        # If diversity is not required, simply return the top configurations by score
        if not ensure_diversity:
            sorted_configs = configs_df.sort_values(by=score_column, ascending=False)
            
            # Apply hardware constraints if provided
            if hardware_constraints is not None:
                batch = self._apply_hardware_constraints(sorted_configs, 
                                                       hardware_constraints, 
                                                       batch_size)
            else:
                batch = sorted_configs.head(batch_size)
                
            logger.info(f"Generated non-diverse batch with {len(batch)} configurations")
            return batch
            
        # For diversity-aware selection, we'll select configurations one by one
        logger.info("Using diversity-aware selection")
        return self._diversity_sampling(configs_df, 
                                       score_column, 
                                       batch_size, 
                                       diversity_weight, 
                                       hardware_constraints)
    
    def _apply_hardware_availability(self, configs_df, hardware_availability, score_column):
        """
        Adjust scores based on hardware availability.
        
        Args:
            configs_df: DataFrame of configurations
            hardware_availability: Dictionary mapping hardware types to availability factor (0-1)
            score_column: Name of the column containing scores
            
        Returns:
            DataFrame with adjusted scores
        """
        # Create a copy so we don't modify the original
        adjusted_df = configs_df.copy()
        
        # Hardware column might be called 'hardware' or 'hardware_platform'
        hardware_column = 'hardware' if 'hardware' in adjusted_df.columns else 'hardware_platform'
        
        # Adjust scores based on hardware availability
        for hw_type, availability in hardware_availability.items():
            # Find configurations with this hardware type
            mask = adjusted_df[hardware_column] == hw_type
            
            # Adjust scores
            adjusted_df.loc[mask, score_column] = adjusted_df.loc[mask, score_column] * availability
            
        return adjusted_df
    
    def _apply_hardware_constraints(self, configs_df, hardware_constraints, batch_size):
        """
        Apply hardware constraints to selection.
        
        Args:
            configs_df: DataFrame of configurations sorted by score
            hardware_constraints: Dictionary mapping hardware types to maximum count in batch
            batch_size: Maximum batch size
            
        Returns:
            DataFrame of selected configurations respecting hardware constraints
        """
        # Hardware column might be called 'hardware' or 'hardware_platform'
        hardware_column = 'hardware' if 'hardware' in configs_df.columns else 'hardware_platform'
        
        # Initialize empty batch and hardware counts
        batch = []
        hw_counts = {hw: 0 for hw in hardware_constraints.keys()}
        total_selected = 0
        
        # Iterate through sorted configurations
        for _, config in configs_df.iterrows():
            hw_type = config[hardware_column]
            
            # Check if we've reached the hardware constraint
            if hw_type in hardware_constraints:
                if hw_counts[hw_type] >= hardware_constraints[hw_type]:
                    continue  # Skip this configuration
                    
                # Increment the hardware count
                hw_counts[hw_type] += 1
            
            # Add configuration to batch
            batch.append(config)
            total_selected += 1
            
            # Check if we've reached the batch size limit
            if total_selected >= batch_size:
                break
                
        # Convert list back to DataFrame
        return pd.DataFrame(batch)
    
    def _diversity_sampling(self, configs_df, score_column, batch_size, diversity_weight, hardware_constraints=None):
        """
        Select diverse configurations with high scores.
        
        Args:
            configs_df: DataFrame of configurations
            score_column: Name of the column containing scores
            batch_size: Maximum number of configurations to select
            diversity_weight: Weight to give diversity vs. score (0-1)
            hardware_constraints: Dictionary mapping hardware types to maximum count in batch
            
        Returns:
            DataFrame of selected diverse configurations
        """
        # Hardware column might be called 'hardware' or 'hardware_platform'
        hardware_column = 'hardware' if 'hardware' in configs_df.columns else 'hardware_platform'
        
        # Get numerical features for diversity calculation
        numeric_columns = [col for col in configs_df.columns if configs_df[col].dtype in [np.int64, np.float64]]
        categorical_columns = [col for col in configs_df.columns if col not in numeric_columns 
                              and col != score_column 
                              and col != 'uncertainty'
                              and col != 'diversity'
                              and col != 'information_gain'
                              and col != 'selection_method']
        
        # Create feature matrix for diversity calculation
        from sklearn.preprocessing import StandardScaler
        feature_df = pd.get_dummies(configs_df[categorical_columns])
        if numeric_columns:
            # Scale numeric columns
            scaler = StandardScaler()
            scaled_numeric = scaler.fit_transform(configs_df[numeric_columns])
            numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_columns)
            feature_df = pd.concat([feature_df, numeric_df], axis=1)
        
        # Convert to numpy array for faster processing
        features = feature_df.values
        scores = configs_df[score_column].values
        
        # Initialize hardware counts if constraints are provided
        hw_counts = {hw: 0 for hw in hardware_constraints.keys()} if hardware_constraints else None
        
        # Initialize selected configurations
        selected_indices = []
        remaining_indices = list(range(len(configs_df)))
        
        # Select first configuration with highest score
        best_idx = np.argmax(scores)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # If hardware constraints are provided, update the count
        if hardware_constraints:
            hw_type = configs_df.iloc[best_idx][hardware_column]
            if hw_type in hw_counts:
                hw_counts[hw_type] += 1
        
        # Select remaining configurations
        while len(selected_indices) < batch_size and remaining_indices:
            best_score = -float('inf')
            best_idx = -1
            
            for idx in remaining_indices:
                # Calculate diversity as minimum distance to already selected points
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    distance = euclidean(features[idx], features[selected_idx])
                    min_distance = min(min_distance, distance)
                
                # Normalize min_distance to [0, 1] range
                # We'll use a simple approach here, assuming distances are roughly in [0, 10] range
                norm_distance = min(min_distance / 10.0, 1.0)
                
                # Calculate combined score as weighted combination of original score and diversity
                norm_score = scores[idx] / max(scores) if max(scores) > 0 else scores[idx]
                combined_score = (1 - diversity_weight) * norm_score + diversity_weight * norm_distance
                
                # Check hardware constraints if provided
                if hardware_constraints:
                    hw_type = configs_df.iloc[idx][hardware_column]
                    if hw_type in hw_counts and hw_counts[hw_type] >= hardware_constraints[hw_type]:
                        continue  # Skip this configuration as we've reached the hardware constraint
                
                # Update best if this is better
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            # If we couldn't find a valid configuration, break
            if best_idx == -1:
                break
                
            # Add best configuration to selected
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            # Update hardware count if constraints are provided
            if hardware_constraints:
                hw_type = configs_df.iloc[best_idx][hardware_column]
                if hw_type in hw_counts:
                    hw_counts[hw_type] += 1
        
        # Extract selected configurations
        selected_configs = configs_df.iloc[selected_indices].copy()
        
        # Add a column indicating selection order
        selected_configs['selection_order'] = range(1, len(selected_configs) + 1)
        
        logger.info(f"Generated diverse batch with {len(selected_configs)} configurations")
        return selected_configs

def test_basic_batch_generation():
    """Test basic batch generation without special constraints."""
    logger.info("Testing basic batch generation")
    batch_generator = TestBatchGenerator()
    
    # Generate a batch with default settings
    batch = batch_generator.suggest_test_batch(
        configurations=batch_generator.configs_df,
        batch_size=10,
        ensure_diversity=True
    )
    
    logger.info(f"Generated batch with {len(batch)} configurations")
    print("\nBasic Batch Generation Results:")
    print(f"Original configurations: {len(batch_generator.configs_df)}")
    print(f"Batch size: {len(batch)}")
    print(f"Selected configurations (first 3):")
    print(batch[['model_type', 'hardware', 'batch_size', 'expected_information_gain', 'selection_order']].head(3))
    
    # Validate that the batch has the right size
    assert len(batch) <= 10, f"Batch size should be ≤ 10, got {len(batch)}"
    
    # Validate that selection_order column was added
    assert 'selection_order' in batch.columns, "Batch should have selection_order column"
    
    return batch

def test_hardware_constrained_batch():
    """Test batch generation with hardware constraints."""
    logger.info("Testing hardware-constrained batch generation")
    batch_generator = TestBatchGenerator()
    
    # Define hardware constraints
    hardware_constraints = {
        'cpu': 2,
        'cuda': 3,
        'openvino': 1,
        'webgpu': 1
    }
    
    # Generate a batch with hardware constraints
    batch = batch_generator.suggest_test_batch(
        configurations=batch_generator.configs_df,
        batch_size=10,
        ensure_diversity=True,
        hardware_constraints=hardware_constraints
    )
    
    logger.info(f"Generated hardware-constrained batch with {len(batch)} configurations")
    print("\nHardware-Constrained Batch Generation Results:")
    print(f"Original configurations: {len(batch_generator.configs_df)}")
    print(f"Batch size: {len(batch)}")
    
    # Check hardware counts
    hw_counts = batch['hardware'].value_counts().to_dict()
    print(f"Hardware counts: {hw_counts}")
    
    # Validate hardware constraints
    for hw, limit in hardware_constraints.items():
        count = hw_counts.get(hw, 0)
        assert count <= limit, f"Hardware {hw} exceeded constraint: {count} > {limit}"
    
    return batch

def test_hardware_availability():
    """Test batch generation with hardware availability factors."""
    logger.info("Testing hardware availability weighting")
    batch_generator = TestBatchGenerator()
    
    # Define hardware availability (probabilities of 0-1)
    hardware_availability = {
        'cpu': 1.0,    # Fully available
        'cuda': 0.8,   # 80% available
        'openvino': 0.5, # 50% available
        'webgpu': 0.2  # 20% available
    }
    
    # Generate a batch with hardware availability weighting
    batch = batch_generator.suggest_test_batch(
        configurations=batch_generator.configs_df,
        batch_size=10,
        ensure_diversity=True,
        hardware_availability=hardware_availability
    )
    
    logger.info(f"Generated availability-weighted batch with {len(batch)} configurations")
    print("\nHardware Availability Batch Generation Results:")
    print(f"Original configurations: {len(batch_generator.configs_df)}")
    print(f"Batch size: {len(batch)}")
    
    # Check hardware counts
    hw_counts = batch['hardware'].value_counts().to_dict()
    print(f"Hardware counts: {hw_counts}")
    
    # No strict validation here, but we can observe the distribution trends
    
    return batch

def test_diversity_weighting():
    """Test batch generation with different diversity weights."""
    logger.info("Testing diversity weighting impact")
    batch_generator = TestBatchGenerator()
    
    results = {}
    
    # Test different diversity weights
    for weight in [0.1, 0.5, 0.9]:
        batch = batch_generator.suggest_test_batch(
            configurations=batch_generator.configs_df,
            batch_size=10,
            ensure_diversity=True,
            diversity_weight=weight
        )
        
        results[weight] = batch
        
        logger.info(f"Generated batch with diversity weight {weight}")
    
    print("\nDiversity Weighting Results:")
    print("Hardware distribution with different diversity weights:")
    
    for weight, batch in results.items():
        hw_counts = batch['hardware'].value_counts().to_dict()
        model_counts = batch['model_type'].value_counts().to_dict()
        print(f"\nDiversity weight {weight}:")
        print(f"Hardware distribution: {hw_counts}")
        print(f"Model type distribution: {model_counts}")
    
    # The higher the diversity weight, the more evenly distributed the configs should be
    
    return results

def test_combined_constraints():
    """Test batch generation with both hardware constraints and availability."""
    logger.info("Testing combined constraints")
    batch_generator = TestBatchGenerator()
    
    # Define constraints
    hardware_constraints = {
        'cpu': 3,
        'cuda': 3,
        'openvino': 2,
        'webgpu': 1
    }
    
    hardware_availability = {
        'cpu': 1.0,
        'cuda': 0.7,
        'openvino': 0.5,
        'webgpu': 0.3
    }
    
    # Generate batch with combined constraints
    batch = batch_generator.suggest_test_batch(
        configurations=batch_generator.configs_df,
        batch_size=10,
        ensure_diversity=True,
        hardware_constraints=hardware_constraints,
        hardware_availability=hardware_availability,
        diversity_weight=0.6
    )
    
    logger.info(f"Generated batch with combined constraints: {len(batch)} configurations")
    print("\nCombined Constraints Batch Generation Results:")
    print(f"Original configurations: {len(batch_generator.configs_df)}")
    print(f"Batch size: {len(batch)}")
    print(f"Hardware counts: {batch['hardware'].value_counts().to_dict()}")
    print(f"Model type counts: {batch['model_type'].value_counts().to_dict()}")
    
    # Validate hardware constraints
    hw_counts = batch['hardware'].value_counts().to_dict()
    for hw, limit in hardware_constraints.items():
        count = hw_counts.get(hw, 0)
        assert count <= limit, f"Hardware {hw} exceeded constraint: {count} > {limit}"
    
    return batch

def run_all_tests():
    """Run all test cases."""
    test_basic_batch_generation()
    test_hardware_constrained_batch()
    test_hardware_availability()
    test_diversity_weighting()
    test_combined_constraints()
    
    logger.info("All tests completed successfully!")

def main():
    """Main function to run tests based on command line arguments."""
    parser = argparse.ArgumentParser(description="Test the Test Batch Generator functionality")
    parser.add_argument("--test", choices=["basic", "hardware", "availability", 
                                          "diversity", "combined", "all"],
                        default="all", help="Test to run")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size for test generation")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.test == "basic":
        test_basic_batch_generation()
    elif args.test == "hardware":
        test_hardware_constrained_batch()
    elif args.test == "availability":
        test_hardware_availability()
    elif args.test == "diversity":
        test_diversity_weighting()
    elif args.test == "combined":
        test_combined_constraints()
    elif args.test == "all":
        run_all_tests()

if __name__ == "__main__":
    main()