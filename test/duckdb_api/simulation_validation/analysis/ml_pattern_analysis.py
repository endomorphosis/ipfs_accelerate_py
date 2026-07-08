#!/usr/bin/env python3
"""
Machine Learning Pattern Analysis for the Simulation Accuracy and Validation Framework.

This module provides machine learning-based pattern analysis for complex relationships
in simulation validation results, including:
- Clustering for identifying similar validation patterns
- Feature importance analysis for determining key simulation parameters
- Dimensional reduction for visualizing complex relationships
- Pattern mining for discovering recurring accuracy patterns
- Automated parameter sensitivity analysis
"""

import logging
import numpy as np
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analysis.ml_pattern")

# Import base class
from data.duckdb.simulation_validation.analysis.base import AnalysisMethod
from data.duckdb.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)

class MLPatternAnalysis(AnalysisMethod):
    """
    Machine learning-based pattern analysis for simulation validation results.
    
    This class extends the basic AnalysisMethod to provide sophisticated
    machine learning techniques for identifying complex patterns in validation results:
    - Clustering for grouping similar validation patterns
    - Feature importance analysis for identifying key simulation parameters
    - Dimensional reduction for visualizing complex relationships
    - Pattern mining for discovering recurring accuracy patterns
    - Automated parameter sensitivity analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML pattern analysis method.
        
        Args:
            config: Configuration options for the analysis method
        """
        super().__init__(config)
        
        # Default configuration
        default_config = {
            # Common metrics to analyze
            "metrics_to_analyze": [
                "throughput_items_per_second",
                "average_latency_ms",
                "memory_peak_mb", 
                "power_consumption_w"
            ],
            
            # Clustering configuration
            "clustering": {
                "enabled": True,
                "methods": ["kmeans", "hierarchical"],
                "max_clusters": 5,           # Maximum number of clusters to consider
                "min_samples_per_cluster": 3,  # Minimum samples per cluster
                "features": ["error", "relative_error"],  # Features to use for clustering
                "visualization": True         # Whether to generate visualization data
            },
            
            # Feature importance configuration
            "feature_importance": {
                "enabled": True,
                "methods": ["random_forest", "permutation", "shap"],
                "target_variable": "error",    # Target variable for importance analysis
                "min_samples_required": 10,   # Minimum samples required for feature importance
                "train_test_split": 0.7       # Train/test split ratio
            },
            
            # Dimensional reduction configuration
            "dimensional_reduction": {
                "enabled": True,
                "methods": ["pca", "tsne", "umap"],
                "n_components": 2,            # Number of components for reduction
                "perplexity": 5,              # Perplexity parameter for t-SNE
                "min_samples_required": 5     # Minimum samples required for dimensional reduction
            },
            
            # Pattern mining configuration
            "pattern_mining": {
                "enabled": True,
                "min_support": 0.3,           # Minimum support for frequent patterns
                "min_confidence": 0.7,        # Minimum confidence for association rules
                "max_patterns": 10,           # Maximum number of patterns to report
                "discretization_bins": 5      # Number of bins for discretizing continuous values
            },
            
            # Parameter sensitivity configuration
            "parameter_sensitivity": {
                "enabled": True,
                "parameters_to_analyze": [
                    "batch_size",
                    "precision"
                ],
                "metrics_to_analyze": [
                    "throughput_items_per_second",
                    "average_latency_ms"
                ],
                "min_samples_per_parameter": 3  # Minimum samples per parameter value
            }
        }
        
        # Apply default config values if not specified
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict) and isinstance(self.config[key], dict):
                # Merge nested dictionaries
                for nested_key, nested_value in value.items():
                    if nested_key not in self.config[key]:
                        self.config[key][nested_key] = nested_value
    
    def analyze(
        self, 
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Perform machine learning-based pattern analysis on validation results.
        
        Args:
            validation_results: List of validation results to analyze
            
        Returns:
            Dictionary containing pattern analysis results and insights
        """
        # Check requirements
        meets_req, error_msg = self.check_requirements(validation_results)
        if not meets_req:
            logger.warning(f"Requirements not met for ML pattern analysis: {error_msg}")
            return {"status": "error", "message": error_msg}
        
        # Initialize results dictionary
        analysis_results = {
            "status": "success",
            "timestamp": datetime.datetime.now().isoformat(),
            "num_validation_results": len(validation_results),
            "metrics_analyzed": self.config["metrics_to_analyze"],
            "analysis_methods": {},
            "insights": {
                "key_findings": [],
                "patterns_identified": [],
                "recommendations": []
            }
        }
        
        # Extract data for analysis
        data_matrix, feature_names, metadata = self._prepare_data_matrix(validation_results)
        
        # Skip analysis if insufficient data
        if data_matrix is None or data_matrix.shape[0] < 3:
            return {
                "status": "error",
                "message": "Insufficient data for ML pattern analysis"
            }
        
        # Perform clustering if enabled
        if self.config["clustering"]["enabled"]:
            try:
                clustering_results = self._perform_clustering(data_matrix, feature_names, metadata)
                analysis_results["analysis_methods"]["clustering"] = clustering_results
                
                # Add key findings from clustering
                if "clusters" in clustering_results and clustering_results["clusters"]:
                    if len(clustering_results["clusters"]) > 1:
                        finding = (f"Identified {len(clustering_results['clusters'])} distinct "
                                  f"clusters in validation results, indicating different error patterns")
                        analysis_results["insights"]["key_findings"].append(finding)
                        analysis_results["insights"]["patterns_identified"].append(
                            f"Distinct validation result clusters: {len(clustering_results['clusters'])}"
                        )
                    
                    # Add findings about cluster characteristics
                    for cluster_id, cluster_info in clustering_results["clusters"].items():
                        if "distinctive_features" in cluster_info:
                            features = ", ".join([f"{f['feature']}: {f['value']:.2f}" 
                                                for f in cluster_info["distinctive_features"][:2]])
                            finding = (f"Cluster {cluster_id} ({cluster_info['size']} samples) "
                                     f"is characterized by {features}")
                            analysis_results["insights"]["patterns_identified"].append(finding)
            except Exception as e:
                logger.error(f"Error performing clustering: {e}")
                analysis_results["analysis_methods"]["clustering"] = {
                    "status": "error", 
                    "message": str(e)
                }
        
        # Perform feature importance analysis if enabled
        if self.config["feature_importance"]["enabled"]:
            try:
                # Check if we have enough data points for feature importance
                min_samples = self.config["feature_importance"]["min_samples_required"]
                if data_matrix.shape[0] >= min_samples:
                    importance_results = self._analyze_feature_importance(
                        data_matrix, feature_names, metadata)
                    analysis_results["analysis_methods"]["feature_importance"] = importance_results
                    
                    # Add key findings from feature importance
                    if "important_features" in importance_results:
                        top_features = importance_results["important_features"][:3]
                        if top_features:
                            features_str = ", ".join([f"{f['feature']} ({f['importance']:.2f})" 
                                                     for f in top_features])
                            finding = (f"Top factors affecting simulation accuracy: {features_str}")
                            analysis_results["insights"]["key_findings"].append(finding)
                            
                            # Add more detailed pattern insights
                            for feature in top_features:
                                if feature["importance"] > 0.2:  # Only include significant features
                                    analysis_results["insights"]["patterns_identified"].append(
                                        f"{feature['feature']} has a significant impact on simulation accuracy "
                                        f"(importance: {feature['importance']:.2f})"
                                    )
                else:
                    analysis_results["analysis_methods"]["feature_importance"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for feature importance analysis. "
                                 f"Required: {min_samples}, Provided: {data_matrix.shape[0]}"
                    }
            except Exception as e:
                logger.error(f"Error analyzing feature importance: {e}")
                analysis_results["analysis_methods"]["feature_importance"] = {
                    "status": "error", 
                    "message": str(e)
                }
        
        # Perform dimensional reduction if enabled
        if self.config["dimensional_reduction"]["enabled"]:
            try:
                # Check if we have enough data points for dimensional reduction
                min_samples = self.config["dimensional_reduction"]["min_samples_required"]
                if data_matrix.shape[0] >= min_samples:
                    reduction_results = self._perform_dimensional_reduction(
                        data_matrix, feature_names, metadata)
                    analysis_results["analysis_methods"]["dimensional_reduction"] = reduction_results
                    
                    # Add key findings from dimensional reduction
                    if "pca" in reduction_results and "explained_variance" in reduction_results["pca"]:
                        var = reduction_results["pca"]["explained_variance"][0] * 100
                        finding = (f"First principal component explains {var:.1f}% of variation "
                                  f"in simulation accuracy")
                        analysis_results["insights"]["key_findings"].append(finding)
                else:
                    analysis_results["analysis_methods"]["dimensional_reduction"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for dimensional reduction. "
                                 f"Required: {min_samples}, Provided: {data_matrix.shape[0]}"
                    }
            except Exception as e:
                logger.error(f"Error performing dimensional reduction: {e}")
                analysis_results["analysis_methods"]["dimensional_reduction"] = {
                    "status": "error", 
                    "message": str(e)
                }
        
        # Perform pattern mining if enabled
        if self.config["pattern_mining"]["enabled"]:
            try:
                pattern_results = self._mine_patterns(validation_results)
                analysis_results["analysis_methods"]["pattern_mining"] = pattern_results
                
                # Add key findings from pattern mining
                if "frequent_patterns" in pattern_results:
                    for pattern in pattern_results["frequent_patterns"][:2]:
                        finding = (f"Discovered frequent pattern: {pattern['description']} "
                                  f"(support: {pattern['support']:.2f})")
                        analysis_results["insights"]["patterns_identified"].append(finding)
                
                if "association_rules" in pattern_results:
                    for rule in pattern_results["association_rules"][:2]:
                        finding = (f"Rule discovered: {rule['description']} "
                                  f"(confidence: {rule['confidence']:.2f})")
                        analysis_results["insights"]["key_findings"].append(finding)
            except Exception as e:
                logger.error(f"Error mining patterns: {e}")
                analysis_results["analysis_methods"]["pattern_mining"] = {
                    "status": "error", 
                    "message": str(e)
                }
        
        # Perform parameter sensitivity analysis if enabled
        if self.config["parameter_sensitivity"]["enabled"]:
            try:
                sensitivity_results = self._analyze_parameter_sensitivity(validation_results)
                analysis_results["analysis_methods"]["parameter_sensitivity"] = sensitivity_results
                
                # Add key findings from parameter sensitivity
                if "parameter_impacts" in sensitivity_results:
                    for param, impacts in sensitivity_results["parameter_impacts"].items():
                        if impacts:
                            # Find most significant impact
                            sorted_impacts = sorted(impacts, key=lambda x: abs(x.get("impact", 0)), reverse=True)
                            if sorted_impacts:
                                impact = sorted_impacts[0]
                                if abs(impact.get("impact", 0)) > 0.2:  # Only include significant impacts
                                    direction = "increases" if impact.get("impact", 0) > 0 else "decreases"
                                    finding = (f"Changing {param} significantly {direction} "
                                             f"{impact.get('metric', 'simulation accuracy')}")
                                    analysis_results["insights"]["key_findings"].append(finding)
                                    analysis_results["insights"]["patterns_identified"].append(
                                        f"{param} has a significant impact on {impact.get('metric', 'accuracy')}"
                                    )
            except Exception as e:
                logger.error(f"Error analyzing parameter sensitivity: {e}")
                analysis_results["analysis_methods"]["parameter_sensitivity"] = {
                    "status": "error", 
                    "message": str(e)
                }
        
        # Generate recommendations based on analysis results
        try:
            recommendations = self._generate_recommendations(analysis_results)
            analysis_results["insights"]["recommendations"] = recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            analysis_results["insights"]["recommendations"] = [
                "Error generating recommendations: " + str(e)
            ]
        
        return analysis_results
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about the capabilities of the ML pattern analysis.
        
        Returns:
            Dictionary describing the capabilities
        """
        return {
            "name": "Machine Learning Pattern Analysis",
            "description": "Identifies complex patterns in simulation validation results using ML techniques",
            "methods": [
                {
                    "name": "Clustering",
                    "description": "Groups similar validation results to identify error patterns",
                    "enabled": self.config["clustering"]["enabled"],
                    "algorithms": self.config["clustering"]["methods"]
                },
                {
                    "name": "Feature Importance",
                    "description": "Identifies which factors most influence simulation accuracy",
                    "enabled": self.config["feature_importance"]["enabled"],
                    "algorithms": self.config["feature_importance"]["methods"]
                },
                {
                    "name": "Dimensional Reduction",
                    "description": "Reduces complexity for visualization and pattern discovery",
                    "enabled": self.config["dimensional_reduction"]["enabled"],
                    "algorithms": self.config["dimensional_reduction"]["methods"]
                },
                {
                    "name": "Pattern Mining",
                    "description": "Discovers recurring patterns in validation results",
                    "enabled": self.config["pattern_mining"]["enabled"]
                },
                {
                    "name": "Parameter Sensitivity",
                    "description": "Analyzes how simulation parameters affect accuracy",
                    "enabled": self.config["parameter_sensitivity"]["enabled"]
                }
            ],
            "output_format": {
                "analysis_results": "Dictionary with analysis results for each method",
                "insights": "Key findings, patterns, and recommendations from the analysis"
            }
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """
        Get information about the requirements of this analysis method.
        
        Returns:
            Dictionary describing the requirements
        """
        # Define minimum requirements
        requirements = {
            "min_validation_results": 3,
            "required_metrics": self.config["metrics_to_analyze"],
            "optimal_validation_results": 15,
            "feature_importance_requirements": {
                "min_samples": self.config["feature_importance"]["min_samples_required"]
            },
            "dimensional_reduction_requirements": {
                "min_samples": self.config["dimensional_reduction"]["min_samples_required"]
            },
            "parameter_sensitivity_requirements": {
                "min_samples_per_parameter": self.config["parameter_sensitivity"]["min_samples_per_parameter"]
            }
        }
        
        return requirements
    
    def _prepare_data_matrix(
        self,
        validation_results: List[ValidationResult]
    ) -> Tuple[Optional[np.ndarray], List[str], List[Dict[str, Any]]]:
        """
        Prepare data matrix for machine learning analysis.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Tuple containing:
                - Data matrix (samples x features) or None if insufficient data
                - List of feature names
                - List of metadata dictionaries for each sample
        """
        if not validation_results:
            return None, [], []
        
        # Define metrics to analyze
        metrics_to_analyze = self.config["metrics_to_analyze"]
        
        # Initialize lists for features, data, and metadata
        feature_names = []
        data_rows = []
        metadata = []
        
        # Extract features from validation results
        for result in validation_results:
            row = []
            meta = {
                "hardware_id": result.hardware_result.hardware_id,
                "model_id": result.hardware_result.model_id,
                "batch_size": result.hardware_result.batch_size,
                "precision": result.hardware_result.precision,
                "timestamp": result.validation_timestamp
            }
            
            # Add metrics from simulation and hardware results
            features_added = False
            
            for metric in metrics_to_analyze:
                if (metric in result.simulation_result.metrics and 
                    metric in result.hardware_result.metrics):
                    
                    sim_val = result.simulation_result.metrics[metric]
                    hw_val = result.hardware_result.metrics[metric]
                    
                    # Skip if either value is None
                    if sim_val is None or hw_val is None:
                        continue
                    
                    # Add feature names the first time
                    if not feature_names:
                        feature_names.append(f"{metric}_sim")
                        feature_names.append(f"{metric}_hw")
                        feature_names.append(f"{metric}_error")
                        feature_names.append(f"{metric}_rel_error")
                    
                    # Calculate error and relative error
                    error = abs(sim_val - hw_val)
                    rel_error = error / hw_val if hw_val != 0 else np.nan
                    
                    # Add values to row
                    row.extend([sim_val, hw_val, error, rel_error])
                    features_added = True
            
            # Add additional parameters as features
            if hasattr(result.simulation_result, "additional_metadata"):
                additional_metadata = result.simulation_result.additional_metadata or {}
                for key, value in additional_metadata.items():
                    if isinstance(value, (int, float)):
                        if not feature_names:
                            feature_names.append(f"param_{key}")
                        row.append(value)
                        meta[key] = value
                        features_added = True
            
            # Only add row if features were added
            if features_added and len(row) > 0:
                # Handle NaN values
                row = [0.0 if np.isnan(x) else x for x in row]
                
                if len(feature_names) == len(row):
                    data_rows.append(row)
                    metadata.append(meta)
                else:
                    logger.warning(f"Feature count mismatch: {len(feature_names)} != {len(row)}")
        
        # Convert to numpy array
        if data_rows:
            data_matrix = np.array(data_rows)
            return data_matrix, feature_names, metadata
        else:
            return None, [], []
    
    def _perform_clustering(
        self,
        data_matrix: np.ndarray,
        feature_names: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform clustering on validation data.
        
        Args:
            data_matrix: Data matrix (samples x features)
            feature_names: List of feature names
            metadata: List of metadata dictionaries for each sample
            
        Returns:
            Dictionary with clustering results
        """
        clustering_results = {
            "methods_used": [],
            "clusters": {},
            "visualization_data": {}
        }
        
        # Skip if insufficient data
        if data_matrix.shape[0] < 3:
            return {"status": "skipped", "message": "Insufficient data for clustering"}
        
        # Filter features to use for clustering
        feature_indices = []
        filtered_feature_names = []
        
        for feature_type in self.config["clustering"]["features"]:
            for i, feature in enumerate(feature_names):
                if feature_type in feature:
                    feature_indices.append(i)
                    filtered_feature_names.append(feature)
        
        # Skip if no features match criteria
        if not feature_indices:
            return {"status": "skipped", "message": "No matching features for clustering"}
        
        # Extract relevant features for clustering
        clustering_data = data_matrix[:, feature_indices]
        
        # Normalize data for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(clustering_data)
        
        # Apply clustering algorithms
        if "kmeans" in self.config["clustering"]["methods"]:
            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                
                # Determine optimal number of clusters
                max_clusters = min(self.config["clustering"]["max_clusters"], 
                                  clustering_data.shape[0] // 2)
                
                if max_clusters < 2:
                    max_clusters = 2
                
                # Calculate silhouette scores for different numbers of clusters
                silhouette_scores = []
                for n_clusters in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(normalized_data)
                    
                    # Skip if any cluster has fewer than min_samples
                    cluster_sizes = np.bincount(cluster_labels)
                    if min(cluster_sizes) < self.config["clustering"]["min_samples_per_cluster"]:
                        silhouette_scores.append(-1)  # Invalid score
                        continue
                    
                    # Calculate silhouette score
                    score = silhouette_score(normalized_data, cluster_labels)
                    silhouette_scores.append(score)
                
                # Select optimal number of clusters
                if silhouette_scores and max(silhouette_scores) > 0:
                    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 since we start from 2
                else:
                    optimal_clusters = 2  # Default to 2 clusters
                
                # Apply KMeans with optimal number of clusters
                kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(normalized_data)
                
                # Calculate cluster centers
                cluster_centers = kmeans.cluster_centers_
                
                # Add cluster info to metadata
                for i, label in enumerate(cluster_labels):
                    metadata[i]["kmeans_cluster"] = int(label)
                
                # Calculate cluster statistics
                clusters = {}
                for label in range(optimal_clusters):
                    # Get samples in this cluster
                    cluster_indices = np.where(cluster_labels == label)[0]
                    cluster_samples = normalized_data[cluster_indices]
                    
                    # Skip if empty cluster
                    if len(cluster_samples) == 0:
                        continue
                    
                    # Calculate cluster center and variance
                    center = cluster_centers[label]
                    variance = np.var(cluster_samples, axis=0)
                    
                    # Find distinctive features for this cluster
                    distinctive_features = []
                    for i, (feat_name, center_val, var_val) in enumerate(
                        zip(filtered_feature_names, center, variance)):
                        # Compare this center to other centers
                        other_centers = [c[i] for j, c in enumerate(cluster_centers) if j != label]
                        
                        if other_centers:
                            # Calculate how distinctive this feature is
                            distinctiveness = abs(center_val - np.mean(other_centers)) / (np.std(other_centers) + 1e-10)
                            
                            distinctive_features.append({
                                "feature": feat_name,
                                "value": float(scaler.inverse_transform([center])[0][i]),
                                "distinctiveness": float(distinctiveness)
                            })
                    
                    # Sort by distinctiveness
                    distinctive_features.sort(key=lambda x: x["distinctiveness"], reverse=True)
                    
                    # Get metadata for samples in this cluster
                    cluster_metadata = [metadata[i] for i in cluster_indices]
                    
                    # Find common characteristics
                    common_characteristics = {}
                    
                    for key in ["hardware_id", "model_id", "batch_size", "precision"]:
                        values = [m[key] for m in cluster_metadata if key in m]
                        if values:
                            most_common = max(set(values), key=values.count)
                            frequency = values.count(most_common) / len(values)
                            
                            if frequency >= 0.7:  # Only include if at least 70% have this value
                                common_characteristics[key] = {
                                    "value": most_common,
                                    "frequency": frequency
                                }
                    
                    # Add cluster info
                    clusters[str(label)] = {
                        "size": len(cluster_indices),
                        "center": center.tolist(),
                        "variance": variance.tolist(),
                        "distinctive_features": distinctive_features,
                        "common_characteristics": common_characteristics,
                        "sample_indices": cluster_indices.tolist()
                    }
                
                # Add cluster info to results
                clustering_results["clusters"] = clusters
                clustering_results["methods_used"].append("kmeans")
                clustering_results["kmeans"] = {
                    "num_clusters": optimal_clusters,
                    "silhouette_scores": silhouette_scores,
                    "optimal_silhouette": max(silhouette_scores) if silhouette_scores else None
                }
                
                # Add visualization data if enabled
                if self.config["clustering"]["visualization"]:
                    clustering_results["visualization_data"]["kmeans"] = {
                        "cluster_labels": cluster_labels.tolist(),
                        "features": filtered_feature_names,
                        "pca_components": self._generate_pca_for_visualization(normalized_data)
                    }
                
            except Exception as e:
                logger.warning(f"Error performing KMeans clustering: {e}")
        
        # Additional clustering methods would be implemented here
        
        return clustering_results
    
    def _generate_pca_for_visualization(
        self,
        data: np.ndarray
    ) -> List[List[float]]:
        """
        Generate PCA components for visualization.
        
        Args:
            data: Input data matrix
            
        Returns:
            List of PCA-transformed data points (2D)
        """
        try:
            from sklearn.decomposition import PCA
            
            # Apply PCA for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data).tolist()
            
            return pca_result
        except Exception as e:
            logger.warning(f"Error generating PCA for visualization: {e}")
            # Return empty list if error occurs
            return []
    
    def _analyze_feature_importance(
        self,
        data_matrix: np.ndarray,
        feature_names: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze feature importance using machine learning methods.
        
        Args:
            data_matrix: Data matrix (samples x features)
            feature_names: List of feature names
            metadata: List of metadata dictionaries for each sample
            
        Returns:
            Dictionary with feature importance results
        """
        importance_results = {
            "methods_used": [],
            "important_features": []
        }
        
        # Skip if insufficient data
        min_samples = self.config["feature_importance"]["min_samples_required"]
        if data_matrix.shape[0] < min_samples:
            return {
                "status": "skipped", 
                "message": f"Insufficient data for feature importance analysis. "
                        f"Required: {min_samples}, Provided: {data_matrix.shape[0]}"
            }
        
        # Determine target variable
        target_variable = self.config["feature_importance"]["target_variable"]
        
        # Find target column index
        target_indices = [i for i, name in enumerate(feature_names) if target_variable in name]
        
        if not target_indices:
            return {
                "status": "error",
                "message": f"Target variable '{target_variable}' not found in features"
            }
        
        # Use the first matching target
        target_index = target_indices[0]
        
        # Separate features and target
        X = np.delete(data_matrix, target_index, axis=1)
        y = data_matrix[:, target_index]
        
        # Update feature names to exclude target
        X_feature_names = feature_names.copy()
        del X_feature_names[target_index]
        
        # Apply Random Forest for feature importance if enabled
        if "random_forest" in self.config["feature_importance"]["methods"]:
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                
                # Split data into train and test sets
                split_ratio = self.config["feature_importance"]["train_test_split"]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=1-split_ratio, random_state=42)
                
                # Train Random Forest
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                
                # Get feature importance
                importances = rf.feature_importances_
                
                # Calculate feature importance scores
                feature_importance = []
                for i, (name, importance) in enumerate(zip(X_feature_names, importances)):
                    feature_importance.append({
                        "feature": name,
                        "importance": float(importance)
                    })
                
                # Sort by importance
                feature_importance.sort(key=lambda x: x["importance"], reverse=True)
                
                # Add to results
                importance_results["random_forest"] = {
                    "feature_importance": feature_importance,
                    "model_score": float(rf.score(X_test, y_test))
                }
                
                # Update methods used
                importance_results["methods_used"].append("random_forest")
                
                # Add to overall important features
                if "important_features" not in importance_results:
                    importance_results["important_features"] = feature_importance
                
            except Exception as e:
                logger.warning(f"Error analyzing feature importance with Random Forest: {e}")
        
        # Apply permutation importance if enabled
        if "permutation" in self.config["feature_importance"]["methods"]:
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.inspection import permutation_importance
                from sklearn.model_selection import train_test_split
                
                # Split data into train and test sets
                split_ratio = self.config["feature_importance"]["train_test_split"]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=1-split_ratio, random_state=42)
                
                # Train a model
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                
                # Calculate permutation importance
                result = permutation_importance(
                    rf, X_test, y_test, n_repeats=10, random_state=42)
                
                # Calculate feature importance scores
                feature_importance = []
                for i, (name, importance) in enumerate(zip(X_feature_names, result.importances_mean)):
                    feature_importance.append({
                        "feature": name,
                        "importance": float(importance),
                        "std": float(result.importances_std[i])
                    })
                
                # Sort by importance
                feature_importance.sort(key=lambda x: x["importance"], reverse=True)
                
                # Add to results
                importance_results["permutation"] = {
                    "feature_importance": feature_importance,
                    "model_score": float(rf.score(X_test, y_test))
                }
                
                # Update methods used
                importance_results["methods_used"].append("permutation")
                
                # Update overall important features if not set yet
                if "important_features" not in importance_results:
                    importance_results["important_features"] = feature_importance
                
            except Exception as e:
                logger.warning(f"Error analyzing feature importance with permutation: {e}")
        
        # SHAP values would be implemented here for a more complete implementation
        
        return importance_results
    
    def _perform_dimensional_reduction(
        self,
        data_matrix: np.ndarray,
        feature_names: List[str],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform dimensional reduction for visualization and pattern discovery.
        
        Args:
            data_matrix: Data matrix (samples x features)
            feature_names: List of feature names
            metadata: List of metadata dictionaries for each sample
            
        Returns:
            Dictionary with dimensional reduction results
        """
        reduction_results = {
            "methods_used": []
        }
        
        # Skip if insufficient data
        min_samples = self.config["dimensional_reduction"]["min_samples_required"]
        if data_matrix.shape[0] < min_samples:
            return {
                "status": "skipped", 
                "message": f"Insufficient data for dimensional reduction. "
                        f"Required: {min_samples}, Provided: {data_matrix.shape[0]}"
            }
        
        # Normalize data
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(data_matrix)
        except Exception as e:
            logger.warning(f"Error normalizing data: {e}")
            normalized_data = data_matrix  # Use original data if normalization fails
        
        # Apply PCA if enabled
        if "pca" in self.config["dimensional_reduction"]["methods"]:
            try:
                from sklearn.decomposition import PCA
                
                # Get number of components
                n_components = min(
                    self.config["dimensional_reduction"]["n_components"],
                    data_matrix.shape[1],
                    data_matrix.shape[0]
                )
                
                # Apply PCA
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(normalized_data)
                
                # Calculate explained variance
                explained_variance = pca.explained_variance_ratio_.tolist()
                cumulative_variance = np.cumsum(explained_variance).tolist()
                
                # Get component feature loadings
                components = []
                for i in range(n_components):
                    # Get top features for this component
                    component = pca.components_[i]
                    
                    # Get absolute loadings
                    abs_loadings = np.abs(component)
                    
                    # Find top features
                    top_indices = np.argsort(abs_loadings)[::-1][:5]  # Top 5 features
                    
                    top_features = []
                    for idx in top_indices:
                        top_features.append({
                            "feature": feature_names[idx],
                            "loading": float(component[idx]),
                            "abs_loading": float(abs_loadings[idx])
                        })
                    
                    components.append({
                        "component_number": i + 1,
                        "explained_variance": float(explained_variance[i]),
                        "top_features": top_features
                    })
                
                # Add PCA results
                reduction_results["pca"] = {
                    "result": pca_result.tolist(),
                    "n_components": n_components,
                    "explained_variance": explained_variance,
                    "cumulative_variance": cumulative_variance,
                    "components": components
                }
                
                # Update methods used
                reduction_results["methods_used"].append("pca")
                
            except Exception as e:
                logger.warning(f"Error performing PCA: {e}")
        
        # t-SNE would be implemented here for a more complete implementation
        # UMAP would be implemented here for a more complete implementation
        
        return reduction_results
    
    def _mine_patterns(
        self,
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Mine patterns and association rules in validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary with pattern mining results
        """
        pattern_results = {
            "frequent_patterns": [],
            "association_rules": []
        }
        
        # Skip if insufficient data
        if len(validation_results) < 3:
            return {"status": "skipped", "message": "Insufficient data for pattern mining"}
        
        # Discretize numerical values
        discretized_data = self._discretize_validation_results(validation_results)
        
        # Skip if discretization failed
        if not discretized_data:
            return {"status": "error", "message": "Error discretizing validation results"}
        
        # Perform simplified pattern mining
        # For a complete implementation, libraries like mlxtend would be used
        
        # Count occurrences of each attribute-value pair
        attribute_counts = defaultdict(int)
        total_records = len(discretized_data)
        
        for record in discretized_data:
            for attr, value in record.items():
                attribute_counts[f"{attr}={value}"] = attribute_counts[f"{attr}={value}"] + 1
        
        # Calculate support for each attribute-value pair
        attribute_support = {}
        for attr, count in attribute_counts.items():
            support = count / total_records
            attribute_support[attr] = support
        
        # Find frequent attribute-value pairs
        min_support = self.config["pattern_mining"]["min_support"]
        frequent_attributes = {attr: support for attr, support in attribute_support.items() 
                              if support >= min_support}
        
        # Sort by support
        sorted_frequent = sorted(frequent_attributes.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to max_patterns
        max_patterns = self.config["pattern_mining"]["max_patterns"]
        for attr, support in sorted_frequent[:max_patterns]:
            pattern_results["frequent_patterns"].append({
                "pattern": attr,
                "support": float(support),
                "count": int(attribute_counts[attr]),
                "description": f"{attr} appears in {support*100:.1f}% of validation results"
            })
        
        # Find simple association rules between attributes
        min_confidence = self.config["pattern_mining"]["min_confidence"]
        
        # Check attribute pairs for rules
        for i, (attr1, support1) in enumerate(sorted_frequent):
            attr1_name = attr1.split('=')[0]
            
            for attr2, support2 in sorted_frequent[i+1:]:
                attr2_name = attr2.split('=')[0]
                
                # Skip if attributes are the same
                if attr1_name == attr2_name:
                    continue
                
                # Count co-occurrences
                co_occurrences = 0
                for record in discretized_data:
                    attr1_val = record.get(attr1_name)
                    attr2_val = record.get(attr2_name)
                    
                    if (attr1_val is not None and attr2_val is not None and
                        f"{attr1_name}={attr1_val}" == attr1 and
                        f"{attr2_name}={attr2_val}" == attr2):
                        co_occurrences += 1
                
                # Calculate confidence in both directions
                conf1 = co_occurrences / attribute_counts[attr1] if attribute_counts[attr1] > 0 else 0
                conf2 = co_occurrences / attribute_counts[attr2] if attribute_counts[attr2] > 0 else 0
                
                # Calculate lift
                expected_co_occurrences = (attribute_counts[attr1] * attribute_counts[attr2]) / total_records
                lift = co_occurrences / expected_co_occurrences if expected_co_occurrences > 0 else 0
                
                # Add rules with sufficient confidence
                if conf1 >= min_confidence:
                    pattern_results["association_rules"].append({
                        "antecedent": attr1,
                        "consequent": attr2,
                        "support": float(co_occurrences / total_records),
                        "confidence": float(conf1),
                        "lift": float(lift),
                        "description": f"If {attr1} then {attr2} (conf: {conf1:.2f}, lift: {lift:.2f})"
                    })
                
                if conf2 >= min_confidence:
                    pattern_results["association_rules"].append({
                        "antecedent": attr2,
                        "consequent": attr1,
                        "support": float(co_occurrences / total_records),
                        "confidence": float(conf2),
                        "lift": float(lift),
                        "description": f"If {attr2} then {attr1} (conf: {conf2:.2f}, lift: {lift:.2f})"
                    })
        
        # Sort rules by confidence
        pattern_results["association_rules"].sort(key=lambda x: x["confidence"], reverse=True)
        
        # Limit to max_patterns
        pattern_results["association_rules"] = pattern_results["association_rules"][:max_patterns]
        
        return pattern_results
    
    def _discretize_validation_results(
        self,
        validation_results: List[ValidationResult]
    ) -> List[Dict[str, str]]:
        """
        Discretize numerical values in validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            List of dictionaries with discretized values
        """
        discretized_data = []
        
        try:
            # Define metrics to analyze
            metrics_to_analyze = self.config["metrics_to_analyze"]
            
            # Get all values for each metric for binning
            metric_values = defaultdict(list)
            
            for result in validation_results:
                for metric in metrics_to_analyze:
                    if (metric in result.simulation_result.metrics and 
                        metric in result.hardware_result.metrics):
                        
                        sim_val = result.simulation_result.metrics[metric]
                        hw_val = result.hardware_result.metrics[metric]
                        
                        if sim_val is not None and hw_val is not None:
                            # Calculate error metrics
                            error = abs(sim_val - hw_val)
                            rel_error = error / hw_val if hw_val != 0 else np.nan
                            
                            if not np.isnan(error):
                                metric_values[f"{metric}_error"].append(error)
                            
                            if not np.isnan(rel_error):
                                metric_values[f"{metric}_rel_error"].append(rel_error)
            
            # Create bins for each metric
            bins = {}
            num_bins = self.config["pattern_mining"]["discretization_bins"]
            
            for metric, values in metric_values.items():
                if values:
                    if min(values) == max(values):
                        # All values are the same, create a single bin
                        bins[metric] = np.array([min(values) - 1, max(values) + 1])
                    else:
                        # Create bins
                        bins[metric] = np.linspace(min(values), max(values), num_bins + 1)
            
            # Discretize values for each validation result
            for result in validation_results:
                record = {
                    "hardware_id": result.hardware_result.hardware_id,
                    "model_id": result.hardware_result.model_id,
                    "batch_size": str(result.hardware_result.batch_size),
                    "precision": result.hardware_result.precision
                }
                
                for metric in metrics_to_analyze:
                    if (metric in result.simulation_result.metrics and 
                        metric in result.hardware_result.metrics):
                        
                        sim_val = result.simulation_result.metrics[metric]
                        hw_val = result.hardware_result.metrics[metric]
                        
                        if sim_val is not None and hw_val is not None:
                            # Calculate error metrics
                            error = abs(sim_val - hw_val)
                            rel_error = error / hw_val if hw_val != 0 else np.nan
                            
                            # Discretize error
                            if not np.isnan(error) and f"{metric}_error" in bins:
                                bin_idx = np.digitize(error, bins[f"{metric}_error"]) - 1
                                bin_idx = min(bin_idx, num_bins - 1)  # Ensure valid bin index
                                record[f"{metric}_error"] = f"bin_{bin_idx + 1}"
                            
                            # Discretize relative error
                            if not np.isnan(rel_error) and f"{metric}_rel_error" in bins:
                                bin_idx = np.digitize(rel_error, bins[f"{metric}_rel_error"]) - 1
                                bin_idx = min(bin_idx, num_bins - 1)  # Ensure valid bin index
                                record[f"{metric}_rel_error"] = f"bin_{bin_idx + 1}"
                
                # Add additional parameters from metadata
                if hasattr(result.simulation_result, "additional_metadata"):
                    additional_metadata = result.simulation_result.additional_metadata or {}
                    for key, value in additional_metadata.items():
                        if isinstance(value, (int, float, str)):
                            record[key] = str(value)
                
                discretized_data.append(record)
            
            return discretized_data
            
        except Exception as e:
            logger.warning(f"Error discretizing validation results: {e}")
            return []
    
    def _analyze_parameter_sensitivity(
        self,
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity of simulation accuracy to different parameters.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary with parameter sensitivity results
        """
        sensitivity_results = {
            "parameter_impacts": {},
            "sensitivity_scores": {}
        }
        
        # Skip if insufficient data
        if len(validation_results) < 3:
            return {"status": "skipped", "message": "Insufficient data for parameter sensitivity analysis"}
        
        # Define parameters and metrics to analyze
        parameters = self.config["parameter_sensitivity"]["parameters_to_analyze"]
        metrics = self.config["parameter_sensitivity"]["metrics_to_analyze"]
        
        # Group validation results by parameter values
        parameter_groups = {}
        
        for param in parameters:
            parameter_groups[param] = defaultdict(list)
            
            for result in validation_results:
                # Get parameter value
                if param == "batch_size":
                    value = result.simulation_result.batch_size
                elif param == "precision":
                    value = result.simulation_result.precision
                elif (hasattr(result.simulation_result, "additional_metadata") and
                      result.simulation_result.additional_metadata and
                      param in result.simulation_result.additional_metadata):
                    value = result.simulation_result.additional_metadata[param]
                else:
                    continue
                
                # Add validation result to group
                parameter_groups[param][value].append(result)
        
        # Analyze impact of parameters on metrics
        for param, groups in parameter_groups.items():
            # Skip if insufficient groups
            if len(groups) < 2:
                sensitivity_results["parameter_impacts"][param] = []
                continue
            
            # Get groups with sufficient samples
            min_samples = self.config["parameter_sensitivity"]["min_samples_per_parameter"]
            valid_groups = {value: results for value, results in groups.items() 
                           if len(results) >= min_samples}
            
            # Skip if insufficient valid groups
            if len(valid_groups) < 2:
                sensitivity_results["parameter_impacts"][param] = []
                continue
            
            # Analyze impact on each metric
            metric_impacts = []
            
            for metric in metrics:
                # Calculate average error for each group
                group_errors = {}
                
                for value, results in valid_groups.items():
                    errors = []
                    
                    for result in results:
                        if (metric in result.simulation_result.metrics and 
                            metric in result.hardware_result.metrics):
                            
                            sim_val = result.simulation_result.metrics[metric]
                            hw_val = result.hardware_result.metrics[metric]
                            
                            if sim_val is not None and hw_val is not None:
                                # Calculate relative error
                                error = abs(sim_val - hw_val)
                                rel_error = error / hw_val if hw_val != 0 else np.nan
                                
                                if not np.isnan(rel_error):
                                    errors.append(rel_error)
                    
                    if errors:
                        group_errors[value] = np.mean(errors)
                
                # Skip if insufficient data
                if len(group_errors) < 2:
                    continue
                
                # Convert string values to numeric if possible
                numeric_values = []
                numeric_errors = []
                
                for value, error in group_errors.items():
                    try:
                        if isinstance(value, str) and value.replace('.', '', 1).isdigit():
                            numeric_values.append(float(value))
                        elif isinstance(value, (int, float)):
                            numeric_values.append(float(value))
                        else:
                            continue
                        
                        numeric_errors.append(error)
                    except:
                        continue
                
                # Skip if insufficient numeric data
                if len(numeric_values) < 2:
                    continue
                
                # Calculate correlation between parameter and error
                try:
                    correlation = np.corrcoef(numeric_values, numeric_errors)[0, 1]
                    
                    # Calculate impact score
                    min_error = min(numeric_errors)
                    max_error = max(numeric_errors)
                    error_range = max_error - min_error
                    
                    # Skip if no variation in error
                    if error_range == 0:
                        continue
                    
                    # Calculate impact as normalized correlation
                    impact = correlation * (error_range / min_error) if min_error > 0 else correlation
                    
                    # Add impact to results
                    metric_impacts.append({
                        "metric": metric,
                        "correlation": float(correlation),
                        "impact": float(impact),
                        "group_errors": {str(k): float(v) for k, v in group_errors.items()}
                    })
                    
                except Exception as e:
                    logger.warning(f"Error calculating parameter impact: {e}")
            
            # Sort by absolute impact
            metric_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
            
            # Add to results
            sensitivity_results["parameter_impacts"][param] = metric_impacts
        
        # Calculate overall sensitivity scores
        for param, impacts in sensitivity_results["parameter_impacts"].items():
            if impacts:
                # Calculate average absolute impact
                avg_impact = np.mean([abs(impact["impact"]) for impact in impacts])
                
                # Calculate sensitivity score
                sensitivity_results["sensitivity_scores"][param] = float(avg_impact)
        
        return sensitivity_results
    
    def _generate_recommendations(
        self,
        analysis_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            analysis_results: Analysis results dictionary
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Generate recommendations based on clustering
        if "clustering" in analysis_results["analysis_methods"]:
            clustering = analysis_results["analysis_methods"]["clustering"]
            
            if "clusters" in clustering and len(clustering["clusters"]) > 1:
                # Find largest cluster
                largest_cluster = None
                max_size = 0
                
                for cluster_id, cluster in clustering["clusters"].items():
                    if cluster["size"] > max_size:
                        max_size = cluster["size"]
                        largest_cluster = (cluster_id, cluster)
                
                if largest_cluster:
                    cluster_id, cluster = largest_cluster
                    
                    # Check if cluster has distinctive features
                    if "distinctive_features" in cluster and cluster["distinctive_features"]:
                        top_feature = cluster["distinctive_features"][0]
                        
                        recommendations.append(
                            f"Focus on validation scenarios similar to cluster {cluster_id} "
                            f"which represents {cluster['size']} results with characteristic "
                            f"{top_feature['feature']} values"
                        )
        
        # Generate recommendations based on feature importance
        if "feature_importance" in analysis_results["analysis_methods"]:
            importance = analysis_results["analysis_methods"]["feature_importance"]
            
            if "important_features" in importance and importance["important_features"]:
                top_feature = importance["important_features"][0]
                
                if top_feature["importance"] > 0.2:  # Only recommend if feature is important
                    recommendations.append(
                        f"Prioritize improvements related to {top_feature['feature']} "
                        f"as it has the highest impact on simulation accuracy "
                        f"(importance: {top_feature['importance']:.2f})"
                    )
        
        # Generate recommendations based on pattern mining
        if "pattern_mining" in analysis_results["analysis_methods"]:
            patterns = analysis_results["analysis_methods"]["pattern_mining"]
            
            if "association_rules" in patterns and patterns["association_rules"]:
                top_rule = patterns["association_rules"][0]
                
                if top_rule["confidence"] > 0.8:  # Only recommend if rule is strong
                    recommendations.append(
                        f"Pay attention to the relationship: {top_rule['description']}"
                    )
        
        # Generate recommendations based on parameter sensitivity
        if "parameter_sensitivity" in analysis_results["analysis_methods"]:
            sensitivity = analysis_results["analysis_methods"]["parameter_sensitivity"]
            
            if "sensitivity_scores" in sensitivity:
                # Find most sensitive parameter
                most_sensitive = None
                max_score = 0
                
                for param, score in sensitivity["sensitivity_scores"].items():
                    if score > max_score:
                        max_score = score
                        most_sensitive = param
                
                if most_sensitive and max_score > 0.3:  # Only recommend if sensitivity is significant
                    recommendations.append(
                        f"Focus on optimizing the {most_sensitive} parameter "
                        f"as it has the highest impact on simulation accuracy "
                        f"(sensitivity score: {max_score:.2f})"
                    )
            
            if "parameter_impacts" in sensitivity:
                for param, impacts in sensitivity["parameter_impacts"].items():
                    if impacts and abs(impacts[0]["impact"]) > 0.5:  # Only recommend if impact is significant
                        impact = impacts[0]
                        direction = "increasing" if impact["impact"] > 0 else "decreasing"
                        
                        recommendations.append(
                            f"Consider {direction} {param} to improve accuracy for {impact['metric']}"
                        )
        
        # Add recommendations based on dimensional reduction insights
        if "dimensional_reduction" in analysis_results["analysis_methods"]:
            reduction = analysis_results["analysis_methods"]["dimensional_reduction"]
            
            if "pca" in reduction and "components" in reduction["pca"]:
                components = reduction["pca"]["components"]
                
                if components and len(components) > 0:
                    comp = components[0]
                    
                    if "top_features" in comp and comp["top_features"]:
                        top_feature = comp["top_features"][0]
                        
                        recommendations.append(
                            f"Consider the importance of {top_feature['feature']} which is a key driver "
                            f"of variance in validation results (loading: {abs(top_feature['loading']):.2f})"
                        )
        
        # Ensure we return at least one recommendation
        if not recommendations:
            recommendations.append(
                "Collect more validation data across different configurations "
                "to enable more detailed pattern analysis"
            )
        
        return recommendations