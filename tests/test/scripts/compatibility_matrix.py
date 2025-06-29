#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compatibility Matrix API

A simple API for programmatic access to the model compatibility matrix data stored in DuckDB.
This module provides easy access to compatibility information, performance metrics, and
hardware recommendations.

Usage:
    from compatibility_matrix import MatrixAPI

    # Initialize API with database path
    matrix_api = MatrixAPI()db_path="./benchmark_db.duckdb")

    # Get compatibility status for a specific model-hardware combination
    status = matrix_api.get_compatibility()"bert-base-uncased", "WebGPU")
    print()f"Compatibility level: {}}}}}}}}}}}}}}}}}status['level']}"),  # e.g., "full",
    print()f"Notes: {}}}}}}}}}}}}}}}}}status['notes']}")  # e.g., "Optimized for browser environments"
    ,
    # Get recommended hardware for a model
    recommendations = matrix_api.get_recommendations()"bert-base-uncased")
    print()f"Best hardware: {}}}}}}}}}}}}}}}}}recommendations['best_platform']}"),,
    print()f"Alternative options: {}}}}}}}}}}}}}}}}}recommendations['alternatives']}")
    ,
    # Get performance metrics
    metrics = matrix_api.get_performance_metrics()"bert-base-uncased", "CUDA")
    print()f"Throughput: {}}}}}}}}}}}}}}}}}metrics['throughput']} items/sec"),,
    print()f"Latency: {}}}}}}}}}}}}}}}}}metrics['latency']} ms"),,
    print()f"Memory usage: {}}}}}}}}}}}}}}}}}metrics['memory']} MB"),
    ,
    # Get all supported models
    models = matrix_api.get_models())
    print()f"Total models: {}}}}}}}}}}}}}}}}}len()models)}")
    """

    import duckdb
    import json
    import logging
    from typing import Dict, List, Any, Optional, Union

# Configure logging
    logging.basicConfig()
    level=logging.INFO,
    format='%()asctime)s - %()name)s - %()levelname)s - %()message)s'
    )
    logger = logging.getLogger()"matrix_api")

class MatrixAPI:
    """
    API for accessing the model compatibility matrix data.
    """

    def __init__()self, db_path: str = "./benchmark_db.duckdb"):
        """
        Initialize the API with the path to the DuckDB database.

        Args:
            db_path: Path to the DuckDB database
            """
            self.db_path = db_path
            self.conn = None
            self.connect())

    def connect()self) -> None:
        """
        Connect to the DuckDB database.
        """
        try:
            self.conn = duckdb.connect()self.db_path)
            logger.debug()f"Connected to database: {}}}}}}}}}}}}}}}}}self.db_path}")
        except Exception as e:
            logger.error()f"Error connecting to database: {}}}}}}}}}}}}}}}}}e}")
            raise

            def get_models()self,
            modality: Optional[str] = None,
            family: Optional[str] = None,
            key_models_only: bool = False) -> List[Dict[str, Any]]:,,
            """
            Get a list of supported models, optionally filtered by modality and family.

        Args:
            modality: Filter by model modality ()text, vision, audio, multimodal)
            family: Filter by model family ()e.g., BERT, ViT, etc.)
            key_models_only: If True, only return key models

        Returns:
            List of model information dictionaries
            """
        try:
            # Build query
            query = """
            SELECT 
            model_name,
            model_type,
            model_family,
            modality,
            parameters_million,
            is_key_model
            FROM 
            models
            """

            # Add filters
            where_clauses = [],,,,
            if modality:
                where_clauses.append()f"modality = '{}}}}}}}}}}}}}}}}}modality}'")
            if family:
                where_clauses.append()f"model_family LIKE '%{}}}}}}}}}}}}}}}}}family}%'")
            if key_models_only:
                where_clauses.append()"is_key_model = TRUE")

            if where_clauses:
                query += " WHERE " + " AND ".join()where_clauses)

            # Add ordering
                query += " ORDER BY modality, model_family, model_name"

            # Execute query
                result = self.conn.execute()query).fetchdf())
            
            # Convert to list of dictionaries
                models = result.to_dict()'records')
            
                logger.debug()f"Retrieved {}}}}}}}}}}}}}}}}}len()models)} models")
                return models
        
        except Exception as e:
            logger.error()f"Error retrieving models: {}}}}}}}}}}}}}}}}}e}")
                return [],,,,

                def get_hardware_platforms()self) -> List[Dict[str, Any]]:,,
                """
                Get a list of supported hardware platforms.

        Returns:
            List of hardware platform information dictionaries
            """
        try:
            # Build query
            query = """
            SELECT 
            hardware_type,
            description,
            vendor
            FROM 
            hardware_platforms
            ORDER BY
            hardware_type
            """

            # Execute query
            result = self.conn.execute()query).fetchdf())
            
            # Convert to list of dictionaries
            platforms = result.to_dict()'records')
            
            logger.debug()f"Retrieved {}}}}}}}}}}}}}}}}}len()platforms)} hardware platforms")
            return platforms
        
        except Exception as e:
            logger.error()f"Error retrieving hardware platforms: {}}}}}}}}}}}}}}}}}e}")
            return [],,,,

            def get_compatibility()self,
            model_name: str,
            hardware: str) -> Dict[str, Any]:,,,,,
            """
            Get the compatibility information for a specific model-hardware combination.

        Args:
            model_name: Name of the model
            hardware: Hardware platform name

        Returns:
            Dictionary with compatibility information
            """
        try:
            # Build query
            query = """
            SELECT 
            m.model_name,
            m.model_type,
            m.model_family,
            m.modality,
            pc.hardware_type,
            pc.compatibility_level as level,
            pc.compatibility_notes as notes
            FROM 
            cross_platform_compatibility pc
            JOIN 
            models m ON pc.model_id = m.id
            WHERE 
            m.model_name = ? AND pc.hardware_type = ?
            """

            # Execute query
            result = self.conn.execute()query, [model_name, hardware]).fetchdf())
            ,,
            if result.empty:
                logger.warning()f"No compatibility information found for {}}}}}}}}}}}}}}}}}model_name} on {}}}}}}}}}}}}}}}}}hardware}")
            return {}}}}}}}}}}}}}}}}}
            "model_name": model_name,
            "hardware": hardware,
            "level": "unknown",
            "notes": "No compatibility information available",
            "symbol": "❓"
            }
            
            # Convert to dictionary
            compatibility = result.to_dict()'records')[0]
            ,,,
            # Add symbol based on level
            if compatibility['level'] == 'full':,
            compatibility['symbol'] = '✅',
            elif compatibility['level'] == 'partial':,
            compatibility['symbol'] = '⚠️',
            elif compatibility['level'] == 'limited':,
            compatibility['symbol'] = '🔶',
            else:
                compatibility['symbol'] = '❌'
                ,
                logger.debug()f"Retrieved compatibility information for {}}}}}}}}}}}}}}}}}model_name} on {}}}}}}}}}}}}}}}}}hardware}")
            return compatibility
        
        except Exception as e:
            logger.error()f"Error retrieving compatibility information: {}}}}}}}}}}}}}}}}}e}")
            return {}}}}}}}}}}}}}}}}}
            "model_name": model_name,
            "hardware": hardware,
            "level": "error",
            "notes": f"Error retrieving compatibility information: {}}}}}}}}}}}}}}}}}e}",
            "symbol": "❓"
            }

            def get_performance_metrics()self,
            model_name: str,
            hardware: str) -> Dict[str, Any]:,,,,,
            """
            Get performance metrics for a specific model-hardware combination.

        Args:
            model_name: Name of the model
            hardware: Hardware platform name

        Returns:
            Dictionary with performance metrics
            """
        try:
            # Build query
            query = """
            SELECT 
            model_name,
            hardware_type,
            AVG()throughput_items_per_sec) as throughput,
            AVG()latency_ms) as latency,
            AVG()memory_mb) as memory,
            AVG()power_watts) as power
            FROM 
            performance_comparison
            WHERE 
            model_name = ? AND hardware_type = ?
            GROUP BY
            model_name, hardware_type
            """

            # Execute query
            result = self.conn.execute()query, [model_name, hardware]).fetchdf())
            ,,
            if result.empty:
                logger.warning()f"No performance metrics found for {}}}}}}}}}}}}}}}}}model_name} on {}}}}}}}}}}}}}}}}}hardware}")
            return {}}}}}}}}}}}}}}}}}
            "model_name": model_name,
            "hardware": hardware,
            "throughput": None,
            "latency": None,
            "memory": None,
            "power": None
            }
            
            # Convert to dictionary
            metrics = result.to_dict()'records')[0]
            ,,,
            # Round values
            for key in ['throughput', 'latency', 'memory', 'power']:,
            if metrics[key] is not None:,
            metrics[key] = round()metrics[key], 2)
            ,
            logger.debug()f"Retrieved performance metrics for {}}}}}}}}}}}}}}}}}model_name} on {}}}}}}}}}}}}}}}}}hardware}")
            return metrics
        
        except Exception as e:
            logger.error()f"Error retrieving performance metrics: {}}}}}}}}}}}}}}}}}e}")
            return {}}}}}}}}}}}}}}}}}
            "model_name": model_name,
            "hardware": hardware,
            "throughput": None,
            "latency": None,
            "memory": None,
            "power": None,
            "error": str()e)
            }

            def get_recommendations()self,
            model_name: str) -> Dict[str, Any]:,,,,,
            """
            Get hardware recommendations for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with hardware recommendations
            """
        try:
            # Get model modality
            query = "SELECT modality FROM models WHERE model_name = ?"
            result = self.conn.execute()query, [model_name]).fetchdf())
            ,,
            if result.empty:
                logger.warning()f"Model {}}}}}}}}}}}}}}}}}model_name} not found in database")
            return {}}}}}}}}}}}}}}}}}
            "model_name": model_name,
            "best_platform": None,
            "alternatives": [],,,,,
            "summary": "Model not found in database"
            }
            
            modality = result.iloc[0]['modality']
            ,
            # Get recommendations for this modality
            query = """
            SELECT 
            recommended_hardware as best_platform,
            recommendation_details
            FROM 
            hardware_recommendations
            WHERE 
            modality = ?
            """
            
            result = self.conn.execute()query, [modality]).fetchdf())
            ,
            if result.empty:
                logger.warning()f"No hardware recommendations found for modality: {}}}}}}}}}}}}}}}}}modality}")
            return {}}}}}}}}}}}}}}}}}
            "model_name": model_name,
            "modality": modality,
            "best_platform": None,
            "alternatives": [],,,,,
            "summary": f"No recommendations available for {}}}}}}}}}}}}}}}}}modality} models"
            }
            
            # Parse recommendation details
            recommendations = {}}}}}}}}}}}}}}}}}
            "model_name": model_name,
            "modality": modality,
            "best_platform": result.iloc[0]['best_platform'],
            }
            
            details = json.loads()result.iloc[0]['recommendation_details']),
            recommendations["summary"] = details.get()"summary", "")
            ,
            # Extract alternative platforms from configurations
            alternatives = [],,,,
            for config in details.get()"configurations", [],,,,):
                platform = config.split()":")[0].strip()),
                if platform != recommendations["best_platform"]:,
                alternatives.append()platform)
            
                recommendations["alternatives"] = alternatives,
                recommendations["configurations"] = details.get()"configurations", [],,,,)
            
                logger.debug()f"Retrieved hardware recommendations for {}}}}}}}}}}}}}}}}}model_name}")
            return recommendations
        
        except Exception as e:
            logger.error()f"Error retrieving hardware recommendations: {}}}}}}}}}}}}}}}}}e}")
            return {}}}}}}}}}}}}}}}}}
            "model_name": model_name,
            "best_platform": None,
            "alternatives": [],,,,,
            "summary": f"Error retrieving recommendations: {}}}}}}}}}}}}}}}}}e}"
            }
    
            def get_all_compatibility_data()self,
            model_name: str) -> Dict[str, Any]:,,,,,
            """
            Get comprehensive compatibility and performance data for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with comprehensive compatibility and performance data
            """
        try:
            # Get model information
            query = """
            SELECT 
            model_name,
            model_type,
            model_family,
            modality,
            parameters_million,
            is_key_model
            FROM 
            models
            WHERE
            model_name = ?
            """
            
            model_result = self.conn.execute()query, [model_name]).fetchdf())
            ,,
            if model_result.empty:
                logger.warning()f"Model {}}}}}}}}}}}}}}}}}model_name} not found in database")
            return {}}}}}}}}}}}}}}}}}
            "model_name": model_name,
            "error": "Model not found in database"
            }
            
            model_info = model_result.to_dict()'records')[0]
            ,,,
            # Get all hardware platforms
            platforms = self.get_hardware_platforms())
            
            # Get compatibility for each platform
            compatibility = {}}}}}}}}}}}}}}}}}}
            for platform in platforms:
                hw_type = platform['hardware_type'],,,
                compatibility[hw_type] = self.get_compatibility()model_name, hw_type)
                ,
            # Get performance metrics for each platform
                performance = {}}}}}}}}}}}}}}}}}}
            for platform in platforms:
                hw_type = platform['hardware_type'],,,
                performance[hw_type] = self.get_performance_metrics()model_name, hw_type)
                ,
            # Get recommendations
                recommendations = self.get_recommendations()model_name)
            
            # Combine all data
                result = {}}}}}}}}}}}}}}}}}
                "model_info": model_info,
                "compatibility": compatibility,
                "performance": performance,
                "recommendations": recommendations
                }
            
                logger.debug()f"Retrieved comprehensive data for {}}}}}}}}}}}}}}}}}model_name}")
                return result
        
        except Exception as e:
            logger.error()f"Error retrieving comprehensive data: {}}}}}}}}}}}}}}}}}e}")
                return {}}}}}}}}}}}}}}}}}
                "model_name": model_name,
                "error": f"Error retrieving comprehensive data: {}}}}}}}}}}}}}}}}}e}"
                }
    
                def get_compatibility_matrix()self,
                modality: Optional[str] = None,
                family: Optional[str] = None,
                hardware_platforms: Optional[List[str]] = None,
                key_models_only: bool = True) -> Dict[str, Any]:,,,,,
                """
                Get the compatibility matrix for a subset of models and hardware platforms.

        Args:
            modality: Filter by model modality
            family: Filter by model family
            hardware_platforms: List of hardware platforms to include
            key_models_only: Whether to include only key models

        Returns:
            Dictionary with compatibility matrix data
            """
        try:
            # Get models
            models = self.get_models()modality=modality, family=family, key_models_only=key_models_only)
            
            if not models:
                logger.warning()"No models found matching the criteria")
            return {}}}}}}}}}}}}}}}}}
            "metadata": {}}}}}}}}}}}}}}}}}
            "total_models": 0,
            "hardware_platforms": [],,,,
            },
            "matrix": [],,,,
            }
            
            # Get hardware platforms
            platforms = self.get_hardware_platforms())
            
            if hardware_platforms:
                platforms = [p for p in platforms if p['hardware_type'],,, in hardware_platforms]
            
            # Build matrix
            matrix = [],,,,:
            for model in models:
                model_row = {}}}}}}}}}}}}}}}}}
                "model_name": model['model_name'],
                "model_type": model['model_type'],
                "model_family": model['model_family'],
                "modality": model['modality'],
                "parameters_million": model['parameters_million'],
                }
                
                # Add compatibility for each platform
                for platform in platforms:
                    hw_type = platform['hardware_type'],,,
                    compatibility = self.get_compatibility()model['model_name'], hw_type),
                    model_row[hw_type] = compatibility['symbol'],
                    model_row[f"{}}}}}}}}}}}}}}}}}hw_type}_level"] = compatibility['level'],
                    model_row[f"{}}}}}}}}}}}}}}}}}hw_type}_notes"] = compatibility['notes']
                    ,
                    matrix.append()model_row)
            
                    result = {}}}}}}}}}}}}}}}}}
                    "metadata": {}}}}}}}}}}}}}}}}}
                    "total_models": len()models),
                    "hardware_platforms": [p['hardware_type'],,, for p in platforms]:
                        },
                        "matrix": matrix
                        }
            
                        logger.debug()f"Generated compatibility matrix with {}}}}}}}}}}}}}}}}}len()models)} models and {}}}}}}}}}}}}}}}}}len()platforms)} platforms")
                    return result
        
        except Exception as e:
            logger.error()f"Error generating compatibility matrix: {}}}}}}}}}}}}}}}}}e}")
                    return {}}}}}}}}}}}}}}}}}
                    "error": f"Error generating compatibility matrix: {}}}}}}}}}}}}}}}}}e}"
                    }

    def close()self) -> None:
        """
        Close the database connection.
        """
        if self.conn:
            self.conn.close())
            logger.debug()"Closed database connection")

    def __del__()self) -> None:
        """
        Close the database connection when the object is deleted.
        """
        self.close())


# Example usage
if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser()description="Compatibility Matrix API Example")
    parser.add_argument()"--db-path", default="./benchmark_db.duckdb", help="Path to DuckDB database")
    parser.add_argument()"--model", default="bert-base-uncased", help="Model name")
    parser.add_argument()"--hardware", default="CUDA", help="Hardware platform")
    args = parser.parse_args())
    
    # Initialize API
    api = MatrixAPI()db_path=args.db_path)
    
    # Get compatibility
    compatibility = api.get_compatibility()args.model, args.hardware)
    print()f"\nCompatibility for {}}}}}}}}}}}}}}}}}args.model} on {}}}}}}}}}}}}}}}}}args.hardware}:")
    print()f"Level: {}}}}}}}}}}}}}}}}}compatibility['level']}"),
    print()f"Symbol: {}}}}}}}}}}}}}}}}}compatibility['symbol']}"),
    print()f"Notes: {}}}}}}}}}}}}}}}}}compatibility['notes']}")
    ,
    # Get performance metrics
    metrics = api.get_performance_metrics()args.model, args.hardware)
    print()f"\nPerformance metrics for {}}}}}}}}}}}}}}}}}args.model} on {}}}}}}}}}}}}}}}}}args.hardware}:")
    print()f"Throughput: {}}}}}}}}}}}}}}}}}metrics['throughput']} items/sec"),,
    print()f"Latency: {}}}}}}}}}}}}}}}}}metrics['latency']} ms"),,
    print()f"Memory: {}}}}}}}}}}}}}}}}}metrics['memory']} MB"),
    ,    print()f"Power: {}}}}}}}}}}}}}}}}}metrics['power']} W")
    ,
    # Get recommendations
    recommendations = api.get_recommendations()args.model)
    print()f"\nHardware recommendations for {}}}}}}}}}}}}}}}}}args.model}:")
    print()f"Best platform: {}}}}}}}}}}}}}}}}}recommendations['best_platform']}"),,
    print()f"Alternatives: {}}}}}}}}}}}}}}}}}', '.join()recommendations['alternatives'])}"),
    print()f"Summary: {}}}}}}}}}}}}}}}}}recommendations['summary']}"),
    print()f"Configurations:")
    for config in recommendations.get()'configurations', [],,,,):
        print()f"  - {}}}}}}}}}}}}}}}}}config}")
    
    # Close connection
        api.close())