#!/usr/bin/env python3

"""
Vision-Text DuckDB Integration

This script integrates vision-text model (CLIP, BLIP) test results into the DuckDB database
for tracking model performance and compatibility across hardware platforms.

Features:
1. Loads test results from JSON files
2. Processes results into a structured format
3. Stores results in the DuckDB database
4. Generates model compatibility matrix
5. Supports both CLIP and BLIP model families

Usage:
  python vision_text_duckdb_integration.py --import-results [directory]
  python vision_text_duckdb_integration.py --generate-matrix
  python vision_text_duckdb_integration.py --list-models
"""

import os
import sys
import json
import glob
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"vision_text_db_integration_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Path constants
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = ROOT_DIR / "skills"
FIXED_TESTS_DIR = SKILLS_DIR / "fixed_tests"
COLLECTED_RESULTS_DIR = FIXED_TESTS_DIR / "collected_results"
DB_PATH = ROOT_DIR / "benchmark_db.duckdb"
REPORTS_DIR = ROOT_DIR / "reports"

# Ensure reports directory exists
REPORTS_DIR.mkdir(exist_ok=True)

# Hardware platforms mapping
HARDWARE_PLATFORMS = {
    "cpu": "CPU",
    "cuda": "CUDA (NVIDIA GPU)",
    "openvino": "OpenVINO (Intel)",
    "rocm": "ROCm (AMD GPU)",
    "mps": "MPS (Apple Silicon)",
    "webnn": "WebNN",
    "webgpu": "WebGPU"
}

# Vision-Text model families
VISION_TEXT_FAMILIES = {
    "clip": {
        "description": "CLIP (Contrastive Language-Image Pre-Training)",
        "architecture_type": "vision_text",
        "model_type": "clip",
        "task": "zero-shot-image-classification",
        "popular_models": [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        ]
    },
    "blip": {
        "description": "BLIP (Bootstrapping Language-Image Pre-training)",
        "architecture_type": "vision_text",
        "model_type": "blip",
        "task": "image-to-text",
        "popular_models": [
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip-image-captioning-large",
            "Salesforce/blip-vqa-base"
        ]
    }
}

def connect_to_db() -> Any:
    """Connect to DuckDB database."""
    try:
        import duckdb
        conn = duckdb.connect(str(DB_PATH))
        return conn
    except ImportError:
        logger.error("DuckDB not installed. Please install with: pip install duckdb")
        return None
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def create_tables(conn) -> bool:
    """Create necessary tables if they don't exist."""
    try:
        # Check if vision_text_results table exists
        result = conn.execute("""
        SELECT name FROM sqlite_master WHERE name='vision_text_results'
        """).fetchall()
        
        if not result:
            # Create vision_text_results table
            conn.execute("""
            CREATE TABLE vision_text_results (
                id VARCHAR PRIMARY KEY,
                model_id VARCHAR NOT NULL,
                model_type VARCHAR NOT NULL,  -- 'clip' or 'blip'
                task VARCHAR NOT NULL,
                hardware_platform VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                success BOOLEAN NOT NULL,
                error_type VARCHAR,
                avg_inference_time DOUBLE,
                memory_usage_mb DOUBLE,
                results JSON,
                metadata JSON
            )
            """)
            logger.info("Created vision_text_results table")
        
        # Check if model_compatibility table exists
        result = conn.execute("""
        SELECT name FROM sqlite_master WHERE name='model_compatibility'
        """).fetchall()
        
        if not result:
            # Create model_compatibility table
            conn.execute("""
            CREATE TABLE model_compatibility (
                model_id VARCHAR PRIMARY KEY,
                model_family VARCHAR NOT NULL,
                model_type VARCHAR NOT NULL,
                architecture_type VARCHAR NOT NULL,
                task VARCHAR NOT NULL,
                cpu BOOLEAN DEFAULT FALSE,
                cuda BOOLEAN DEFAULT FALSE,
                openvino BOOLEAN DEFAULT FALSE,
                rocm BOOLEAN DEFAULT FALSE,
                mps BOOLEAN DEFAULT FALSE,
                webnn BOOLEAN DEFAULT FALSE,
                webgpu BOOLEAN DEFAULT FALSE,
                last_tested TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            logger.info("Created model_compatibility table")
            
        return True
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False

def import_result_file(conn, file_path: str) -> bool:
    """Import a single result file into the database."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract metadata
        metadata = data.get("metadata", {})
        model_id = metadata.get("model", "unknown")
        model_type = metadata.get("model_type", "unknown")
        task = metadata.get("task", "unknown")
        timestamp = metadata.get("timestamp", datetime.datetime.now().isoformat())
        
        # Process results
        results = data.get("results", {})
        success = False
        error_type = None
        avg_inference_time = None
        memory_usage_mb = None
        
        # Look for success in any of the test methods
        for test_name, test_result in results.items():
            if test_result.get("pipeline_success", False):
                success = True
                avg_inference_time = test_result.get("pipeline_avg_time")
                break
            elif test_result.get("from_pretrained_success", False):
                success = True
                avg_inference_time = test_result.get("from_pretrained_avg_time")
                break
            elif test_result.get("openvino_success", False):
                success = True
                avg_inference_time = test_result.get("openvino_inference_time")
                break
        
        # If no success found, get the error type from the first test
        if not success and results:
            first_test = next(iter(results.values()))
            error_type = first_test.get("pipeline_error_type") or first_test.get("from_pretrained_error_type") or first_test.get("openvino_error_type")
        
        # Extract hardware platform from the filename
        hardware_platform = "cpu"  # Default
        if "cuda" in file_path:
            hardware_platform = "cuda"
        elif "openvino" in file_path:
            hardware_platform = "openvino"
        
        # Generate a unique ID with microseconds to prevent duplicates
        import_id = f"{model_id.replace('/', '_')}_{model_type}_{hardware_platform}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Insert into database
        conn.execute("""
        INSERT INTO vision_text_results 
        (id, model_id, model_type, task, hardware_platform, timestamp, success, error_type, 
         avg_inference_time, memory_usage_mb, results, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            import_id, model_id, model_type, task, hardware_platform, timestamp, success, error_type,
            avg_inference_time, memory_usage_mb, json.dumps(results), json.dumps(metadata)
        ])
        
        # Update model_compatibility table
        # First, check if model exists
        existing = conn.execute("""
        SELECT model_id FROM model_compatibility WHERE model_id = ?
        """, [model_id]).fetchall()
        
        if existing:
            # Update existing record using a dynamic update based on hardware platform
            if hardware_platform in ["cpu", "cuda", "openvino", "rocm", "mps", "webnn", "webgpu"]:
                # Create a parameterized SQL statement that updates only the specific hardware column
                update_sql = f"""
                UPDATE model_compatibility 
                SET {hardware_platform} = ?, last_tested = ?, last_updated = CURRENT_TIMESTAMP
                WHERE model_id = ?
                """
                conn.execute(update_sql, [success, timestamp, model_id])
            else:
                logger.warning(f"Unknown hardware platform: {hardware_platform}, not updating database")
        else:
            # Insert new record
            # Determine model family
            model_family = "unknown"
            architecture_type = "vision_text"
            
            if "clip" in model_id.lower() or model_type == "clip":
                model_family = "clip"
            elif "blip" in model_id.lower() or model_type == "blip":
                model_family = "blip"
            
            # Initialize all hardware flags to False
            hardware_flags = {
                "cpu": False,
                "cuda": False,
                "openvino": False,
                "rocm": False,
                "mps": False,
                "webnn": False,
                "webgpu": False
            }
            
            # Set the current hardware platform flag based on the test result
            if hardware_platform in hardware_flags:
                hardware_flags[hardware_platform] = success
            
            conn.execute("""
            INSERT INTO model_compatibility
            (model_id, model_family, model_type, architecture_type, task, 
             cpu, cuda, openvino, rocm, mps, webnn, webgpu, last_tested)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                model_id, model_family, model_type, architecture_type, task,
                hardware_flags["cpu"], hardware_flags["cuda"], hardware_flags["openvino"],
                hardware_flags["rocm"], hardware_flags["mps"], hardware_flags["webnn"], 
                hardware_flags["webgpu"], timestamp
            ])
        
        logger.info(f"Imported {file_path} for {model_id} on {hardware_platform}")
        return True
    except Exception as e:
        logger.error(f"Error importing {file_path}: {e}")
        return False

def import_results(conn, directory: str) -> Dict:
    """Import all results from a directory into the database."""
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return {"success": False, "imported": 0, "failed": 0}
    
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    # Filter for vision-text models (CLIP and BLIP)
    vision_text_files = [f for f in json_files if "clip" in f.lower() or "blip" in f.lower()]
    
    logger.info(f"Found {len(vision_text_files)} vision-text result files in {directory}")
    
    imported = 0
    failed = 0
    
    for file_path in vision_text_files:
        if import_result_file(conn, file_path):
            imported += 1
        else:
            failed += 1
    
    return {"success": True, "imported": imported, "failed": failed}

def generate_compatibility_matrix(conn) -> Dict:
    """Generate a compatibility matrix for vision-text models."""
    try:
        # Check which columns exist in the model_compatibility table
        column_info = conn.execute("""
        SELECT name FROM pragma_table_info('model_compatibility')
        """).fetchall()
        column_names = [row[0] for row in column_info]
        
        # Build the query dynamically based on available columns
        base_query = """
        SELECT 
            model_id, 
            model_family, 
            model_type, 
            task, 
            cpu, 
            cuda, 
            openvino"""
        
        # Add optional columns if they exist
        for col in ['rocm', 'mps', 'webnn', 'webgpu']:
            if col in column_names:
                base_query += f",\n            {col}"
            else:
                base_query += f",\n            FALSE as {col}"
                
        base_query += ",\n            last_tested, last_updated"
        
        # Complete the query
        full_query = base_query + """
        FROM model_compatibility
        WHERE model_type IN ('clip', 'blip')
        ORDER BY model_family, model_id
        """
        
        # Execute the query
        results = conn.execute(full_query).fetchall()
        
        if not results:
            logger.warning("No vision-text models found in the database")
            return {"success": False, "message": "No vision-text models found"}
        
        # Create matrix dictionary
        matrix = {
            "timestamp": datetime.datetime.now().isoformat(),
            "models": {}
        }
        
        # Organize by model family
        for row in results:
            model_id = row[0]
            model_family = row[1]
            model_type = row[2]
            task = row[3]
            cpu_compat = row[4]
            cuda_compat = row[5]
            openvino_compat = row[6]
            rocm_compat = row[7]
            mps_compat = row[8]
            webnn_compat = row[9]
            webgpu_compat = row[10]
            last_tested = row[11]
            
            # Convert datetime to string if needed
            if isinstance(last_tested, datetime.datetime):
                last_tested_str = last_tested.isoformat()
            else:
                last_tested_str = last_tested
                
            matrix["models"][model_id] = {
                "model_id": model_id,
                "family": model_family,
                "type": model_type,
                "task": task,
                "compatibility": {
                    "cpu": cpu_compat,
                    "cuda": cuda_compat,
                    "openvino": openvino_compat,
                    "rocm": rocm_compat,
                    "mps": mps_compat,
                    "webnn": webnn_compat,
                    "webgpu": webgpu_compat
                },
                "last_tested": last_tested_str
            }
        
        # Save matrix to JSON file
        json_file = REPORTS_DIR / f"vision_text_compatibility_matrix_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, "w") as f:
            json.dump(matrix, f, indent=2)
        
        # Save matrix to Markdown file
        md_file = REPORTS_DIR / f"vision_text_compatibility_matrix_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(md_file, "w") as f:
            f.write("# Vision-Text Model Compatibility Matrix\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Header
            f.write("## Model Compatibility\n\n")
            
            # Group by model type
            for model_type in ["clip", "blip"]:
                # Skip if no models of this type
                type_models = {k: v for k, v in matrix["models"].items() if v["type"] == model_type}
                if not type_models:
                    continue
                
                f.write(f"### {model_type.upper()} Models\n\n")
                
                # Table header with all hardware platforms
                f.write("| Model | Task | CPU | CUDA | OpenVINO | ROCm | MPS | WebNN | WebGPU | Last Tested |\n")
                f.write("|-------|------|-----|------|----------|------|-----|-------|--------|-------------|\n")
                
                # Table rows
                for model_id, info in sorted(type_models.items()):
                    task = info["task"]
                    cpu = "✅" if info["compatibility"]["cpu"] else "❌"
                    cuda = "✅" if info["compatibility"]["cuda"] else "❌"
                    openvino = "✅" if info["compatibility"]["openvino"] else "❌"
                    rocm = "✅" if info["compatibility"]["rocm"] else "❌"
                    mps = "✅" if info["compatibility"]["mps"] else "❌"
                    webnn = "✅" if info["compatibility"]["webnn"] else "❌"
                    webgpu = "✅" if info["compatibility"]["webgpu"] else "❌"
                    
                    last_tested = "Unknown"
                    if info["last_tested"]:
                        try:
                            last_tested = datetime.datetime.fromisoformat(info["last_tested"]).strftime("%Y-%m-%d")
                        except:
                            pass
                    
                    f.write(f"| {model_id} | {task} | {cpu} | {cuda} | {openvino} | {rocm} | {mps} | {webnn} | {webgpu} | {last_tested} |\n")
                
                f.write("\n")
            
            # Summary
            f.write("## Summary\n\n")
            
            clip_count = len([v for v in matrix["models"].values() if v["type"] == "clip"])
            blip_count = len([v for v in matrix["models"].values() if v["type"] == "blip"])
            
            f.write(f"- Total models: {len(matrix['models'])}\n")
            f.write(f"- CLIP models: {clip_count}\n")
            f.write(f"- BLIP models: {blip_count}\n\n")
            
            # Hardware compatibility summary
            cpu_compat = len([v for v in matrix["models"].values() if v["compatibility"]["cpu"]])
            cuda_compat = len([v for v in matrix["models"].values() if v["compatibility"]["cuda"]])
            openvino_compat = len([v for v in matrix["models"].values() if v["compatibility"]["openvino"]])
            rocm_compat = len([v for v in matrix["models"].values() if v["compatibility"]["rocm"]])
            mps_compat = len([v for v in matrix["models"].values() if v["compatibility"]["mps"]])
            webnn_compat = len([v for v in matrix["models"].values() if v["compatibility"]["webnn"]])
            webgpu_compat = len([v for v in matrix["models"].values() if v["compatibility"]["webgpu"]])
            
            total_models = len(matrix["models"])
            
            f.write("### Hardware Compatibility\n\n")
            f.write(f"- CPU: {cpu_compat}/{total_models} models ({cpu_compat/total_models*100 if total_models > 0 else 0:.1f}%)\n")
            f.write(f"- CUDA: {cuda_compat}/{total_models} models ({cuda_compat/total_models*100 if total_models > 0 else 0:.1f}%)\n")
            f.write(f"- OpenVINO: {openvino_compat}/{total_models} models ({openvino_compat/total_models*100 if total_models > 0 else 0:.1f}%)\n")
            f.write(f"- ROCm: {rocm_compat}/{total_models} models ({rocm_compat/total_models*100 if total_models > 0 else 0:.1f}%)\n")
            f.write(f"- MPS: {mps_compat}/{total_models} models ({mps_compat/total_models*100 if total_models > 0 else 0:.1f}%)\n")
            f.write(f"- WebNN: {webnn_compat}/{total_models} models ({webnn_compat/total_models*100 if total_models > 0 else 0:.1f}%)\n")
            f.write(f"- WebGPU: {webgpu_compat}/{total_models} models ({webgpu_compat/total_models*100 if total_models > 0 else 0:.1f}%)\n")
        
        logger.info(f"Compatibility matrix saved to {json_file} and {md_file}")
        
        return {
            "success": True,
            "json_file": str(json_file),
            "md_file": str(md_file),
            "model_count": len(matrix["models"])
        }
    except Exception as e:
        logger.error(f"Error generating compatibility matrix: {e}")
        return {"success": False, "message": str(e)}

def generate_performance_report(conn) -> Dict:
    """Generate a performance report for vision-text models."""
    try:
        # Query performance statistics from the database with safer join handling
        try:
            # First check if there's data in vision_text_results
            count = conn.execute("SELECT COUNT(*) FROM vision_text_results WHERE success = TRUE").fetchone()[0]
            
            if count == 0:
                logger.warning("No successful test results found in the database")
                return {"success": False, "message": "No successful test results found"}
            
            # Use a safer query that checks for JOIN matches first
            results = conn.execute("""
            SELECT 
                vr.model_id, 
                vr.model_type,
                vr.hardware_platform, 
                AVG(vr.avg_inference_time) as avg_time,
                COUNT(*) as test_count,
                COALESCE(mc.task, vr.task) as task
            FROM vision_text_results vr
            LEFT JOIN model_compatibility mc ON vr.model_id = mc.model_id
            WHERE vr.success = TRUE
            GROUP BY vr.model_id, vr.model_type, vr.hardware_platform, COALESCE(mc.task, vr.task)
            ORDER BY vr.model_type, avg_time
            """).fetchall()
        except Exception as e:
            logger.error(f"Database query error: {e}")
            # Fallback to a simpler query without the join
            results = conn.execute("""
            SELECT 
                model_id, 
                model_type,
                hardware_platform, 
                AVG(avg_inference_time) as avg_time,
                COUNT(*) as test_count,
                task
            FROM vision_text_results
            WHERE success = TRUE
            GROUP BY model_id, model_type, hardware_platform, task
            ORDER BY model_type, avg_time
            """).fetchall()
        
        if not results:
            logger.warning("No performance data found in the database")
            return {"success": False, "message": "No performance data found"}
        
        # Prepare report data
        report_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "models": {}
        }
        
        # Process results
        for row in results:
            model_id = row[0]
            model_type = row[1]
            hardware = row[2]
            avg_time = row[3]
            test_count = row[4]
            task = row[5]
            
            if model_id not in report_data["models"]:
                report_data["models"][model_id] = {
                    "model_id": model_id,
                    "type": model_type,
                    "task": task,
                    "performance": {}
                }
            
            report_data["models"][model_id]["performance"][hardware] = {
                "avg_inference_time": avg_time,
                "test_count": test_count
            }
        
        # Generate CSV file for visualization
        csv_file = REPORTS_DIR / f"vision_text_performance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_file, "w") as f:
            # Header
            f.write("Model ID,Model Type,Task,Hardware,Average Inference Time (s),Tests\n")
            
            # Data rows
            for model_id, info in report_data["models"].items():
                model_type = info["type"]
                task = info["task"]
                
                for hardware, perf in info["performance"].items():
                    avg_time = perf["avg_inference_time"]
                    test_count = perf["test_count"]
                    
                    f.write(f"{model_id},{model_type},{task},{hardware},{avg_time},{test_count}\n")
        
        # Generate markdown report
        md_file = REPORTS_DIR / f"vision_text_performance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(md_file, "w") as f:
            f.write("# Vision-Text Model Performance Report\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group by model type
            for model_type in ["clip", "blip"]:
                models_of_type = {k: v for k, v in report_data["models"].items() if v["type"] == model_type}
                if not models_of_type:
                    continue
                
                f.write(f"## {model_type.upper()} Models\n\n")
                
                # Table header
                f.write("| Model | Task | Hardware | Avg Inference Time (s) | Tests |\n")
                f.write("|-------|------|----------|------------------------|-------|\n")
                
                # Table rows
                for model_id, info in sorted(models_of_type.items()):
                    task = info["task"]
                    
                    # First row for this model
                    first_row = True
                    
                    for hardware, perf in sorted(info["performance"].items()):
                        avg_time = perf["avg_inference_time"]
                        test_count = perf["test_count"]
                        
                        if first_row:
                            f.write(f"| {model_id} | {task} | {hardware} | {avg_time:.4f} | {test_count} |\n")
                            first_row = False
                        else:
                            f.write(f"| | | {hardware} | {avg_time:.4f} | {test_count} |\n")
                
                f.write("\n")
            
            # Performance summary
            f.write("## Performance Summary\n\n")
            
            # Calculate average performance by hardware
            hardware_perf = {}
            for model_info in report_data["models"].values():
                for hw, perf in model_info["performance"].items():
                    if hw not in hardware_perf:
                        hardware_perf[hw] = {"times": [], "total_tests": 0}
                    
                    hardware_perf[hw]["times"].append(perf["avg_inference_time"])
                    hardware_perf[hw]["total_tests"] += perf["test_count"]
            
            # Create performance table
            f.write("| Hardware | Avg Inference Time (s) | Total Tests |\n")
            f.write("|----------|------------------------|-------------|\n")
            
            for hw, perf in sorted(hardware_perf.items()):
                avg = sum(perf["times"]) / len(perf["times"]) if perf["times"] else 0
                f.write(f"| {hw} | {avg:.4f} | {perf['total_tests']} |\n")
        
        logger.info(f"Performance report saved to {csv_file} and {md_file}")
        
        return {
            "success": True,
            "csv_file": str(csv_file),
            "md_file": str(md_file)
        }
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        return {"success": False, "message": str(e)}

def list_models(conn) -> Dict:
    """List all vision-text models in the database."""
    try:
        # Query the database for all vision-text models with safer column selection
        results = conn.execute("""
        SELECT 
            model_id, 
            model_family, 
            model_type, 
            task,
            (cpu OR cuda OR openvino) AS has_compatibility,
            last_tested
        FROM model_compatibility
        WHERE model_type IN ('clip', 'blip')
        ORDER BY model_family, model_id
        """).fetchall()
        
        if not results:
            logger.warning("No vision-text models found in the database")
            print("No vision-text models found in the database")
            return {"success": False, "message": "No vision-text models found"}
        
        # Print results
        print("\nVision-Text Models in Database:\n")
        
        print(f"{'Model ID':<40} {'Type':<8} {'Task':<30} {'Hardware Tested':<15} {'Last Tested'}")
        print("-" * 100)
        
        for row in results:
            model_id = row[0]
            model_type = row[2].upper()
            task = row[3]
            has_compat = "Yes" if row[4] else "No"
            
            last_tested = "Unknown"
            if row[5]:
                try:
                    last_tested = datetime.datetime.fromisoformat(row[5]).strftime("%Y-%m-%d")
                except:
                    pass
            
            print(f"{model_id:<40} {model_type:<8} {task[:30]:<30} {has_compat:<15} {last_tested}")
        
        # Get summary counts
        clip_count = len([r for r in results if r[2] == "clip"])
        blip_count = len([r for r in results if r[2] == "blip"])
        
        print("\nSummary:")
        print(f"- Total models: {len(results)}")
        print(f"- CLIP models: {clip_count}")
        print(f"- BLIP models: {blip_count}")
        
        return {
            "success": True,
            "total": len(results),
            "clip_count": clip_count,
            "blip_count": blip_count
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        print(f"Error: {e}")
        return {"success": False, "message": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Vision-Text DuckDB Integration")
    
    # Action options
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--import-results", type=str, metavar="DIRECTORY",
                            help="Import test results from a directory")
    action_group.add_argument("--generate-matrix", action="store_true",
                            help="Generate compatibility matrix for vision-text models")
    action_group.add_argument("--list-models", action="store_true",
                            help="List all vision-text models in the database")
    action_group.add_argument("--performance-report", action="store_true",
                            help="Generate performance report for vision-text models")
    
    args = parser.parse_args()
    
    # Connect to database
    conn = connect_to_db()
    if not conn:
        return 1
    
    # Create tables if needed
    if not create_tables(conn):
        return 1
    
    if args.import_results:
        # Import results
        result = import_results(conn, args.import_results)
        print(f"\nImported {result['imported']} files, {result['failed']} failed")
    
    elif args.generate_matrix:
        # Generate compatibility matrix
        result = generate_compatibility_matrix(conn)
        if result["success"]:
            print(f"\nGenerated compatibility matrix with {result['model_count']} models:")
            print(f"- JSON: {result['json_file']}")
            print(f"- Markdown: {result['md_file']}")
        else:
            print(f"\nError generating matrix: {result['message']}")
    
    elif args.list_models:
        # List models
        list_models(conn)
    
    elif args.performance_report:
        # Generate performance report
        result = generate_performance_report(conn)
        if result["success"]:
            print(f"\nGenerated performance report:")
            print(f"- CSV: {result['csv_file']}")
            print(f"- Markdown: {result['md_file']}")
        else:
            print(f"\nError generating performance report: {result['message']}")
    
    # Close database connection
    conn.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())