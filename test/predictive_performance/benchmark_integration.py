#!/usr/bin/env python3
"""
Benchmark Integration module for the Predictive Performance System.

This module provides utilities to integrate prediction recommendations
with the benchmark execution system, allowing for automated testing of
high-value configurations.
"""

import os
import sys
import json
import logging
import subprocess
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("predictive_performance.benchmark_integration")

class BenchmarkScheduler:
    """
    Scheduler for running benchmark jobs based on active learning recommendations.
    
    This class takes high-value configurations identified by the ActiveLearningSystem
    and converts them into executable benchmark commands.
    """
    
    def __init__(self, db_path: Optional[str] = None, benchmark_script: Optional[str] = None):
        """
        Initialize the benchmark scheduler.
        
        Args:
            db_path: Path to the benchmark database
            benchmark_script: Path to the benchmark script
        """
        self.db_path = db_path or "./benchmark_db.duckdb"
        self.benchmark_script = benchmark_script or "run_benchmark_with_db.py"
        
        # Track executed benchmark jobs
        self.executed_jobs = []
        self.job_results = []
        
    def load_recommendations(self, recommendations_file: str) -> List[Dict[str, Any]]:
        """
        Load recommendations from a JSON file.
        
        Args:
            recommendations_file: Path to JSON file containing recommendations
            
        Returns:
            List of configuration dictionaries
        """
        if not os.path.exists(recommendations_file):
            logger.error(f"Recommendations file {recommendations_file} not found")
            return []
            
        try:
            with open(recommendations_file, 'r') as f:
                recommendations = json.load(f)
                
            logger.info(f"Loaded {len(recommendations)} recommendations from {recommendations_file}")
            return recommendations
        except Exception as e:
            logger.error(f"Failed to load recommendations: {e}")
            return []
    
    def generate_benchmark_commands(self, configurations: List[Dict[str, Any]]) -> List[str]:
        """
        Generate benchmark commands for the given configurations.
        
        Args:
            configurations: List of configuration dictionaries
            
        Returns:
            List of command strings to execute
        """
        commands = []
        
        for config in configurations:
            # Extract configuration parameters
            model_name = config.get("model_name")
            hardware = config.get("hardware")
            batch_size = config.get("batch_size")
            
            if not model_name or not hardware or not batch_size:
                logger.warning(f"Skipping invalid configuration: {config}")
                continue
                
            # Create command
            command = f"python {self.benchmark_script} --model {model_name} --hardware {hardware} --batch-size {batch_size} --db-path {self.db_path}"
            
            # Add additional parameters if available
            if "precision" in config:
                command += f" --precision {config['precision']}"
                
            if "sequence_length" in config:
                command += f" --sequence-length {config['sequence_length']}"
                
            commands.append(command)
            
        logger.info(f"Generated {len(commands)} benchmark commands")
        return commands
        
    def schedule_benchmarks(self, configurations: List[Dict[str, Any]], 
                          execute: bool = False, 
                          parallel: bool = False,
                          max_parallel: int = 2) -> List[str]:
        """
        Schedule benchmark jobs for the given configurations.
        
        Args:
            configurations: List of configuration dictionaries
            execute: Whether to actually execute the commands (default: False)
            parallel: Whether to execute commands in parallel (default: False)
            max_parallel: Maximum number of parallel jobs (default: 2)
            
        Returns:
            List of command strings that were scheduled
        """
        commands = self.generate_benchmark_commands(configurations)
        
        if execute:
            if parallel and len(commands) > 1:
                self._execute_parallel(commands, max_parallel)
            else:
                self._execute_sequential(commands)
                    
        return commands
    
    def _execute_sequential(self, commands: List[str]) -> None:
        """Execute commands sequentially."""
        for i, command in enumerate(commands):
            logger.info(f"Executing command {i+1}/{len(commands)}: {command}")
            try:
                start_time = datetime.now()
                
                # Execute command
                result = subprocess.run(command, shell=True, check=True, 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)
                
                # Record job info
                job_info = {
                    "command": command,
                    "exit_code": result.returncode,
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
                # Parse benchmark results if possible
                try:
                    # Extract model, hardware, batch_size from command
                    import re
                    model_match = re.search(r'--model\s+(\S+)', command)
                    hardware_match = re.search(r'--hardware\s+(\S+)', command)
                    batch_size_match = re.search(r'--batch-size\s+(\d+)', command)
                    
                    if model_match and hardware_match and batch_size_match:
                        model = model_match.group(1)
                        hardware = hardware_match.group(1)
                        batch_size = int(batch_size_match.group(1))
                        
                        # Parse stdout for results
                        # This is just a simple example - actual parsing would depend on
                        # the output format of your benchmark script
                        throughput_match = re.search(r'Throughput:\s+([\d\.]+)', result.stdout)
                        latency_match = re.search(r'Latency:\s+([\d\.]+)', result.stdout)
                        memory_match = re.search(r'Memory:\s+([\d\.]+)', result.stdout)
                        
                        # Create result dict
                        benchmark_results = {
                            "model_name": model,
                            "hardware": hardware,
                            "batch_size": batch_size,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Add metrics if found
                        if throughput_match:
                            benchmark_results["throughput"] = float(throughput_match.group(1))
                        if latency_match:
                            benchmark_results["latency"] = float(latency_match.group(1))
                        if memory_match:
                            benchmark_results["memory"] = float(memory_match.group(1))
                            
                        self.job_results.append(benchmark_results)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse benchmark results: {e}")
                
                self.executed_jobs.append(job_info)
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}: {e}")
                
                # Record failed job
                job_info = {
                    "command": command,
                    "exit_code": e.returncode,
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "stdout": e.stdout if hasattr(e, 'stdout') else "",
                    "stderr": e.stderr if hasattr(e, 'stderr') else "",
                    "error": str(e)
                }
                self.executed_jobs.append(job_info)
                
            except Exception as e:
                logger.error(f"Error executing command: {e}")
    
    def _execute_parallel(self, commands: List[str], max_parallel: int) -> None:
        """Execute commands in parallel using subprocess."""
        import concurrent.futures
        
        logger.info(f"Executing {len(commands)} commands in parallel (max {max_parallel} at once)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for command in commands:
                # Start the command
                logger.info(f"Scheduling command: {command}")
                future = executor.submit(self._execute_command, command)
                futures[future] = command
            
            # Wait for commands to complete
            for future in concurrent.futures.as_completed(futures):
                command = futures[future]
                try:
                    job_info = future.result()
                    logger.info(f"Completed command: {command}")
                    self.executed_jobs.append(job_info)
                    
                    # Add benchmark results if available
                    if "benchmark_results" in job_info and job_info["benchmark_results"]:
                        self.job_results.append(job_info["benchmark_results"])
                        
                except Exception as e:
                    logger.error(f"Command failed: {command}, Error: {e}")
    
    def _execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a single command and return job info."""
        start_time = datetime.now()
        job_info = {
            "command": command,
            "start_time": start_time.isoformat(),
            "benchmark_results": None
        }
        
        try:
            # Execute command
            result = subprocess.run(command, shell=True, check=True, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
            
            # Update job info
            job_info.update({
                "exit_code": result.returncode,
                "end_time": datetime.now().isoformat(),
                "stdout": result.stdout,
                "stderr": result.stderr
            })
            
            # Parse benchmark results if possible
            try:
                # Extract model, hardware, batch_size from command
                import re
                model_match = re.search(r'--model\s+(\S+)', command)
                hardware_match = re.search(r'--hardware\s+(\S+)', command)
                batch_size_match = re.search(r'--batch-size\s+(\d+)', command)
                
                if model_match and hardware_match and batch_size_match:
                    model = model_match.group(1)
                    hardware = hardware_match.group(1)
                    batch_size = int(batch_size_match.group(1))
                    
                    # Parse stdout for results
                    throughput_match = re.search(r'Throughput:\s+([\d\.]+)', result.stdout)
                    latency_match = re.search(r'Latency:\s+([\d\.]+)', result.stdout)
                    memory_match = re.search(r'Memory:\s+([\d\.]+)', result.stdout)
                    
                    # Create result dict
                    benchmark_results = {
                        "model_name": model,
                        "hardware": hardware,
                        "batch_size": batch_size,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Add metrics if found
                    if throughput_match:
                        benchmark_results["throughput"] = float(throughput_match.group(1))
                    if latency_match:
                        benchmark_results["latency"] = float(latency_match.group(1))
                    if memory_match:
                        benchmark_results["memory"] = float(memory_match.group(1))
                        
                    job_info["benchmark_results"] = benchmark_results
                    
            except Exception as e:
                logger.warning(f"Failed to parse benchmark results: {e}")
            
        except subprocess.CalledProcessError as e:
            # Record error info
            job_info.update({
                "exit_code": e.returncode,
                "end_time": datetime.now().isoformat(),
                "stdout": e.stdout if hasattr(e, 'stdout') else "",
                "stderr": e.stderr if hasattr(e, 'stderr') else "",
                "error": str(e)
            })
            
        except Exception as e:
            # Record general error
            job_info.update({
                "exit_code": -1,
                "end_time": datetime.now().isoformat(),
                "error": str(e)
            })
        
        return job_info
    
    def get_benchmark_results(self) -> List[Dict[str, Any]]:
        """Get benchmark results from executed jobs."""
        return self.job_results
    
    def save_job_report(self, output_file: str) -> bool:
        """
        Save job execution report to file.
        
        Args:
            output_file: Path to output file
            
        Returns:
            Success flag
        """
        if not self.executed_jobs:
            logger.warning("No jobs executed yet")
            return False
            
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Create report
            report = {
                "timestamp": datetime.now().isoformat(),
                "jobs": self.executed_jobs,
                "job_count": len(self.executed_jobs),
                "success_count": sum(1 for job in self.executed_jobs if job.get("exit_code", -1) == 0),
                "failure_count": sum(1 for job in self.executed_jobs if job.get("exit_code", -1) != 0)
            }
            
            # Save report
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Saved job report to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save job report: {e}")
            return False
            
    def save_benchmark_results(self, output_file: str) -> bool:
        """
        Save benchmark results to file.
        
        Args:
            output_file: Path to output file
            
        Returns:
            Success flag
        """
        if not self.job_results:
            logger.warning("No benchmark results available")
            return False
            
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(self.job_results)
            
            # Save results
            results_df.to_csv(output_file, index=False)
                
            logger.info(f"Saved benchmark results to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
            return False