#!/usr/bin/env python
"""
Implementation of OpenAI Fine-tuning API for ipfs_accelerate_py.
This enhances the existing OpenAI API implementation with fine-tuning capabilities.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()))))))

# Add parent directory to path to import the ipfs_accelerate_py module
sys.path.append())))))os.path.join())))))os.path.dirname())))))os.path.dirname())))))__file__)), 'ipfs_accelerate_py'))

try:
    # Import the OpenAI API implementation to extend it
    from api_backends import openai_api
    import openai
except ImportError as e:
    print())))))f"Failed to import required modules: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
    sys.exit())))))1)

class OpenAIFineTuning:
    """
    Extension to add Fine-tuning capabilities to the ipfs_accelerate_py OpenAI implementation.
    This class provides methods for creating and managing fine-tuned models.
    """
    
    def __init__())))))self, api_key: Optional[]]]]],,,,str] = None, resources: Dict[]]]]],,,,str, Any] = None, metadata: Dict[]]]]],,,,str, Any] = None):,
    """
    Initialize the OpenAI Fine-tuning extension.
        
        Args:
            api_key: OpenAI API key ())))))optional, will use environment variable if not provided):
                resources: Resources dictionary for ipfs_accelerate_py
                metadata: Metadata dictionary for ipfs_accelerate_py
                """
                self.resources = resources if resources else {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                self.metadata = metadata if metadata else {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Get API key from parameters, metadata, or environment variable
                self.api_key = api_key or ())))))metadata.get())))))"openai_api_key") if metadata else None) or os.environ.get())))))"OPENAI_API_KEY")
        :
        if not self.api_key:
            print())))))"Warning: No OpenAI API key provided. Fine-tuning will not work without an API key.")
        
        # Set up the OpenAI API client
            openai.api_key = self.api_key
        
        # Store main OpenAI API implementation for delegation
            self.openai_api_impl = openai_api())))))resources=self.resources, metadata=self.metadata)
        
        # Initialize tracking for rate limiting and backoff
            self.current_requests = 0
            self.max_concurrent_requests = 5
            self.request_queue = []]]]],,,,],,
            self.max_retries = 3
            self.base_wait_time = 1.0  # in seconds
        
            print())))))"OpenAI Fine-tuning extension initialized")
    
    # === File Management for Fine-tuning ===
    
            def prepare_jsonl_from_messages())))))self, messages_list: List[]]]]],,,,List[]]]]],,,,Dict[]]]]],,,,str, Any]]],
            file_path: str) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
            """
            Convert a list of message sequences into a JSONL file for fine-tuning.
        
        Args:
            messages_list: List of message sequences, where each sequence is a list of message dictionaries
            file_path: Path to save the JSONL file
            
        Returns:
            Status of file creation
            """
        try:
            # Convert messages to fine-tuning format
            with open())))))file_path, 'w') as f:
                for messages in messages_list:
                    # Create a fine-tuning example
                    example = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "messages": messages
                    }
                    
                    # Write as JSONL
                    f.write())))))json.dumps())))))example) + '\n')
            
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": True,
                "file_path": file_path,
                "message": f"Created fine-tuning file with {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}len())))))messages_list)} examples."
                }
        
        except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": str())))))e)
                }
    
                def validate_fine_tuning_file())))))self, file_path: str) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
                """
                Validate a JSONL file for fine-tuning, checking for common errors.
        
        Args:
            file_path: Path to the JSONL file to validate
            
        Returns:
            Validation results
            """
        try:
            # Initialize validation results
            results = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "valid": True,
            "examples_count": 0,
            "errors": []]]]],,,,],,,
            "warnings": []]]]],,,,],,
            }
            
            # Read and validate the file
            with open())))))file_path, 'r') as f:
                line_number = 0
                
                for line in f:
                    line_number += 1
                    
                    try:
                        # Check if line is valid JSON
                        example = json.loads())))))line)
                        results[]]]]],,,,"examples_count"] += 1
                        ,
                        # Check if example has the correct structure:
                        if not isinstance())))))example, dict):
                            results[]]]]],,,,"errors"].append()))))),f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}: Example must be a JSON object"),
                            results[]]]]],,,,"valid"] = False,,,,,,,,,
                        continue
                        
                        # Check for messages array
                        if "messages" not in example:
                            results[]]]]],,,,"errors"].append()))))),f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}: Missing 'messages' field"),
                            results[]]]]],,,,"valid"] = False,,,,,,,,,
                        continue
                        
                        if not isinstance())))))example[]]]]],,,,"messages"], list):,
                        results[]]]]],,,,"errors"].append()))))),f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}: 'messages' must be an array"),
                        results[]]]]],,,,"valid"] = False,,,,,,,,,
                    continue
                        
                    if len())))))example[]]]]],,,,"messages"]) < 1:,
                    results[]]]]],,,,"errors"].append()))))),f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}: 'messages' array must have at least one message"),
                    results[]]]]],,,,"valid"] = False,,,,,,,,,
                continue
                        
                        # Check each message
                for i, message in enumerate())))))example[]]]]],,,,"messages"]):,
                            if not isinstance())))))message, dict):
                                results[]]]]],,,,"errors"].append()))))),f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}, message {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}i+1}: Message must be a JSON object"),
                                results[]]]]],,,,"valid"] = False,,,,,,,,,
                continue
                            
                            # Check required fields
                            if "role" not in message:
                                results[]]]]],,,,"errors"].append()))))),f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}, message {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}i+1}: Missing 'role' field"),
                                results[]]]]],,,,"valid"] = False,,,,,,,,,
                continue
                            
                            if "content" not in message:
                                results[]]]]],,,,"errors"].append()))))),f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}, message {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}i+1}: Missing 'content' field"),
                                results[]]]]],,,,"valid"] = False,,,,,,,,,
                continue
                            
                            # Check role values
                if message[]]]]],,,,"role"] not in []]]]],,,,"system", "user", "assistant"]:,
                results[]]]]],,,,"errors"].append()))))),
                f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}, message {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}i+1}: Invalid 'role' value ())))))must be 'system', 'user', or 'assistant')"
                )
                results[]]]]],,,,"valid"] = False,,,,,,,,,
            continue
                            
                            # Check if content is a string:
            if not isinstance())))))message[]]]]],,,,"content"], str):,
            results[]]]]],,,,"errors"].append()))))),f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}, message {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}i+1}: 'content' must be a string")
            results[]]]]],,,,"valid"] = False,,,,,,,,,
            continue
                            
                            # Check content length
            if len())))))message[]]]]],,,,"content"]) < 1:,
            results[]]]]],,,,"warnings"],.append())))))f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}, message {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}i+1}: 'content' is empty"),
            continue
                        
                        # Check for coherent conversation structure
            roles = []]]]],,,,m[]]]]],,,,"role"] for m in example[]]]]],,,,"messages"]]:,
                        # Check if there's a user message:
                        if "user" not in roles:
                            results[]]]]],,,,"warnings"],.append())))))f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}: No user message in the conversation")
                            ,
                        # Check if there's an assistant message:
                        if "assistant" not in roles:
                            results[]]]]],,,,"warnings"],.append())))))f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}: No assistant message in the conversation")
                            ,
                        # Check for alternating user-assistant pattern
                        for i in range())))))1, len())))))roles)):
                            if roles[]]]]],,,,i-1] == "assistant" and roles[]]]]],,,,i] == "assistant":,
                            results[]]]]],,,,"warnings"],.append())))))f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}: Two consecutive assistant messages")
                            ,
                            if roles[]]]]],,,,i-1] == "user" and roles[]]]]],,,,i] == "user":,
                            results[]]]]],,,,"warnings"],.append())))))f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}: Two consecutive user messages")
                            ,
                    except json.JSONDecodeError:
                        results[]]]]],,,,"errors"].append()))))),f"Line {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}line_number}: Invalid JSON")
                        results[]]]]],,,,"valid"] = False,,,,,,,,,
            
            # Check if file is empty:
                        if results[]]]]],,,,"examples_count"] == 0:,
                        results[]]]]],,,,"errors"].append()))))),"File is empty")
                        results[]]]]],,,,"valid"] = False,,,,,,,,,
            
            # Check if there are too few examples:
                        if results[]]]]],,,,"examples_count"] < 10:,
                        results[]]]]],,,,"warnings"],.append())))))f"Only {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}results[]]]]],,,,'examples_count']} examples. Recommended minimum is 10.")
                        ,
            # Calculate total tokens as an estimate
                        total_chars = 0
            with open())))))file_path, 'r') as f:
                for line in f:
                    example = json.loads())))))line)
                    for message in example[]]]]],,,,"messages"]:,
                    total_chars += len())))))message[]]]]],,,,"content"])
                    ,
            # Rough estimate: 4 characters per token
                    estimated_tokens = total_chars // 4
                    results[]]]]],,,,"estimated_tokens"] = estimated_tokens
                    ,
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": True,
                "valid": results[]]]]],,,,"valid"],
                "examples_count": results[]]]]],,,,"examples_count"],
                "estimated_tokens": estimated_tokens,
                "errors": results[]]]]],,,,"errors"],
                "warnings": results[]]]]],,,,"warnings"],
                }
        
        except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": str())))))e)
                }
    
                def upload_fine_tuning_file())))))self, file_path: str) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
                """
                Upload a file for fine-tuning.
        
        Args:
            file_path: Path to the JSONL file to upload
            
        Returns:
            File upload status
            """
        try:
            # Validate file before upload
            validation = self.validate_fine_tuning_file())))))file_path)
            
            if not validation[]]]]],,,,"success"]:,
            return validation
            
            if not validation[]]]]],,,,"valid"]:,
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": "File validation failed",
            "validation": validation
            }
            
            # Upload the file
            with open())))))file_path, "rb") as file:
                response = openai.files.create())))))
                file=file,
                purpose="fine-tune"
                )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "file_id": response.id,
            "filename": response.filename,
            "purpose": response.purpose,
            "created_at": response.created_at,
            "status": response.status,
            "validation": validation
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))e)
            }
    
            def list_files())))))self, purpose: Optional[]]]]],,,,str] = "fine-tune") -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,,
            """
            List files available for fine-tuning.
        
        Args:
            purpose: Filter files by purpose ())))))fine-tune, assistants, etc.)
            
        Returns:
            List of files
            """
        try:
            # Get files
            if purpose:
                files = openai.files.list())))))purpose=purpose)
            else:
                files = openai.files.list()))))))
            
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": True,
                "files": files.data,
                "count": len())))))files.data)
                }
        
        except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": str())))))e)
                }
    
                def get_file_info())))))self, file_id: str) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
                """
                Get information about a specific file.
        
        Args:
            file_id: ID of the file to retrieve
            
        Returns:
            File information
            """
        try:
            file = openai.files.retrieve())))))file_id)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "file_id": file.id,
            "filename": file.filename,
            "purpose": file.purpose,
            "created_at": file.created_at,
            "status": file.status
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))e)
            }
    
            def delete_file())))))self, file_id: str) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
            """
            Delete a file.
        
        Args:
            file_id: ID of the file to delete
            
        Returns:
            Deletion status
            """
        try:
            deletion = openai.files.delete())))))file_id)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "deleted": deletion.deleted,
            "id": deletion.id
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))e)
            }
    
    # === Fine-tuning Jobs ===
    
            def create_fine_tuning_job())))))self, training_file_id: str,
            model: str = "gpt-3.5-turbo",
            validation_file_id: Optional[]]]]],,,,str] = None,
            hyperparameters: Optional[]]]]],,,,Dict[]]]]],,,,str, Any]] = None,
            suffix: Optional[]]]]],,,,str] = None) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,,
            """
            Create a new fine-tuning job.
        
        Args:
            training_file_id: ID of the training data file
            model: Base model to fine-tune
            validation_file_id: Optional ID of the validation data file
            hyperparameters: Optional hyperparameters for fine-tuning
            suffix: Optional custom suffix for the fine-tuned model name
            
        Returns:
            Fine-tuning job creation status
            """
        try:
            # Set up parameters
            params = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "training_file": training_file_id,
            "model": model
            }
            
            if validation_file_id:
                params[]]]]],,,,"validation_file"] = validation_file_id
                ,
            if hyperparameters:
                params[]]]]],,,,"hyperparameters"] = hyperparameters
                ,
            if suffix:
                params[]]]]],,,,"suffix"] = suffix
                ,
            # Create the fine-tuning job
                job = openai.fine_tuning.jobs.create())))))**params)
            
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": True,
                "job_id": job.id,
                "model": job.model,
                "status": job.status,
                "created_at": job.created_at,
                "fine_tuned_model": job.fine_tuned_model,
                "object": job.object
                }
        
        except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": str())))))e)
                }
    
                def list_fine_tuning_jobs())))))self, limit: int = 10) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
                """
                List fine-tuning jobs.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of fine-tuning jobs
            """
        try:
            jobs = openai.fine_tuning.jobs.list())))))limit=limit)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "jobs": jobs.data,
            "count": len())))))jobs.data)
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))e)
            }
    
            def get_fine_tuning_job())))))self, job_id: str) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
            """
            Get information about a specific fine-tuning job.
        
        Args:
            job_id: ID of the fine-tuning job
            
        Returns:
            Fine-tuning job information
            """
        try:
            job = openai.fine_tuning.jobs.retrieve())))))job_id)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "job_id": job.id,
            "model": job.model,
            "status": job.status,
            "created_at": job.created_at,
            "fine_tuned_model": job.fine_tuned_model,
            "object": job.object,
            "training_file": job.training_file,
            "validation_file": job.validation_file,
            "hyperparameters": job.hyperparameters,
            "result_files": job.result_files,
            "trained_tokens": job.trained_tokens
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))e)
            }
    
            def cancel_fine_tuning_job())))))self, job_id: str) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
            """
            Cancel a fine-tuning job.
        
        Args:
            job_id: ID of the fine-tuning job to cancel
            
        Returns:
            Cancellation status
            """
        try:
            job = openai.fine_tuning.jobs.cancel())))))job_id)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "job_id": job.id,
            "status": job.status,
            "model": job.model
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))e)
            }
    
            def list_fine_tuning_events())))))self, job_id: str, limit: int = 10) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
            """
            List events for a fine-tuning job.
        
        Args:
            job_id: ID of the fine-tuning job
            limit: Maximum number of events to return
            
        Returns:
            List of events for the fine-tuning job
            """
        try:
            events = openai.fine_tuning.jobs.list_events())))))
            fine_tuning_job_id=job_id,
            limit=limit
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "job_id": job_id,
            "events": events.data,
            "count": len())))))events.data)
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))e)
            }
    
    # === Using Fine-tuned Models ===
    
            def list_fine_tuned_models())))))self) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
            """
            List all available fine-tuned models for the organization.
        
        Returns:
            List of fine-tuned models
            """
        try:
            models = openai.models.list()))))))
            
            # Filter for fine-tuned models
            fine_tuned_models = []]]]],,,,
                model for model in models.data:
                    if model.id.startswith())))))"ft:") or ":ft-" in model.id
                    ]
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "models": fine_tuned_models,
            "count": len())))))fine_tuned_models)
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))e)
            }
    
            def delete_fine_tuned_model())))))self, model_id: str) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
            """
            Delete a fine-tuned model.
        
        Args:
            model_id: ID of the fine-tuned model to delete
            
        Returns:
            Deletion status
            """
        try:
            deletion = openai.models.delete())))))model_id)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "deleted": deletion.deleted,
            "id": deletion.id
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))e)
            }
    
            def use_fine_tuned_model())))))self, model: str, messages: List[]]]]],,,,Dict[]]]]],,,,str, Any]],
            temperature: float = 0.7,
            max_tokens: Optional[]]]]],,,,int] = None) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
            """
            Use a fine-tuned model for generating completions.
        
        Args:
            model: ID of the fine-tuned model to use
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Model response
            """
        try:
            # Check API key
            if not self.api_key:
            raise ValueError())))))"No OpenAI API key provided.")
            
            # Create completion with fine-tuned model
            response = openai.chat.completions.create())))))
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "model": response.model,
            "id": response.id,
            "content": response.choices[]]]]],,,,0].message.content,
            "finish_reason": response.choices[]]]]],,,,0].finish_reason
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str())))))e)
            }
    
    # === High-level Helper Methods ===
    
            def wait_for_fine_tuning_completion())))))self, job_id: str,
            polling_interval: int = 60,
            max_wait_time: int = 7200) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
            """
            Wait for a fine-tuning job to complete, with progress updates.
        
        Args:
            job_id: ID of the fine-tuning job to wait for
            polling_interval: Time between status checks in seconds
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            Final job status
            """
        try:
            start_time = time.time()))))))
            elapsed_time = 0
            last_status = None
            
            print())))))f"Waiting for fine-tuning job {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}job_id} to complete...")
            print())))))f"Will check status every {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}polling_interval} seconds, for up to {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}max_wait_time//60} minutes.")
            
            while elapsed_time < max_wait_time:
                # Get job status
                job_info = self.get_fine_tuning_job())))))job_id)
                
                if not job_info[]]]]],,,,"success"]:,
            return job_info
                
            current_status = job_info[]]]]],,,,"status"]
                
                # Print status update if status has changed:
                if current_status != last_status:
                    timestamp = time.strftime())))))"%Y-%m-%d %H:%M:%S")
                    print())))))f"[]]]]],,,,{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}timestamp}] Job status: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}current_status}")
                    last_status = current_status
                
                # Check if job is complete:
                if current_status in []]]]],,,,"succeeded", "failed", "cancelled"]:
                    print())))))f"Fine-tuning job completed with status: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}current_status}")
                    
                    # Get final metrics
                    events = self.list_fine_tuning_events())))))job_id, limit=100)
                    if events[]]]]],,,,"success"]:,
                        # Extract training metrics from events
                    metrics = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                        for event in events[]]]]],,,,"events"]:
                            if event.type == "metrics":
                                for key, value in event.data.items())))))):
                                    metrics[]]]]],,,,key] = value
                        
                                    job_info[]]]]],,,,"metrics"] = metrics
                    
                                return job_info
                
                # Wait before checking again
                                time.sleep())))))polling_interval)
                                elapsed_time = time.time())))))) - start_time
            
            # If we get here, we've timed out
                            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "success": False,
                            "error": f"Timed out waiting for fine-tuning job to complete after {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}max_wait_time//60} minutes.",
                            "job_id": job_id,
                            "last_status": last_status
                            }
        
        except Exception as e:
                            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                            "success": False,
                            "error": str())))))e)
                            }
    
                            def complete_fine_tuning_workflow())))))self, training_data_path: str,
                            validation_data_path: Optional[]]]]],,,,str] = None,
                            model: str = "gpt-3.5-turbo",
                            suffix: Optional[]]]]],,,,str] = None,
                            hyperparameters: Optional[]]]]],,,,Dict[]]]]],,,,str, Any]] = None,
                            wait_for_completion: bool = True,
                            polling_interval: int = 60,
                            max_wait_time: int = 7200) -> Dict[]]]]],,,,str, Any]:,,,,,,,,,,
                            """
                            Execute a complete fine-tuning workflow from data to model.
        
        Args:
            training_data_path: Path to the training data file
            validation_data_path: Optional path to the validation data file
            model: Base model to fine-tune
            suffix: Optional custom suffix for the fine-tuned model name
            hyperparameters: Optional hyperparameters for fine-tuning
            wait_for_completion: Whether to wait for the job to complete
            polling_interval: Time between status checks in seconds
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            Complete workflow status
            """
        try:
            workflow_results = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "steps": {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            }
            
            # Step 1: Validate training data
            print())))))"Step 1: Validating training data...")
            validation = self.validate_fine_tuning_file())))))training_data_path)
            workflow_results[]]]]],,,,"steps"][]]]]],,,,"validation"] = validation
            
            if not validation[]]]]],,,,"success"] or not validation[]]]]],,,,"valid"]:,
            workflow_results[]]]]],,,,"error"] = "Training data validation failed."
            return workflow_results
            
            # Step 2: Upload training file
            print())))))"Step 2: Uploading training file...")
            training_upload = self.upload_fine_tuning_file())))))training_data_path)
            workflow_results[]]]]],,,,"steps"][]]]]],,,,"training_upload"] = training_upload
            
            if not training_upload[]]]]],,,,"success"]:,
            workflow_results[]]]]],,,,"error"] = "Training file upload failed."
            return workflow_results
            
            training_file_id = training_upload[]]]]],,,,"file_id"]
            print())))))f"Training file uploaded successfully with ID: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}training_file_id}")
            
            # Step 3: Upload validation file if provided
            validation_file_id = None:
            if validation_data_path:
                print())))))"Step 3: Uploading validation file...")
                validation_upload = self.upload_fine_tuning_file())))))validation_data_path)
                workflow_results[]]]]],,,,"steps"][]]]]],,,,"validation_upload"] = validation_upload
                
                if not validation_upload[]]]]],,,,"success"]:,
                workflow_results[]]]]],,,,"error"] = "Validation file upload failed."
                return workflow_results
                
                validation_file_id = validation_upload[]]]]],,,,"file_id"]
                print())))))f"Validation file uploaded successfully with ID: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}validation_file_id}")
            else:
                print())))))"Step 3: No validation file provided, skipping upload.")
            
            # Step 4: Create fine-tuning job
                print())))))"Step 4: Creating fine-tuning job...")
                job_creation = self.create_fine_tuning_job())))))
                training_file_id=training_file_id,
                validation_file_id=validation_file_id,
                model=model,
                suffix=suffix,
                hyperparameters=hyperparameters
                )
                workflow_results[]]]]],,,,"steps"][]]]]],,,,"job_creation"] = job_creation
            
                if not job_creation[]]]]],,,,"success"]:,
                workflow_results[]]]]],,,,"error"] = "Fine-tuning job creation failed."
                return workflow_results
            
                job_id = job_creation[]]]]],,,,"job_id"]
                print())))))f"Fine-tuning job created successfully with ID: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}job_id}")
            
            # Step 5: Wait for job completion if requested:
            if wait_for_completion:
                print())))))"Step 5: Waiting for fine-tuning job to complete...")
                job_completion = self.wait_for_fine_tuning_completion())))))
                job_id=job_id,
                polling_interval=polling_interval,
                max_wait_time=max_wait_time
                )
                workflow_results[]]]]],,,,"steps"][]]]]],,,,"job_completion"] = job_completion
                
                if not job_completion[]]]]],,,,"success"]:,
                workflow_results[]]]]],,,,"error"] = "Error waiting for job completion."
                return workflow_results
                
                if job_completion[]]]]],,,,"status"] == "succeeded":
                    print())))))f"Fine-tuning completed successfully!")
                    print())))))f"Fine-tuned model ID: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}job_completion[]]]]],,,,'fine_tuned_model']}")
                    workflow_results[]]]]],,,,"fine_tuned_model"] = job_completion[]]]]],,,,"fine_tuned_model"]
                else:
                    workflow_results[]]]]],,,,"error"] = f"Fine-tuning failed with status: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}job_completion[]]]]],,,,'status']}"
                    return workflow_results
            else:
                print())))))"Step 5: Skipping wait for completion as requested.")
                print())))))f"You can check the status later with get_fine_tuning_job())))))'{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}job_id}')")
            
            # Set overall success
                workflow_results[]]]]],,,,"success"] = True
                workflow_results[]]]]],,,,"job_id"] = job_id
            
                    return workflow_results
        
        except Exception as e:
                    return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                    "success": False,
                    "error": str())))))e)
                    }

def main())))))):
    """
    Example usage of the OpenAI Fine-tuning extension.
    """
    # Check if we have an API key
    api_key = os.environ.get())))))"OPENAI_API_KEY"):
    if not api_key:
        print())))))"No OpenAI API key found in environment variables.")
        print())))))"Please set OPENAI_API_KEY in your .env file.")
        return
    
    # Initialize the Fine-tuning extension
        fine_tuning = OpenAIFineTuning())))))api_key=api_key)
    
    # Example 1: Create a simple training dataset
        print())))))"\nExample 1: Creating a simple training dataset")
    
    # Create a temporary directory if it doesn't exist
        import tempfile
        temp_dir = tempfile.gettempdir()))))))
        training_file_path = os.path.join())))))temp_dir, "fine_tuning_data.jsonl")
    
    # Example training data - simple Q&A format
        training_data = []]]]],,,,
        # Example 1
        []]]]],,,,:
            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "system", "content": "You are a helpful assistant that provides information about astronomy."},
            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "user", "content": "What is a black hole?"},
            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "assistant", "content": "A black hole is a region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves, can escape from it. They form when massive stars collapse at the end of their life cycle."}
            ],
        # Example 2
            []]]]],,,,
            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "system", "content": "You are a helpful assistant that provides information about astronomy."},
            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "user", "content": "How far is the Moon from Earth?"},
            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "assistant", "content": "The Moon is approximately 238,855 miles ())))))384,400 kilometers) away from Earth on average. However, because the Moon's orbit is elliptical, its distance from Earth varies throughout its orbit, ranging from about 225,700 miles ())))))363,300 kilometers) at its closest point ())))))perigee) to 252,000 miles ())))))405,500 kilometers) at its farthest point ())))))apogee)."}
            ],
        # Example 3
            []]]]],,,,
            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "system", "content": "You are a helpful assistant that provides information about astronomy."},
            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "user", "content": "What are the planets in our solar system?"},
            {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}"role": "assistant", "content": "The eight planets in our solar system, in order from the Sun outward, are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was considered the ninth planet until 2006, when it was reclassified as a dwarf planet by the International Astronomical Union ())))))IAU)."}
            ]
            ]
    
    # Create the training file
            result = fine_tuning.prepare_jsonl_from_messages())))))training_data, training_file_path)
    
            if result[]]]]],,,,"success"]:,
            print())))))f"Created training file at {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}training_file_path}")
            print())))))f"Number of examples: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}len())))))training_data)}")
        
        # Validate the file
            validation = fine_tuning.validate_fine_tuning_file())))))training_file_path)
        
            if validation[]]]]],,,,"success"] and validation[]]]]],,,,"valid"]:,
            print())))))"File validation passed!")
            print())))))f"Examples: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}validation[]]]]],,,,'examples_count']}")
            print())))))f"Estimated tokens: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}validation[]]]]],,,,'estimated_tokens']}")
            
            for warning in validation.get())))))"warnings", []]]]],,,,],,):
                print())))))f"Warning: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}warning}")
        else:
            print())))))"File validation failed!")
            for error in validation.get())))))"errors", []]]]],,,,],,):
                print())))))f"Error: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}error}")
    else:
        print())))))f"Error creating training file: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}result.get())))))'error')}")
    
    # Example 2: List existing fine-tuned models
        print())))))"\nExample 2: Listing existing fine-tuned models")
    
        models = fine_tuning.list_fine_tuned_models()))))))
    
        if models[]]]]],,,,"success"]:,
        print())))))f"Found {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}models[]]]]],,,,'count']} fine-tuned models:")
        for i, model in enumerate())))))models[]]]]],,,,"models"], 1):
            print())))))f"{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}i}. {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model.id} ())))))created at: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}model.created})")
    else:
        print())))))f"Error listing models: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}models.get())))))'error')}")
    
    # Note: We won't actually create a fine-tuning job or use a fine-tuned model in this example
    # since it requires API credits and takes time to complete.
    
        print())))))"\nFor a complete fine-tuning workflow, you would use:")
        print())))))"fine_tuning.complete_fine_tuning_workflow())))))training_file_path)")
        print())))))"\nThis would:")
        print())))))"1. Validate your training data")
        print())))))"2. Upload it to OpenAI")
        print())))))"3. Create a fine-tuning job")
        print())))))"4. Wait for completion ())))))optional)")
        print())))))"5. Return the fine-tuned model ID for use")

if __name__ == "__main__":
    main()))))))