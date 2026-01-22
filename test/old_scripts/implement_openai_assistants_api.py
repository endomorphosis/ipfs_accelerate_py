#!/usr/bin/env python
"""
Implementation of the OpenAI Assistants API for ipfs_accelerate_py.
This enhances the existing OpenAI API implementation with the Assistants API functionality.
"""

import os
import sys
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv())))))))))))))))))))

# Add parent directory to path to import the ipfs_accelerate_py module
sys.path.append()))))))))))))))))))os.path.join()))))))))))))))))))os.path.dirname()))))))))))))))))))os.path.dirname()))))))))))))))))))__file__)), 'ipfs_accelerate_py'))

try:
    # Import the OpenAI API implementation to extend it
    from api_backends import openai_api
    import openai
except ImportError as e:
    print()))))))))))))))))))f"Failed to import required modules: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}e}")
    sys.exit()))))))))))))))))))1)

class OpenAIAssistantsAPI:
    """
    Extension to add Assistants API capabilities to the ipfs_accelerate_py OpenAI implementation.
    This class provides methods for creating and managing assistants, threads, and messages.
    """
    
    def __init__()))))))))))))))))))self, api_key: Optional[]],,str] = None, resources: Dict[]],,str, Any] = None, metadata: Dict[]],,str, Any] = None):,
    """
    Initialize the OpenAI Assistants API extension.
        
        Args:
            api_key: OpenAI API key ()))))))))))))))))))optional, will use environment variable if not provided):
                resources: Resources dictionary for ipfs_accelerate_py
                metadata: Metadata dictionary for ipfs_accelerate_py
                """
                self.resources = resources if resources else {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                self.metadata = metadata if metadata else {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Get API key from parameters, metadata, or environment variable
                self.api_key = api_key or ()))))))))))))))))))metadata.get()))))))))))))))))))"openai_api_key") if metadata else None) or os.environ.get()))))))))))))))))))"OPENAI_API_KEY")
        :
        if not self.api_key:
            print()))))))))))))))))))"Warning: No OpenAI API key provided. Assistants API will not work without an API key.")
        
        # Set up the OpenAI API client
            openai.api_key = self.api_key
        
        # Initialize tracking for rate limiting and backoff
            self.current_requests = 0
            self.max_concurrent_requests = 5
            self.request_queue = []],,],,
            self.max_retries = 3
            self.base_wait_time = 1.0  # in seconds
        
        # Store main OpenAI API implementation for delegation
            self.openai_api_impl = openai_api()))))))))))))))))))resources=self.resources, metadata=self.metadata)
        
            print()))))))))))))))))))"OpenAI Assistants API extension initialized")
    
    # === Assistant Management ===
    
            def create_assistant()))))))))))))))))))self, model: str, name: Optional[]],,str] = None,
            description: Optional[]],,str] = None,
            instructions: Optional[]],,str] = None,
            tools: Optional[]],,List[]],,Dict[]],,str, Any]]] = None,
            file_ids: Optional[]],,List[]],,str]] = None) -> Dict[]],,str, Any]:,,,,,
            """
            Create a new assistant with the specified configuration.
        
        Args:
            model: The model to use for the assistant ()))))))))))))))))))e.g., "gpt-4o")
            name: The name of the assistant
            description: A description of the assistant
            instructions: Instructions for the assistant
            tools: A list of tools the assistant can use
            file_ids: A list of file IDs that the assistant can access
            
        Returns:
            The created assistant object
            """
        try:
            # Check API key
            if not self.api_key:
            raise ValueError()))))))))))))))))))"No OpenAI API key provided.")
                
            # Create the assistant
            assistant = openai.beta.assistants.create()))))))))))))))))))
            model=model,
            name=name,
            description=description,
            instructions=instructions,
            tools=tools or []],,],,,
            file_ids=file_ids or []],,],,
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "assistant": assistant,
            "assistant_id": assistant.id,
            "model": assistant.model,
            "name": assistant.name,
            "created_at": assistant.created_at
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def list_assistants()))))))))))))))))))self, limit: int = 20, order: str = "desc") -> Dict[]],,str, Any]:,,,
            """
            List all assistants for the current organization.
        
        Args:
            limit: The maximum number of assistants to return
            order: The order to return assistants in ()))))))))))))))))))"asc" or "desc")
            
        Returns:
            A list of assistant objects
            """
        try:
            assistants = openai.beta.assistants.list()))))))))))))))))))
            limit=limit,
            order=order
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "assistants": assistants.data,
            "count": len()))))))))))))))))))assistants.data)
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def retrieve_assistant()))))))))))))))))))self, assistant_id: str) -> Dict[]],,str, Any]:,,,
            """
            Retrieve a specific assistant by ID.
        
        Args:
            assistant_id: The ID of the assistant to retrieve
            
        Returns:
            The assistant object
            """
        try:
            assistant = openai.beta.assistants.retrieve()))))))))))))))))))assistant_id)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "assistant": assistant,
            "assistant_id": assistant.id,
            "model": assistant.model,
            "name": assistant.name,
            "created_at": assistant.created_at
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def update_assistant()))))))))))))))))))self, assistant_id: str,
            model: Optional[]],,str] = None,
            name: Optional[]],,str] = None,
            description: Optional[]],,str] = None,
            instructions: Optional[]],,str] = None,
            tools: Optional[]],,List[]],,Dict[]],,str, Any]]] = None,
            file_ids: Optional[]],,List[]],,str]] = None) -> Dict[]],,str, Any]:,,,,,
            """
            Update an existing assistant.
        
        Args:
            assistant_id: The ID of the assistant to update
            model: The model to use for the assistant
            name: The name of the assistant
            description: A description of the assistant
            instructions: Instructions for the assistant
            tools: A list of tools the assistant can use
            file_ids: A list of file IDs that the assistant can access
            
        Returns:
            The updated assistant object
            """
        try:
            # Create a dictionary of update parameters, only including non-None values
            update_params = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            if model is not None:
                update_params[]],,"model"] = model,
            if name is not None:
                update_params[]],,"name"] = name,
            if description is not None:
                update_params[]],,"description"] = description,
            if instructions is not None:
                update_params[]],,"instructions"] = instructions,
            if tools is not None:
                update_params[]],,"tools"] = tools,
            if file_ids is not None:
                update_params[]],,"file_ids"] = file_ids
                ,
            # Update the assistant
                assistant = openai.beta.assistants.update()))))))))))))))))))
                assistant_id=assistant_id,
                **update_params
                )
            
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": True,
                "assistant": assistant,
                "assistant_id": assistant.id,
                "model": assistant.model,
                "name": assistant.name,
                "created_at": assistant.created_at
                }
        
        except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": str()))))))))))))))))))e)
                }
    
                def delete_assistant()))))))))))))))))))self, assistant_id: str) -> Dict[]],,str, Any]:,,,
                """
                Delete an assistant by ID.
        
        Args:
            assistant_id: The ID of the assistant to delete
            
        Returns:
            Deletion status
            """
        try:
            deletion = openai.beta.assistants.delete()))))))))))))))))))assistant_id)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "deleted": deletion.deleted,
            "id": deletion.id
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
    # === Thread Management ===
    
            def create_thread()))))))))))))))))))self, messages: Optional[]],,List[]],,Dict[]],,str, Any]]] = None) -> Dict[]],,str, Any]:,,,,
            """
            Create a new thread for conversation.
        
        Args:
            messages: Optional initial messages for the thread
            
        Returns:
            The created thread object
            """
        try:
            # Format messages for the API if provided:
            formatted_messages = None:
            if messages:
                formatted_messages = []],,
                {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "role": msg.get()))))))))))))))))))"role", "user"),
                "content": msg.get()))))))))))))))))))"content", "")
                }
                    for msg in messages:
                        ]
            
            # Create the thread
                        thread = openai.beta.threads.create()))))))))))))))))))
                        messages=formatted_messages
                        )
            
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": True,
                "thread": thread,
                "thread_id": thread.id,
                "created_at": thread.created_at
                }
        
        except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": str()))))))))))))))))))e)
                }
    
                def retrieve_thread()))))))))))))))))))self, thread_id: str) -> Dict[]],,str, Any]:,,,
                """
                Retrieve a thread by ID.
        
        Args:
            thread_id: The ID of the thread to retrieve
            
        Returns:
            The thread object
            """
        try:
            thread = openai.beta.threads.retrieve()))))))))))))))))))thread_id)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "thread": thread,
            "thread_id": thread.id,
            "created_at": thread.created_at
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def delete_thread()))))))))))))))))))self, thread_id: str) -> Dict[]],,str, Any]:,,,
            """
            Delete a thread by ID.
        
        Args:
            thread_id: The ID of the thread to delete
            
        Returns:
            Deletion status
            """
        try:
            deletion = openai.beta.threads.delete()))))))))))))))))))thread_id)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "deleted": deletion.deleted,
            "id": deletion.id
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
    # === Message Management ===
    
            def create_message()))))))))))))))))))self, thread_id: str, role: str, content: str,
            file_ids: Optional[]],,List[]],,str]] = None) -> Dict[]],,str, Any]:,,,,,
            """
            Create a new message in a thread.
        
        Args:
            thread_id: The ID of the thread to add the message to
            role: The role of the message sender ()))))))))))))))))))user)
            content: The content of the message
            file_ids: A list of file IDs to attach to the message
            
        Returns:
            The created message object
            """
        try:
            message = openai.beta.threads.messages.create()))))))))))))))))))
            thread_id=thread_id,
            role=role,
            content=content,
            file_ids=file_ids or []],,],,
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "message": message,
            "message_id": message.id,
            "thread_id": thread_id,
            "role": message.role,
            "created_at": message.created_at
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def list_messages()))))))))))))))))))self, thread_id: str, limit: int = 20, order: str = "desc") -> Dict[]],,str, Any]:,,,
            """
            List all messages in a thread.
        
        Args:
            thread_id: The ID of the thread to list messages from
            limit: The maximum number of messages to return
            order: The order to return messages in ()))))))))))))))))))"asc" or "desc")
            
        Returns:
            A list of message objects
            """
        try:
            messages = openai.beta.threads.messages.list()))))))))))))))))))
            thread_id=thread_id,
            limit=limit,
            order=order
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "messages": messages.data,
            "count": len()))))))))))))))))))messages.data)
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def retrieve_message()))))))))))))))))))self, thread_id: str, message_id: str) -> Dict[]],,str, Any]:,,,
            """
            Retrieve a specific message by ID.
        
        Args:
            thread_id: The ID of the thread the message belongs to
            message_id: The ID of the message to retrieve
            
        Returns:
            The message object
            """
        try:
            message = openai.beta.threads.messages.retrieve()))))))))))))))))))
            thread_id=thread_id,
            message_id=message_id
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "message": message,
            "message_id": message.id,
            "thread_id": thread_id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
    # === Run Management ===
    
            def create_run()))))))))))))))))))self, thread_id: str, assistant_id: str,
            instructions: Optional[]],,str] = None,
            tools: Optional[]],,List[]],,Dict[]],,str, Any]]] = None) -> Dict[]],,str, Any]:,,,,
            """
            Create a run to generate a response from the assistant in a thread.
        
        Args:
            thread_id: The ID of the thread to run the assistant on
            assistant_id: The ID of the assistant to use
            instructions: Additional instructions for the assistant
            tools: A list of tools the assistant can use for this run
            
        Returns:
            The created run object
            """
        try:
            # Create parameters dictionary
            params = {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "thread_id": thread_id,
            "assistant_id": assistant_id
            }
            
            # Add optional parameters if provided:
            if instructions is not None:
                params[]],,"instructions"] = instructions,
            if tools is not None:
                params[]],,"tools"] = tools,
            
            # Create the run
                run = openai.beta.threads.runs.create()))))))))))))))))))**params)
            
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": True,
                "run": run,
                "run_id": run.id,
                "thread_id": thread_id,
                "assistant_id": assistant_id,
                "status": run.status,
                "created_at": run.created_at
                }
        
        except Exception as e:
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": False,
                "error": str()))))))))))))))))))e)
                }
    
                def retrieve_run()))))))))))))))))))self, thread_id: str, run_id: str) -> Dict[]],,str, Any]:,,,
                """
                Retrieve a run by ID.
        
        Args:
            thread_id: The ID of the thread the run belongs to
            run_id: The ID of the run to retrieve
            
        Returns:
            The run object
            """
        try:
            run = openai.beta.threads.runs.retrieve()))))))))))))))))))
            thread_id=thread_id,
            run_id=run_id
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "run": run,
            "run_id": run.id,
            "thread_id": thread_id,
            "assistant_id": run.assistant_id,
            "status": run.status,
            "created_at": run.created_at
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def list_runs()))))))))))))))))))self, thread_id: str, limit: int = 20, order: str = "desc") -> Dict[]],,str, Any]:,,,
            """
            List all runs in a thread.
        
        Args:
            thread_id: The ID of the thread to list runs from
            limit: The maximum number of runs to return
            order: The order to return runs in ()))))))))))))))))))"asc" or "desc")
            
        Returns:
            A list of run objects
            """
        try:
            runs = openai.beta.threads.runs.list()))))))))))))))))))
            thread_id=thread_id,
            limit=limit,
            order=order
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "runs": runs.data,
            "count": len()))))))))))))))))))runs.data)
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def cancel_run()))))))))))))))))))self, thread_id: str, run_id: str) -> Dict[]],,str, Any]:,,,
            """
            Cancel a run that is in progress.
        
        Args:
            thread_id: The ID of the thread the run belongs to
            run_id: The ID of the run to cancel
            
        Returns:
            The cancelled run object
            """
        try:
            run = openai.beta.threads.runs.cancel()))))))))))))))))))
            thread_id=thread_id,
            run_id=run_id
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "run": run,
            "run_id": run.id,
            "thread_id": thread_id,
            "status": run.status,
            "created_at": run.created_at
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
    # === Function Handling ===
    
            def submit_tool_outputs()))))))))))))))))))self, thread_id: str, run_id: str,
            tool_outputs: List[]],,Dict[]],,str, Any]]) -> Dict[]],,str, Any]:,,,
            """
            Submit tool outputs to a run that requires function calls.
        
        Args:
            thread_id: The ID of the thread the run belongs to
            run_id: The ID of the run to submit tool outputs for
            tool_outputs: A list of tool outputs with function call results
            
        Returns:
            The updated run object
            """
        try:
            run = openai.beta.threads.runs.submit_tool_outputs()))))))))))))))))))
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "run": run,
            "run_id": run.id,
            "thread_id": thread_id,
            "status": run.status,
            "created_at": run.created_at
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
    # === High-Level Helper Methods ===
    
            async def wait_for_run_completion()))))))))))))))))))self, thread_id: str, run_id: str,
            timeout: int = 300, poll_interval: int = 1) -> Dict[]],,str, Any]:,,,
            """
            Wait for a run to complete and handle required actions.
        
        Args:
            thread_id: The ID of the thread the run belongs to
            run_id: The ID of the run to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks in seconds
            
        Returns:
            The completed run object
            """
            start_time = time.time())))))))))))))))))))
        
        while time.time()))))))))))))))))))) - start_time < timeout:
            # Get the current run status
            response = self.retrieve_run()))))))))))))))))))thread_id, run_id)
            
            if not response[]],,"success"]:
            return response
            
            run = response[]],,"run"]
            status = run.status
            
            # Check if the run is complete:
            if status in []],,"completed", "failed", "cancelled", "expired"]:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "run": run,
            "status": status,
            "thread_id": thread_id,
            "run_id": run_id,
            "elapsed_time": time.time()))))))))))))))))))) - start_time
            }
            
            # Handle required actions ()))))))))))))))))))function calls)
            if status == "requires_action":
                # Get required actions
                required_action = run.required_action
                if required_action.type == "submit_tool_outputs":
                    # This is where you would handle function calls
                    # For this example, we'll just return that action is required
                return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
                "success": True,
                "status": "requires_action",
                "run": run,
                "thread_id": thread_id,
                "run_id": run_id,
                "required_action": required_action,
                "tool_calls": required_action.submit_tool_outputs.tool_calls,
                "elapsed_time": time.time()))))))))))))))))))) - start_time
                }
            
            # Wait before checking again
                await asyncio.sleep()))))))))))))))))))poll_interval)
        
        # If we get here, the run has timed out
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": f"Run timed out after {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}timeout} seconds",
            "thread_id": thread_id,
            "run_id": run_id
            }
    
            def simple_conversation()))))))))))))))))))self, assistant_id: str, messages: List[]],,str],
            instructions: Optional[]],,str] = None,
            tools: Optional[]],,List[]],,Dict[]],,str, Any]]] = None,
            timeout: int = 300) -> Dict[]],,str, Any]:,,,
            """
            Have a simple conversation with an assistant.
        
        Args:
            assistant_id: The ID of the assistant to use
            messages: A list of user message strings
            instructions: Additional instructions for the assistant
            tools: A list of tools the assistant can use
            timeout: Maximum time to wait for completion in seconds:
        Returns:
            The conversation history and results
            """
        try:
            # Create a new thread
            thread_response = self.create_thread())))))))))))))))))))
            if not thread_response[]],,"success"]:
            return thread_response
            
            thread_id = thread_response[]],,"thread_id"]
            
            # Add all messages to the thread
            for message in messages:
                message_response = self.create_message()))))))))))))))))))thread_id, "user", message)
                if not message_response[]],,"success"]:
                return message_response
            
            # Create a run
                run_response = self.create_run()))))))))))))))))))
                thread_id=thread_id,
                assistant_id=assistant_id,
                instructions=instructions,
                tools=tools
                )
            
            if not run_response[]],,"success"]:
                return run_response
            
                run_id = run_response[]],,"run_id"]
            
            # Wait for the run to complete
                run_completion_response = asyncio.run()))))))))))))))))))
                self.wait_for_run_completion()))))))))))))))))))thread_id, run_id, timeout=timeout)
                )
            
            if not run_completion_response[]],,"success"]:
                return run_completion_response
            
            # If the run requires action, return that
            if run_completion_response.get()))))))))))))))))))"status") == "requires_action":
                return run_completion_response
            
            # Get the messages from the thread
                messages_response = self.list_messages()))))))))))))))))))thread_id)
            
            if not messages_response[]],,"success"]:
                return messages_response
            
            # Return the conversation history
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "thread_id": thread_id,
            "run_id": run_id,
            "assistant_id": assistant_id,
            "messages": messages_response[]],,"messages"],
            "run_status": run_completion_response.get()))))))))))))))))))"status", "unknown")
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
    # === File Management ===
    
            def upload_file()))))))))))))))))))self, file_path: str, purpose: str = "assistants") -> Dict[]],,str, Any]:,,,
            """
            Upload a file to use with the Assistants API.
        
        Args:
            file_path: Path to the file to upload
            purpose: The purpose of the file ()))))))))))))))))))assistants, etc.)
            
        Returns:
            The uploaded file object
            """
        try:
            with open()))))))))))))))))))file_path, "rb") as file:
                # Upload the file
                response = openai.files.create()))))))))))))))))))
                file=file,
                purpose=purpose
                )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "file": response,
            "file_id": response.id,
            "filename": response.filename,
            "purpose": response.purpose,
            "created_at": response.created_at
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def list_files()))))))))))))))))))self, purpose: str = "assistants") -> Dict[]],,str, Any]:,,,
            """
            List all files uploaded to OpenAI.
        
        Args:
            purpose: Filter files by purpose
            
        Returns:
            A list of file objects
            """
        try:
            files = openai.files.list()))))))))))))))))))purpose=purpose)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "files": files.data,
            "count": len()))))))))))))))))))files.data)
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def retrieve_file()))))))))))))))))))self, file_id: str) -> Dict[]],,str, Any]:,,,
            """
            Retrieve a file by ID.
        
        Args:
            file_id: The ID of the file to retrieve
            
        Returns:
            The file object
            """
        try:
            file = openai.files.retrieve()))))))))))))))))))file_id)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "file": file,
            "file_id": file.id,
            "filename": file.filename,
            "purpose": file.purpose,
            "created_at": file.created_at
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def delete_file()))))))))))))))))))self, file_id: str) -> Dict[]],,str, Any]:,,,
            """
            Delete a file by ID.
        
        Args:
            file_id: The ID of the file to delete
            
        Returns:
            Deletion status
            """
        try:
            deletion = openai.files.delete()))))))))))))))))))file_id)
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "deleted": deletion.deleted,
            "id": deletion.id
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
    # === Assistant Files ===
    
            def attach_file_to_assistant()))))))))))))))))))self, assistant_id: str, file_id: str) -> Dict[]],,str, Any]:,,,
            """
            Attach a file to an assistant.
        
        Args:
            assistant_id: The ID of the assistant to attach the file to
            file_id: The ID of the file to attach
            
        Returns:
            The assistant file object
            """
        try:
            file = openai.beta.assistants.files.create()))))))))))))))))))
            assistant_id=assistant_id,
            file_id=file_id
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "file": file,
            "assistant_id": assistant_id,
            "file_id": file.id
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def list_assistant_files()))))))))))))))))))self, assistant_id: str) -> Dict[]],,str, Any]:,,,
            """
            List all files attached to an assistant.
        
        Args:
            assistant_id: The ID of the assistant to list files for
            
        Returns:
            A list of assistant file objects
            """
        try:
            files = openai.beta.assistants.files.list()))))))))))))))))))
            assistant_id=assistant_id
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "files": files.data,
            "count": len()))))))))))))))))))files.data)
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }
    
            def remove_file_from_assistant()))))))))))))))))))self, assistant_id: str, file_id: str) -> Dict[]],,str, Any]:,,,
            """
            Remove a file from an assistant.
        
        Args:
            assistant_id: The ID of the assistant to remove the file from
            file_id: The ID of the file to remove
            
        Returns:
            Deletion status
            """
        try:
            deletion = openai.beta.assistants.files.delete()))))))))))))))))))
            assistant_id=assistant_id,
            file_id=file_id
            )
            
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": True,
            "deleted": deletion.deleted,
            "id": deletion.id
            }
        
        except Exception as e:
            return {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
            "success": False,
            "error": str()))))))))))))))))))e)
            }

def main()))))))))))))))))))):
    """
    Example usage of the OpenAI Assistants API extension.
    """
    # Check if we have an API key
    api_key = os.environ.get()))))))))))))))))))"OPENAI_API_KEY"):
    if not api_key:
        print()))))))))))))))))))"No OpenAI API key found in environment variables.")
        print()))))))))))))))))))"Please set OPENAI_API_KEY in your .env file.")
        return
    
    # Initialize the Assistants API extension
        assistants_api = OpenAIAssistantsAPI()))))))))))))))))))api_key=api_key)
    
    # Example 1: Create a new assistant
        print()))))))))))))))))))"\nCreating a new assistant...")
        assistant_response = assistants_api.create_assistant()))))))))))))))))))
        model="gpt-4o",
        name="Research Assistant",
        description="A helpful assistant for research tasks",
        instructions="You are a research assistant. Help users find and analyze information."
        )
    
    if not assistant_response[]],,"success"]:
        print()))))))))))))))))))f"Error creating assistant: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}assistant_response.get()))))))))))))))))))'error')}")
        return
    
        assistant_id = assistant_response[]],,"assistant_id"]
        print()))))))))))))))))))f"Assistant created with ID: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}assistant_id}")
    
    # Example 2: Create a thread and have a conversation
        print()))))))))))))))))))"\nStarting a conversation...")
        conversation = assistants_api.simple_conversation()))))))))))))))))))
        assistant_id=assistant_id,
        messages=[]],,
        "Hello! I'm researching climate change. Can you help me understand its impact on agriculture?",
        "What are the main crops affected by rising temperatures?"
        ]
        )
    
    if not conversation[]],,"success"]:
        print()))))))))))))))))))f"Error in conversation: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}conversation.get()))))))))))))))))))'error')}")
    else:
        print()))))))))))))))))))"\nConversation results:")
        messages = conversation.get()))))))))))))))))))"messages", []],,],,)
        for message in messages:
            role = message.role
            content_list = message.content
            
            # Extract the text content
            text_content = []],,],,
            for content_item in content_list:
                if hasattr()))))))))))))))))))content_item, 'text') and hasattr()))))))))))))))))))content_item.text, 'value'):
                    text_content.append()))))))))))))))))))content_item.text.value)
            
                    print()))))))))))))))))))f"\n{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}role.upper())))))))))))))))))))}: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}' '.join()))))))))))))))))))text_content)}")
    
    # Example 3: Clean up by deleting the assistant
                    print()))))))))))))))))))"\nCleaning up...")
                    deletion = assistants_api.delete_assistant()))))))))))))))))))assistant_id)
    
    if deletion[]],,"success"]:
        print()))))))))))))))))))f"Assistant {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}assistant_id} deleted successfully.")
    else:
        print()))))))))))))))))))f"Error deleting assistant: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}deletion.get()))))))))))))))))))'error')}")

if __name__ == "__main__":
    main())))))))))))))))))))