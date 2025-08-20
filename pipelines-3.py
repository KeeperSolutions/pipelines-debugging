"""
title: Langfuse Filter Pipeline
author: open-webui
date: 2025-06-16
version: 1.7.1
license: MIT
description: A filter pipeline that uses Langfuse.
requirements: langfuse<3.0.0
"""

from typing import List, Optional
import os
import uuid
import json

from utils.pipelines.main import get_last_assistant_message
from pydantic import BaseModel
from langfuse import Langfuse
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError


def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    """Retrieve the last assistant message from the message list."""
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    return {}


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        secret_key: str
        public_key: str
        host: str
        insert_tags: bool = True
        use_model_name_instead_of_id_for_generation: bool = False
        debug: bool = False

    def __init__(self):
        self.type = "filter"
        self.name = "Langfuse Filter"

        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "your-secret-key-here"),
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "your-public-key-here"),
                "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                "use_model_name_instead_of_id_for_generation": os.getenv("USE_MODEL_NAME", "false").lower() == "true",
                "debug": os.getenv("DEBUG_MODE", "false").lower() == "true",
            }
        )

        self.langfuse = None
        self.chat_traces = {}
        self.suppressed_logs = set()
        self.model_names = {}
        self.GENERATION_TASKS = {"llm_response"}

    def log(self, message: str, suppress_repeats: bool = False):
        if self.valves.debug:
            if suppress_repeats:
                if message in self.suppressed_logs:
                    return
                self.suppressed_logs.add(message)
            print(f"[DEBUG] {message}")

    async def on_startup(self):
        self.log(f"on_startup triggered for {__name__}")
        self.set_langfuse()

    async def on_shutdown(self):
        self.log(f"on_shutdown triggered for {__name__}")
        if self.langfuse:
            self.langfuse.flush()

    async def on_valves_updated(self):
        self.log("Valves updated, resetting Langfuse client.")
        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=self.valves.debug,
            )
            self.langfuse.auth_check()
            self.log("Langfuse client initialized successfully.")
        except UnauthorizedError:
            print(
                "Langfuse credentials incorrect. Please re-enter your Langfuse credentials in the pipeline settings."
            )
        except Exception as e:
            print(
                f"Langfuse error: {e} Please re-enter your Langfuse credentials in the pipeline settings."
            )

    def _build_tags(self, task_name: str) -> list:
        """
        Builds a list of tags based on valve settings, ensuring we always add
        'open-webui' and skip user_response / llm_response from becoming tags themselves.
        """
        tags_list = []
        if self.valves.insert_tags:
            tags_list.append("open-webui")
            if task_name not in ["user_response", "llm_response"]:
                tags_list.append(task_name)
        return tags_list

    def _extract_metadata(self, body: dict) -> dict:
        """Extract metadata from various possible locations in the body."""
        # Try to get metadata from the top level
        if "metadata" in body:
            return body["metadata"]
        
        # If no metadata, try to extract from other fields
        metadata = {}
        
        # Extract chat_id from various possible locations
        chat_id = (
            body.get("chat_id") or 
            body.get("metadata", {}).get("chat_id") or
            str(uuid.uuid4())
        )
        
        # Handle temporary chats
        if chat_id == "local":
            session_id = body.get("session_id") or body.get("metadata", {}).get("session_id")
            if session_id:
                chat_id = f"temporary-session-{session_id}"
        
        metadata["chat_id"] = chat_id
        
        # Extract other useful metadata
        if "user_id" in body:
            metadata["user_id"] = body["user_id"]
        if "message_id" in body:
            metadata["message_id"] = body["message_id"]
        if "session_id" in body:
            metadata["session_id"] = body["session_id"]
        if "interface" in body:
            metadata["interface"] = body["interface"]
        if "type" in body:
            metadata["type"] = body["type"]
        
        # Extract model info
        if "model" in body:
            if isinstance(body["model"], dict):
                metadata["model"] = body["model"]
            elif isinstance(body["model"], str):
                metadata["model"] = {"id": body["model"]}
        
        return metadata

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if self.valves.debug:
            print(f"[DEBUG] Received request: {json.dumps(body, indent=2)}")

        self.log(f"Inlet function called with body keys: {list(body.keys())} and user: {user}")

        # Extract metadata using the helper function
        metadata = self._extract_metadata(body)
        chat_id = metadata.get("chat_id")

        # Ensure metadata is set in body
        body["metadata"] = metadata

        # Extract and store model information
        model_info = metadata.get("model", {})
        model_id = body.get("model")
        
        if isinstance(model_id, dict):
            # If model is a dict, extract the id
            actual_model_id = model_id.get("id", "unknown")
            model_name = model_id.get("name", "unknown")
        elif isinstance(model_id, str):
            actual_model_id = model_id
            model_name = model_info.get("name", "unknown") if isinstance(model_info, dict) else "unknown"
        else:
            actual_model_id = "unknown"
            model_name = "unknown"
        
        # Store model information for this chat
        if chat_id not in self.model_names:
            self.model_names[chat_id] = {"id": actual_model_id, "name": model_name}
        else:
            self.model_names[chat_id].update({"id": actual_model_id, "name": model_name})
            
        self.log(f"Stored model info - name: '{model_name}', id: '{actual_model_id}' for chat_id: {chat_id}")

        # Check for required keys - be more flexible about messages
        if "messages" not in body and "input" not in body:
            # Try to construct messages from the body if possible
            if "prompt" in body:
                body["messages"] = [{"role": "user", "content": body["prompt"]}]
            else:
                self.log("Warning: No messages or input found in body")
                body["messages"] = []

        user_email = user.get("email") if user else metadata.get("user_id")
        task_name = metadata.get("task", metadata.get("type", "user_response"))

        # Build tags
        tags_list = self._build_tags(task_name)

        if chat_id not in self.chat_traces:
            self.log(f"Creating new trace for chat_id: {chat_id}")

            trace_payload = {
                "name": f"chat:{chat_id}",
                "input": body.get("messages", body),
                "user_id": user_email,
                "metadata": metadata,
                "session_id": chat_id,
            }

            if tags_list:
                trace_payload["tags"] = tags_list

            if self.valves.debug:
                print(f"[DEBUG] Langfuse trace request: {json.dumps(trace_payload, indent=2)}")

            try:
                trace = self.langfuse.trace(**trace_payload)
                self.chat_traces[chat_id] = trace
            except Exception as e:
                self.log(f"Error creating trace: {e}")
                return body
        else:
            trace = self.chat_traces[chat_id]
            self.log(f"Reusing existing trace for chat_id: {chat_id}")
            if tags_list:
                try:
                    trace.update(tags=tags_list)
                except Exception as e:
                    self.log(f"Error updating trace tags: {e}")

        # Update metadata with type and interface
        metadata["type"] = task_name
        metadata["interface"] = metadata.get("interface", "open-webui")

        # Handle generation or event logging
        try:
            if task_name in self.GENERATION_TASKS:
                # Determine which model value to use based on the use_model_name valve
                model_value = model_name if self.valves.use_model_name_instead_of_id_for_generation else actual_model_id
                
                metadata["model_id"] = actual_model_id
                metadata["model_name"] = model_name
                
                generation_payload = {
                    "name": f"{task_name}:{str(uuid.uuid4())}",
                    "model": model_value,
                    "input": body.get("messages", []),
                    "metadata": metadata,
                }
                if tags_list:
                    generation_payload["tags"] = tags_list

                if self.valves.debug:
                    print(f"[DEBUG] Langfuse generation request: {json.dumps(generation_payload, indent=2)}")

                trace.generation(**generation_payload)
            else:
                # Log as an event
                event_payload = {
                    "name": f"{task_name}:{str(uuid.uuid4())}",
                    "metadata": metadata,
                    "input": body.get("messages", body),
                }
                if tags_list:
                    event_payload["tags"] = tags_list

                if self.valves.debug:
                    print(f"[DEBUG] Langfuse event request: {json.dumps(event_payload, indent=2)}")

                trace.event(**event_payload)
        except Exception as e:
            self.log(f"Error logging to Langfuse: {e}")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log(f"Outlet function called with body keys: {list(body.keys())}")
        try:
            # Extract chat_id from multiple possible locations
            chat_id = (
                body.get("chat_id") or
                body.get("metadata", {}).get("chat_id")
            )

            # Handle temporary chats
            if chat_id == "local":
                session_id = body.get("session_id") or body.get("metadata", {}).get("session_id")
                if session_id:
                    chat_id = f"temporary-session-{session_id}"

            if not chat_id:
                self.log("Warning: No chat_id found in outlet")
                return body

            metadata = body.get("metadata", {})
            task_name = metadata.get("task", metadata.get("type", "llm_response"))

            # Build tags
            tags_list = self._build_tags(task_name)

            if chat_id not in self.chat_traces:
                self.log(f"[WARNING] No matching trace found for chat_id: {chat_id}")
                return body

            trace = self.chat_traces[chat_id]

            # Extract assistant message and usage info
            messages = body.get("messages", [])
            assistant_message = get_last_assistant_message(messages) if messages else ""
            assistant_message_obj = get_last_assistant_message_obj(messages) if messages else {}

            usage = None
            if assistant_message_obj:
                info = assistant_message_obj.get("usage", {})
                if isinstance(info, dict):
                    input_tokens = info.get("prompt_eval_count") or info.get("prompt_tokens")
                    output_tokens = info.get("eval_count") or info.get("completion_tokens")
                    if input_tokens is not None and output_tokens is not None:
                        usage = {
                            "input": input_tokens,
                            "output": output_tokens,
                            "unit": "TOKENS",
                        }
                        self.log(f"Usage data extracted: {usage}")

            # Update the trace output
            try:
                trace.update(output=assistant_message or body)
            except Exception as e:
                self.log(f"Error updating trace output: {e}")

            metadata["type"] = task_name
            metadata["interface"] = metadata.get("interface", "open-webui")

            # Handle generation completion or event logging
            try:
                if task_name in self.GENERATION_TASKS:
                    model_id = self.model_names.get(chat_id, {}).get("id", "unknown")
                    model_name = self.model_names.get(chat_id, {}).get("name", "unknown")
                    
                    model_value = model_name if self.valves.use_model_name_instead_of_id_for_generation else model_id
                    
                    metadata["model_id"] = model_id
                    metadata["model_name"] = model_name
                    
                    generation_payload = {
                        "name": f"{task_name}:{str(uuid.uuid4())}",
                        "model": model_value,
                        "input": messages,
                        "output": assistant_message,
                        "metadata": metadata,
                    }
                    
                    if usage:
                        generation_payload["usage"] = usage
                    if tags_list:
                        generation_payload["tags"] = tags_list

                    if self.valves.debug:
                        print(f"[DEBUG] Langfuse generation end request: {json.dumps(generation_payload, indent=2)}")

                    trace.generation().end(**generation_payload)
                    self.log(f"Generation ended for chat_id: {chat_id}")
                else:
                    # Log as an event
                    event_payload = {
                        "name": f"{task_name}:{str(uuid.uuid4())}",
                        "metadata": metadata,
                        "input": messages,
                        "output": assistant_message,
                    }
                    if usage:
                        event_payload["metadata"]["usage"] = usage
                    if tags_list:
                        event_payload["tags"] = tags_list

                    if self.valves.debug:
                        print(f"[DEBUG] Langfuse event end request: {json.dumps(event_payload, indent=2)}")

                    trace.event(**event_payload)
                    self.log(f"Event logged for chat_id: {chat_id}")
            except Exception as e:
                self.log(f"Error completing Langfuse logging: {e}")

            return body
        except Exception as e:
            self.log(f"Error in outlet: {e}")
            return body
        finally:
            try:
                if self.langfuse:
                    self.log("Forcing Langfuse flush in finally block.")
                    self.langfuse.flush()
            except Exception as e:
                print(f"[ERROR] Failed to flush Langfuse: {e}")
