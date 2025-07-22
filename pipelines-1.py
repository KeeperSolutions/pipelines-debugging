"""
title: Langfuse Filter Pipeline
 new
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
                "debug": os.getenv("DEBUG_MODE", "true").lower() == "true",  # Force debug mode
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
            print(f"[DEBUG][Langfuse Pipeline] {message}")

    async def on_startup(self):
        self.log(f"on_startup triggered for {self.name}")
        self.set_langfuse()

    async def on_shutdown(self):
        self.log(f"on_shutdown triggered for {self.name}")
        if self.langfuse:
            self.langfuse.flush()

    async def on_valves_updated(self):
        self.log("Valves updated, resetting Langfuse client.")
        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.log(f"Initializing Langfuse with host: {self.valves.host}")
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=self.valves.debug,
            )
            self.langfuse.auth_check()
            self.log("Langfuse client initialized successfully.")
        except UnauthorizedError as e:
            self.log(f"UnauthorizedError: Invalid Langfuse credentials - {str(e)}")
            raise
        except Exception as e:
            self.log(f"Langfuse initialization error: {str(e)}")
            raise

    def _build_tags(self, task_name: str) -> list:
        tags_list = []
        if self.valves.insert_tags:
            tags_list.append("open-webui")
            if task_name not in ["user_response", "llm_response"]:
                tags_list.append(task_name)
        self.log(f"Built tags: {tags_list}")
        return tags_list

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log(f"Inlet called with body: {json.dumps(body, indent=2)}")
        self.log(f"User: {user}")

        metadata = body.get("metadata", {})
        chat_id = metadata.get("chat_id", str(uuid.uuid4()))

        # Handle temporary chats
        if chat_id == "local":
            session_id = metadata.get("session_id", str(uuid.uuid4()))
            chat_id = f"temporary-session-{session_id}"
            self.log(f"Generated temporary chat_id: {chat_id}")

        metadata["chat_id"] = chat_id
        body["metadata"] = metadata

        # Extract and store model info
        model_info = metadata.get("model", {})
        model_id = body.get("model")
        if chat_id not in self.model_names:
            self.model_names[chat_id] = {"id": model_id}
        else:
            self.model_names[chat_id]["id"] = model_id
        if isinstance(model_info, dict) and "name" in model_info:
            self.model_names[chat_id]["name"] = model_info["name"]
            self.log(f"Stored model info - name: '{model_info.get('name', 'unknown')}', id: '{model_id}' for chat_id: {chat_id}")

        required_keys = ["model", "messages"]
        missing_keys = [key for key in required_keys if key not in body]
        if missing_keys:
            error_message = f"Error: Missing keys in request body: {', '.join(missing_keys)}"
            self.log(error_message)
            raise ValueError(error_message)

        user_email = user.get("email") if user else None
        task_name = metadata.get("task", "user_response")
        self.log(f"Task name: {task_name}")

        tags_list = self._build_tags(task_name)

        if chat_id not in self.chat_traces:
            self.log(f"Creating new trace for chat_id: {chat_id}")
            trace_payload = {
                "name": f"chat:{chat_id}",
                "input": body,
                "user_id": user_email,
                "metadata": metadata,
                "session_id": chat_id,
            }
            if tags_list:
                trace_payload["tags"] = tags_list

            self.log(f"Langfuse trace request: {json.dumps(trace_payload, indent=2)}")
            try:
                trace = self.langfuse.trace(**trace_payload)
                self.chat_traces[chat_id] = trace
            except Exception as e:
                self.log(f"Failed to create Langfuse trace: {str(e)}")
                raise
        else:
            trace = self.chat_traces[chat_id]
            self.log(f"Reusing existing trace for chat_id: {chat_id}")
            if tags_list:
                trace.update(tags=tags_list)

        metadata["type"] = task_name
        metadata["interface"] = "open-webui"

        if task_name in self.GENERATION_TASKS:
            model_id = self.model_names.get(chat_id, {}).get("id", body["model"])
            model_name = self.model_names.get(chat_id, {}).get("name", "unknown")
            model_value = model_name if self.valves.use_model_name_instead_of_id_for_generation else model_id
            metadata["model_id"] = model_id
            metadata["model_name"] = model_name

            generation_payload = {
                "name": f"{task_name}:{str(uuid.uuid4())}",
                "model": model_value,
                "input": body["messages"],
                "metadata": metadata,
            }
            if tags_list:
                generation_payload["tags"] = tags_list

            self.log(f"Langfuse generation request: {json.dumps(generation_payload, indent=2)}")
            try:
                trace.generation(**generation_payload)
            except Exception as e:
                self.log(f"Failed to create Langfuse generation: {str(e)}")
                raise

        else:
            event_payload = {
                "name": f"{task_name}:{str(uuid.uuid4())}",
                "metadata": metadata,
                "input": body["messages"],
            }
            if tags_list:
                event_payload["tags"] = tags_list

            self.log(f"Langfuse event request: {json.dumps(event_payload, indent=2)}")
            try:
                trace.event(**event_payload)
            except Exception as e:
                self.log(f"Failed to create Langfuse event: {str(e)}")
                raise

        self.log("Inlet processing completed")
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log(f"Outlet called with body: {json.dumps(body, indent=2)}")
        self.log(f"User: {user}")

        chat_id = body.get("chat_id")
        if chat_id == "local":
            session_id = body.get("session_id", str(uuid.uuid4()))
            chat_id = f"temporary-session-{session_id}"
            self.log(f"Generated temporary chat_id: {chat_id}")

        metadata = body.get("metadata", {})
        task_name = metadata.get("task", "llm_response")
        self.log(f"Task name: {task_name}")

        tags_list = self._build_tags(task_name)

        if chat_id not in self.chat_traces:
            self.log(f"[WARNING] No matching trace for chat_id: {chat_id}, re-running inlet")
            return await self.inlet(body, user)

        trace = self.chat_traces[chat_id]

        assistant_message = get_last_assistant_message(body["messages"])
        assistant_message_obj = get_last_assistant_message_obj(body["messages"])

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

        trace.update(output=assistant_message)

        metadata["type"] = task_name
        metadata["interface"] = "open-webui"

        if task_name in self.GENERATION_TASKS:
            model_id = self.model_names.get(chat_id, {}).get("id", body.get("model"))
            model_name = self.model_names.get(chat_id, {}).get("name", "unknown")
            model_value = model_name if self.valves.use_model_name_instead_of_id_for_generation else model_id
            metadata["model_id"] = model_id
            metadata["model_name"] = model_name

            generation_payload = {
                "name": f"{task_name}:{str(uuid.uuid4())}",
                "model": model_value,
                "input": body["messages"],
                "metadata": metadata,
                "usage": usage,
            }
            if tags_list:
                generation_payload["tags"] = tags_list

            self.log(f"Langfuse generation end request: {json.dumps(generation_payload, indent=2)}")
            try:
                trace.generation().end(**generation_payload)
                self.log(f"Generation ended for chat_id: {chat_id}")
            except Exception as e:
                self.log(f"Failed to end Langfuse generation: {str(e)}")
                raise
        else:
            event_payload = {
                "name": f"{task_name}:{str(uuid.uuid4())}",
                "metadata": metadata,
                "input": body["messages"],
            }
            if usage:
                event_payload["metadata"]["usage"] = usage
            if tags_list:
                event_payload["tags"] = tags_list

            self.log(f"Langfuse event end request: {json.dumps(event_payload, indent=2)}")
            try:
                trace.event(**event_payload)
                self.log(f"Event logged for chat_id: {chat_id}")
            except Exception as e:
                self.log(f"Failed to log Langfuse event: {str(e)}")
                raise

        self.log("Outlet processing completed")
        return body