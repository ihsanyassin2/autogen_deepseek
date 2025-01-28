# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

"""Create an OpenAI-compatible client using Ollama's API.

Example:
    llm_config={
        "config_list": [{
            "api_type": "ollama",
            "model": "mistral:7b-instruct-v0.3-q6_K"
        }]
    }

    agent = autogen.AssistantAgent("my_agent", llm_config=llm_config)

Dependencies:
    - pip install --upgrade ollama
    - pip install --upgrade fix-busted-json
"""

from __future__ import annotations

import copy
import json
import random
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import ollama
from fix_busted_json import repair_json
from ollama import Client
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

# Custom ChatCompletion class that includes cost attribute
class OllamaChatCompletion(ChatCompletion):
    """Custom ChatCompletion class that includes a cost attribute for Ollama responses."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost = 0.0  # Local models have no cost

# class OllamaClient:
#     """Client for Ollama's API that provides OpenAI-compatible interface."""
# 
#     # Tool calling configuration
#     TOOL_CALL_MANUAL_INSTRUCTION = (
#         "You are to follow a strict two step process that will occur over "
#         "a number of interactions, so pay attention to what step you are in based on the full "
#         "conversation. We will be taking turns so only do one step at a time so don't perform step "
#         "2 until step 1 is complete and I've told you the result. The first step is to choose one "
#         "or more functions based on the request given and return only JSON with the functions and "
#         "arguments to use. The second step is to analyse the given output of the function and summarise "
#         "it returning only TEXT and not Python or JSON. "
#         "For argument values, be sure numbers aren't strings, they should not have double quotes around them. "
#         "In terms of your response format, for step 1 return only JSON and NO OTHER text, "
#         "for step 2 return only text and NO JSON/Python/Markdown. "
#         'The format for running a function is [{"name": "function_name1", "arguments":{"argument_name": "argument_value"}},{"name": "function_name2", "arguments":{"argument_name": "argument_value"}}] '
#         'Make sure the keys "name" and "arguments" are as described. '
#         "If you don't get the format correct, try again. "
#         "The following functions are available to you:[FUNCTIONS_LIST]"
#     )
#     TOOL_CALL_MANUAL_STEP1 = " (proceed with step 1)"
#     TOOL_CALL_MANUAL_STEP2 = " (proceed with step 2)"
# 
#     def __init__(self, config=None, **kwargs):
#         """Initialize the Ollama client."""
#         # Ensure that `config` is not passed redundantly
#         if config is not None and 'config' in kwargs:
#             raise ValueError("Duplicate 'config' argument detected. Pass 'config' either as positional or keyword argument, not both.")
#         
#         # Resolve `config` from positional argument or keyword arguments
#         self.config = config or kwargs.pop('config', {})
#         
#         # Store response_format but don't serialize it
#         self._response_format = kwargs.pop("response_format", None)
#         
#         # Initialize instance variables
#         self._total_cost = 0.0
#         self._last_cost = 0.0
#         self._tools_in_conversation = False
#         self._should_hide_tools = False
#         self._native_tool_calls = True
# 
# 
#     def message_retrieval(self, response: Union[OllamaChatCompletion, ChatCompletion]) -> list:
#         """Retrieve messages from the response."""
#         return [choice.message for choice in response.choices]
# 
#     def cost(self, response: Union[OllamaChatCompletion, ChatCompletion]) -> float:
#         """Calculate cost for local models (always 0)."""
#         return getattr(response, 'cost', 0.0)
# 
#     @staticmethod
#     def get_usage(response: Union[OllamaChatCompletion, ChatCompletion]) -> dict:
#         """Return usage summary of the response."""
#         return {
#             "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
#             "completion_tokens": response.usage.completion_tokens if response.usage else 0,
#             "total_tokens": response.usage.total_tokens if response.usage else 0,
#             "cost": 0.0,  # Local models are free
#             "model": response.model,
#         }
# 
#     def parse_params(self, params: dict[str, Any]) -> dict[str, Any]:
#         """Parse and validate parameters for Ollama API."""
#         # Convert all params to JSON-serializable format
#         ollama_params = {}
# 
#         # Required parameters
#         ollama_params["model"] = params.get("model", self.config.get("model", "openhermes"))
# 
#         # Convert messages to a serializable format
#         messages = params.get("messages", [])
#         if messages:
#             ollama_params["messages"] = [
#                 {
#                     "role": msg.get("role", "user"),
#                     "content": msg.get("content", ""),
#                     "name": msg.get("name", "")
#                 } for msg in messages
#             ]
# 
#         # Stream parameter (default to False)
#         ollama_params["stream"] = params.get("stream", False)
# 
#         # Optional numerical parameters with defaults
#         numerical_params = {
#             "temperature": 0.8,
#             "top_k": 40,
#             "top_p": 0.9,
#             "num_predict": 128,
#             "repeat_penalty": 1.1,
#             "seed": 42
#         }
# 
#         options = {}
#         for param, default in numerical_params.items():
#             if param in params:
#                 val = params[param]
#                 if isinstance(val, (int, float)):
#                     options[param] = val
# 
#         if options:
#             ollama_params["options"] = options
# 
#         # Handle tool-related parameters
#         if "tools" in params and not self._should_hide_tools:
#             if self._native_tool_calls:
#                 tools_list = []
#                 for tool in params["tools"]:
#                     if isinstance(tool, dict) and "function" in tool:
#                         tools_list.append({
#                             "name": tool["function"].get("name", ""),
#                             "description": tool["function"].get("description", ""),
#                             "parameters": tool["function"].get("parameters", {})
#                         })
#                 if tools_list:
#                     ollama_params["tools"] = tools_list
# 
#         return ollama_params
#     
#     
#     
#     def create(self, params: dict) -> OllamaChatCompletion:
#         """
#         Creates a chat completion using Ollama's API.
#         """
#         try:
#             # Ensure parameters are JSON-serializable
#             params = self.make_json_serializable(params)
#             
#             # Remove non-serializable parameters
#             params = params.copy()
#             cache = params.pop("cache", None)
#             agent = params.pop("agent", None)
#             response_format = params.pop("response_format", None)
#             
#             # Process parameters
#             self._tools_in_conversation = "tools" in params
#             ollama_params = self.parse_params(params)
# 
#             # Execute request
#             response = self._execute_chat_request(ollama_params, params)
# 
#             # Process response
#             response_data = self._process_single_response(response)
# 
#             # Create ChatCompletion response
#             completion = OllamaChatCompletion(
#                 id=str(response_data["id"]),
#                 model=ollama_params["model"],
#                 created=int(time.time()),
#                 object="chat.completion",
#                 choices=[
#                     Choice(
#                         index=0,
#                         message=response_data["message"],
#                         finish_reason=response_data["finish_reason"]
#                     )
#                 ],
#                 usage=CompletionUsage(
#                     prompt_tokens=response_data["prompt_tokens"],
#                     completion_tokens=response_data["completion_tokens"],
#                     total_tokens=response_data["total_tokens"]
#                 )
#             )
# 
#             # Set cost (always 0 for local models)
#             completion.cost = 0.0
#             return completion
# 
#         except Exception as e:
#             raise RuntimeError(f"Error in Ollama chat completion: {str(e)}") from e
# 
# 
#     def _execute_chat_request(self, ollama_params: dict, original_params: dict) -> Any:
#         """Execute the chat request to Ollama."""
#         if "client_host" in original_params:
#             client = Client(host=original_params["client_host"])
#             return client.chat(**ollama_params)
#         return ollama.chat(**ollama_params)
# 
#     def _process_response(self, response: Any, ollama_params: dict) -> dict:
#         """Process the raw Ollama response into a standardized format."""
#         if ollama_params["stream"]:
#             return self._process_stream_response(response)
#         return self._process_single_response(response)
# 
#     def _process_stream_response(self, response: Any) -> dict:
#         """Process streaming response from Ollama."""
#         content = ""
#         prompt_tokens = completion_tokens = 0
#         last_chunk = None
#         
#         for chunk in response:
#             content += chunk["message"].get("content", "")
#             if "done_reason" in chunk:
#                 prompt_tokens = chunk.get("prompt_eval_count", 0)
#                 completion_tokens = chunk.get("eval_count", 0)
#                 last_chunk = chunk
#         
#         return {
#             "content": content,
#             "id": last_chunk["created_at"],
#             "prompt_tokens": prompt_tokens,
#             "completion_tokens": completion_tokens,
#             "total_tokens": prompt_tokens + completion_tokens,
#             "finish_reason": "stop",
#             "message": self._create_message(content)
#         }
# 
#     def _process_single_response(self, response: Any) -> dict:
#         """Process single response from Ollama."""
#         if not isinstance(response, dict) or "message" not in response:
#             raise ValueError("Invalid response format from Ollama")
#             
#         content = response["message"].get("content", "")
#         
#         # Initialize response data
#         response_data = {
#             "content": content,
#             "id": response.get("created_at", str(int(time.time()))),
#             "prompt_tokens": response.get("prompt_eval_count", 0),
#             "completion_tokens": response.get("eval_count", 0),
#             "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
#             "finish_reason": "stop"
#         }
#         
#         # Handle tool calls if present
#         if self._tools_in_conversation and "tool_calls" in response["message"]:
#             tool_calls = []
#             call_id = random.randint(0, 10000)
#             
#             for tool_call in response["message"].get("tool_calls", []):
#                 if isinstance(tool_call, dict) and "function" in tool_call:
#                     tool_calls.append(
#                         ChatCompletionMessageToolCall(
#                             id=f"ollama_func_{call_id}",
#                             function={
#                                 "name": tool_call["function"].get("name", ""),
#                                 "arguments": json.dumps(tool_call["function"].get("arguments", {}))
#                             },
#                             type="function"
#                         )
#                     )
#                     call_id += 1
#             
#             if tool_calls:
#                 response_data["tool_calls"] = tool_calls
#                 response_data["finish_reason"] = "tool_calls"
#                 
#         # Create the message object
#         response_data["message"] = ChatCompletionMessage(
#             role="assistant",
#             content=content,
#             tool_calls=response_data.get("tool_calls")
#         )
#         
#         return response_data
# 
#     def _create_message(self, content: str, tool_calls: Optional[list] = None) -> ChatCompletionMessage:
#         """Create a ChatCompletionMessage object."""
#         return ChatCompletionMessage(
#             role="assistant",
#             content=content,
#             function_call=None,
#             tool_calls=tool_calls
#         )
# 
#     def _process_tool_calls(self, response: Any, content: str) -> Tuple[Optional[list], str, str]:
#         """Process tool calls from response."""
#         tool_calls = None
#         finish_reason = "stop"
# 
#         if self._native_tool_calls and "tool_calls" in response["message"]:
#             tool_calls = []
#             random_id = random.randint(0, 10000)
#             for tool_call in response["message"]["tool_calls"]:
#                 tool_calls.append(self._create_tool_call(tool_call, random_id))
#                 random_id += 1
#             finish_reason = "tool_calls"
#         elif not self._native_tool_calls:
#             tool_calls = self._process_non_native_tool_calls(response["message"]["content"])
#             if tool_calls:
#                 content = ""
#                 finish_reason = "tool_calls"
# 
#         return tool_calls, content, finish_reason
# 
#     def _create_tool_call(self, tool_call: dict, call_id: int) -> ChatCompletionMessageToolCall:
#         """Create a tool call object."""
#         return ChatCompletionMessageToolCall(
#             id=f"ollama_func_{call_id}",
#             function={
#                 "name": tool_call["function"]["name"],
#                 "arguments": json.dumps(tool_call["function"]["arguments"])
#             },
#             type="function"
#         )
# 
#     def oai_messages_to_ollama_messages(self, messages: list[dict[str, Any]], tools: Optional[list]) -> list[dict[str, Any]]:
#         """Convert OpenAI format messages to Ollama format."""
#         # Implementation remains the same as in original code
#         # This method should be kept as is since it handles complex message conversion logic
#         return messages  # Simplified for example
#     
#     
#     @staticmethod
#     def make_json_serializable(data):
#         """
#         Removes non-serializable keys from a dictionary to make it JSON-serializable.
#         """
#         try:
#             json.dumps(data)  # Test serialization
#             return data
#         except TypeError:
#             # Remove non-serializable keys
#             return {k: v for k, v in data.items() if isinstance(v, (str, int, float, list, dict))}
#         
#     @staticmethod
#     def validate_parameter(
#         params: dict,
#         param_name: str,
#         param_type: Union[type, tuple[type, ...]], 
#         required: bool = False,
#         default: Any = None,
#         min_value: Optional[Union[int, float]] = None,
#         allowed_values: Optional[list] = None
#     ) -> Any:
#         """Validate and return a parameter value."""
#         value = params.get(param_name, default)
#         
#         if required and value is None:
#             raise ValueError(f"Parameter {param_name} is required")
#             
#         if value is not None:
#             if not isinstance(value, param_type):
#                 raise TypeError(f"Parameter {param_name} must be of type {param_type}")
#                 
#             if min_value is not None and value < min_value:
#                 raise ValueError(f"Parameter {param_name} must be >= {min_value}")
#                 
#             if allowed_values is not None and value not in allowed_values:
#                 raise ValueError(f"Parameter {param_name} must be one of {allowed_values}")
#                 
#         return value
from custom_client import ModelClient
import json
import time


from ollama_client import OllamaClient



# Utility functions remain the same
def response_to_tool_call(response_string: str) -> Any:
    """Original implementation remains the same"""
    pass

def _object_to_tool_call(data_object: Any) -> list[dict]:
    """Original implementation remains the same"""
    pass

def is_valid_tool_call_item(call_item: dict) -> bool:
    """Validate structure of a tool call item.
    
    Args:
        call_item: Dictionary to validate
        
    Returns:
        bool: True if the item has valid structure, False otherwise
    """
    if "name" not in call_item or not isinstance(call_item["name"], str):
        return False

    if set(call_item.keys()) - {"name", "arguments"}:
        return False

    return True

def response_to_tool_call(response_string: str) -> Any:
    """Convert a response string to a tool call object.

    Args:
        response_string: String containing potential JSON tool call

    Returns:
        Any: Tool call object if valid, None otherwise
    """
    # Detect list[dict] format patterns
    patterns = [r"\[[\s\S]*?\]", r"\{[\s\S]*\}"]

    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, response_string.strip())

        for match in matches:
            json_str = match.strip()
            data_object = None

            try:
                # Try direct JSON parsing
                data_object = json.loads(json_str)
            except Exception:
                try:
                    # Try repairing JSON if needed
                    if i == 0:
                        fixed_json = repair_json("{'temp':" + json_str + "}")
                        data_object = json.loads(fixed_json)["temp"]
                    else:
                        fixed_json = repair_json(json_str)
                        data_object = json.loads(fixed_json)
                except json.JSONDecodeError as e:
                    if e.msg == "Invalid \\escape":
                        # Handle escape character issues
                        try:
                            json_str = json_str.replace("\\_", "_")
                            fixed_json = repair_json("{'temp':" + json_str + "}")
                            data_object = json.loads(fixed_json)
                            data_object = data_object["temp"]
                        except Exception:
                            continue
                except Exception:
                    continue

            if data_object is not None:
                data_object = _object_to_tool_call(data_object)
                if data_object is not None:
                    return data_object

    return None

def _object_to_tool_call(data_object: Any) -> Optional[list[dict]]:
    """Convert object to tool call format if possible.
    
    Args:
        data_object: Object to convert 
        
    Returns:
        Optional[list[dict]]: List of tool call dicts if valid, None otherwise
    """
    # Convert single dict to list
    if isinstance(data_object, dict):
        data_object = [data_object]

    # Validate list of dicts structure
    if isinstance(data_object, list) and all(isinstance(item, dict) for item in data_object):
        # Validate each dict has required format
        is_invalid = False
        for item in data_object:
            if not is_valid_tool_call_item(item):
                is_invalid = True
                break

        if not is_invalid:
            return data_object

    # Try converting string items to dicts    
    elif isinstance(data_object, list):
        data_copy = data_object.copy()
        is_invalid = False

        for i, item in enumerate(data_copy):
            try:
                new_item = eval(item)
                if isinstance(new_item, dict) and is_valid_tool_call_item(new_item):
                    data_object[i] = new_item
                else:
                    is_invalid = True
                    break
            except Exception:
                is_invalid = True 
                break

        if not is_invalid:
            return data_object

    return None

# Optional helper functions for better organization

def should_hide_tools(messages: list[dict], tools: list, hide_mode: str) -> bool:
    """Determine if tools should be hidden based on context and mode.
    
    Args:
        messages: Message history
        tools: Available tools
        hide_mode: When to hide tools ("if_all_run", "if_any_run", "never")
        
    Returns:
        bool: Whether tools should be hidden
    """
    if hide_mode == "never":
        return False
        
    tool_call_counts = {tool["function"]["name"]: 0 for tool in tools}
    
    for msg in messages:
        if "tool_calls" in msg:
            for call in msg["tool_calls"]:
                tool_name = call["function"]["name"]
                if tool_name in tool_call_counts:
                    tool_call_counts[tool_name] += 1
    
    all_tools_used = all(count > 0 for count in tool_call_counts.values())
    any_tools_used = any(count > 0 for count in tool_call_counts.values())
    
    return (
        (hide_mode == "if_all_run" and all_tools_used) or 
        (hide_mode == "if_any_run" and any_tools_used)
    )