from custom_model_client import ModelClient
from typing import Any, Dict, List, Union
import json
import time
from datetime import datetime

class DictObject(dict):
    """A dictionary-like object that supports attribute-style access."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(self)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

class OllamaClient(ModelClient):
    """
    Client for Ollama's API that provides OpenAI-compatible interface.
    Implements the ModelClient protocol.
    """

    def __init__(self, config=None, **kwargs):
        """Initialize the Ollama client."""
        self.config = config or kwargs.get("config", {})
        self.base_url = self.config.get("api_base", "http://localhost:11434/api")
        self.model = self.config.get("model", "deepseek-r1")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 4096)

    def create(self, params: Dict[str, Any]) -> ModelClient.ModelClientResponseProtocol:
        """Creates a chat completion using Ollama's API."""
        try:
            # Clean and prepare parameters
            params = self.make_json_serializable(params)
            ollama_params = self.parse_params(params)
            
            # Execute request
            response = self._execute_chat_request(ollama_params)
            
            # Handle both regular and stream responses
            processed_response = self._process_response(response, ollama_params)

            # Add cost attributes (local models are free)
            processed_response['cost'] = 0.0
            processed_response['total_cost'] = 0.0

            # Convert the response and nested structures to DictObject
            result = self._convert_to_dict_object(processed_response)
            return result

        except Exception as e:
            raise RuntimeError(f"Error in Ollama chat completion: {str(e)}") from e

    def message_retrieval(
        self, 
        response: ModelClient.ModelClientResponseProtocol
    ) -> List[Dict[str, Any]]:
        """Retrieve messages from the response."""
        try:
            if hasattr(response, 'choices') and isinstance(response.choices, list):
                return [choice.get("message", {}) for choice in response.choices]
            return []
        except Exception:
            raise AttributeError("Response does not contain valid 'choices'.")

    def cost(self, response: ModelClient.ModelClientResponseProtocol) -> float:
        """Calculate cost for local models (always 0)."""
        return 0.0

    @staticmethod
    def get_usage(response: Union[dict, ModelClient.ModelClientResponseProtocol]) -> dict:
        """Return usage summary of the response."""
        try:
            if isinstance(response, dict):
                usage = response.get('usage', {})
                model = response.get('model', 'unknown')
            else:
                usage = getattr(response, 'usage', {})
                model = getattr(response, 'model', 'unknown')

            return {
                "prompt_tokens": getattr(usage, 'prompt_tokens', 0) if not isinstance(usage, dict) else usage.get('prompt_tokens', 0),
                "completion_tokens": getattr(usage, 'completion_tokens', 0) if not isinstance(usage, dict) else usage.get('completion_tokens', 0),
                "total_tokens": getattr(usage, 'total_tokens', 0) if not isinstance(usage, dict) else usage.get('total_tokens', 0),
                "cost": 0.0,
                "total_cost": 0.0,
                "model": model
            }
        except Exception:
            # Fallback to default values if any error occurs
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "total_cost": 0.0,
                "model": "unknown"
            }

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate parameters for Ollama API."""
        return {
            "model": params.get("model", self.model),
            "messages": [
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                    "name": msg.get("name", "") if "name" in msg else None
                }
                for msg in params.get("messages", [])
                if msg.get("content") is not None  # Skip messages with None content
            ],
            "stream": params.get("stream", False),
            "options": {
                "temperature": params.get("temperature", self.temperature),
                "num_predict": params.get("max_tokens", self.max_tokens),
                "top_k": params.get("top_k", 40),
                "top_p": params.get("top_p", 0.9),
                "repeat_penalty": params.get("repeat_penalty", 1.1),
                "seed": self.config.get("seed", 42),
            },
        }

    def _execute_chat_request(self, ollama_params: Dict[str, Any]) -> Any:
        """Execute the chat request to Ollama."""
        import ollama
        response = ollama.chat(**ollama_params)
        return response

    def _process_response(
        self, response: Any, ollama_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the response into a standardized format."""
        if ollama_params.get("stream", False):
            return self._process_stream_response(response)
        return self._process_single_response(response)

    def _process_single_response(self, response: Any) -> Dict[str, Any]:
        """Process single response from Ollama."""
        try:
            # Handle both dict and object-style responses
            if isinstance(response, dict):
                message = response.get("message", {})
                content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
                eval_counts = {
                    "prompt": response.get("prompt_eval_count", 0),
                    "completion": response.get("eval_count", 0)
                }
            else:
                message = getattr(response, "message", {})
                content = getattr(message, "content", "")
                eval_counts = {
                    "prompt": getattr(response, "prompt_eval_count", 0),
                    "completion": getattr(response, "eval_count", 0)
                }

            created_timestamp = int(time.time())
            
            return {
                "id": str(created_timestamp),
                "model": self.model,
                "created": created_timestamp,
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }
                ],
                "usage": {
                    "prompt_tokens": eval_counts["prompt"],
                    "completion_tokens": eval_counts["completion"],
                    "total_tokens": eval_counts["prompt"] + eval_counts["completion"],
                },
                "cost": 0.0,
                "total_cost": 0.0,
            }
        except Exception as e:
            raise ValueError(f"Failed to process Ollama response: {str(e)}")

    def _process_stream_response(self, response: Any) -> Dict[str, Any]:
        """Process streaming response from Ollama."""
        content = ""
        prompt_tokens = completion_tokens = 0
        last_chunk = None

        try:
            for chunk in response:
                # Extract content from chunk
                if isinstance(chunk, dict):
                    message = chunk.get("message", {})
                    chunk_content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
                else:
                    chunk_content = getattr(getattr(chunk, "message", {}), "content", "")
                
                content += chunk_content

                # Update token counts if available
                if isinstance(chunk, dict):
                    if "done_reason" in chunk:
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        completion_tokens = chunk.get("eval_count", 0)
                        last_chunk = chunk
                else:
                    if hasattr(chunk, "done_reason"):
                        prompt_tokens = getattr(chunk, "prompt_eval_count", 0)
                        completion_tokens = getattr(chunk, "eval_count", 0)
                        last_chunk = chunk

            created_timestamp = int(time.time())
            
            return {
                "id": str(created_timestamp),
                "model": self.model,
                "created": created_timestamp,
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "cost": 0.0,
                "total_cost": 0.0,
            }
        except Exception as e:
            raise ValueError(f"Failed to process stream response: {str(e)}")

    def _convert_to_dict_object(self, data: Dict[str, Any]) -> DictObject:
        """Convert a dictionary and its nested structures to DictObject."""
        result = DictObject(data)

        if 'choices' in result:
            result['choices'] = [DictObject(choice) for choice in result['choices']]
            for choice in result['choices']:
                if 'message' in choice:
                    choice['message'] = DictObject(choice['message'])

        if 'usage' in result:
            result['usage'] = DictObject(result['usage'])

        return result

    def make_json_serializable(self, data: Any) -> Any:
        """Make a dictionary JSON-serializable by removing non-serializable values."""
        if isinstance(data, dict):
            return {k: self.make_json_serializable(v) for k, v in data.items() if v is not None}
        elif isinstance(data, list):
            return [self.make_json_serializable(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        return str(data)