from __future__ import annotations

import inspect
import logging
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional, Protocol, Union, runtime_checkable
from pydantic import BaseModel, schema_json_of
from custom_model_client import ModelClient



# Remove all non-ollama imports and keep only essential ones
try:
    print("Attempting to import ollama...")
    from ollama import (
        RequestError as ollama_RequestError,
        ResponseError as ollama_ResponseError,
    )
    print("Ollama import successful")
    print("Attempting to import custom_ollama_client...")
    from custom_ollama_client import OllamaClient
    print("custom_ollama_client import successful")
    ollama_import_exception = None
except ImportError as e:
    print(f"Import failed with error: {str(e)}")
    print(f"Error type: {type(e)}")
    print(f"Error details: {e.__dict__}")
    ollama_RequestError = ollama_ResponseError = Exception
    ollama_import_exception = e

from autogen.cache import Cache
from autogen.io.base import IOStream
from autogen.logger.logger_utils import get_current_ts
from autogen.oai.client_utils import logging_formatter
from autogen.oai.openai_utils import get_key, is_valid_api_key

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    _ch = logging.StreamHandler(stream=sys.stdout)
    _ch.setFormatter(logging_formatter)
    logger.addHandler(_ch)

LEGACY_DEFAULT_CACHE_SEED = 41
LEGACY_CACHE_DIR = ".cache"

@runtime_checkable
class FormatterProtocol(Protocol):
    def format(self) -> str: ...

class PlaceHolderClient:
    def __init__(self, config):
        self.config = config

class OpenAIWrapper:
    extra_kwargs = {
        "agent",
        "cache",
        "cache_seed",
        "filter_func",
        "allow_format_str_template",
        "context",
        "api_version",
        "api_type",
        "tags",
        "price",
    }

    def __init__(
        self,
        *,
        config_list: Optional[list[dict[str, Any]]] = None,
        **base_config: Any,
    ):
        self._clients: list[ModelClient] = []
        self._config_list: list[dict[str, Any]] = []
        self.wrapper_id = id(self)
        self.total_usage_summary: Optional[dict[str, Any]] = None
        self.actual_usage_summary: Optional[dict[str, Any]] = None

        if config_list:
            for config in config_list:
                if config.get("api_type", "").startswith("ollama"):
                    self._register_ollama_client(config.copy())
                    self._config_list.append(config)
        elif base_config.get("api_type", "").startswith("ollama"):
            self._register_ollama_client(base_config)
            self._config_list = [base_config]

    def _register_ollama_client(self, config: dict[str, Any]) -> None:
        if ollama_import_exception:
            raise ImportError("Please install `ollama` and `fix-busted-json` to use the Ollama API.")
        response_format = config.get("response_format")
        client = OllamaClient(response_format=response_format, **config)
        self._clients.append(client)

    def _separate_create_config(self, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        create_config = {k: v for k, v in config.items() if k not in self.extra_kwargs}
        extra_kwargs = {k: v for k, v in config.items() if k in self.extra_kwargs}
        return create_config, extra_kwargs

    def _instantiate(
        self,
        template: Optional[Union[str, Callable[[dict[str, Any]], str]]],
        context: Optional[dict[str, Any]] = None,
        allow_format_str_template: Optional[bool] = False,
    ) -> Optional[str]:
        if not context or template is None:
            return template
        if isinstance(template, str):
            return template.format(**context) if allow_format_str_template else template
        return template(context)

    def _construct_create_params(self, create_config: dict[str, Any], extra_kwargs: dict[str, Any]) -> dict[str, Any]:
        prompt: Optional[str] = create_config.get("prompt")
        messages: Optional[list[dict[str, Any]]] = create_config.get("messages")
        if (prompt is None) == (messages is None):
            raise ValueError("Either prompt or messages should be in create config but not both.")
        
        context = extra_kwargs.get("context")
        if context is None:
            return create_config

        allow_format_str_template = extra_kwargs.get("allow_format_str_template", False)
        params = create_config.copy()
        
        if prompt is not None:
            params["prompt"] = self._instantiate(prompt, context, allow_format_str_template)
        elif context:
            params["messages"] = [
                {
                    **m,
                    "content": self._instantiate(m["content"], context, allow_format_str_template),
                }
                if m.get("content")
                else m
                for m in messages  # type: ignore
            ]
        return params

    def _update_usage(self, actual_usage, total_usage):
        def update_usage(usage_summary, response_usage):
            if not response_usage:
                return usage_summary
            
            for key in ModelClient.RESPONSE_USAGE_KEYS:
                if key not in response_usage:
                    return usage_summary

            model = response_usage["model"]
            cost = response_usage["cost"]
            prompt_tokens = response_usage["prompt_tokens"]
            completion_tokens = response_usage["completion_tokens"] or 0
            total_tokens = response_usage["total_tokens"]

            if usage_summary is None:
                usage_summary = {"total_cost": cost}
            else:
                usage_summary["total_cost"] += cost

            usage_summary[model] = {
                "cost": usage_summary.get(model, {}).get("cost", 0) + cost,
                "prompt_tokens": usage_summary.get(model, {}).get("prompt_tokens", 0) + prompt_tokens,
                "completion_tokens": usage_summary.get(model, {}).get("completion_tokens", 0) + completion_tokens,
                "total_tokens": usage_summary.get(model, {}).get("total_tokens", 0) + total_tokens,
            }
            return usage_summary

        if total_usage is not None:
            self.total_usage_summary = update_usage(self.total_usage_summary, total_usage)
        if actual_usage is not None:
            self.actual_usage_summary = update_usage(self.actual_usage_summary, actual_usage)

    def create(self, **config: Any) -> ModelClient.ModelClientResponseProtocol:
        if not self._clients:
            raise RuntimeError("No Ollama clients registered.")

        invocation_id = str(uuid.uuid4())
        last = len(self._clients) - 1

        for i, client in enumerate(self._clients):
            full_config = {**config, **self._config_list[i]}
            create_config, extra_kwargs = self._separate_create_config(full_config)
            params = self._construct_create_params(create_config, extra_kwargs)

            cache_seed = extra_kwargs.get("cache_seed", LEGACY_DEFAULT_CACHE_SEED)
            cache = extra_kwargs.get("cache")
            filter_func = extra_kwargs.get("filter_func")
            context = extra_kwargs.get("context")
            price = extra_kwargs.get("price")

            if isinstance(price, list):
                price = tuple(price)
            elif isinstance(price, (float, int)):
                price = (price, price)

            total_usage = None
            actual_usage = None

            cache_client = cache if cache is not None else Cache.disk(cache_seed, LEGACY_CACHE_DIR) if cache_seed is not None else None

            if cache_client is not None:
                with cache_client as cache:
                    key = get_key(params)
                    request_ts = get_current_ts()
                    response = cache.get(key, None)

                    if response is not None:
                        response.message_retrieval_function = client.message_retrieval
                        try:
                            response.cost
                        except AttributeError:
                            response.cost = client.cost(response)
                            cache.set(key, response)
                        
                        total_usage = client.get_usage(response)
                        pass_filter = filter_func is None or filter_func(context=context, response=response)
                        
                        if pass_filter or i == last:
                            response.config_id = i
                            response.pass_filter = pass_filter
                            self._update_usage(actual_usage=actual_usage, total_usage=total_usage)
                            return response
                        continue

            try:
                response = client.create(params)
                if price is not None:
                    response.cost = self._cost_with_customized_price(response, price)
                else:
                    response.cost = client.cost(response)

                actual_usage = client.get_usage(response)
                total_usage = actual_usage.copy() if actual_usage is not None else None
                self._update_usage(actual_usage=actual_usage, total_usage=total_usage)

                if cache_client is not None:
                    with cache_client as cache:
                        cache.set(key, response)

                response.message_retrieval_function = client.message_retrieval
                pass_filter = filter_func is None or filter_func(context=context, response=response)
                
                if pass_filter or i == last:
                    response.config_id = i
                    response.pass_filter = pass_filter
                    return response
                
            except (ollama_RequestError, ollama_ResponseError) as err:
                logger.debug(f"Ollama client {i} failed", exc_info=True)
                if i == last:
                    raise err

        raise RuntimeError("All Ollama clients failed.")

    @staticmethod
    def _cost_with_customized_price(
        response: ModelClient.ModelClientResponseProtocol, 
        price_1k: tuple[float, float]
    ) -> float:
        n_input_tokens = response.usage.prompt_tokens if response.usage is not None else 0
        n_output_tokens = response.usage.completion_tokens if response.usage is not None else 0
        if n_output_tokens is None:
            n_output_tokens = 0
        return (n_input_tokens * price_1k[0] + n_output_tokens * price_1k[1]) / 1000

    @classmethod
    def extract_text_or_completion_object(
        cls, response: ModelClient.ModelClientResponseProtocol
    ) -> Union[list[str], list[ModelClient.ModelClientResponseProtocol.Choice.Message]]:
        return response.message_retrieval_function(response)