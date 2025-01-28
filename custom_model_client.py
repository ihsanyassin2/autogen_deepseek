from __future__ import annotations
from typing import Optional, Protocol, Union, runtime_checkable

@runtime_checkable
class ModelClient(Protocol):
    """Protocol defining the interface for model clients."""
    
    RESPONSE_USAGE_KEYS = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    class ModelClientResponseProtocol(Protocol):
        class Choice(Protocol):
            class Message(Protocol):
                content: Optional[str]
            message: Message
        choices: list[Choice]
        model: str

    def message_retrieval(
        self, response: ModelClientResponseProtocol
    ) -> Union[list[str], list[ModelClient.ModelClientResponseProtocol.Choice.Message]]: ...

    def cost(self, response: ModelClientResponseProtocol) -> float: ...

    @staticmethod
    def get_usage(response: ModelClientResponseProtocol) -> dict: ...