import logging
import ollama
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from custom_conversable_agent import Agent, ConversableAgent

logger = logging.getLogger(__name__)

class GPTResearcherAgent(ConversableAgent):
    """An agent with local LLM research capabilities using Ollama."""

    DEFAULT_PROMPT = (
        "You are a helpful research assistant with access to a local LLM as your personal ChatGPT. Your task is to analyze questions "
        "and provide thoughtful responses by consulting the LLM when needed. Today's date is "
        + datetime.now().date().isoformat()
    )

    def __init__(
        self,
        name: str,
        system_message: str = DEFAULT_PROMPT,
        llm_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs,
        )

    def _extract_message_content(self, response: Any) -> str:
        """Helper method to safely extract message content from Ollama response"""
        try:
            if isinstance(response, dict):
                message = response.get('message')
                if hasattr(message, 'content'):  # Message object
                    return message.content
                elif isinstance(message, dict):  # Dictionary
                    return message.get('content', '')
                elif isinstance(message, str):  # String
                    return message
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                return response.message.content
            elif hasattr(response, 'message') and isinstance(response.message, dict):
                return response.message.get('content', '')
            return ''
        except Exception as e:
            logger.error(f"Error extracting message content: {str(e)} - Response: {response}")
            return ''

    def generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional[Agent] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any], None]:
        """Generate a reply with a focus on responding to team broadcasts."""
        try:
            if messages is None:
                messages = []

            # Check for the most recent broadcast
            recent_broadcasts = [
                msg for msg in messages if msg["role"] == "assistant" and msg.get("name") != self.name
            ]
            if recent_broadcasts:
                last_broadcast = recent_broadcasts[-1]
                return {
                    "role": "assistant",
                    "content": f"{self.name} responding to {last_broadcast['name']}: {last_broadcast['content']}"
                }

            # If no relevant broadcast, fall back to LLM-based generation
            ollama_messages = [{"role": "system", "content": self.system_message}]
            for msg in messages:
                if "content" in msg:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role not in ["system", "user", "assistant"]:
                        role = "user"
                    ollama_messages.append({"role": role, "content": content})

            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            response = ollama.chat(model=model, messages=ollama_messages)

            content = self._extract_message_content(response)
            if content:
                return {
                    "role": "assistant",
                    "content": content
                }

            raise ValueError("Could not generate valid content")

        except Exception as e:
            logger.error(f"Error in generate_reply: {str(e)}")
            return {
                "role": "assistant",
                "content": "I apologize, but something went wrong. Please try again."
            }
