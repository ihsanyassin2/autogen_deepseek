from typing import Dict, Optional, Any, Union
# @dataclass
class OllamaConfig:
    """Configuration for Ollama API settings."""
    api_base: str = "http://localhost:11434/api"
    api_key: str = "ollama"
    model: str = "llama3"
    seed: int = 42
    description: Optional[str] = None

    def validate(self) -> None:
        if not self.api_base.startswith("http"):
            raise ValueError("Invalid API base URL.")
        if not self.api_key:
            raise ValueError("API key cannot be empty.")
        if not self.model:
            raise ValueError("Model name must be specified.")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Seed must be a non-negative integer.")

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}