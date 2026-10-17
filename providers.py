
import os
from abc import ABC, abstractmethod
import anthropic

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system: str, prompt: str) -> str:
        """Return the model's text response."""
        raise NotImplementedError

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv(
            "ANTHROPIC_MODEL", "claude-sonnet-4-6"
        )
        self.client = anthropic.Anthropic()

    def generate(self, system: str, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return "\n".join(
            block.text
            for block in response.content
            if block.type == "text"
        )

def create_provider(name: str, model: str, api_key: str) -> LLMProvider:
    providers = {
        "anthropic": AnthropicProvider,
    }
    try:
        return providers[name](model=model)
    except KeyError:
        raise ValueError(f"Unsupported provider: {name}") from None