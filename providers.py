
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import yaml


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system: str, prompt: str) -> str:
        raise NotImplementedError

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Yield the answer in pieces. Providers without streaming
        support yield the whole answer at once."""
        yield self.generate(system, prompt)

    @staticmethod
    @contextmanager
    def open_json_request(url: str, headers: dict, payload: dict):
        body = json.dumps(payload).encode("utf-8")

        request = Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                **headers,
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                yield response

        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"API returned HTTP {exc.code}: {details}"
            ) from None

        except URLError as exc:
            raise RuntimeError(
                f"HTTP request failed: {exc.reason}"
            ) from None

    @classmethod
    def post_json(cls, url: str, headers: dict, payload: dict) -> dict:
        with cls.open_json_request(url, headers, payload) as response:
            result = response.read().decode("utf-8")

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            raise RuntimeError("API returned invalid JSON.") from None

    @classmethod
    def post_sse(
        cls, url: str, headers: dict, payload: dict
    ) -> Iterator[dict]:
        """POST a request and yield each Server-Sent Event's JSON data."""
        with cls.open_json_request(url, headers, payload) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue

                try:
                    yield json.loads(line[len("data:"):])
                except json.JSONDecodeError:
                    raise RuntimeError(
                        "API returned invalid JSON."
                    ) from None


class GeminiProvider(LLMProvider):
    API_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models"
    )

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def payload(self, system: str, prompt: str) -> dict:
        return {
            "systemInstruction": {
                "parts": [{"text": system}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 4096,
            },
        }

    def url(self, method: str) -> str:
        model = quote(self.model, safe="")
        return f"{self.API_URL}/{model}:{method}"

    @staticmethod
    def response_text(response: dict) -> str:
        if "error" in response:
            raise RuntimeError(f"Gemini returned an error: {response['error']}")

        candidates = response.get("candidates", [])
        if not candidates:
            feedback = response.get("promptFeedback", {})
            raise RuntimeError(
                f"Gemini returned no candidates: {feedback or response}"
            )

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        return "".join(
            part["text"]
            for part in parts
            if isinstance(part.get("text"), str)
        )

    def generate(self, system: str, prompt: str) -> str:
        response = self.post_json(
            self.url("generateContent"),
            headers={"x-goog-api-key": self.api_key},
            payload=self.payload(system, prompt),
        )
        return self.response_text(response)

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        for chunk in self.post_sse(
            self.url("streamGenerateContent?alt=sse"),
            headers={"x-goog-api-key": self.api_key},
            payload=self.payload(system, prompt),
        ):
            # Trailing chunks may carry only usage metadata.
            blocked = chunk.get("promptFeedback", {}).get("blockReason")
            if not chunk.get("candidates") and "error" not in chunk and not blocked:
                continue

            text = self.response_text(chunk)
            if text:
                yield text


PROVIDERS = {
    "gemini": GeminiProvider,
}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    for name, settings in config.get("providers", {}).items():
        api_key = settings.get("api_key", "")

        if (
            isinstance(api_key, str)
            and api_key.startswith("${")
            and api_key.endswith("}")
        ):
            env_name = api_key[2:-1]
            api_key = os.environ.get(env_name, "")

        settings["api_key"] = api_key

    return config


def create_provider(name: str, config: dict) -> LLMProvider:
    name = name.lower()

    try:
        provider_class = PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"Unsupported provider: {name}"
        ) from None

    settings = config.get("providers", {}).get(name)
    if not settings:
        raise ValueError(
            f"Missing configuration for provider: {name}"
        )

    model = settings.get("model")
    api_key = settings.get("api_key")

    if not model:
        raise ValueError(f"Missing model for provider: {name}")

    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(
            f"Missing API key for provider: {name}. "
            "Check config.yaml or your environment variables."
        )

    return provider_class(model=model, api_key=api_key)


def create_default_provider(config: dict) -> LLMProvider:
    name = config.get("default_provider")
    if not name:
        raise ValueError("Missing default_provider in config")

    return create_provider(name, config)
