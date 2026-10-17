import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

from providers import create_provider

BASE_DIR = Path(__file__).resolve().parent

# Searched in order when --config is not given.
CONFIG_SEARCH_PATHS = (
    Path.home() / ".config" / "gary" / "config.yaml",
    BASE_DIR / "config.yaml",
)

DEFAULT_MAX_INPUT_CHARS = 100_000


def find_config() -> Path:
    for path in CONFIG_SEARCH_PATHS:
        if path.is_file():
            return path

    searched = ", ".join(str(p) for p in CONFIG_SEARCH_PATHS)
    raise FileNotFoundError(f"Config file not found. Searched: {searched}")


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    for key in ("default_provider", "providers", "system_prompt", "files"):
        if key not in config:
            raise ValueError(f"Missing config setting: {key}")

    if not isinstance(config["providers"], dict):
        raise ValueError("'providers' must be a YAML mapping.")

    if not isinstance(config["files"], dict):
        raise ValueError("'files' must be a YAML mapping.")

    default_provider = config["default_provider"]
    if default_provider not in config["providers"]:
        raise ValueError(
            f"Default provider '{default_provider}' is not configured."
        )

    # Resolve environment variable references in provider API keys.
    for name, settings in config["providers"].items():
        if not isinstance(settings, dict):
            raise ValueError(
                f"Configuration for provider '{name}' must be a mapping."
            )

        api_key = settings.get("api_key", "")

        if isinstance(api_key, str):
            match = re.fullmatch(r"\$\{([^}]+)\}", api_key)
            if match:
                api_key = os.environ.get(match.group(1), "")

        settings["api_key"] = api_key

    return config


def resolve_path(value: str, config_path: Path) -> Path:
    """Resolve a configured file path relative to the config file."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path


def main():
    parser = argparse.ArgumentParser(
        prog="gary",
        epilog="Example: cat app.log | gary explain the latest error",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text")
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    parser.add_argument("--provider", help="Override default provider")
    parser.add_argument("--model", help="Override configured provider model")
    args = parser.parse_args()

    config_path = (args.config or find_config()).expanduser().resolve()
    config = load_config(config_path)

    provider_name = (args.provider or config["default_provider"]).lower()

    if provider_name not in config["providers"]:
        raise ValueError(
            f"Provider '{provider_name}' is not configured in "
            "the providers section."
        )

    provider_config = config["providers"][provider_name]
    model = args.model or provider_config.get("model")

    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"No model configured for provider '{provider_name}'.")

    api_key = provider_config.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(
            f"No API key configured for '{provider_name}'. "
            "Check config.yaml or your environment variables."
        )

    log_file = resolve_path(config["files"].get("log", "log.jsonl"), config_path)

    prompt = " ".join(args.prompt).strip()

    # Piped input (e.g. `cat app.log | gary explain this`) is sent as data
    # alongside the prompt. Without a prompt, the piped text is the prompt.
    piped = ""
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()

    if not prompt and not piped and sys.stdin.isatty():
        prompt = input("Prompt: ").strip()
    elif not prompt:
        prompt, piped = piped, ""

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    max_input_chars = int(
        config.get("max_input_chars", DEFAULT_MAX_INPUT_CHARS)
    )
    if len(piped) > max_input_chars:
        print(
            f"Warning: input is {len(piped):,} characters; "
            f"sending only the last {max_input_chars:,}.",
            file=sys.stderr,
        )
        piped = piped[-max_input_chars:]

    # Create the selected provider with its configured API key.
    provider = create_provider(
        provider_name,
        {
            **config,
            "providers": {
                **config["providers"],
                provider_name: {
                    **provider_config,
                    "model": model,
                    "api_key": api_key,
                },
            },
        },
    )

    request_parts = []
    if piped:
        request_parts.append(f"Input:\n<input>\n{piped}\n</input>")
    request_parts.append(f"Current prompt:\n<prompt>\n{prompt}\n</prompt>")
    request = "\n\n".join(request_parts)

    # Only log once generation succeeds.
    answer = provider.generate(config["system_prompt"], request)

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    record = {
        "timestamp": timestamp,
        "provider": provider_name,
        "model": model,
        "prompt": prompt,
        "input_chars": len(piped),
        "answer": answer
    }

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(answer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
