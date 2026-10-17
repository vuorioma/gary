
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

from providers import create_provider

CONFIG_FILE = Path("config.yaml")


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    for key in ("default_provider", "default_model",
                "api_keys", "system_prompt", "files"):
        if key not in config:
            raise ValueError(f"Missing config setting: {key}")

    return config


def parse_response(raw: str) -> dict:
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from the model.")

    for key in ("answer", "updated_context"):
        if not isinstance(data.get(key), str):
            raise ValueError(f"Missing or invalid response field: {key}")

    return data


def main():
    parser = argparse.ArgumentParser(
        description="LLM CLI with YAML configuration and persistent context"
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--provider", help="Override default provider")
    parser.add_argument("--model", help="Override default model")
    args = parser.parse_args()

    config = load_config(args.config)

    provider_name = args.provider or config["default_provider"]
    model = args.model or config["default_model"]

    api_keys = config["api_keys"]
    api_key = api_keys.get(provider_name)

    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"No API key configured for {provider_name}")

    files = config["files"]
    context_file = Path(files.get("context", "context.txt"))
    log_file = Path(files.get("log", "log.jsonl"))

    # Create context file if missing.
    context_file.parent.mkdir(parents=True, exist_ok=True)
    context_file.touch(exist_ok=True)
    context = context_file.read_text(encoding="utf-8")

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        if sys.stdin.isatty():
            prompt = input("Prompt: ").strip()
        else:
            prompt = sys.stdin.read().strip()

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    provider = create_provider(provider_name, model, api_key)

    request = (
        f"Existing context:\n<context>\n{context}\n</context>\n\n"
        f"Current prompt:\n<prompt>\n{prompt}\n</prompt>"
    )

    # API failure or invalid JSON leaves context unchanged.
    raw = provider.generate(config["system_prompt"], request)
    result = parse_response(raw)

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    record = {
        "timestamp": timestamp,
        "provider": provider_name,
        "model": model,
        "prompt": prompt,
        "answer": result["answer"],
    }

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    context_file.write_text(
        result["updated_context"].rstrip() + "\n",
        encoding="utf-8",
    )

    print(result["answer"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
