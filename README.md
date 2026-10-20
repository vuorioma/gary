# gary

Gary is a command-line tool for third party LLM inference.

If Claude Code is a Swiss Army knife, gary is a butter knife. It doesn't
edit files, run commands or remember previous conversations. It sends one
prompt to an LLM and prints the answer.

```sh
gary what is the difference between a process and a thread
cat app.log | gary explain the latest error
git diff | gary --persona terminal write a commit message for this
```

## Features

- Gemini, OpenAI and Groq providers, switchable per command.
- Streams the answer when printing to a terminal; prints it in one go when
  the output is piped or redirected.
- Piped input is sent along with the prompt as data to answer about.
- Personas: system prompts kept as plain text files.
- Every answer is appended to a JSONL log.
- One dependency (PyYAML); HTTP is done with the Python standard library.

## Installation

Requires Python 3.9 or newer.

```sh
git clone git@github.com:vuorioma/gary.git
cd gary
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config_example.yaml config.yaml
```

Set the API key for at least one provider:

```sh
export GEMINI_API_KEY=...   # and/or OPENAI_API_KEY, GROQ_API_KEY
```

To run `gary` from any directory, link the launcher onto your `PATH`:

```sh
ln -s "$PWD/gary" ~/.local/bin/gary
```

The launcher runs `main.py` with the project's `.venv`.

## Usage

```
gary [--provider NAME] [--model MODEL] [--persona NAME] [--config PATH] [prompt ...]
```

| Option       | Description                                              |
|--------------|----------------------------------------------------------|
| `--provider` | Provider to use instead of `default_provider`.           |
| `--model`    | Model to use instead of the provider's configured model. |
| `--persona`  | Persona to use instead of `default_persona`.             |
| `--config`   | Path to a config file (see below for the default).       |

The prompt is the rest of the command line. Quote it if it contains
characters your shell would interpret, such as `?`, `*` or `'`.

- With a prompt and piped input, the piped text is sent as data and the
  prompt asks about it.
- With only piped input, the piped text is the prompt.
- With neither, gary asks for a prompt interactively.

Piped input longer than `max_input_chars` is truncated, keeping the end
(the most recent lines of a log, for example), and gary prints a warning.

Press Ctrl-C to stop an answer. A stopped answer is not logged.

## Configuration

gary uses the first config file it finds:

1. The path given with `--config`
2. `~/.config/gary/config.yaml`
3. `config.yaml` next to `main.py`

`config_example.yaml` lists every setting:

```yaml
default_provider: gemini

providers:
  gemini:
    model: gemini-3.8-flash
    api_key: "${GEMINI_API_KEY}"
  openai:
    model: gpt-5-mini
    api_key: "${OPENAI_API_KEY}"
  groq:
    model: openai/gpt-oss-120b
    api_key: "${GROQ_API_KEY}"

default_persona: generic
max_input_chars: 100000

files:
  log: log.jsonl
  # personas: personas
```

- `api_key` can be the key itself or `${VAR}` to read it from an
  environment variable. Prefer the environment variable; `config.yaml` is
  gitignored, but plain-text keys are easy to leak.
- Paths under `files` are resolved relative to the config file. Without
  `files.personas`, gary uses the `personas` folder next to `main.py`.

Model names change often. If a provider returns `model_not_found`, check
its model list and update `model` in your config.

## Personas

A persona is a system prompt stored as `personas/<name>.txt`. Two come
with gary:

- `generic`: a general-purpose assistant.
- `terminal`: plain-text answers for reading in a terminal: no Markdown,
  lines under 80 characters, commands indented so they're easy to copy.

To add one, create a new text file in `personas/` and use it with
`--persona <name>`, or make it the default with `default_persona`.
gary adds its instructions for handling piped input to every persona, so
persona files don't need to mention it.

## Log

Each answer is appended to the log file as one JSON object per line:

```json
{"timestamp": "2026-10-20T09:15:02+03:00", "provider": "groq", "model": "openai/gpt-oss-120b", "persona": "terminal", "prompt": "explain the latest error", "input_chars": 5120, "answer": "..."}
```

The piped input itself is not logged, only its length.

## Adding a provider

Providers live in `providers.py`. Subclass `LLMProvider`, implement
`generate()` (and `stream()` if the API supports streaming), then register
the class in `PROVIDERS`. If the API is OpenAI-compatible, subclassing
`OpenAIProvider` and changing `API_URL` is enough; see `GroqProvider`.
