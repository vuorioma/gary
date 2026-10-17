import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
import providers  # noqa: E402
from providers import GeminiProvider, LLMProvider, create_provider  # noqa: E402


class MockProvider(LLMProvider):
    """Records every call instead of contacting an API."""

    calls = []
    reply = "mock answer"

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def generate(self, system: str, prompt: str) -> str:
        MockProvider.calls.append(
            {"model": self.model, "system": system, "prompt": prompt}
        )
        return MockProvider.reply


class FakeStdin(io.StringIO):
    def __init__(self, text: str = "", tty: bool = False):
        super().__init__(text)
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


CONFIG_TEMPLATE = """\
default_provider: mock

providers:
  mock:
    model: mock-model
    api_key: "{api_key}"

system_prompt: |
  You are a test.

max_input_chars: {max_input_chars}

files:
  log: log.jsonl
"""


class GaryTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.config = self.write_config()

        MockProvider.calls = []
        MockProvider.reply = "mock answer"

        patcher = mock.patch.dict(providers.PROVIDERS, {"mock": MockProvider})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def write_config(self, api_key="test-key", max_input_chars=1000):
        path = self.dir / "config.yaml"
        path.write_text(
            CONFIG_TEMPLATE.format(
                api_key=api_key, max_input_chars=max_input_chars
            ),
            encoding="utf-8",
        )
        return path

    def run_gary(self, *args, stdin="", tty=False):
        """Run main() and return (stdout, stderr)."""
        argv = ["gary", "--config", str(self.config), *args]
        out, err = io.StringIO(), io.StringIO()

        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(sys, "stdin", FakeStdin(stdin, tty)), \
                mock.patch.object(sys, "stdout", out), \
                mock.patch.object(sys, "stderr", err):
            main.main()

        return out.getvalue(), err.getvalue()

    def log_records(self):
        log = self.dir / "log.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()]


class TestInput(GaryTestCase):
    def test_prompt_from_args(self):
        out, _ = self.run_gary("hello", "there", tty=True)

        self.assertEqual(out, "mock answer\n")
        self.assertEqual(len(MockProvider.calls), 1)
        request = MockProvider.calls[0]["prompt"]
        self.assertIn("<prompt>\nhello there\n</prompt>", request)
        self.assertNotIn("<input>", request)

    def test_piped_input_with_prompt(self):
        self.run_gary("explain", "this", stdin="ERROR disk full\n")

        request = MockProvider.calls[0]["prompt"]
        self.assertIn("<input>\nERROR disk full\n</input>", request)
        self.assertIn("<prompt>\nexplain this\n</prompt>", request)

    def test_piped_input_without_prompt_becomes_prompt(self):
        self.run_gary(stdin="what is 2+2?")

        request = MockProvider.calls[0]["prompt"]
        self.assertIn("<prompt>\nwhat is 2+2?\n</prompt>", request)
        self.assertNotIn("<input>", request)

    def test_interactive_prompt_when_no_args_and_tty(self):
        with mock.patch("builtins.input", return_value="typed prompt"):
            self.run_gary(tty=True)

        self.assertIn("typed prompt", MockProvider.calls[0]["prompt"])

    def test_empty_prompt_raises(self):
        with self.assertRaisesRegex(ValueError, "Prompt cannot be empty"):
            self.run_gary(stdin="   \n")
        self.assertEqual(MockProvider.calls, [])

    def test_long_input_keeps_the_end(self):
        self.config = self.write_config(max_input_chars=10)

        _, err = self.run_gary("summarize", stdin="AAAAAAAAAA" + "0123456789")

        request = MockProvider.calls[0]["prompt"]
        self.assertIn("<input>\n0123456789\n</input>", request)
        self.assertIn("Warning: input is 20 characters", err)

    def test_system_prompt_is_passed(self):
        self.run_gary("hi", tty=True)

        self.assertEqual(MockProvider.calls[0]["system"], "You are a test.\n")

    def test_no_memory_between_calls(self):
        self.run_gary("first", tty=True)
        self.run_gary("second", tty=True)

        second = MockProvider.calls[1]["prompt"]
        self.assertNotIn("first", second)
        self.assertNotIn("mock answer", second)


class TestProviderSelection(GaryTestCase):
    def test_model_override(self):
        self.run_gary("--model", "other-model", "hi", tty=True)

        self.assertEqual(MockProvider.calls[0]["model"], "other-model")
        self.assertEqual(self.log_records()[0]["model"], "other-model")

    def test_unknown_provider_raises(self):
        with self.assertRaisesRegex(ValueError, "not configured"):
            self.run_gary("--provider", "nope", "hi", tty=True)

    def test_missing_api_key_raises(self):
        self.config = self.write_config(api_key="")

        with self.assertRaisesRegex(ValueError, "No API key"):
            self.run_gary("hi", tty=True)

    def test_api_key_from_environment(self):
        self.config = self.write_config(api_key="${GARY_TEST_KEY}")

        with mock.patch.dict("os.environ", {"GARY_TEST_KEY": "from-env"}):
            config = main.load_config(self.config)

        self.assertEqual(config["providers"]["mock"]["api_key"], "from-env")

    def test_unset_environment_key_raises(self):
        self.config = self.write_config(api_key="${GARY_UNSET_KEY}")

        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "No API key"):
                self.run_gary("hi", tty=True)

    def test_create_provider_rejects_unsupported(self):
        with self.assertRaisesRegex(ValueError, "Unsupported provider"):
            create_provider("nope", {"providers": {}})


class TestLogging(GaryTestCase):
    def test_successful_call_is_logged(self):
        self.run_gary("explain", stdin="some log data")

        [record] = self.log_records()
        self.assertEqual(record["provider"], "mock")
        self.assertEqual(record["model"], "mock-model")
        self.assertEqual(record["prompt"], "explain")
        self.assertEqual(record["input_chars"], len("some log data"))
        self.assertEqual(record["answer"], "mock answer")

    def test_log_appends(self):
        self.run_gary("one", tty=True)
        self.run_gary("two", tty=True)

        prompts = [r["prompt"] for r in self.log_records()]
        self.assertEqual(prompts, ["one", "two"])

    def test_failed_call_is_not_logged(self):
        with mock.patch.object(
            MockProvider, "generate", side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                self.run_gary("hi", tty=True)

        self.assertFalse((self.dir / "log.jsonl").exists())


class TestConfig(GaryTestCase):
    def test_missing_config_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            main.load_config(self.dir / "nope.yaml")

    def test_missing_required_setting_raises(self):
        self.config.write_text("default_provider: mock\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Missing config setting"):
            main.load_config(self.config)

    def test_find_config_uses_first_existing_path(self):
        missing = self.dir / "missing.yaml"

        with mock.patch.object(
            main, "CONFIG_SEARCH_PATHS", (missing, self.config)
        ):
            self.assertEqual(main.find_config(), self.config)

    def test_find_config_raises_when_none_exist(self):
        with mock.patch.object(
            main, "CONFIG_SEARCH_PATHS", (self.dir / "missing.yaml",)
        ):
            with self.assertRaises(FileNotFoundError):
                main.find_config()

    def test_relative_log_path_resolves_next_to_config(self):
        path = main.resolve_path("log.jsonl", self.config)

        self.assertEqual(path, self.dir / "log.jsonl")


class TestGeminiProvider(unittest.TestCase):
    """Exercises Gemini request/response handling without the network."""

    def setUp(self):
        self.provider = GeminiProvider(model="gemini-test", api_key="k")

    def test_builds_request_and_joins_text_parts(self):
        response = {
            "candidates": [
                {"content": {"parts": [{"text": "a"}, {"text": "b"}]}}
            ]
        }

        with mock.patch.object(
            GeminiProvider, "post_json", return_value=response
        ) as post:
            answer = self.provider.generate("sys", "user prompt")

        self.assertEqual(answer, "a\nb")

        url = post.call_args.args[0]
        headers = post.call_args.kwargs["headers"]
        payload = post.call_args.kwargs["payload"]
        self.assertTrue(url.endswith("/gemini-test:generateContent"))
        self.assertEqual(headers["x-goog-api-key"], "k")
        self.assertEqual(
            payload["systemInstruction"]["parts"][0]["text"], "sys"
        )
        self.assertEqual(
            payload["contents"][0]["parts"][0]["text"], "user prompt"
        )

    def test_no_candidates_raises(self):
        response = {"promptFeedback": {"blockReason": "SAFETY"}}

        with mock.patch.object(
            GeminiProvider, "post_json", return_value=response
        ):
            with self.assertRaisesRegex(RuntimeError, "SAFETY"):
                self.provider.generate("sys", "prompt")


if __name__ == "__main__":
    unittest.main()
