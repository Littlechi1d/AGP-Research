import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agp_research.config import read_env_file, setting
from agp_research.llm import OpenAICompatibleClient


class ConfigTest(unittest.TestCase):
    def write_env(self, directory: str, content: str) -> Path:
        path = Path(directory) / ".env"
        path.write_text(content, encoding="utf-8")
        return path

    def test_reads_comments_blank_lines_quotes_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_env(
                directory,
                "# comment\n\nOPENAI_API_KEY='secret'\nexport AGP_MODEL=small-model\n",
            )
            self.assertEqual(
                read_env_file(path),
                {"OPENAI_API_KEY": "secret", "AGP_MODEL": "small-model"},
            )

    def test_shell_environment_overrides_file_without_being_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_env(directory, "AGP_MODEL=file-model\n")
            with patch.dict(os.environ, {"AGP_MODEL": "shell-model"}, clear=True):
                self.assertEqual(setting("AGP_MODEL", env_path=path), "shell-model")
                self.assertEqual(os.environ["AGP_MODEL"], "shell-model")

    def test_client_loads_env_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_env(
                directory,
                "OPENAI_API_KEY=test-key\nAGP_MODEL=test-model\n"
                "OPENAI_BASE_URL=https://example.test/v1/\nAGP_LLM_TIMEOUT=12\n",
            )
            with patch.dict(os.environ, {}, clear=True):
                client = OpenAICompatibleClient.from_environment(path)
            self.assertIsNotNone(client)
            self.assertEqual(client.api_key, "test-key")
            self.assertEqual(client.model, "test-model")
            self.assertEqual(client.base_url, "https://example.test/v1")
            self.assertEqual(client.timeout, 12)
            self.assertEqual(client.cache_dir, Path(".agp_cache/llm"))
            self.assertEqual(client.log_path, Path(".agp_logs/llm_requests.jsonl"))

    def test_missing_key_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_env(directory, "AGP_MODEL=test-model\n")
            with patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(OpenAICompatibleClient.from_environment(path))

    def test_invalid_timeout_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_env(
                directory, "OPENAI_API_KEY=test-key\nAGP_LLM_TIMEOUT=slow\n"
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "AGP_LLM_TIMEOUT"):
                    OpenAICompatibleClient.from_environment(path)


if __name__ == "__main__":
    unittest.main()
