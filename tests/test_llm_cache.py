import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agp_research.llm import OpenAICompatibleClient


class FakeResponse:
    def __init__(self, result):
        self.result = result

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.result).encode()


class LLMCacheTest(unittest.TestCase):
    def test_second_identical_call_uses_cache_and_logs_usage(self):
        result = {
            "choices": [{"message": {"content": '{"depth": 2}'}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = OpenAICompatibleClient(
                api_key="test-key",
                model="test-model",
                cache_dir=root / "cache",
                log_path=root / "requests.jsonl",
            )
            urlopen = Mock(return_value=FakeResponse(result))
            with patch("agp_research.llm.urllib.request.urlopen", urlopen):
                self.assertEqual(client.complete_json("system", "question"), {"depth": 2})
                self.assertEqual(client.complete_json("system", "question"), {"depth": 2})

            self.assertEqual(urlopen.call_count, 1)
            records = [
                json.loads(line)
                for line in (root / "requests.jsonl").read_text().splitlines()
            ]
            self.assertFalse(records[0]["cache_hit"])
            self.assertTrue(records[0]["network_request"])
            self.assertEqual(records[0]["total_tokens"], 17)
            self.assertTrue(records[1]["cache_hit"])
            self.assertFalse(records[1]["network_request"])
            self.assertNotIn("test-key", json.dumps(records))
            self.assertNotIn("question", json.dumps(records))

    def test_different_response_modes_have_different_cache_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            client = OpenAICompatibleClient(
                api_key="test-key", model="test-model", cache_dir=Path(directory)
            )
            json_key = client._cache_key(
                {"temperature": 0}, "system", "question", True
            )
            text_key = client._cache_key(
                {"temperature": 0}, "system", "question", False
            )
            self.assertNotEqual(json_key, text_key)


if __name__ == "__main__":
    unittest.main()
