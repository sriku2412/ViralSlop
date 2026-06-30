import unittest
from unittest.mock import patch

import requests

from viral_slop.config import AppConfig
from viral_slop.ollama_client import OllamaClient


class FakeStreamingResponse:
    status_code = 200

    def __init__(self, lines: list[str]):
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = False):
        return iter(self._lines)


class OllamaClientTests(unittest.TestCase):
    def test_generate_accumulates_streamed_chunks(self) -> None:
        response = FakeStreamingResponse(
            [
                '{"response":"{\\"final_answer\\":","done":false}',
                '{"response":" \\"42\\"}","done":false}',
                '{"response":"","done":true}',
            ]
        )

        with patch("requests.post", return_value=response) as post:
            generated = OllamaClient(AppConfig()).generate("Solve it")

        self.assertEqual(generated, '{"final_answer": "42"}')
        _, kwargs = post.call_args
        self.assertTrue(kwargs["stream"])
        self.assertTrue(kwargs["json"]["stream"])
        self.assertEqual(kwargs["json"]["options"]["num_predict"], 2200)

    def test_generate_reports_timeout_as_runtime_error(self) -> None:
        with patch("requests.post", side_effect=requests.Timeout):
            with self.assertRaisesRegex(RuntimeError, "Timed out waiting for Ollama"):
                OllamaClient(AppConfig()).generate("Solve it")


if __name__ == "__main__":
    unittest.main()
