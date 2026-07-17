import http.client
import json
import unittest
from unittest.mock import patch

from src.text2cad.llm import _chat_completion_code


class _FakeResponse:
    def __init__(self, pieces=("result", " = 1")):
        self.pieces = pieces

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        lines = []
        for piece in self.pieces:
            payload = {
                "choices": [
                    {
                        "delta": {
                            "content": piece,
                            "reasoning_content": "ignored reasoning",
                        }
                    }
                ]
            }
            lines.append(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
        lines.append(b"data: [DONE]\n\n")
        return iter(lines)


class _DisconnectingResponse(_FakeResponse):
    def __iter__(self):
        payload = {"choices": [{"delta": {"content": "partial code"}}]}
        yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        raise http.client.RemoteDisconnected("disconnected during stream")


class LLMRetryTests(unittest.TestCase):
    @patch("src.text2cad.llm.time.sleep")
    @patch("src.text2cad.llm.urllib.request.urlopen")
    def test_remote_disconnect_is_retried(self, urlopen, sleep):
        urlopen.side_effect = [
            http.client.RemoteDisconnected("first disconnect"),
            http.client.RemoteDisconnected("second disconnect"),
            _FakeResponse(),
        ]
        progress = []

        code = _chat_completion_code(
            [{"role": "user", "content": "test"}],
            api_key="test-key",
            base_url="https://example.invalid/v1",
            max_retries=3,
            progress_callback=progress.append,
        )

        self.assertEqual(code, "result = 1")
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])
        retry_messages = [message for message in progress if "Retrying" in message]
        self.assertEqual(len(retry_messages), 2)

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertIs(payload["stream"], True)

    @patch("src.text2cad.llm.urllib.request.urlopen")
    def test_streaming_chunks_are_joined(self, urlopen):
        urlopen.return_value = _FakeResponse(("import cadquery as cq\n", "result = cq.Workplane()"))

        code = _chat_completion_code(
            [{"role": "user", "content": "test"}],
            api_key="test-key",
            base_url="https://example.invalid/v1",
        )

        self.assertEqual(code, "import cadquery as cq\nresult = cq.Workplane()")

    @patch("src.text2cad.llm.time.sleep")
    @patch("src.text2cad.llm.urllib.request.urlopen")
    def test_partial_stream_is_discarded_before_retry(self, urlopen, sleep):
        urlopen.side_effect = [_DisconnectingResponse(), _FakeResponse(("final code",))]

        code = _chat_completion_code(
            [{"role": "user", "content": "test"}],
            api_key="test-key",
            base_url="https://example.invalid/v1",
            max_retries=1,
        )

        self.assertEqual(code, "final code")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
