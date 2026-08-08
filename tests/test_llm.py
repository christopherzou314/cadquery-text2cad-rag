import http.client
import json
import os
import unittest
from unittest.mock import patch

from src.text2cad.llm import _chat_completion_code, generate_clarification_response


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


class _UsageResponse(_FakeResponse):
    def __iter__(self):
        content_payload = {
            "id": "response-123",
            "request_id": "provider-request-456",
            "model": "qwen-test",
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "short reasoning",
                        "content": "result = 1",
                    },
                    "finish_reason": None,
                }
            ],
            "usage": None,
        }
        finish_payload = {
            "id": "response-123",
            "model": "qwen-test",
            "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
            "usage": None,
        }
        usage_payload = {
            "id": "response-123",
            "model": "qwen-test",
            "choices": [],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "prompt_tokens_details": {"cached_tokens": 100},
                "completion_tokens_details": {
                    "reasoning_tokens": 300,
                    "text_tokens": 200,
                },
            },
        }
        yield f"data: {json.dumps(content_payload)}\n\n".encode("utf-8")
        yield f"data: {json.dumps(finish_payload)}\n\n".encode("utf-8")
        yield f"data: {json.dumps(usage_payload)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"


class LLMRetryTests(unittest.TestCase):
    @patch("src.text2cad.llm.urllib.request.urlopen")
    def test_clarification_response_remains_natural_language(self, urlopen):
        urlopen.return_value = _FakeResponse(
            ("The two dimensions conflict. ", "Which value should take priority?")
        )

        response = generate_clarification_response(
            "A plate must be both 20 mm and 30 mm long.",
            api_key="test-key",
            base_url="https://example.invalid/v1",
        )

        self.assertEqual(
            response,
            "The two dimensions conflict. Which value should take priority?",
        )
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertIn("natural-language text only", payload["messages"][0]["content"])
        self.assertNotIn("assign it to a variable named `result`", payload["messages"][0]["content"])

    @patch("src.text2cad.llm.time.sleep")
    @patch("src.text2cad.llm.urllib.request.urlopen")
    def test_remote_disconnect_is_retried(self, urlopen, sleep):
        urlopen.side_effect = [
            http.client.RemoteDisconnected("first disconnect"),
            http.client.RemoteDisconnected("second disconnect"),
            _FakeResponse(),
        ]
        progress = []
        metrics = []

        code = _chat_completion_code(
            [{"role": "user", "content": "test"}],
            api_key="test-key",
            base_url="https://example.invalid/v1",
            max_retries=3,
            progress_callback=progress.append,
            metrics_callback=metrics.append,
        )

        self.assertEqual(code, "result = 1")
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])
        retry_messages = [message for message in progress if "Retrying" in message]
        self.assertEqual(len(retry_messages), 2)
        self.assertEqual(metrics[0]["transport_attempt_count"], 3)
        self.assertEqual(metrics[0]["transport_retry_count"], 2)
        self.assertFalse(metrics[0]["transport_attempts"][0]["success"])
        self.assertTrue(metrics[0]["transport_attempts"][2]["success"])

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertIs(payload["stream"], True)
        self.assertNotIn("stream_options", payload)

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

    @patch.dict(
        os.environ,
        {
            "LLM_INPUT_COST_PER_1M_TOKENS": "2",
            "LLM_OUTPUT_COST_PER_1M_TOKENS": "4",
            "LLM_COST_CURRENCY": "CNY",
            "LLM_ENABLE_THINKING": "false",
        },
        clear=False,
    )
    @patch("src.text2cad.llm.urllib.request.urlopen")
    def test_usage_cost_and_unsent_seed_are_logged(self, urlopen):
        urlopen.return_value = _UsageResponse()
        metrics = []

        code = _chat_completion_code(
            [{"role": "user", "content": "test"}],
            api_key="test-key",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            seed=42,
            send_seed=False,
            metrics_callback=metrics.append,
        )

        self.assertEqual(code, "result = 1")
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["usage"]["total_tokens"], 1500)
        self.assertEqual(metrics[0]["usage"]["cached_prompt_tokens"], 100)
        self.assertEqual(metrics[0]["usage"]["reasoning_tokens"], 300)
        self.assertEqual(metrics[0]["usage"]["text_completion_tokens"], 200)
        self.assertEqual(metrics[0]["reasoning_characters"], len("short reasoning"))
        self.assertEqual(metrics[0]["estimated_cost"], 0.004)
        self.assertFalse(metrics[0]["seed_sent_to_provider"])
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertNotIn("seed", payload)
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertFalse(payload["enable_thinking"])

    @patch("src.text2cad.llm.urllib.request.urlopen")
    def test_seed_is_only_sent_when_explicitly_enabled(self, urlopen):
        urlopen.return_value = _UsageResponse()

        _chat_completion_code(
            [{"role": "user", "content": "test"}],
            model="qwen3.7-max",
            api_key="test-key",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            seed=42,
            send_seed=True,
        )

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertTrue(payload["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
