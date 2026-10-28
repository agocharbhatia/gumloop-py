from __future__ import annotations

import json
from typing import Any

import pytest
import respx
from typer.testing import CliRunner

from gumloop.cli.credentials import Credentials
from gumloop.cli.credentials import save_credentials
from gumloop.cli.main import app
from tests.cli.test_chat import STREAM_BASE
from tests.cli.test_chat import _sse

# ---------------------------------------------------------------------------
# Helpers: wire-format chunks, never mocked SDK return objects.
# ---------------------------------------------------------------------------


def _chunk(content: str = "", finish_reason: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "id": "synthetic-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}],
        **extra,
    }


# ---------------------------------------------------------------------------
# CLI failure status must reflect typed SSE errors, not just HTTP status.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@respx.mock
def test_chat_stream_error_exits_nonzero_and_preserves_chunks(
    cli_runner: CliRunner, json_output: bool, partial: bool
) -> None:
    error = {"code": 502, "message": "Provider disconnected mid-stream"}
    chunks = [_chunk("Partial answer")] if partial else []
    chunks.append(_chunk(finish_reason="error", error=error))
    # Accounting may trail the error; keep every NDJSON frame for consumers.
    chunks.append(_chunk(finish_reason="error", usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}))
    route = respx.post(f"{STREAM_BASE}/chat/completions").respond(
        200, text=_sse([*chunks, "[DONE]"]), headers={"content-type": "text/event-stream"}
    )
    save_credentials(Credentials(api_key="synthetic-key"))

    result = cli_runner.invoke(
        app, ["chat", "completions", "create", "hello", "-m", "m", "--stream", *(["--json"] if json_output else [])]
    )

    assert result.exit_code == 1, (result.stdout, result.stderr)
    assert route.call_count == 1
    assert "Provider disconnected mid-stream" in result.stderr
    if json_output:
        frames = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(frames) == len(chunks)
        assert frames[-2]["error"] == error
        assert frames[-1]["usage"]["total_tokens"] == 6
        assert json.loads(result.stderr)["error"]["message"] == error["message"]
        if partial:
            assert frames[0]["choices"][0]["delta"]["content"] == "Partial answer"
    else:
        assert result.stdout == ("Partial answer\n" if partial else "\n")


@pytest.mark.parametrize("json_output", [False, True])
@respx.mock
def test_chat_stream_error_finish_reason_has_fallback_message(cli_runner: CliRunner, json_output: bool) -> None:
    respx.post(f"{STREAM_BASE}/chat/completions").respond(
        200, text=_sse([_chunk(finish_reason="error"), "[DONE]"]), headers={"content-type": "text/event-stream"}
    )
    save_credentials(Credentials(api_key="synthetic-key"))

    result = cli_runner.invoke(
        app, ["chat", "completions", "create", "hello", "-m", "m", "--stream", *(["--json"] if json_output else [])]
    )

    assert result.exit_code == 1
    assert "Chat completion stream failed." in result.stderr
    if json_output:
        assert json.loads(result.stdout)["choices"][0]["finish_reason"] == "error"


@pytest.mark.parametrize("finish_reason", ["stop", "length", "tool_calls", "content_filter"])
@respx.mock
def test_chat_stream_success_keeps_usage_and_other_finish_reasons(cli_runner: CliRunner, finish_reason: str) -> None:
    chunks = [
        _chunk("Complete answer", finish_reason=finish_reason),
        _chunk(usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}),
    ]
    respx.post(f"{STREAM_BASE}/chat/completions").respond(
        200, text=_sse([*chunks, "[DONE]"]), headers={"content-type": "text/event-stream"}
    )
    save_credentials(Credentials(api_key="synthetic-key"))

    result = cli_runner.invoke(app, ["chat", "completions", "create", "hello", "-m", "m", "--stream", "--json"])

    assert result.exit_code == 0, result.output
    frames = [json.loads(line) for line in result.stdout.splitlines()]
    assert frames[0]["choices"][0]["finish_reason"] == finish_reason
    assert frames[-1]["usage"]["total_tokens"] == 6
    assert not result.stderr
