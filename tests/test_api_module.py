from __future__ import annotations

from types import SimpleNamespace

import pytest

import framework.api as api_mod


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.choices = [SimpleNamespace(finish_reason=payload["choices"][0]["finish_reason"])]

    def model_dump(self) -> dict:
        return self._payload


class _FakeCompletions:
    def __init__(self, create_fn) -> None:
        self.create = create_fn


class _FakeOpenAIClient:
    def __init__(self, create_fn) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(create_fn))


def test_chat_passes_tools_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_openai(**kwargs):
        seen["init"] = kwargs

        def create(**create_kwargs):
            seen["create"] = create_kwargs
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        return _FakeOpenAIClient(create)

    monkeypatch.setattr(api_mod, "OpenAI", fake_openai)
    client = api_mod.AgentAPI(
        base_url="http://localhost:11434/v1",
        api_key="benchbot",
        model="hermes3",
        temperature=0.4,
        max_tokens=512,
        timeout=15,
    )

    payload = client.chat(
        [{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "ping"}}],
    )

    assert seen["init"]["base_url"] == "http://localhost:11434/v1"
    assert seen["create"]["model"] == "hermes3"
    assert seen["create"]["temperature"] == 0.4
    assert seen["create"]["max_tokens"] == 512
    assert seen["create"]["tool_choice"] == "auto"
    assert payload["choices"][0]["message"]["content"] == "done"


def test_chat_wraps_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTimeout(Exception):
        pass

    monkeypatch.setattr(api_mod, "APITimeoutError", FakeTimeout)
    monkeypatch.setattr(
        api_mod,
        "OpenAI",
        lambda **kwargs: _FakeOpenAIClient(lambda **create_kwargs: (_ for _ in ()).throw(FakeTimeout("slow"))),
    )
    client = api_mod.AgentAPI("http://localhost:11434/v1", "benchbot", "hermes3")

    with pytest.raises(RuntimeError, match="API timeout: slow"):
        client.chat([{"role": "user", "content": "hello"}])


def test_stream_chat_yields_content_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [
        SimpleNamespace(choices=[]),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hello "))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))]
        ),
    ]
    monkeypatch.setattr(
        api_mod,
        "OpenAI",
        lambda **kwargs: _FakeOpenAIClient(lambda **create_kwargs: chunks),
    )
    client = api_mod.AgentAPI("http://localhost:11434/v1", "benchbot", "hermes3")

    assert list(client.stream_chat([{"role": "user", "content": "hello"}])) == [
        "hello ",
        "world",
    ]


def test_chat_with_stream_assembles_tool_calls_and_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deltas: list[str] = []
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="Plan: ",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                type="function",
                                function=SimpleNamespace(name="set_", arguments='{"k'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="apply config",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                type=None,
                                function=SimpleNamespace(name="mode", arguments='ey":"v"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ]
        ),
    ]
    monkeypatch.setattr(
        api_mod,
        "OpenAI",
        lambda **kwargs: _FakeOpenAIClient(lambda **create_kwargs: chunks),
    )
    client = api_mod.AgentAPI("http://localhost:11434/v1", "benchbot", "hermes3")

    payload = client.chat_with_stream(
        [{"role": "user", "content": "configure it"}],
        on_text_delta=deltas.append,
    )

    assert deltas == ["Plan: ", "apply config"]
    message = payload["choices"][0]["message"]
    assert message["content"] == "Plan: apply config"
    assert message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "set_mode", "arguments": '{"key":"v"}'},
        }
    ]
    assert payload["choices"][0]["finish_reason"] == "tool_calls"


def test_chat_with_stream_wraps_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAPIError(Exception):
        def __init__(self, status_code: int, message: str) -> None:
            super().__init__(message)
            self.status_code = status_code
            self.message = message

    monkeypatch.setattr(api_mod, "APIError", FakeAPIError)
    monkeypatch.setattr(
        api_mod,
        "OpenAI",
        lambda **kwargs: _FakeOpenAIClient(
            lambda **create_kwargs: (_ for _ in ()).throw(FakeAPIError(503, "boom"))
        ),
    )
    client = api_mod.AgentAPI("http://localhost:11434/v1", "benchbot", "hermes3")

    with pytest.raises(RuntimeError, match="API error 503: boom"):
        client.chat_with_stream([{"role": "user", "content": "hello"}])


def test_from_harness_prefers_env_over_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_BASE_URL", "http://env-host:9000/v1")
    monkeypatch.setenv("CUSTOM_API_KEY", "secret")
    monkeypatch.setenv("BENCHBOT_MODEL", "global-model")
    monkeypatch.setattr(api_mod, "OpenAI", lambda **kwargs: SimpleNamespace())

    client = api_mod.AgentAPI.from_harness(
        {
            "name": "hermes",
            "endpoint": {
                "base_url": "http://yaml-host:11434/v1",
                "api_key_env": "CUSTOM_API_KEY",
            },
            "model_defaults": {
                "model": "yaml-model",
                "temperature": 0.6,
                "max_tokens": 2048,
            },
        },
        defaults={"temperature": 0.1, "max_tokens": 1024, "timeout_s": 33},
    )

    assert client.model == "global-model"
    assert client.temperature == 0.6
    assert client.max_tokens == 2048

