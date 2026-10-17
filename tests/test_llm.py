from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.rag import AnthropicLLM, OpenAICompatLLM, make_llm


def _fake_client(content):
    msg = SimpleNamespace(content=content)
    create = lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=msg)])  # noqa: E731
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_default_provider_is_gemini(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    s = get_settings()
    assert s.llm_provider == "gemini" and "generativelanguage" in s.llm_base_url and s.llm_key_env == "GEMINI_API_KEY"


@pytest.mark.parametrize("provider,env", [("groq", "GROQ_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")])
def test_provider_presets_and_key_lookup(monkeypatch, provider, env):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv(env, "k-123")
    llm = make_llm(get_settings())
    assert isinstance(llm, OpenAICompatLLM) and llm.api_key == "k-123" and llm.configured


def test_model_can_be_overridden(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "my-model")
    assert get_settings().llm_model == "my-model"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        get_settings()


def test_anthropic_provider_still_supported(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(make_llm(get_settings()), AnthropicLLM)


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert make_llm(get_settings()).configured


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    llm = make_llm(get_settings())
    assert not llm.configured
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.generate("sys", "user")


def test_generate_returns_text_and_rejects_empty(monkeypatch):
    llm = OpenAICompatLLM("m", "http://x", "key", "K")
    llm._client = _fake_client("  hello  ")
    assert llm.generate("s", "u") == "hello"
    llm._client = _fake_client(None)
    with pytest.raises(RuntimeError, match="empty"):
        llm.generate("s", "u")
