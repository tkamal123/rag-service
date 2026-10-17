"""Settings are read from environment variables at call time, so tests can override them."""
import os
from dataclasses import dataclass

# Free-tier providers expose an OpenAI-compatible API, so one client class covers all of them.
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "key_env": "OPENROUTER_API_KEY",
    },
    "ollama": {  # runs on your own machine: no key, no limits
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2",
        "key_env": None,
    },
    "anthropic": {"base_url": None, "model": "claude-sonnet-5-5", "key_env": "ANTHROPIC_API_KEY"},
}


@dataclass(frozen=True)
class Settings:
    chroma_dir: str
    collection: str
    top_k: int
    max_distance: float
    llm_provider: str
    llm_model: str
    llm_base_url: str | None
    llm_key_env: str | None
    max_input_chars: int
    api_key: str | None
    rate_limit_per_min: int


def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got '{provider}'")
    preset = PROVIDERS[provider]
    return Settings(
        chroma_dir=os.getenv("CHROMA_DIR", "chroma_db"),
        collection=os.getenv("COLLECTION", "docs"),
        top_k=int(os.getenv("TOP_K", "4")),
        # Cosine distance cut-off: if the best chunk is farther than this, we refuse
        # instead of letting the LLM guess. Tune it with the eval set.
        max_distance=float(os.getenv("MAX_DISTANCE", "0.75")),
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL") or preset["model"],
        llm_base_url=os.getenv("LLM_BASE_URL") or preset["base_url"],
        llm_key_env=preset["key_env"],
        max_input_chars=int(os.getenv("MAX_INPUT_CHARS", "1000")),
        api_key=os.getenv("API_KEY") or None,  # protects /query (not the LLM key)
        rate_limit_per_min=int(os.getenv("RATE_LIMIT_PER_MIN", "20")),
    )
