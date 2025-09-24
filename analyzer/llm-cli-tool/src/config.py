import os
from .settings import get_saved_api_key

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
AUTHORIZATION_HEADER_PREFIX = "Bearer "

DEFAULT_MODEL = "qwen-free"
MODEL_LIST = {
    "qwen-free": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "r1": "deepseek-ai/DeepSeek-R1",
    "qwen": "Qwen/QwQ-32B-Preview",
}
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.7
DEFAULT_TOP_K = 50
DEFAULT_FREQUENCY_PENALTY = 0.5
DEFAULT_N = 1

# Try to get API key from:
# 1. Environment variable
# 2. Saved settings
_api_key = os.environ.get("SILICONFLOW_API_KEY", "") or get_saved_api_key() or ""


def set_api_key(key):
    """Set the API key for requests."""
    global _api_key
    _api_key = key


def get_api_key():
    """Get the current API key."""
    return _api_key


def get_api_config(api_key=None):
    """
    Get API configuration with optional API key override.

    Args:
        api_key (str, optional): Override the current API key.

    Returns:
        dict: API configuration
    """
    # Use provided key or the global one
    key = api_key if api_key is not None else _api_key

    return {
        "url": API_URL,
        "headers": {
            "Authorization": f"{AUTHORIZATION_HEADER_PREFIX}{key}",
            "Content-Type": "application/json",
        },
        "default_model": DEFAULT_MODEL,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
        "default_temperature": DEFAULT_TEMPERATURE,
        "default_top_p": DEFAULT_TOP_P,
        "default_top_k": DEFAULT_TOP_K,
        "default_frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        "default_n": DEFAULT_N,
    }
