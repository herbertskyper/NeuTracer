import os
import json
from pathlib import Path

# Define the settings directory in the user's home folder
SETTINGS_DIR = os.path.expanduser("~/.llm-cli-tool")
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "settings.json")


def ensure_settings_dir():
    """Ensure the settings directory exists."""
    if not os.path.exists(SETTINGS_DIR):
        os.makedirs(SETTINGS_DIR, exist_ok=True)


def load_settings():
    """Load settings from the settings file."""
    ensure_settings_dir()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_settings(settings):
    """Save settings to the settings file."""
    ensure_settings_dir()
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        # Set permissions to be readable only by the owner
        os.chmod(SETTINGS_FILE, 0o600)
        return True
    except IOError:
        return False


def get_saved_api_key():
    """Get the saved API key if it exists."""
    settings = load_settings()
    return settings.get("api_key")


def save_api_key(api_key):
    """Save the API key to the settings file."""
    settings = load_settings()
    settings["api_key"] = api_key
    return save_settings(settings)
