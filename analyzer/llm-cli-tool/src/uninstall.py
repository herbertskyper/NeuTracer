import os
import shutil


def main():
    """Remove all settings files created by llm-cli-tool."""
    settings_dir = os.path.expanduser("~/.llm-cli-tool")
    if os.path.exists(settings_dir):
        try:
            shutil.rmtree(settings_dir)
            print(f"Successfully removed settings directory: {settings_dir}")
        except Exception as e:
            print(f"Failed to remove settings directory: {e}")
    else:
        print(f"No settings directory found at: {settings_dir}")


if __name__ == "__main__":
    main()
