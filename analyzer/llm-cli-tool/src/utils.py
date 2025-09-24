from src import config


def validate_content(content):
    if not content or not isinstance(content, str):
        raise ValueError("Content must be a non-empty string.")
    return content


def validate_model(model):
    if model not in config.MODEL_LIST.values():
        raise ValueError("Model must be a string from the model list.")
    return model


def format_response(response):
    try:
        return response.json()
    except ValueError:
        return {"error": "Invalid JSON response"}


def process_stream(chunk: dict):
    """
    Process a streaming response chunk.

    Args:
        chunk (dict): The response chunk from the API

    Returns:
        dict: Processed chunk with the extracted content
    """
    try:
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("reasoning_content", "")
            return {"content": content}
        return {"content": ""}
    except Exception as e:
        return {"error": f"Failed to process stream chunk: {str(e)}"}
