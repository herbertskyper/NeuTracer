from typing import Generator, Dict, Any, Union, Optional
import requests
import json
from .config import get_api_config, MODEL_LIST, get_api_key
from .utils import validate_content, validate_model, format_response, process_stream


def send_request(
    content: str, model: Optional[str] = None, stream: bool = True
) -> Union[Generator[Dict[str, Any], None, None], Dict[str, Any]]:
    """
    Send a request to the language model API.

    Args:
        content (str): The content to send to the API.
        model (str, optional): The model to use. Default from config if not provided.
        stream (bool, optional): Whether to stream the response. Default is True.

    Returns:
        If stream=True: Generator yielding response chunks
        If stream=False: dict: The API response as a dictionary.
    """
    try:
        config: Dict[str, Any] = get_api_config()
        validated_content: str = validate_content(content)

        if model is None:
            model = config["default_model"]
            model = validate_model(model)
        else:
            model = validate_model(MODEL_LIST.get(model, model))

        payload: Dict[str, Any] = {
            "model": model,
            "stream": stream,
            "max_tokens": config["default_max_tokens"],
            "temperature": config["default_temperature"],
            "top_p": config["default_top_p"],
            "top_k": config["default_top_k"],
            "frequency_penalty": config["default_frequency_penalty"],
            "n": config["default_n"],
            "messages": [{"role": "user", "content": validated_content}],
        }

        if stream:
            return stream_request(config["url"], payload, config["headers"])
        else:
            response: requests.Response = requests.post(
                config["url"], json=payload, headers=config["headers"]
            )
            response.raise_for_status()
            return format_response(response)
    except requests.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


def stream_request(
    url: str, payload: Dict[str, Any], headers: Dict[str, str]
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream the request and yield chunks as they arrive.

    Args:
        url (str): API endpoint URL
        payload (dict): Request payload
        headers (dict): Request headers

    Yields:
        dict: Parsed response chunks
    """
    with requests.post(url, json=payload, headers=headers, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            # print(line)
            if line:
                # Skip the "data: " prefix and empty lines
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data: str = line[6:]  # Remove 'data: ' prefix
                    if data == "[DONE]":
                        break
                    try:
                        chunk: Dict[str, Any] = json.loads(data)
                        yield process_stream(chunk)
                    except json.JSONDecodeError:
                        yield {"error": f"Failed to decode JSON: {data}"}
