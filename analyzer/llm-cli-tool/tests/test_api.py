import unittest
from unittest.mock import patch
from src.api import send_request


class TestAPI(unittest.TestCase):

    @patch("src.api.requests.request")
    def test_send_request_success(self, mock_request):
        mock_request.return_value.text = '{"success": true, "data": "response data"}'
        content = "What opportunities and challenges will the Chinese large model industry face in 2025?"
        model = "r1"

        response = send_request(content, model)

        self.assertEqual(response, {"success": True, "data": "response data"})
        mock_request.assert_called_once()

    @patch("src.api.requests.request")
    def test_send_request_failure(self, mock_request):
        mock_request.return_value.text = '{"success": false, "error": "some error"}'
        content = "What opportunities and challenges will the Chinese large model industry face in 2025?"
        model = "qwen"

        response = send_request(content, model)

        self.assertEqual(response, {"success": False, "error": "some error"})
        mock_request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
