"""Tests for multimodal extraction chain (mocked HTTP)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, Field

from extraction_chain import MultimodalExtractionChain, VisionChatModel
from media_server_config import MediaServerConfig


# ---------------------------------------------------------------------------
# Test model
# ---------------------------------------------------------------------------


class Invoice(BaseModel):
    vendor: str
    total: float
    date: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_vision_response(content: str) -> httpx.Response:
    """Build a fake vision /v1/chat/completions response."""
    return httpx.Response(
        status_code=200,
        json={
            "model": "vision",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVisionChatModel:
    def test_invoke_with_images(self):
        """VisionChatModel sends multimodal messages correctly."""
        model = VisionChatModel(base_url="http://test:8080")
        with patch.object(model, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_vision_response(
                "The image shows a cat sitting on a chair."
            )
            mock_get.return_value = mock_client

            result = model.invoke_with_images(
                prompt="What's in this image?",
                images=["data:image/png;base64,fakeimage=="],
            )

            assert "cat" in result.lower()

            # Verify the request body
            call_args = mock_client.post.call_args
            sent_messages = call_args[1]["json"]["messages"]
            user_msg = sent_messages[-1]

            assert user_msg["role"] == "user"
            # Content should be a list of parts
            content_parts = user_msg["content"]
            assert isinstance(content_parts, list)
            assert content_parts[0]["type"] == "text"
            assert content_parts[1]["type"] == "image_url"

    def test_invoke_with_system_prompt(self):
        """System prompt is included in the message list."""
        model = VisionChatModel(base_url="http://test:8080")
        with patch.object(model, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_vision_response("OK")
            mock_get.return_value = mock_client

            model.invoke_with_images(
                prompt="Analyze",
                images=["data:image/png;base64,img=="],
                system_prompt="You are a document analyst.",
            )

            call_args = mock_client.post.call_args
            sent = call_args[1]["json"]["messages"]
            assert sent[0]["role"] == "system"
            assert sent[0]["content"] == "You are a document analyst."


class TestMultimodalChain:
    def test_extract_raw_text(self):
        """Extract returns raw text when no schema is provided."""
        chain = MultimodalExtractionChain(base_url="http://test:8080")
        with patch.object(chain._model, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_vision_response(
                "This is a receipt from Acme Corp."
            )
            mock_get.return_value = mock_client

            result = chain.extract("data:image/png;base64,fake==")
            assert isinstance(result, str)
            assert "Acme Corp" in result

    def test_extract_structured(self):
        """Extract returns a Pydantic object when schema is set."""
        chain = MultimodalExtractionChain(
            base_url="http://test:8080",
            schema=Invoice,
        )
        with patch.object(chain._model, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_vision_response(
                json.dumps({"vendor": "Acme Corp", "total": 150.00, "date": "2025-01-15"})
            )
            mock_get.return_value = mock_client

            result = chain.extract("data:image/png;base64,fake==")
            assert isinstance(result, Invoice)
            assert result.vendor == "Acme Corp"
            assert result.total == 150.00
            assert result.date == "2025-01-15"

    def test_extract_with_code_fences(self):
        """Model output wrapped in ```json fences is still parsed."""
        chain = MultimodalExtractionChain(
            base_url="http://test:8080",
            schema=Invoice,
        )
        with patch.object(chain._model, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_vision_response(
                "```json\n{\"vendor\": \"Beta Inc\", \"total\": 75.50, \"date\": \"2025-06-01\"}\n```"
            )
            mock_get.return_value = mock_client

            result = chain.extract("data:image/png;base64,fake==")
            assert result.vendor == "Beta Inc"
            assert result.total == 75.50

    def test_extract_batch(self):
        """extract_batch processes multiple images."""
        chain = MultimodalExtractionChain(base_url="http://test:8080")
        with patch.object(chain._model, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_vision_response("Result")
            mock_get.return_value = mock_client

            results = chain.extract_batch([
                "data:image/png;base64,img1==",
                "data:image/png;base64,img2==",
            ])
            assert len(results) == 2
            assert mock_client.post.call_count == 2

    def test_extract_with_context(self):
        """extract_with_context includes context in the prompt."""
        chain = MultimodalExtractionChain(base_url="http://test:8080")
        with patch.object(chain._model, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_vision_response("Context-aware result")
            mock_get.return_value = mock_client

            result = chain.extract_with_context(
                "data:image/png;base64,img==",
                context="Page 2 of 5",
                question="What is the total?",
            )
            assert isinstance(result, str)
            # Verify context was included
            call_args = mock_client.post.call_args
            sent = call_args[1]["json"]["messages"]
            user_content = sent[-1]["content"]
            # Should be a list of parts; check the text part
            text_parts = [p["text"] for p in user_content if p["type"] == "text"]
            assert any("Page 2 of 5" in p for p in text_parts)

    def test_parse_json_error(self):
        """Invalid JSON raises ValueError."""
        chain = MultimodalExtractionChain(
            base_url="http://test:8080",
            schema=Invoice,
        )
        with patch.object(chain._model, "_get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.post.return_value = make_vision_response("not json at all")
            mock_get.return_value = mock_client

            with pytest.raises(ValueError, match="Failed to parse"):
                chain.extract("data:image/png;base64,fake==")


class TestMediaServerConfig:
    def test_cli_args(self):
        """CLI args are correctly formatted."""
        config = MediaServerConfig(media_dir="/data/images")
        args = config.cli_args
        assert "--media-path /data/images" in args
        assert ".png" in args
        assert ".jpg" in args

    def test_file_url(self):
        """File URLs are correctly generated."""
        config = MediaServerConfig(
            media_dir="/data/images",
            host="127.0.0.1",
            port=8081,
        )
        url = config.file_url("invoice.png")
        assert url == "http://127.0.0.1:8081/media/invoice.png"

    def test_check_exists(self):
        """check_exists returns True/False correctly."""
        import tempfile
        import os
        tmpdir = tempfile.mkdtemp()
        try:
            config = MediaServerConfig(media_dir=tmpdir)

            # File doesn't exist yet
            assert not config.check_exists("nonexistent.png")

            # Create the file
            Path(os.path.join(tmpdir, "exists.png")).touch()
            assert config.check_exists("exists.png")
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
