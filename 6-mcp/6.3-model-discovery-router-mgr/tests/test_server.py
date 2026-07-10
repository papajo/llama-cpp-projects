"""Tests for MCP server tools."""

import json
from unittest.mock import MagicMock, patch

import pytest

from model_router.server import get_model_info, list_models, route_request


def _mock_urlopen(body: dict):
    mc = MagicMock()
    mc.read.return_value = json.dumps(body).encode()
    cm = MagicMock()
    cm.__enter__.return_value = mc
    return cm


class TestServerTools:
    def test_list_models(self):
        models_resp = {"data": [
            {"id": "gpt-4", "object": "model", "owned_by": "openai"},
        ]}
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_urlopen(models_resp).__enter__.return_value
            result = list_models()
            assert "gpt-4" in result
            assert "completion" in result

    def test_get_model_info_found(self):
        models_resp = {"data": [
            {"id": "gpt-4", "object": "model", "owned_by": "openai"},
        ]}
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_urlopen(models_resp).__enter__.return_value
            result = get_model_info("gpt-4")
            assert "gpt-4" in result
            assert "openai" in result

    def test_get_model_info_not_found(self):
        models_resp = {"data": [
            {"id": "gpt-4", "object": "model", "owned_by": "openai"},
        ]}
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_urlopen(models_resp).__enter__.return_value
            result = get_model_info("nonexistent")
            assert "not found" in result

    def test_route_request(self):
        models_resp = {"data": [
            {"id": "gpt-4", "object": "model", "owned_by": "openai"},
        ]}
        with patch("urllib.request.urlopen") as m:
            m.return_value.__enter__.return_value = _mock_urlopen(models_resp).__enter__.return_value
            result = route_request(task_type="chat")
            assert "Routed" in result
            assert "gpt-4" in result
