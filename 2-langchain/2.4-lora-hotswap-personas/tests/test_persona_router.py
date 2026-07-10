"""Tests for PersonaRouter."""

import pytest

from adapters import AdapterDef, AdapterRegistry
from persona_router import PersonaRouter


@pytest.fixture
def registry() -> AdapterRegistry:
    reg = AdapterRegistry()
    reg["legal"] = AdapterDef(
        name="legal",
        lora_id=0,
        gguf_path="/legal.gguf",
        tags=["legal", "formal"],
        keywords=["contract", "lawsuit", "clause", "legal"],
        is_default=False,
    )
    reg["chat"] = AdapterDef(
        name="chat",
        lora_id=1,
        gguf_path="/chat.gguf",
        tags=["chat", "general"],
        keywords=["hello", "hi", "general"],
        is_default=True,
    )
    return reg


class TestPersonaRouter:
    def test_route_by_keywords_match(self, registry):
        router = PersonaRouter(registry)
        adapter = router.route_by_keywords("I need to review this contract")
        assert adapter.name == "legal"

    def test_route_by_keywords_fallback_to_default(self, registry):
        router = PersonaRouter(registry)
        adapter = router.route_by_keywords("Hello, how are you?")
        assert adapter.name == "chat"  # default

    def test_route_by_tag(self, registry):
        router = PersonaRouter(registry)
        adapter = router.route_by_tags("legal")
        assert adapter.name == "legal"

    def test_route_by_tag_not_found(self, registry):
        router = PersonaRouter(registry)
        with pytest.raises(ValueError, match="No adapter found"):
            router.route_by_tags("nonexistent")

    def test_list_personas(self, registry):
        router = PersonaRouter(registry)
        personas = router.list_personas()
        assert len(personas) == 2
        names = [p["name"] for p in personas]
        assert "legal" in names
        assert "chat" in names


    def test_set_adapters(self):
        """set_adapters sends correct JSON array."""
        from unittest.mock import patch, MagicMock
        from agent import LoraManager

        mgr = LoraManager(base_url="http://test:8080")
        with patch.object(mgr._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok"}
            mock_post.return_value = mock_response

            result = mgr.set_adapters([
                {"id": 0, "scale": 1.0},
                {"id": 1, "scale": 0.5},
            ])
            assert result == {"status": "ok"}

            call_args = mock_post.call_args
            assert call_args[1]["json"] == [
                {"id": 0, "scale": 1.0},
                {"id": 1, "scale": 0.5},
            ]

    def test_disable_adapter(self):
        """disable_adapter sends POST with scale=0."""
        from unittest.mock import patch, MagicMock
        from agent import LoraManager

        mgr = LoraManager(base_url="http://test:8080")
        with patch.object(mgr._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok"}
            mock_post.return_value = mock_response

            result = mgr.disable_adapter(lora_id=0)
            assert result == {"status": "ok"}

            call_args = mock_post.call_args
            assert call_args[1]["json"] == [{"id": 0, "scale": 0.0}]

class TestLoraManager:
    def test_apply_adapter(self):
        """LoraManager sends correct POST request."""
        from unittest.mock import patch, MagicMock
        from agent import LoraManager

        mgr = LoraManager(base_url="http://test:8080")
        with patch.object(mgr._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_post.return_value = mock_response

            result = mgr.apply_adapter(lora_id=0, scale=1.0)
            assert result == {"success": True}

            # Verify the request (API expects a JSON array)
            call_args = mock_post.call_args
            assert "/lora-adapters" in call_args[0][0]
            assert call_args[1]["json"] == [{"id": 0, "scale": 1.0}]

    def test_remove_adapter(self, registry):
        from unittest.mock import patch, MagicMock
        from agent import LoraManager

        mgr = LoraManager(base_url="http://test:8080")
        with patch.object(mgr._client, "delete") as mock_delete:
            mock_response = MagicMock()
            mock_response.status_code = 204
            mock_delete.return_value = mock_response

            result = mgr.remove_adapter(lora_id=0)
            assert result == {}

    def test_list_adapters(self):
        from unittest.mock import patch, MagicMock
        from agent import LoraManager

        mgr = LoraManager(base_url="http://test:8080")
        with patch.object(mgr._client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"adapters": []}
            mock_get.return_value = mock_response

            result = mgr.list_adapters()
            assert result == {"adapters": []}


class TestLoraAgent:
    def test_select_persona_by_keyword(self):
        """LoraAgent selects the right adapter based on user input."""
        from unittest.mock import patch, MagicMock
        from langchain_core.messages import HumanMessage
        from agent import LoraAgent

        registry = AdapterRegistry()
        registry["legal"] = AdapterDef(
            name="legal", lora_id=0, gguf_path="/legal.gguf",
            keywords=["contract", "lawsuit"],
            is_default=False,
        )
        registry["chat"] = AdapterDef(
            name="chat", lora_id=1, gguf_path="/chat.gguf",
            keywords=["hello"],
            is_default=True,
        )
        router = PersonaRouter(registry)

        agent = LoraAgent(router=router, base_url="http://test:8080")

        with patch.object(agent, "_get_client") as mock_get_client:
            with patch.object(agent._lora_manager, "apply_adapter") as mock_apply:
                mock_client = MagicMock()
                mock_client.post.return_value = MagicMock(
                    status_code=200,
                    json=lambda: {
                        "choices": [
                            {
                                "message": {"content": "Legal analysis result"},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                )
                mock_get_client.return_value = mock_client

                result = agent._generate(
                    messages=[HumanMessage(content="Review this contract")]
                )
                assert len(result.generations) == 1
                # Should have applied the legal adapter
                mock_apply.assert_called_once_with(lora_id=0, scale=1.0)

    def test_switch_persona_manually(self):
        """Manual persona switch works."""
        from unittest.mock import patch
        from agent import LoraAgent

        registry = AdapterRegistry()
        registry["legal"] = AdapterDef(
            name="legal", lora_id=0, gguf_path="/legal.gguf",
        )
        registry["chat"] = AdapterDef(
            name="chat", lora_id=1, gguf_path="/chat.gguf",
        )
        router = PersonaRouter(registry)

        agent = LoraAgent(router=router, base_url="http://test:8080")

        with patch.object(agent._lora_manager, "apply_adapter") as mock_apply:
            agent.switch_persona("legal")
            mock_apply.assert_called_once_with(lora_id=0, scale=1.0)
            assert agent.current_persona == "legal"

    def test_switch_persona_not_found(self):
        """Switching to a nonexistent persona raises KeyError."""
        from agent import LoraAgent

        registry = AdapterRegistry()
        router = PersonaRouter(registry)
        agent = LoraAgent(router=router, base_url="http://test:8080")

        with pytest.raises(KeyError):
            agent.switch_persona("nonexistent")

    def test_list_personas(self):
        """List all available personas."""
        from agent import LoraAgent

        registry = AdapterRegistry()
        registry["legal"] = AdapterDef(
            name="legal", lora_id=0, gguf_path="/legal.gguf",
            description="Legal style",
        )
        router = PersonaRouter(registry)
        agent = LoraAgent(router=router, base_url="http://test:8080")

        personas = agent.list_personas()
        assert len(personas) == 1
        assert personas[0]["name"] == "legal"
