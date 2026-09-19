"""
LoRA adapter manager — handles ``POST /lora-adapters`` API calls to the
llama.cpp server for hot-swapping adapters at runtime.

Usage::

    from agent import LoraManager

    mgr = LoraManager(base_url="http://127.0.0.1:8080")
    mgr.apply_adapter(lora_id=0, scale=1.0)   # Apply legal-tone
    mgr.remove_adapter(lora_id=0)              # Remove it
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class LoraManager:
    """
    Manages LoRA adapter lifecycles on a llama.cpp server.

    Wraps ``POST /lora-adapters`` (apply/set scale) and
    ``DELETE /lora-adapters`` (remove).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        request_timeout: float = 30.0,
    ):
        """
        Args:
            base_url: llama.cpp server URL.
            request_timeout: HTTP request timeout.
        """
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self._client = httpx.Client(timeout=request_timeout)

    def apply_adapter(self, lora_id: int, scale: float = 1.0) -> Dict[str, Any]:
        """
        Apply a LoRA adapter (or set of adapters) by slot ID.

        Corresponds to ``POST /lora-adapters`` which accepts a JSON
        **array** of ``{"id": …, "scale": …}`` objects. This method
        sends a single-entry list for convenience.

        Args:
            lora_id: LoRA slot ID (0-indexed, matching ``--lora`` order).
            scale: Adapter scale factor.

        Returns:
            JSON response from the server.

        Raises:
            httpx.HTTPStatusError: If the server rejects the request.
        """
        return self.set_adapters([{"id": lora_id, "scale": scale}])

    def set_adapters(self, adapters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Set the global scale for multiple LoRA adapters at once.

        Corresponds to ``POST /lora-adapters`` with a JSON array body::

            [
              {"id": 0, "scale": 1.0},
              {"id": 1, "scale": 0.0},
            ]

        Setting an adapter's scale to 0 effectively disables it.

        Args:
            adapters: List of ``{"id": …, "scale": …}`` dicts.

        Returns:
            JSON response from the server.

        Raises:
            httpx.HTTPStatusError: If the server rejects the request.
        """
        response = self._client.post(
            f"{self.base_url}/lora-adapters",
            json=adapters,
        )
        if response.status_code >= 400:
            try:
                req = response.request
            except RuntimeError:
                req = None
            raise httpx.HTTPStatusError(
                f"Failed to set LoRA adapters: {response.status_code}",
                request=req,
                response=response,
            )
        return response.json()

    def disable_adapter(self, lora_id: int) -> Dict[str, Any]:
        """
        Disable a LoRA adapter by setting its scale to 0.

        This is the documented way to "remove" an adapter. The
        ``DELETE /lora-adapters/<id>`` endpoint may also be available
        in newer builds but is not guaranteed across all versions.

        Args:
            lora_id: LoRA slot ID to disable.
        """
        return self.apply_adapter(lora_id=lora_id, scale=0.0)

    def remove_adapter(self, lora_id: int) -> Dict[str, Any]:
        """
        Remove a LoRA adapter by slot ID.

        Uses ``DELETE /lora-adapters/<id>`` if available in the server
        build. Falls back to :meth:`disable_adapter` (POST with
        scale=0) on 404/405.

        Args:
            lora_id: LoRA slot ID to remove.
        """
        try:
            response = self._client.delete(
                f"{self.base_url}/lora-adapters/{lora_id}",
            )
            if response.status_code >= 400:
                try:
                    req = response.request
                except RuntimeError:
                    req = None
                raise httpx.HTTPStatusError(
                    f"Failed to remove LoRA adapter {lora_id}: "
                    f"{response.status_code}",
                    request=req,
                    response=response,
                )
            return response.json() if response.status_code != 204 else {}
        except httpx.HTTPStatusError:
            # Fallback: disable via scale=0
            return self.disable_adapter(lora_id)

    def list_adapters(self) -> List[Dict[str, Any]]:
        """
        List currently loaded LoRA adapters.

        Corresponds to ``GET /lora-adapters``, which returns a JSON **array**
        of adapter objects -- not an object. Verified against a real server:
        with no ``--lora`` flags it returns ``[]``.
        """
        response = self._client.get(f"{self.base_url}/lora-adapters")
        if response.status_code >= 400:
            try:
                req = response.request
            except RuntimeError:
                req = None
            raise httpx.HTTPStatusError(
                f"Failed to list LoRA adapters: {response.status_code}",
                request=req,
                response=response,
            )
        return response.json()

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:
        self.close()
