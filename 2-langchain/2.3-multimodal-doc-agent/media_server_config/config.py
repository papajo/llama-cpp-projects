"""
Configuration helper for llama.cpp ``--media-path`` file serving.

When using vision models, llama.cpp can serve local files via a built-in
HTTP media server.  This module generates the correct CLI flags and
``file://`` URL mappings.

Usage::

    from media_server_config import MediaServerConfig

    config = MediaServerConfig(media_dir="/path/to/images")
    print(config.cli_args)
    # -> --media-path /path/to/images --media-path-allow-list *.png,*.jpg,*.pdf

    url = config.file_url("invoice-001.png")
    # -> http://127.0.0.1:8081/media/invoice-001.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class MediaServerConfig:
    """
    Generates llama.cpp ``--media-path`` CLI flags and file URL mappings.

    The media server serves files from a directory so that vision models
    can reference them via ``file://`` or HTTP URLs in multimodal messages.
    """

    def __init__(
        self,
        media_dir: str | Path,
        host: str = "127.0.0.1",
        port: int = 8081,
        allow_extensions: Optional[list[str]] = None,
    ):
        """
        Args:
            media_dir: Directory containing media files to serve.
            host: Media server bind host.
            port: Media server port (default: 8081, offset from main server).
            allow_extensions: Allowed file extensions for media serving.
                Default: ``[".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf"]``.
        """
        self.media_dir = Path(media_dir).resolve()
        self.host = host
        self.port = port
        self.allow_extensions = allow_extensions or [
            ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf"
        ]

    @property
    def cli_args(self) -> str:
        """Return llama.cpp CLI flags for media serving."""
        ext_list = ",".join(self.allow_extensions)
        return (
            f"--media-path {self.media_dir} "
            f"--media-path-allow-list {ext_list}"
        )

    @property
    def cli_args_list(self) -> list[str]:
        """Return llama.cpp CLI flags as a list (for subprocess)."""
        ext_list = ",".join(self.allow_extensions)
        return [
            "--media-path", str(self.media_dir),
            "--media-path-allow-list", ext_list,
        ]

    def file_url(self, filename: str) -> str:
        """
        Generate the HTTP URL for a file served by the media server.

        This URL can be passed as ``image_url`` in multimodal messages.
        """
        return f"http://{self.host}:{self.port}/media/{filename}"

    def check_exists(self, filename: str) -> bool:
        """Check if a file exists in the media directory."""
        return (self.media_dir / filename).exists()
