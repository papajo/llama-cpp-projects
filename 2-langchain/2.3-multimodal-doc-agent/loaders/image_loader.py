"""
Image loader — converts images to base64 data URIs for llama.cpp vision.

llama.cpp's ``/v1/chat/completions`` endpoint supports multimodal messages
with ``image_url`` content.  This loader reads image files from disk and
encodes them as base64 data URIs that can be passed directly in the
request body — no need for ``--media-path`` file serving.

Usage::

    from loaders import ImageLoader

    loader = ImageLoader()
    images = loader.load_folder("invoices/")
    for img in images:
        print(img.path, img.width, img.height, img.data_uri[:80] + "...")
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image


@dataclass
class LoadedImage:
    """A single loaded image, ready for multimodal inference."""

    path: str
    """Original file path."""

    data_uri: str
    """Base64 data URI (e.g. ``data:image/png;base64,iVBOR...``)."""

    media_type: str
    """MIME type (e.g. ``image/png``, ``image/jpeg``)."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    file_size_bytes: int
    """Original file size on disk."""

    metadata: Dict[str, str] = field(default_factory=dict)
    """Optional metadata (e.g. ``{"source": "scanned-invoice-001.png"}``)."""


# Map file extensions to MIME types
MIME_MAP: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

SUPPORTED_EXTENSIONS = set(MIME_MAP.keys())


class ImageLoader:
    """
    Loads images from disk and encodes them as base64 data URIs.

    Supports PNG, JPEG, GIF, WebP, and BMP.
    Optionally resizes images to stay within llama.cpp's token budget
    (``--image-max-tokens``).
    """

    def __init__(
        self,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        max_file_size_mb: float = 20.0,
    ):
        """
        Args:
            max_width: Optional max width in pixels (aspect-ratio preserved).
            max_height: Optional max height in pixels (aspect-ratio preserved).
            max_file_size_mb: Skip files larger than this (default: 20 MB).
        """
        self.max_width = max_width
        self.max_height = max_height
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

    def load_file(self, path: str | Path) -> LoadedImage:
        """
        Load a single image file.

        Args:
            path: Path to the image file.

        Returns:
            A ``LoadedImage`` with base64-encoded data URI.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file type is unsupported or too large.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format {ext}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        file_size = path.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise ValueError(
                f"File too large: {file_size / 1024**2:.1f} MB "
                f"(max {self.max_file_size_bytes / 1024**2:.0f} MB)"
            )

        media_type = MIME_MAP[ext]

        img = Image.open(path)
        original_width, original_height = img.size

        # Resize if needed (preserving aspect ratio)
        if self.max_width or self.max_height:
            img = self._resize(img)

        # Encode as base64 data URI
        buffer = img if img.format == path.suffix[1:].upper() else img
        # Re-save to bytes to get the encoded version
        from io import BytesIO

        buf = BytesIO()
        # Determine format — use original or fall back to PNG
        save_format = img.format or "PNG"
        img.save(buf, format=save_format)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = f"data:{media_type};base64,{b64}"

        return LoadedImage(
            path=str(path),
            data_uri=data_uri,
            media_type=media_type,
            width=img.width,
            height=img.height,
            file_size_bytes=file_size,
            metadata={"original_width": str(original_width),
                      "original_height": str(original_height)},
        )

    def load_folder(
        self,
        folder: str | Path,
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[LoadedImage]:
        """
        Load all supported images from a folder.

        Args:
            folder: Directory path.
            pattern: Glob pattern (default: ``*``).
            recursive: Search subdirectories.

        Returns:
            List of ``LoadedImage`` objects.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")

        results: List[LoadedImage] = []
        glob_method = folder.rglob if recursive else folder.glob

        for ext in SUPPORTED_EXTENSIONS:
            for file_path in glob_method(f"*{ext}"):
                try:
                    loaded = self.load_file(file_path)
                    results.append(loaded)
                except (ValueError, FileNotFoundError) as e:
                    # Skip files that fail validation
                    continue

        return results

    def _resize(self, img: Image.Image) -> Image.Image:
        """Resize image to fit within max dimensions, preserving aspect ratio."""
        w, h = img.size
        ratio = 1.0

        if self.max_width and w > self.max_width:
            ratio = min(ratio, self.max_width / w)
        if self.max_height and h > self.max_height:
            ratio = min(ratio, self.max_height / h)

        if ratio < 1.0:
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        return img
