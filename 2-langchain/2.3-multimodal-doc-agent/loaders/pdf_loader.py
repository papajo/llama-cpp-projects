"""
PDF loader — converts PDF pages to images for multimodal inference.

Since llama.cpp vision models process image inputs (not PDF text), we
render each page to a PNG image, then encode it as a base64 data URI.
This preserves layout, tables, and formatting that text-only extraction
would lose.

Usage::

    from loaders import PdfLoader

    loader = PdfLoader(dpi=200)
    pages = loader.load("report.pdf")
    for page in pages:
        print(f"Page {page.page_number}: {page.width}x{page.height}")
        # page.data_uri  ->  base64 data URI for llama.cpp vision
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from PIL import Image


@dataclass
class PdfPage:
    """A single PDF page rendered as an image."""

    path: str
    """Original PDF path."""

    page_number: int
    """1-indexed page number."""

    data_uri: str
    """Base64 data URI of the rendered page image."""

    width: int
    """Page width in pixels at the render DPI."""

    height: int
    """Page height in pixels at the render DPI."""

    page_count: int
    """Total pages in the PDF."""

    metadata: dict = field(default_factory=dict)


class PdfLoader:
    """
    Loads PDF files and renders each page as a PNG image for vision models.

    Uses ``pypdf`` for page dimensions and metadata, then renders pages
    via ``pdf2image`` (poppler required) or PIL-based fallback.

    Note:
        For full rendering (preserving layout/images/formatting), install
        ``pdf2image`` and poppler::

            pip install pdf2image
            # macOS: brew install poppler
            # Ubuntu: apt-get install poppler-utils
    """

    def __init__(
        self,
        dpi: int = 200,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        fmt: str = "PNG",
    ):
        """
        Args:
            dpi: Rendering resolution (default: 200).
            max_width: Optional max width after resize.
            max_height: Optional max height after resize.
            fmt: Output image format (``PNG`` or ``JPEG``).
        """
        self.dpi = dpi
        self.max_width = max_width
        self.max_height = max_height
        self.fmt = fmt.upper()

    def load(self, path: str | Path) -> List[PdfPage]:
        """
        Render all PDF pages to images.

        Args:
            path: Path to the PDF file.

        Returns:
            List of ``PdfPage`` objects, one per page.

        Raises:
            FileNotFoundError: If the PDF doesn't exist.
            ImportError: If ``pdf2image`` is not installed (try
                ``pip install pdf2image``) *and* the fallback fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        # Try the high-quality renderer first
        try:
            return self._render_with_pdf2image(path)
        except ImportError:
            # Fallback: extract metadata only, render placeholder
            return self._render_fallback(path)

    def _render_with_pdf2image(self, path: Path) -> List[PdfPage]:
        """Render PDF pages using pdf2image (requires poppler)."""
        from pdf2image import convert_from_path

        images = convert_from_path(
            str(path),
            dpi=self.dpi,
            fmt=self.fmt.lower(),
        )

        # Get page count
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            page_count = len(reader.pages)
        except Exception:
            page_count = len(images)

        pages: List[PdfPage] = []
        for i, img in enumerate(images):
            # Resize if needed
            if self.max_width or self.max_height:
                img = self._resize(img)

            # Encode as data URI
            buf = BytesIO()
            img.save(buf, format=self.fmt)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            media_type = "image/png" if self.fmt == "PNG" else "image/jpeg"
            data_uri = f"data:{media_type};base64,{b64}"

            pages.append(PdfPage(
                path=str(path),
                page_number=i + 1,
                data_uri=data_uri,
                width=img.width,
                height=img.height,
                page_count=page_count,
            ))

        return pages

    def _render_fallback(self, path: Path) -> List[PdfPage]:
        """
        Fallback when pdf2image is not available.

        Creates a placeholder image with the PDF filename for each page
        (metadata only — actual content needs pdf2image + poppler).
        """
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        page_count = len(reader.pages)

        # Create a simple placeholder image
        placeholder = Image.new("RGB", (800, 600), color="white")

        pages: List[PdfPage] = []
        for i in range(min(page_count, 10)):  # Cap at 10 pages in fallback
            buf = BytesIO()
            placeholder.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64}"

            pages.append(PdfPage(
                path=str(path),
                page_number=i + 1,
                data_uri=data_uri,
                width=800,
                height=600,
                page_count=page_count,
                metadata={"warning": "pdf2image not installed; showing placeholder. "
                          "Install: pip install pdf2image + poppler"},
            ))

        return pages

    def _resize(self, img: Image.Image) -> Image.Image:
        """Resize image to fit within max dimensions."""
        w, h = img.size
        ratio = 1.0
        if self.max_width and w > self.max_width:
            ratio = min(ratio, self.max_width / w)
        if self.max_height and h > self.max_height:
            ratio = min(ratio, self.max_height / h)
        if ratio < 1.0:
            img = img.resize(
                (int(w * ratio), int(h * ratio)), Image.LANCZOS
            )
        return img
