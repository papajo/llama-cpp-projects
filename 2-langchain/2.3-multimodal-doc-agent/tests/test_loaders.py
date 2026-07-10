"""Tests for document loaders."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from loaders import ImageLoader, PdfLoader


class TestImageLoader:
    def test_load_png(self):
        """Load a PNG image and get a valid data URI."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name

        try:
            # Create a small test PNG
            img = Image.new("RGB", (100, 50), color="red")
            img.save(tmp_path, format="PNG")

            loader = ImageLoader()
            loaded = loader.load_file(tmp_path)

            assert loaded.width == 100
            assert loaded.height == 50
            assert loaded.data_uri.startswith("data:image/png;base64,")
            assert loaded.file_size_bytes > 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_jpeg(self):
        """Load a JPEG image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name

        try:
            img = Image.new("RGB", (200, 100), color="blue")
            img.save(tmp_path, format="JPEG")

            loader = ImageLoader()
            loaded = loader.load_file(tmp_path)

            assert loaded.media_type == "image/jpeg"
            assert loaded.data_uri.startswith("data:image/jpeg;base64,")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_unsupported_format(self):
        """Unsupported formats raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
            tmp_path = f.name

        try:
            loader = ImageLoader()
            with pytest.raises(ValueError, match="Unsupported image format"):
                loader.load_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Non-existent file raises FileNotFoundError."""
        loader = ImageLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/image.png")

    def test_resize_max_dimensions(self):
        """Images are resized when max_width/max_height is set."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name

        try:
            img = Image.new("RGB", (1000, 800), color="green")
            img.save(tmp_path, format="PNG")

            loader = ImageLoader(max_width=200, max_height=200)
            loaded = loader.load_file(tmp_path)

            assert loaded.width <= 200
            assert loaded.height <= 200
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_folder(self):
        """Load all images from a directory."""
        import os
        tmpdir = tempfile.mkdtemp()

        try:
            # Create a couple of test images
            for name in ["a.png", "b.jpg", "c.gif"]:
                img = Image.new("RGB", (10, 10), color="white")
                img.save(os.path.join(tmpdir, name))

            loader = ImageLoader()
            images = loader.load_folder(tmpdir)

            assert len(images) == 3
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_folder_skip_invalid(self):
        """Invalid files in folder are silently skipped."""
        import os
        tmpdir = tempfile.mkdtemp()

        try:
            # Create valid and invalid files
            img = Image.new("RGB", (10, 10), color="white")
            img.save(os.path.join(tmpdir, "valid.png"))

            # Create a text file
            with open(os.path.join(tmpdir, "readme.txt"), "w") as f:
                f.write("not an image")

            loader = ImageLoader()
            images = loader.load_folder(tmpdir)

            assert len(images) == 1
            assert images[0].path.endswith("valid.png")
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_folder_not_found(self):
        """Non-existent folder raises NotADirectoryError."""
        loader = ImageLoader()
        with pytest.raises(NotADirectoryError):
            loader.load_folder("/nonexistent/folder")


class TestPdfLoader:
    def test_load_nonexistent(self):
        """Non-existent PDF raises FileNotFoundError."""
        loader = PdfLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/report.pdf")
