"""Tests for PDF service."""

import pytest

from app.backend.services.pdf_service import PDFConversionError, PDFService


class TestPDFService:
    """Tests for PDFService class."""

    def test_init_default_values(self):
        """Test PDFService initializes with default values."""
        service = PDFService()
        assert service.dpi == 200
        assert service.image_format == "PNG"

    def test_init_custom_values(self):
        """Test PDFService accepts custom configuration."""
        service = PDFService(dpi=300, image_format="JPEG")
        assert service.dpi == 300
        assert service.image_format == "JPEG"

    def test_convert_empty_file_raises_error(self):
        """Test that empty file raises PDFConversionError."""
        service = PDFService()
        with pytest.raises(PDFConversionError) as exc_info:
            service.convert_pdf_to_images(b"")
        assert "Empty" in str(exc_info.value)

    def test_convert_invalid_pdf_raises_error(self):
        """Test that non-PDF content raises PDFConversionError."""
        service = PDFService()
        with pytest.raises(PDFConversionError) as exc_info:
            service.convert_pdf_to_images(b"This is not a PDF")
        assert "Invalid PDF" in str(exc_info.value) or "does not start" in str(
            exc_info.value
        )

    def test_image_to_bytes_png(self):
        """Test converting PIL Image to bytes."""
        from PIL import Image

        service = PDFService()
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        result = service.image_to_bytes(img, format="PNG")

        assert isinstance(result, bytes)
        assert len(result) > 0
        # PNG magic bytes
        assert result[:4] == b"\x89PNG"

    def test_image_to_bytes_jpeg(self):
        """Test converting PIL Image to JPEG bytes."""
        from PIL import Image

        service = PDFService()
        img = Image.new("RGB", (100, 100), color="blue")
        result = service.image_to_bytes(img, format="JPEG", quality=85)

        assert isinstance(result, bytes)
        assert len(result) > 0
        # JPEG magic bytes
        assert result[:2] == b"\xff\xd8"

