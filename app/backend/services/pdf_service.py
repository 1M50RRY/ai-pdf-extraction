"""
PDF processing service using pdf2image (poppler).

Handles conversion of PDF documents to PIL Images for AI processing.
"""

import io
import logging
from typing import BinaryIO

from PIL import Image

logger = logging.getLogger(__name__)


class PDFConversionError(Exception):
    """Raised when PDF conversion fails."""

    pass


class PDFService:
    """
    Service for PDF processing operations.

    Uses pdf2image (backed by poppler) to convert PDF pages to images.
    """

    def __init__(self, dpi: int = 200, image_format: str = "PNG"):
        """
        Initialize the PDF service.

        Args:
            dpi: Resolution for PDF to image conversion. Higher = better quality but slower.
            image_format: Output image format (PNG recommended for quality).
        """
        self.dpi = dpi
        self.image_format = image_format

    def convert_pdf_to_images(
        self,
        file_bytes: bytes | BinaryIO,
        first_page: int | None = None,
        last_page: int | None = None,
    ) -> list[Image.Image]:
        """
        Convert PDF pages to PIL Images.

        Args:
            file_bytes: PDF file as bytes or file-like object.
            first_page: First page to convert (1-indexed, inclusive). None for first page.
            last_page: Last page to convert (1-indexed, inclusive). None for last page.

        Returns:
            List of PIL Image objects, one per page.

        Raises:
            PDFConversionError: If conversion fails for any reason.
        """
        try:
            # Import here to provide clear error if poppler not installed
            from pdf2image import convert_from_bytes
            from pdf2image.exceptions import (
                PDFInfoNotInstalledError,
                PDFPageCountError,
                PDFSyntaxError,
            )
        except ImportError as e:
            logger.error("pdf2image not installed: %s", e)
            raise PDFConversionError(
                "pdf2image library not installed. Run: pip install pdf2image"
            ) from e

        # Ensure we have bytes
        if hasattr(file_bytes, "read"):
            pdf_bytes = file_bytes.read()
        else:
            pdf_bytes = file_bytes

        if not pdf_bytes:
            raise PDFConversionError("Empty PDF file provided")

        # Validate PDF magic bytes
        if not pdf_bytes[:4] == b"%PDF":
            raise PDFConversionError(
                "Invalid PDF file: does not start with PDF header"
            )

        try:
            logger.info(
                "Converting PDF to images (dpi=%d, pages=%s-%s)",
                self.dpi,
                first_page or "first",
                last_page or "last",
            )

            images = convert_from_bytes(
                pdf_bytes,
                dpi=self.dpi,
                fmt=self.image_format.lower(),
                first_page=first_page,
                last_page=last_page,
                thread_count=2,  # Parallel processing for multi-page PDFs
            )

            logger.info("Successfully converted %d page(s)", len(images))
            return images

        except PDFInfoNotInstalledError as e:
            logger.error("Poppler not installed: %s", e)
            raise PDFConversionError(
                "Poppler not installed. Install poppler-utils: "
                "brew install poppler (macOS) or apt-get install poppler-utils (Linux)"
            ) from e

        except PDFPageCountError as e:
            logger.error("Could not get PDF page count: %s", e)
            raise PDFConversionError(
                f"Could not determine PDF page count: {e}"
            ) from e

        except PDFSyntaxError as e:
            logger.error("PDF syntax error: %s", e)
            raise PDFConversionError(f"Invalid or corrupted PDF file: {e}") from e

        except Exception as e:
            logger.exception("Unexpected error during PDF conversion")
            raise PDFConversionError(f"PDF conversion failed: {e}") from e

    def convert_first_page(self, file_bytes: bytes | BinaryIO) -> Image.Image:
        """
        Convenience method to convert only the first page.

        Args:
            file_bytes: PDF file as bytes or file-like object.

        Returns:
            PIL Image of the first page.

        Raises:
            PDFConversionError: If conversion fails.
        """
        images = self.convert_pdf_to_images(file_bytes, first_page=1, last_page=1)
        if not images:
            raise PDFConversionError("No pages found in PDF")
        return images[0]

    def get_page_count(self, file_bytes: bytes | BinaryIO) -> int:
        """
        Get the total number of pages in a PDF.

        Args:
            file_bytes: PDF file as bytes or file-like object.

        Returns:
            Number of pages in the PDF.

        Raises:
            PDFConversionError: If page count cannot be determined.
        """
        try:
            from pdf2image import pdfinfo_from_bytes
        except ImportError as e:
            raise PDFConversionError(
                "pdf2image library not installed"
            ) from e

        # Ensure we have bytes
        if hasattr(file_bytes, "read"):
            pdf_bytes = file_bytes.read()
        else:
            pdf_bytes = file_bytes

        try:
            info = pdfinfo_from_bytes(pdf_bytes)
            return info.get("Pages", 0)
        except Exception as e:
            logger.error("Could not get page count: %s", e)
            raise PDFConversionError(f"Could not get page count: {e}") from e

    def image_to_bytes(
        self, image: Image.Image, format: str = "PNG", quality: int = 95
    ) -> bytes:
        """
        Convert a PIL Image to bytes.

        Args:
            image: PIL Image to convert.
            format: Output format (PNG, JPEG, etc.).
            quality: Quality for lossy formats (1-100).

        Returns:
            Image as bytes.
        """
        buffer = io.BytesIO()
        save_kwargs = {"format": format}
        if format.upper() in ("JPEG", "JPG", "WEBP"):
            save_kwargs["quality"] = quality
        image.save(buffer, **save_kwargs)
        buffer.seek(0)
        return buffer.getvalue()


# Singleton instance for convenience
_pdf_service: PDFService | None = None


def get_pdf_service() -> PDFService:
    """Get or create the PDF service singleton."""
    global _pdf_service
    if _pdf_service is None:
        _pdf_service = PDFService()
    return _pdf_service

