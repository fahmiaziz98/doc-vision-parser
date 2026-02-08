import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import fitz
import numpy as np
from PIL import Image

from ..core.types import ImageFormat
from .crop import ContentCropper
from .rotate import ImageRotator


class ImageProcessor:
    """
    Handles PDF to image conversion and subsequent image optimization including
    auto-rotation, cropping, resizing, and base64 encoding.
    """

    def __init__(
        self,
        dpi: int = 300,
        auto_crop: bool = False,
        auto_rotate: bool = False,
        resize: bool = True,
        max_dimension: int = 2048,
        crop_padding: int = 10,
        crop_ignore_bottom_percent: float = 12.0,
        crop_footer_gap_threshold: int = 100,
        crop_column_ink_ratio: float = 0.01,
        crop_row_ink_ratio: float = 0.002,
        rotate_score_diff_threshold: float = 0.35,
        rotate_small_angle_threshold: float = 0.1,
        rotate_analysis_max_size: int = 1500,
        debug_save_path: Optional[str] = None,
    ):
        """
        Initialize the ImageProcessor.

        Args:
            dpi: Dots per inch for PDF rendering.
            auto_crop: Whether to automatically crop the image content.
            auto_rotate: Whether to automatically correct the image orientation.
            resize: Whether to resize the image to a fixed or maximum dimension.
            max_dimension: Maximum dimension (width or height) for the processed image.
            crop_padding: Padding for the auto-crop.
            crop_ignore_bottom_percent: Footer ignore percentage for auto-crop.
            crop_footer_gap_threshold: Vertical gap threshold for auto-crop.
            crop_column_ink_ratio: Column ink ratio for auto-crop.
            crop_row_ink_ratio: Row ink ratio for auto-crop.
            rotate_score_diff_threshold: Minimum score difference for rotation confidence.
            rotate_small_angle_threshold: Minimum angle for micro-rotation correction.
            rotate_analysis_max_size: Max dimension for rotation analysis.
            debug_save_path: Path to save intermediate images for debugging.
        """
        self.dpi = dpi
        self.auto_crop = auto_crop
        self.auto_rotate = auto_rotate
        self.resize = resize
        self.max_dimension = max_dimension
        self.debug_save_path = Path(debug_save_path) if debug_save_path else None

        if self.debug_save_path:
            self.debug_save_path.mkdir(parents=True, exist_ok=True)

        self.cropper = (
            ContentCropper(
                padding=crop_padding,
                ignore_bottom_percent=crop_ignore_bottom_percent,
                footer_gap_threshold=crop_footer_gap_threshold,
                column_ink_ratio=crop_column_ink_ratio,
                row_ink_ratio=crop_row_ink_ratio,
            )
            if auto_crop
            else None
        )

        self.rotator = (
            ImageRotator(
                score_diff_threshold=rotate_score_diff_threshold,
                small_angle_threshold=rotate_small_angle_threshold,
                analysis_max_size=rotate_analysis_max_size,
            )
            if auto_rotate
            else None
        )

    def pdf_to_images(
        self,
        pdf_path: Union[str, Path],
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Convert specific pages of a PDF file into a list of PIL Images.

        Args:
            pdf_path: Path to the PDF file.
            start_page: Index of the first page to convert (0-indexed).
            end_page: Index of the last page to convert (inclusive).

        Returns:
            A list of PIL Color images.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        total_pages = doc.page_count

        start = start_page or 0
        end = end_page or (total_pages - 1)

        start = max(0, min(start, total_pages - 1))
        end = max(start, min(end, total_pages - 1))

        images = []
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(start, end + 1):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()
        return images

    def process_image(
        self, image: Image.Image, page_num: int = 0, doc_name: str = "image"
    ) -> Image.Image:
        """
        Apply processing operations such as auto-rotation, cropping, and resizing to an image.

        Args:
            image: PIL Image input.
            page_num: Page index for logging/debugging.
            doc_name: Document identifier for debugging filenames.

        Returns:
            The processed PIL Image.
        """
        img_np = np.array(image)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 1. Auto-Rotate
        if self.auto_rotate and self.rotator:
            img_bgr = self.rotator.rotate(img_bgr)

        # 2. Auto-Crop
        if self.auto_crop and self.cropper:
            img_bgr = self.cropper.crop(img_bgr)

        # 3. Resize
        if self.resize:
            img_bgr = self._resize_image(img_bgr)
        else:
            # Thumbnail if exceeds max dimension
            h, w = img_bgr.shape[:2]
            if max(h, w) > self.max_dimension:
                scale = self.max_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Convert back to PIL RGB
        processed_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # Debug Save
        if self.debug_save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filename = f"{doc_name}_page{page_num:03d}_{timestamp}.png"
            debug_path = self.debug_save_path / debug_filename
            processed_image.save(debug_path, "PNG")

        return processed_image

    def encode_image(
        self,
        image: Image.Image,
        image_format: ImageFormat = ImageFormat.JPEG,
        jpeg_quality: int = 95,
    ) -> Tuple[str, str]:
        """
        Encode a PIL Image into a base64 string.

        Args:
            image: PIL Image to encode.
            image_format: Targeted image format (JPEG or PNG).
            jpeg_quality: Quality factor for JPEG encoding (0-100).

        Returns:
            A tuple containing (base64_string, mime_type).
        """
        buffer = BytesIO()

        if image_format == ImageFormat.JPEG:
            image.save(
                buffer,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
                subsampling=0,
            )
            mime_type = "image/jpeg"
        else:
            image.save(buffer, format="PNG", optimize=True)
            mime_type = "image/png"

        b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return b64_data, mime_type

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize large images to prevent memory issues and optimize for VLMs.

        Args:
            image: Image as a numpy array.

        Returns:
            Resized image as a numpy array.
        """
        h, w = image.shape[:2]

        if max(h, w) > self.max_dimension:
            scale = self.max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return image
