import cv2
import numpy as np


class ImageRotator:
    """
    Handles automatic image rotation detection and correction for document images.

    Uses Hough Transform to detect line angles and heuristic scoring to determine
    the optimal orientation for document readability.
    """

    def __init__(
        self,
        score_diff_threshold: float = 0.35,
        small_angle_threshold: float = 0.1,
        analysis_max_size: int = 1500,
    ):
        """
        Initialize the ImageRotator.

        Args:
            score_diff_threshold: Minimum score difference required to confidently rotate.
            small_angle_threshold: Minimum angle (degrees) to apply micro-rotation correction.
            analysis_max_size: Maximum dimension for downsampled analysis (speeds up detection).
        """
        self.score_diff_threshold = score_diff_threshold
        self.small_angle_threshold = small_angle_threshold
        self.analysis_max_size = analysis_max_size

    def rotate(self, img: np.ndarray) -> np.ndarray:
        """
        Automatically detect and correct image orientation.

        Args:
            img: Input image (numpy array in BGR format).

        Returns:
            Corrected image (numpy array in BGR format).
        """
        # Pre-process: downsample if too large to speed up analysis
        h, w = img.shape[:2]
        analysis_img = img

        if max(h, w) > self.analysis_max_size:
            scale = self.analysis_max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            analysis_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Step 1: Detect angle using Hough Transform
        median_angle = self._detect_angle(analysis_img)
        if median_angle is None:
            return img  # No lines detected, return original

        # Step 2: Handle ~90 degree rotations with scoring
        if 85 <= abs(median_angle) <= 95:
            return self._handle_90_degree_rotation(img, analysis_img)

        # Step 3: Handle small skew corrections
        if abs(median_angle) <= 5:
            if abs(median_angle) > self.small_angle_threshold:
                return self._rotate_arbitrary(img, median_angle)
            return img

        # Step 4: Handle 180 degree flip
        if abs(median_angle) > 175:
            return cv2.rotate(img, cv2.ROTATE_180)

        # Default: apply detected angle
        return self._rotate_arbitrary(img, median_angle)

    def _detect_angle(self, img: np.ndarray) -> float:
        """
        Detect the dominant line angle in the image using Hough Transform.

        Args:
            img: Input image (numpy array).

        Returns:
            Median angle in degrees, or None if no lines detected.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return None

        angles = []
        for line in lines[:50]:  # Analyze top 50 strongest lines
            theta = line[0][1]
            angle = np.degrees(theta) - 90

            # Normalize angle to -90 to +90 range
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            angles.append(angle)

        return float(np.median(angles)) if angles else None

    def _handle_90_degree_rotation(self, img: np.ndarray, analysis_img: np.ndarray) -> np.ndarray:
        """
        Determine the correct 90-degree rotation using orientation scoring.

        Args:
            img: Original full-resolution image.
            analysis_img: Downsampled image for analysis.

        Returns:
            Rotated image.
        """
        # Test both 90-degree orientations
        test_cw = cv2.rotate(analysis_img, cv2.ROTATE_90_CLOCKWISE)
        test_ccw = cv2.rotate(analysis_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        score_cw = self._score_orientation(test_cw)
        score_ccw = self._score_orientation(test_ccw)

        score_diff = abs(score_cw - score_ccw)

        # Require significant difference to justify rotation
        if score_diff < self.score_diff_threshold:
            return img  # Too ambiguous, keep original

        # Apply rotation to original full-res image
        if score_cw > score_ccw:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _score_orientation(self, img: np.ndarray) -> float:
        """
        Calculate orientation score based on text density and structure.

        Higher score indicates "upright" orientation likelihood.

        Args:
            img: Input image (numpy array).

        Returns:
            Orientation score.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoise before analysis
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Aggressive thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological cleaning (remove noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Factor 1: Top-heavy density (text usually starts at top)
        top_third = binary[: h // 3, :]
        middle_third = binary[h // 3 : 2 * h // 3, :]
        bottom_third = binary[2 * h // 3 :, :]

        top_density = np.sum(top_third) / (top_third.size + 1e-6)
        middle_density = np.sum(middle_third) / (middle_third.size + 1e-6)
        bottom_density = np.sum(bottom_third) / (bottom_third.size + 1e-6)

        top_heavy_score = (top_density + middle_density * 0.5) / (bottom_density + 1e-6)

        # Factor 2: Horizontal projection variance
        projection = np.sum(binary, axis=1)

        # Smooth projection to reduce noise impact
        from scipy.ndimage import gaussian_filter1d

        projection = gaussian_filter1d(projection, sigma=2)
        variance_score = np.var(projection) / 1000

        # Factor 3: Detect large text blocks at top (headers)
        contours, _ = cv2.findContours(top_third, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        header_blocks = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:  # Minimum size threshold
                x, y, w_cont, h_cont = cv2.boundingRect(c)
                aspect_ratio = w_cont / (h_cont + 1e-6)

                # Text blocks usually wider than tall
                if aspect_ratio > 2:
                    header_blocks += 1

        header_score = header_blocks / 10.0

        # Weighted combination with capping to prevent extreme values
        total_score = (
            min(top_heavy_score, 5.0) * 0.5
            + min(variance_score, 2.0) * 0.3
            + min(header_score, 1.0) * 0.2
        )

        return total_score

    def _rotate_arbitrary(self, img: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by arbitrary angle (for small skew corrections).

        Args:
            img: Input image (numpy array).
            angle: Rotation angle in degrees (positive = counterclockwise).

        Returns:
            Rotated image (numpy array).
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image bounds to prevent cropping
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust translation to center the image
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # Perform rotation with white border
        rotated = cv2.warpAffine(
            img,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        return rotated
