import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


class PitchLineDetector:
    """
    Pitch line detector that supports multiple edge detection methods.
    The choice of method is configured via a parameter in the configuration.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        """
        Initialize the pitch line detector with configuration parameters.

        Args:
            config: Dictionary with configuration parameters, e.g.,
                - detection_method: Method for edge detection ('canny', 'adaptive', 'sobel', 'combined')
                - canny_low: Lower threshold for Canny edge detection
                - canny_high: Higher threshold for Canny edge detection
                - hough_threshold: Threshold for Hough line detection
                - min_line_length: Minimum line length for Hough detection
                - max_line_gap: Maximum gap between line segments
                - binary_threshold: Threshold for binary image conversion (for fixed thresholding)
                - adaptive_block_size: Block size for adaptive thresholding (must be odd)
                - adaptive_C: Constant subtracted in adaptive thresholding
        """
        # Default configuration parameters
        self.config: Dict = {
            'detection_method': 'canny',  # Options: 'canny', 'adaptive', 'sobel', 'combined'
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 50,
            'min_line_length': 100,
            'max_line_gap': 10,
            'binary_threshold': 200,
            'adaptive_block_size': 11,  # Must be odd
            'adaptive_C': 2
        }
        if config is not None:
            self.config.update(config)

        # Select detection method from config
        self.detection_method: str = self.config.get('detection_method', 'canny')

        # Initialize kernels for line enhancement
        self.horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        self.vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    def create_line_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a binary mask highlighting the pitch lines based on the selected detection method.

        Args:
            frame: Input video frame.

        Returns:
            Binary mask with detected pitch lines.
        """
        if self.detection_method == 'canny':
            return self.create_line_mask_canny(frame)
        elif self.detection_method == 'adaptive':
            return self.create_line_mask_adaptive(frame)
        elif self.detection_method == 'sobel':
            return self.create_line_mask_sobel(frame)
        elif self.detection_method == 'combined':
            mask_canny = self.create_line_mask_canny(frame)
            mask_sobel = self.create_line_mask_sobel(frame)
            combined_mask = cv2.bitwise_or(mask_canny, mask_sobel)
            return combined_mask
        else:
            raise ValueError(f"Invalid detection_method: {self.detection_method}")

    def create_line_mask_canny(self, frame: np.ndarray) -> np.ndarray:
        """
        Create line mask using the traditional Canny edge detection approach.

        Args:
            frame: Input video frame.

        Returns:
            Binary mask with detected edges.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        # Fixed binary thresholding to isolate bright (white) lines
        _, binary = cv2.threshold(
            smooth, self.config['binary_threshold'], 255, cv2.THRESH_BINARY
        )

        # Enhance horizontal and vertical structures
        horizontal = cv2.erode(binary, self.horizontal_kernel)
        horizontal = cv2.dilate(horizontal, self.horizontal_kernel)
        vertical = cv2.erode(binary, self.vertical_kernel)
        vertical = cv2.dilate(vertical, self.vertical_kernel)
        combined = cv2.bitwise_or(horizontal, vertical)

        # Apply Canny edge detection
        edges = cv2.Canny(combined, self.config['canny_low'], self.config['canny_high'])
        return edges

    def create_line_mask_adaptive(self, frame: np.ndarray) -> np.ndarray:
        """
        Create line mask using adaptive thresholding to account for varying lighting conditions.

        Args:
            frame: Input video frame.

        Returns:
            Binary mask with detected edges.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use adaptive thresholding instead of a fixed threshold
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, self.config['adaptive_block_size'],
            self.config['adaptive_C']
        )

        # Enhance lines using morphological operations
        horizontal = cv2.erode(adaptive_thresh, self.horizontal_kernel)
        horizontal = cv2.dilate(horizontal, self.horizontal_kernel)
        vertical = cv2.erode(adaptive_thresh, self.vertical_kernel)
        vertical = cv2.dilate(vertical, self.vertical_kernel)
        combined = cv2.bitwise_or(horizontal, vertical)

        # Optionally refine edges with Canny
        edges = cv2.Canny(combined, self.config['canny_low'], self.config['canny_high'])
        return edges

    def create_line_mask_sobel(self, frame: np.ndarray) -> np.ndarray:
        """
        Create line mask using the Sobel operator to detect gradient edges.

        Args:
            frame: Input video frame.

        Returns:
            Binary mask with detected edges.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)

        # Compute gradients in x and y directions
        grad_x = cv2.Sobel(smooth, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smooth, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # Threshold the gradient image to obtain binary edges
        _, binary = cv2.threshold(
            grad, self.config['binary_threshold'], 255, cv2.THRESH_BINARY
        )
        return binary

    def detect_lines(self, frame: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
        """
        Detect pitch lines in the frame using the probabilistic Hough transform.

        Args:
            frame: Input video frame.

        Returns:
            A tuple containing:
                - List of detected line segments as [x1, y1, x2, y2]
                - The line mask used for detection.
        """
        line_mask = self.create_line_mask(frame)

        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(
            line_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config['hough_threshold'],
            minLineLength=self.config['min_line_length'],
            maxLineGap=self.config['max_line_gap']
        )

        if lines is None:
            return [], line_mask

        # Filter lines based on angle (keep near-horizontal or near-vertical)
        filtered_lines: List[List[int]] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 10 or (80 < angle < 100):
                filtered_lines.append([x1, y1, x2, y2])
        return filtered_lines, line_mask

    def draw_detected_lines(self, frame: np.ndarray, lines: List[List[int]]) -> np.ndarray:
        """
        Draw detected lines on the frame for visualization.

        Args:
            frame: Input video frame.
            lines: List of detected line segments.

        Returns:
            Frame with drawn lines.
        """
        vis_frame = frame.copy()
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return vis_frame

    def get_features_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a mask suitable for feature detection that focuses on pitch lines.

        Args:
            frame: Input video frame.

        Returns:
            Binary mask for feature detection.
        """
        lines, line_mask = self.detect_lines(frame)
        features_mask = np.zeros_like(line_mask)

        # Dilate the line mask to cover regions around detected lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        features_mask = cv2.dilate(line_mask, kernel, iterations=2)

        # Enhance intersections by adding extra weight where near-perpendicular lines meet
        for i, line1 in enumerate(lines):
            x1_1, y1_1, x2_1, y2_1 = line1
            for line2 in lines[i + 1:]:
                x1_2, y1_2, x2_2, y2_2 = line2
                angle1 = np.degrees(np.arctan2(y2_1 - y1_1, x2_1 - x1_1))
                angle2 = np.degrees(np.arctan2(y2_2 - y1_2, x2_2 - x1_2))
                if abs(abs(angle1 - angle2) - 90) < 10:
                    intersection = self._line_intersection(line1, line2)
                    if intersection is not None:
                        x, y = intersection
                        cv2.circle(features_mask, (int(x), int(y)), 10, 255, -1)
        return features_mask

    def _line_intersection(self, line1: List[int], line2: List[int]) -> Optional[Tuple[float, float]]:
        """
        Calculate the intersection point of two lines if it exists.

        Args:
            line1: First line segment as [x1, y1, x2, y2].
            line2: Second line segment as [x1, y1, x2, y2].

        Returns:
            (x, y) coordinates of the intersection point or None if lines are parallel.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator

        if 0 <= t <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        return None
