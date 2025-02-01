import cv2
import numpy as np

class PitchLineDetector:
    def __init__(self, config=None):
        """
        Initialize the pitch line detector with configuration parameters
        
        Args:
            config: Dictionary with configuration parameters:
                - canny_low: Lower threshold for Canny edge detection
                - canny_high: Higher threshold for Canny edge detection
                - hough_threshold: Threshold for Hough line detection
                - min_line_length: Minimum line length for Hough detection
                - max_line_gap: Maximum gap between line segments
                - binary_threshold: Threshold for binary image conversion
        """
        # Default configuration
        self.config = {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 50,
            'min_line_length': 100,
            'max_line_gap': 10,
            'binary_threshold': 200
        }
        
        if config is not None:
            self.config.update(config)
            
        # Initialize kernels for line enhancement
        self.horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        self.vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
    def create_line_mask(self, frame):
        """
        Create a binary mask highlighting the pitch lines
        
        Args:
            frame: Input video frame
            
        Returns:
            Binary mask with pitch lines
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Create binary image to isolate white lines
        _, binary = cv2.threshold(smooth, self.config['binary_threshold'], 255, cv2.THRESH_BINARY) # white_pixels = np.sum(binary == 255)
        
        # Enhance horizontal and vertical lines separately
        horizontal = cv2.erode(binary, self.horizontal_kernel)
        horizontal = cv2.dilate(horizontal, self.horizontal_kernel)
        
        vertical = cv2.erode(binary, self.vertical_kernel)
        vertical = cv2.dilate(vertical, self.vertical_kernel)
        
        # Combine horizontal and vertical lines
        combined = cv2.bitwise_or(horizontal, vertical)
        
        # Apply Canny edge detection
        edges = cv2.Canny(combined, 
                         self.config['canny_low'], 
                         self.config['canny_high'])
        
        return edges
        
    def detect_lines(self, frame):
        """
        Detect pitch lines in the frame
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detected line segments and the line mask
        """
        # Get the line mask
        line_mask = self.create_line_mask(frame)
        
        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(line_mask,
                               rho=1,
                               theta=np.pi/180,
                               threshold=self.config['hough_threshold'],
                               minLineLength=self.config['min_line_length'],
                               maxLineGap=self.config['max_line_gap'])
        
        if lines is None:
            return [], line_mask
            
        # Filter lines based on angle to remove diagonal lines
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            # Keep only near-horizontal (0±10°) or near-vertical (90±10°) lines
            if angle < 10 or (angle > 80 and angle < 100):
                filtered_lines.append(line[0])
                
        return filtered_lines, line_mask
        
    def draw_detected_lines(self, frame, lines):
        """
        Draw detected lines on the frame for visualization
        
        Args:
            frame: Input video frame
            lines: List of detected line segments
            
        Returns:
            Frame with drawn lines
        """
        vis_frame = frame.copy()
        
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        return vis_frame
        
    def get_features_mask(self, frame):
        """
        Create a mask suitable for feature detection that focuses on pitch lines
        
        Args:
            frame: Input video frame
            
        Returns:
            Binary mask for feature detection
        """
        # Detect lines and get the line mask
        lines, line_mask = self.detect_lines(frame)
        
        # Create a mask that's black everywhere except near the detected lines
        features_mask = np.zeros_like(line_mask)
        
        # Dilate the line mask to allow feature detection near the lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        features_mask = cv2.dilate(line_mask, kernel, iterations=2)
        
        # Add extra weight to intersections of lines
        for i, line1 in enumerate(lines):
            x1_1, y1_1, x2_1, y2_1 = line1
            for line2 in lines[i+1:]:
                x1_2, y1_2, x2_2, y2_2 = line2
                
                # Check if lines are perpendicular (one horizontal, one vertical)
                angle1 = np.degrees(np.arctan2(y2_1 - y1_1, x2_1 - x1_1))
                angle2 = np.degrees(np.arctan2(y2_2 - y1_2, x2_2 - x1_2))
                
                if abs(abs(angle1 - angle2) - 90) < 10:  # Near perpendicular
                    # Find intersection point
                    intersection = self._line_intersection(line1, line2)
                    if intersection is not None:
                        x, y = intersection
                        # Add extra weight to intersection point
                        cv2.circle(features_mask, (int(x), int(y)), 10, 255, -1)
        
        return features_mask
        
    def _line_intersection(self, line1, line2):
        """Calculate intersection point of two lines if it exists"""
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