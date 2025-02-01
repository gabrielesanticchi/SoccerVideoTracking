import cv2
import numpy as np

class DebugVisualizer:
    """
    A utility class for creating debug visualizations during algorithm development.
    Focused on creating side-by-side comparisons and maintaining frame sequences.
    """
    
    def __init__(self):
        """Initialize the debug visualizer."""
        self.frames = []
    
    def create_side_by_side(self, original_frame, processed_frame, titles=None):
        """
        Create a side-by-side comparison of original and processed frames.
        
        Args:
            original_frame: Original video frame
            processed_frame: Processed frame (e.g., mask or detection result)
            titles: Optional tuple of (left_title, right_title)
            
        Returns:
            Combined frame with both images side by side
        """
        # Ensure processed frame is in the same format as original
        if len(processed_frame.shape) == 2:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            
        # Get dimensions
        height = max(original_frame.shape[0], processed_frame.shape[0])
        width = original_frame.shape[1] + processed_frame.shape[1]
        
        # Create combined image
        combined = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Copy frames into combined image
        combined[:original_frame.shape[0], :original_frame.shape[1]] = original_frame
        combined[:processed_frame.shape[0], original_frame.shape[1]:] = processed_frame
        
        # Add titles if provided
        if titles:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            
            # Add left title with shadow for better visibility
            self._add_text_with_shadow(combined, titles[0], 
                                     (10, 30), font, font_scale, thickness)
            
            # Add right title with shadow
            self._add_text_with_shadow(combined, titles[1],
                                     (original_frame.shape[1] + 10, 30), 
                                     font, font_scale, thickness)
            
        return combined
    
    def _add_text_with_shadow(self, image, text, position, font, scale, thickness):
        """Add text with a shadow effect for better visibility."""
        # Draw shadow (black)
        cv2.putText(image, text, 
                   (position[0] + 2, position[1] + 2),
                   font, scale, (0, 0, 0), thickness + 1)
        # Draw main text (white)
        cv2.putText(image, text,
                   position, font, scale, (255, 255, 255), thickness)
    
    def add_frame(self, frame):
        """
        Add a frame to the visualization sequence.
        
        Args:
            frame: Frame to add to the sequence
        """
        self.frames.append(frame.copy())
    
    def get_frames(self):
        """
        Get the accumulated frames.
        
        Returns:
            List of frames
        """
        return self.frames
    
    def clear_frames(self):
        """Clear accumulated frames from memory."""
        self.frames = []