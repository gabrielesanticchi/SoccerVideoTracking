import cv2
import numpy as np
from PIL import Image
import os

class VideoIO:
    """
    A utility class for handling video and image sequence I/O operations.
    Provides methods for reading videos, saving videos, and creating GIFs.
    """
    
    @staticmethod
    def read_video(video_path, frame_rate_reduction=1, resize_factor=1):
        """
        Read a video file and return its frames, with optional frame rate reduction.
        
        Args:
            video_path: Path to the video file
            frame_rate_reduction: Factor by which to reduce frame rate. For example:
                - 1: Keep all frames (default)
                - 2: Keep every 2nd frame (half frame rate)
                - 3: Keep every 3rd frame (third frame rate)
                Must be a positive integer.
            
        Returns:
            List of video frames
            
        Raises:
            ValueError: If frame_rate_reduction is less than 1 or not an integer
        """
        # Validate frame rate reduction parameter
        if not isinstance(frame_rate_reduction, int) or frame_rate_reduction < 1:
            raise ValueError("frame_rate_reduction must be a positive integer")

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
            
        # Get original video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Resize frames if requested
        # if resize_factor != 1:
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_width = int(original_width * resize_factor)
        new_height = int(original_height * resize_factor)

        # Calculate new properties after reduction
        new_fps = original_fps / frame_rate_reduction
        estimated_frames = total_frames // frame_rate_reduction
        
        print(f"Video properties:")
        print(f"- Original: {total_frames} frames @ {original_fps:.2f} fps")
        print(f"- Reduced: ~{estimated_frames} frames @ {new_fps:.2f} fps")
        print(f"- Resized: {original_width}x{original_height} -> {new_width}x{new_height}")
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame if requested
            if resize_factor != 1:
                frame = cv2.resize(frame, (new_width, new_height))

            # Only keep frames based on reduction factor
            if frame_count % frame_rate_reduction == 0:
                frames.append(frame)
                
            frame_count += 1
        
        cap.release()
        
        print(f"Actually loaded {len(frames)} frames")
        return frames

    @staticmethod
    def save_video(frames, output_path, fps=30):
        """
        Save a sequence of frames as a video file.
        
        Args:
            frames: List of frames to save
            output_path: Path where the video will be saved
            fps: Frames per second for the output video
        """
        if not frames:
            print("No frames to save!")
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get dimensions from first frame
        height, width = frames[0].shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
            
        out.release()
        print(f"Saved video to {output_path}")

    @staticmethod
    def save_gif(frames, output_path, duration=100):
        """
        Save a sequence of frames as an animated GIF.
        
        Args:
            frames: List of frames to save
            output_path: Path where the GIF will be saved
            duration: Duration for each frame in milliseconds
        """
        if not frames:
            print("No frames to save!")
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert frames to PIL images
        pil_frames = []
        for frame in frames:
            # Convert to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Convert to P mode with adaptive palette for better GIF compression
            pil_image = pil_image.convert('P', palette=Image.ADAPTIVE, colors=256)
            pil_frames.append(pil_image)
        
        # Save as GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            optimize=True
        )
        print(f"Saved GIF to {output_path}")