import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from src.utils import BoxMath
from src.utils import VideoIO

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Optional, Tuple, Union

class CameraMovementEstimator:
    """
    Estimates camera movement in video sequences using feature tracking and optical flow.
    Tracks features across frames to estimate camera motion, with configurable parameters
    for feature detection and tracking.
    """
    
    def __init__(
        self, 
        frame: np.ndarray,
        video_filename: str,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the camera movement estimator.
        
        Args:
            frame: Initial video frame for feature detection setup
            video_filename: Name of the video being processed
            config: Optional configuration dictionary from YAML
            visualizer: Optional visualization component
        """
        # Store basic information
        self.video_filename = video_filename
        self.visualizer = CameraMovementVisualizer(video_filename = video_filename)
        
        # Set up default configuration
        self._setup_default_config()
        
        # Update with provided configuration if any
        if config is not None:
            self._update_config(config)
            
        # Initialize tracking parameters
        self._initialize_tracking_params()
        
        # Initialize feature detection mask
        self._initialize_feature_mask(frame)
    
    def _setup_default_config(self) -> None:
        """Set up default configuration values."""
        self.config = {
            # Core parameters
            'minimum_distance': 5,
            
            # Feature detection settings
            'features': {
                'max_corners': 25,
                'quality_level': 0.3,
                'min_distance': 3,
                'block_size': 7,
                'mask': {
                    'left_border': 20,
                    'right_border': [900, 1050]
                }
            },
            
            # Lucas-Kanade parameters
            'lk_params': {
                'win_size': (15, 15),
                'max_level': 2,
                'max_count': 10,
                'epsilon': 0.03
            },
            
            # Caching and visualization
            'read_from_stub': False,
            'stub_path': f'stubs/{self.video_filename}_camera_movement_stubs.pkl',
            'save_visualization': False,
        }
    
    def _update_config(self, new_config: Dict) -> None:
        """
        Update configuration with new values, preserving nested structure.
        
        Args:
            new_config: New configuration values from YAML
        """
        def update_nested(target: Dict, source: Dict) -> None:
            for key, value in source.items():
                if isinstance(value, dict) and key in target:
                    update_nested(target[key], value)
                else:
                    target[key] = value
                    
        update_nested(self.config, new_config)
    
    def _initialize_tracking_params(self) -> None:
        """Initialize parameters for feature detection and tracking."""
        # Set up Lucas-Kanade parameters
        lk_config = self.config['lk_params']
        self.lk_params = {
            'winSize': self._ensure_tuple(lk_config['win_size']),
            'maxLevel': lk_config['max_level'],
            'criteria': (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                lk_config['max_count'],
                lk_config['epsilon']
            )
        }
        
        # Set up feature detection parameters (mask will be added later)
        feat_config = self.config['features']
        self.features = {
            'maxCorners': feat_config['max_corners'],
            'qualityLevel': feat_config['quality_level'],
            'minDistance': feat_config['min_distance'],
            'blockSize': feat_config['block_size']
        }
        
        # Store minimum distance for movement detection
        self.minimum_distance = self.config['minimum_distance']
    
    def _initialize_feature_mask(self, frame: np.ndarray) -> None:
        """
        Initialize the mask for feature detection in specific image regions.
        
        Args:
            frame: Initial video frame
        """
        # Convert frame to grayscale for mask creation
        frame_gray = (
            frame if len(frame.shape) == 2 
            else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
        
        # Create mask with specified borders
        mask_features = np.zeros_like(frame_gray)
        mask_config = self.config['features']['mask']
        
        # Set left border
        left_width = mask_config['left_border']
        mask_features[:, 0:left_width] = 1
        
        # Set right border
        right_start, right_end = mask_config['right_border']
        mask_features[:, right_start:right_end] = 1
        
        # Add mask to feature detection parameters
        self.features['mask'] = mask_features
    
    @staticmethod
    def _ensure_tuple(value: Union[Tuple, list]) -> Tuple:
        """
        Convert value to tuple if it's a list.
        
        Args:
            value: Input value (tuple or list)
            
        Returns:
            Tuple version of the input
        """
        return tuple(value) if isinstance(value, list) else value
    

    def get_camera_movement(self, frames):
        """
        Get camera movement between frames using configured settings
        
        Args:
            frames: List of video frames
        
        Returns:
            List of [x,y] camera movements for each frame
        """
        # Check if should read from stub
        if self.config['read_from_stub'] and self.config['stub_path'] is not None:
            if os.path.exists(self.config['stub_path']):
                with open(self.config['stub_path'], 'rb') as f:
                    return pickle.load(f)
        
        camera_movement = [[0,0]]*len(frames)
        old_gray, old_features = self._initialize_tracking(frames[0])

        # Initial visualization if enabled
        if self.config['save_visualization']:
            self._visualize_frame(self.visualizer, frames[0], old_features)

        # Process frames
        for frame_num in range(1, len(frames)):
            frame_gray, new_features = self._process_frame(frames[frame_num], old_gray, old_features)
            movement = self._calculate_frame_movement(new_features, old_features)
            
            if movement['max_distance'] > self.minimum_distance:
                camera_movement[frame_num] = [movement['x'], movement['y']]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            
            if self.config['save_visualization']:
                self._visualize_frame(
                    self.visualizer,
                    frames[frame_num],
                    old_features,
                    new_features,
                    camera_movement[frame_num]
                )
            
            old_gray = frame_gray.copy()
        
        # Save visualization if enabled
        if self.config['save_visualization']:
            print(f'... Saving visualization to {self.visualizer.gif_path}')
            self.visualizer.save_visualization()
        
        # Cache results if path provided
        if self.config['stub_path'] is not None:
            with open(self.config['stub_path'], 'wb') as f:
                pickle.dump(camera_movement, f)
        
        return camera_movement
    
    def _convert_to_gray(self, frame):
        """Convert frame to grayscale if it is not already"""
        if frame.shape[-1] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    
    def _initialize_tracking(self, first_frame):
        """Initialize tracking for the first frame"""
        old_gray = self._convert_to_gray(first_frame)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        return old_gray, old_features
    
    def _process_frame(self, frame, old_gray, old_features):
        """Process a single frame and get new features"""
        frame_gray = self._convert_to_gray(frame)
        new_features, _, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, 
            old_features, None, 
            **self.lk_params
        )
        return frame_gray, new_features
    
    def _calculate_frame_movement(self, new_features, old_features):
        """Calculate camera movement between feature sets"""
        max_distance = 0
        camera_movement_x, camera_movement_y = 0, 0
        
        for new, old in zip(new_features, old_features):
            new_point = new.ravel()
            old_point = old.ravel()
            
            distance = BoxMath.measure_distance(new_point, old_point)
            if distance > max_distance:
                max_distance = distance
                camera_movement_x, camera_movement_y = BoxMath.measure_xy_distance(
                    old_point, new_point)
        
        return {
            'max_distance': max_distance,
            'x': camera_movement_x,
            'y': camera_movement_y
        }

    def _visualize_frame(self, visualizer, frame, old_features, new_features=None, movement=None):
        """Handle frame visualization"""
        vis_frame = visualizer.draw_features_and_movement(
            frame, 
            old_features, 
            new_features, 
            movement
        )
        visualizer.add_frame(vis_frame)

    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames
    
class CameraMovementVisualizer:
    """
    Handles visualization of camera movement tracking, including feature points,
    movement vectors, and frame-by-frame visualization. Uses VideoIO for file operations.
    """
    
    def __init__(self, video_filename: str = ''):
        """
        Initialize the visualizer with output path configuration.
        
        Args:
            video_filename: Base name of the video being processed
        """
        # Set up output paths for different visualization formats
        self.output_dir = 'output_videos'
        self.base_filename = f'{video_filename}_camera_movement_visualization'
        self.gif_path = f'{self.output_dir}/{self.base_filename}.gif'
        self.video_path = f'{self.output_dir}/{self.base_filename}.mp4'
        
        # Store frames for batch processing
        self.frames = []
        
    def draw_features_and_movement(self, frame, old_features, new_features=None, movement=None):
        """
        Draw tracked features and movement vectors on a frame.
        
        Args:
            frame: Video frame to draw on
            old_features: Previously detected feature points
            new_features: Currently detected feature points (optional)
            movement: Camera movement vector [dx, dy] (optional)
            
        Returns:
            Frame with visualization overlays
        """
        # Create a copy of the frame to avoid modifying the original
        visualization = frame.copy()
        
        # Draw original feature points in red
        if old_features is not None:
            for point in old_features:
                x, y = point.ravel()
                cv2.circle(
                    visualization, 
                    (int(x), int(y)), 
                    2, 
                    (0, 0, 255),  # Red color
                    -1
                )
        
        # Draw feature movement lines and new points if available
        if new_features is not None:
            for old, new in zip(old_features, new_features):
                # Extract point coordinates
                x1, y1 = old.ravel()
                x2, y2 = new.ravel()
                
                # Draw movement line in yellow
                cv2.line(
                    visualization,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 255, 0),  # Yellow color
                    1
                )
                
                # Draw new point position in blue
                cv2.circle(
                    visualization,
                    (int(x2), int(y2)),
                    2,
                    (255, 0, 0),  # Blue color
                    -1
                )
        
        # Draw overall camera movement vector if available
        if movement is not None:
            # Calculate vector endpoints
            center = (visualization.shape[1]//2, visualization.shape[0]//2)
            end_point = (
                int(center[0] + movement[0]*5),
                int(center[1] + movement[1]*5)
            )
            
            # Draw movement vector
            cv2.arrowedLine(
                visualization,
                center,
                end_point,
                (0, 255, 0),  # Green color
                2
            )
            
            # Add movement values as text
            cv2.putText(
                visualization,
                f"dx: {movement[0]:.1f}, dy: {movement[1]:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),  # Yellow color
                1
            )
        
        return visualization
    
    def add_frame(self, frame):
        """
        Add a frame to the visualization sequence.
        
        Args:
            frame: Frame to add to the sequence
        """
        self.frames.append(frame)
    
    def save_visualization(self, duration: int = 100, fps: int = 30):
        """
        Save visualization as both GIF and MP4 using VideoIO.
        
        Args:
            duration: Frame duration for GIF in milliseconds
            fps: Frames per second for MP4 output
        """
        if not self.frames:
            print("No frames to save!")
            return
        
        # Save as GIF using VideoIO
        VideoIO.save_gif(
            frames=self.frames,
            output_path=self.gif_path,
            duration=duration
        )
        
        # Save as MP4 using VideoIO
        VideoIO.save_video(
            frames=self.frames,
            output_path=self.video_path,
            fps=fps
        )