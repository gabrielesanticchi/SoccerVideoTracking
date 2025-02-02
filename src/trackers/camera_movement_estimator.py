import pickle
import cv2
import numpy as np
import os
import sys
from typing import Dict, Optional, Tuple, Union, List

sys.path.append('../')
from src.utils import BoxMath
from src.utils import VideoIO

class CameraMovementEstimator:
    """
    Estimates camera movement in video sequences using feature tracking. Depending on the
    configuration, it can use either Lucas-Kanade optical flow or SIFT-based feature matching.
    It also applies a smoothing filter to the movement estimates.
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
            frame: Initial video frame for feature detection setup.
            video_filename: Name of the video being processed.
            config: Optional configuration dictionary from YAML.
        """
        # Store basic information
        self.video_filename = video_filename
        self.visualizer = CameraMovementVisualizer(video_filename=video_filename)
        
        # Set up default configuration
        self._setup_default_config()
        
        # Update with provided configuration if any
        if config is not None:
            self._update_config(config)
            
        # New: Select estimation method from configuration (optical_flow or sift)
        self.estimation_method: str = self.config.get('estimation_method', 'optical_flow')
        
        # Initialize tracking parameters (only used for optical flow)
        self._initialize_tracking_params()
        
        # Initialize feature detection mask for optical flow (not used for SIFT)
        self._initialize_feature_mask(frame)
    
    def _setup_default_config(self) -> None:
        """Set up default configuration values."""
        self.config = {
            # Core parameters
            'minimum_distance': 5,
            
            # Feature detection settings (for optical flow)
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
            
            # Lucas-Kanade parameters (for optical flow)
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
            
            # New: Estimation method can be 'optical_flow' or 'sift'
            'estimation_method': 'optical_flow',
            
            # New: Smoothing settings for camera movement estimates
            'smoothing': {
                'enabled': False,
                'window_size': 5  # Number of frames for moving average smoothing
            }
        }
    
    def _update_config(self, new_config: Dict) -> None:
        """
        Update configuration with new values, preserving nested structure.
        
        Args:
            new_config: New configuration values from YAML.
        """
        def update_nested(target: Dict, source: Dict) -> None:
            for key, value in source.items():
                if isinstance(value, dict) and key in target:
                    update_nested(target[key], value)
                else:
                    target[key] = value
                    
        update_nested(self.config, new_config)
    
    def _initialize_tracking_params(self) -> None:
        """Initialize parameters for feature detection and tracking (optical flow)."""
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
        
        feat_config = self.config['features']
        self.features = {
            'maxCorners': feat_config['max_corners'],
            'qualityLevel': feat_config['quality_level'],
            'minDistance': feat_config['min_distance'],
            'blockSize': feat_config['block_size']
        }
        self.minimum_distance = self.config['minimum_distance']
    
    def _initialize_feature_mask(self, frame: np.ndarray) -> None:
        """
        Initialize the mask for feature detection in specific image regions (optical flow).
        
        Args:
            frame: Initial video frame.
        """
        frame_gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(frame_gray)
        mask_config = self.config['features']['mask']
        
        # Set left border
        left_width = mask_config['left_border']
        mask_features[:, 0:left_width] = 1
        
        # Set right border
        right_start, right_end = mask_config['right_border']
        mask_features[:, right_start:right_end] = 1
        
        self.features['mask'] = mask_features
    
    @staticmethod
    def _ensure_tuple(value: Union[Tuple, list]) -> Tuple:
        """
        Convert value to tuple if it's a list.
        
        Args:
            value: Input value (tuple or list).
            
        Returns:
            Tuple version of the input.
        """
        return tuple(value) if isinstance(value, list) else value

    def get_camera_movement(self, frames: List[np.ndarray]) -> List[List[float]]:
        """
        Get camera movement between frames using the configured estimation method and applies smoothing.
        
        Args:
            frames: List of video frames.
        
        Returns:
            List of [x, y] camera movements for each frame.
        """
        # Check if should read from stub
        if self.config['read_from_stub'] and self.config['stub_path'] is not None:
            if os.path.exists(self.config['stub_path']):
                with open(self.config['stub_path'], 'rb') as f:
                    return pickle.load(f)
        
        # Preallocate movement list (one per frame)
        camera_movement = [[0, 0] for _ in range(len(frames))]
        
        # Choose estimation method based on config
        if self.estimation_method == 'optical_flow':
            old_gray, old_features = self._initialize_tracking(frames[0])
            if self.config['save_visualization']:
                self._visualize_frame(self.visualizer, frames[0], old_features)
            
            for frame_num in range(1, len(frames)):
                frame_gray, new_features = self._process_frame(frames[frame_num], old_gray, old_features)
                movement = self._calculate_frame_movement(new_features, old_features)
                
                if movement['max_distance'] > self.minimum_distance:
                    camera_movement[frame_num] = [movement['x'], movement['y']]
                    # Update features for next iteration
                    old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                else:
                    camera_movement[frame_num] = [0, 0]
                
                if self.config['save_visualization']:
                    self._visualize_frame(
                        self.visualizer,
                        frames[frame_num],
                        old_features,
                        new_features,
                        camera_movement[frame_num]
                    )
                
                old_gray = frame_gray.copy()
        
        elif self.estimation_method == 'sift':
            # SIFT-based estimation: no need for prior feature tracking initialization.
            old_gray = self._convert_to_gray(frames[0])
            if self.config['save_visualization']:
                kp_old, _ = self._get_sift_features(old_gray)
                old_features_arr = self._keypoints_to_array(kp_old)
                self._visualize_frame(self.visualizer, frames[0], old_features_arr)
            
            for frame_num in range(1, len(frames)):
                frame_gray = self._convert_to_gray(frames[frame_num])
                movement = self._get_movement_sift(old_gray, frame_gray)
                camera_movement[frame_num] = [movement['x'], movement['y']]
                
                if self.config['save_visualization']:
                    kp_old, _ = self._get_sift_features(old_gray)
                    kp_new, _ = self._get_sift_features(frame_gray)
                    old_features_arr = self._keypoints_to_array(kp_old)
                    new_features_arr = self._keypoints_to_array(kp_new)
                    self._visualize_frame(
                        self.visualizer,
                        frames[frame_num],
                        old_features_arr,
                        new_features_arr,
                        camera_movement[frame_num]
                    )
                old_gray = frame_gray.copy()
        
        else:
            raise ValueError(f"Invalid estimation method: {self.estimation_method}")
        
        # Apply smoothing if enabled
        if self.config.get('smoothing', {}).get('enabled', False):
            window_size = self.config['smoothing'].get('window_size', 5)
            camera_movement = self._smooth_movements(camera_movement, window_size)
        
        # Cache results if stub_path is provided
        if self.config['stub_path'] is not None:
            with open(self.config['stub_path'], 'wb') as f:
                pickle.dump(camera_movement, f)
        
        return camera_movement
    
    def _smooth_movements(self, movements: List[List[float]], window_size: int) -> List[List[float]]:
        """
        Smooth the sequence of movement vectors using a moving average.
        
        Args:
            movements: List of [x, y] movement vectors.
            window_size: Number of frames over which to average.
            
        Returns:
            Smoothed list of [x, y] movement vectors.
        """
        smoothed = []
        for i in range(len(movements)):
            window = movements[max(0, i - window_size + 1): i + 1]
            avg = np.mean(window, axis=0)
            smoothed.append(avg.tolist())
        return smoothed
    
    def _convert_to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale if it is not already."""
        if frame.ndim == 3 and frame.shape[-1] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    
    def _initialize_tracking(self, first_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize tracking for the first frame using optical flow."""
        old_gray = self._convert_to_gray(first_frame)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        return old_gray, old_features
    
    def _process_frame(self, frame: np.ndarray, old_gray: np.ndarray, old_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single frame and get new features using optical flow."""
        frame_gray = self._convert_to_gray(frame)
        new_features, _, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, 
            old_features, None, 
            **self.lk_params
        )
        return frame_gray, new_features
    
    def _calculate_frame_movement(self, new_features: np.ndarray, old_features: np.ndarray) -> Dict[str, float]:
        """Calculate camera movement between feature sets (optical flow)."""
        max_distance = 0
        camera_movement_x, camera_movement_y = 0, 0
        
        for new, old in zip(new_features, old_features):
            new_point = new.ravel()
            old_point = old.ravel()
            distance = BoxMath.measure_distance(new_point, old_point)
            if distance > max_distance:
                max_distance = distance
                camera_movement_x, camera_movement_y = BoxMath.measure_xy_distance(old_point, new_point)
        
        return {
            'max_distance': max_distance,
            'x': camera_movement_x,
            'y': camera_movement_y
        }
    
    def _get_sift_features(self, gray_frame: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract SIFT keypoints and descriptors from a grayscale frame.
        
        Args:
            gray_frame: Grayscale image.
            
        Returns:
            Tuple of (keypoints, descriptors).
        """
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_frame, None)
        return keypoints, descriptors
    
    def _get_movement_sift(self, old_gray: np.ndarray, new_gray: np.ndarray) -> Dict[str, float]:
        """
        Estimate movement between two frames using SIFT feature matching.
        
        Args:
            old_gray: Previous grayscale frame.
            new_gray: Current grayscale frame.
            
        Returns:
            Dictionary with keys 'x', 'y', and 'max_distance' indicating the median movement.
        """
        kp1, des1 = self._get_sift_features(old_gray)
        kp2, des2 = self._get_sift_features(new_gray)
        
        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            return {'x': 0, 'y': 0, 'max_distance': 0}
        
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        
        if len(good) < 1:
            return {'x': 0, 'y': 0, 'max_distance': 0}
        
        dxs, dys, distances = [], [], []
        for m in good:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            dxs.append(dx)
            dys.append(dy)
            distances.append(np.sqrt(dx*dx + dy*dy))
        
        median_dx = float(np.median(dxs))
        median_dy = float(np.median(dys))
        max_distance = float(np.max(distances)) if distances else 0
        
        return {'x': median_dx, 'y': median_dy, 'max_distance': max_distance}
    
    def _keypoints_to_array(self, keypoints: List[cv2.KeyPoint]) -> Optional[np.ndarray]:
        """
        Convert a list of SIFT keypoints to a NumPy array of points.
        
        Args:
            keypoints: List of cv2.KeyPoint.
            
        Returns:
            Array of shape (N, 1, 2) if keypoints exist; otherwise, None.
        """
        if not keypoints:
            return None
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return pts.reshape(-1, 1, 2)
    
    def _visualize_frame(self, visualizer, frame, old_features, new_features=None, movement=None):
        """Handle frame visualization."""
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
    movement vectors, and frame-by-frame visualization.
    """
    
    def __init__(self, video_filename: str = ''):
        """
        Initialize the visualizer with output path configuration.
        
        Args:
            video_filename: Base name of the video being processed.
        """
        self.output_dir = 'output_videos'
        self.base_filename = f'{video_filename}_camera_movement_visualization'
        self.gif_path = f'{self.output_dir}/{self.base_filename}.gif'
        self.video_path = f'{self.output_dir}/{self.base_filename}.mp4'
        self.frames = []
        
    def draw_features_and_movement(self, frame, old_features, new_features=None, movement=None):
        """
        Draw tracked features and movement vectors on a frame.
        
        Args:
            frame: Video frame to draw on.
            old_features: Previously detected feature points.
            new_features: Currently detected feature points (optional).
            movement: Camera movement vector [dx, dy] (optional).
            
        Returns:
            Frame with visualization overlays.
        """
        visualization = frame.copy()
        
        # Draw original feature points in red
        if old_features is not None:
            for point in old_features:
                x, y = point.ravel()
                cv2.circle(visualization, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        # Draw feature movement lines and new points if available
        if new_features is not None:
            for old, new in zip(old_features, new_features):
                x1, y1 = old.ravel()
                x2, y2 = new.ravel()
                cv2.line(visualization, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
                cv2.circle(visualization, (int(x2), int(y2)), 2, (255, 0, 0), -1)
        
        # Draw overall camera movement vector if available
        if movement is not None:
            center = (visualization.shape[1] // 2, visualization.shape[0] // 2)
            end_point = (int(center[0] + movement[0] * 5), int(center[1] + movement[1] * 5))
            cv2.arrowedLine(visualization, center, end_point, (0, 255, 0), 2)
            cv2.putText(visualization, f"dx: {movement[0]:.1f}, dy: {movement[1]:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return visualization
    
    def add_frame(self, frame):
        """Add a frame to the visualization sequence."""
        self.frames.append(frame)
    
    def save_visualization(self, duration: int = 100, fps: int = 30):
        """
        Save visualization as both GIF and MP4 using VideoIO.
        
        Args:
            duration: Frame duration for GIF in milliseconds.
            fps: Frames per second for MP4 output.
        """
        if not self.frames:
            print("No frames to save!")
            return
        
        VideoIO.save_media(self.frames, self.video_path, fps=fps, formats=['gif'])
