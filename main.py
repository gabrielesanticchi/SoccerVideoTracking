from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from pathlib import Path
import yaml
from dataclasses import dataclass
from enum import Enum

from src.trackers import Tracker, CameraMovementEstimator
from src.pitch.pitch_lines_detector import BasicPitchLineDetector
from src.pitch.no_bells_just_whistles import NoBellsJustWhistles
from src.utils import VideoIO
from tests.debug_visualizer import DebugVisualizer

@dataclass
class VideoConfig:
    frame_rate_reduction: int
    resize_factor: float
    target_width: int = 960
    target_height: int = 480

class ModelType(Enum):
    BASIC = "basic"
    NETWORK = "no_bells_just_whistles"

class SoccerVideoProcessor:
    """Process soccer videos with object tracking and pitch analysis."""
    
    def __init__(self, config_path: str):
        # Load config and initialize components
        self.config = yaml.safe_load(open(config_path, 'r'))
        self.video_config = VideoConfig(**self.config['input_video'])
        
        # Components will be initialized when processing starts
        self.tracker = None
        self.pitch_detector = None
        self.camera_estimator = None
        
        # Setup debug output
        self.debug = DebugVisualizer()
        self.output_dir = Path("debug_output")
        self.output_dir.mkdir(exist_ok=True)

    def load_video(self, video_path: str) -> Tuple[List[np.ndarray], dict]:
        """Load and prepare video frames for processing."""
        print("Loading video...")
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
            
        props = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        cap.release()
        
        # Read frames with specified reduction
        frames = VideoIO.read_video(
            video_path,
            frame_rate_reduction=self.video_config.frame_rate_reduction,
            resize_factor=self.video_config.resize_factor
        )
        
        return frames, props

    def preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Preprocess frames to standard size and format."""
        processed = []
        for frame in frames:
            if frame.shape[:2] != (self.video_config.target_height, self.video_config.target_width):
                frame = cv2.resize(frame, (self.video_config.target_width, 
                                         self.video_config.target_height))
            processed.append(frame)
        return processed

    def setup_models(self, first_frame: np.ndarray, video_name: str):
        """Initialize all detection and tracking models."""
        print("Setting up models...")
        
        # Initialize tracker
        self.tracker = Tracker(
            model_path=self.config['tracker']['model_path'],
            stub_path=self.config['tracker']['stub_path'],
        )
        
        # Initialize pitch detector based on config type
        model_type = ModelType(self.config['pitch_lines_detection']['model'])
        pitch_config = self.config['pitch_lines_detection']
        if model_type == ModelType.BASIC:
            self.pitch_detector = BasicPitchLineDetector(
                pitch_config['basic']
            )

        else:
            self.pitch_detector = NoBellsJustWhistles(
                pitch_config['no_bells_just_whistles'],
                self.config['device'],
            )
        
        # Initialize camera movement estimation
        self.camera_estimator = CameraMovementEstimator(
            frame=first_frame,
            video_filename=video_name,
            config=self.config['camera_movement']
        )
        
        # Set initial pitch estimation
        if model_type == ModelType.BASIC:
            mask = self.pitch_detector.get_features_mask(first_frame)
            self.camera_estimator.features['mask'] = mask
        else:
            self.pitch_detector.inference(first_frame)
            self.camera_estimator.features['mask'] = None

    def detect_objects(self, frames: List[np.ndarray], video_name: str) -> Dict:
        """Detect and track players, referees, and ball."""
        print("... Detecting objects")
        
        # Generate stub path based on video config
        stub_path = (f"stubs/{video_name}_track_stubs_"
                    f"{self.video_config.frame_rate_reduction}_"
                    f"{self.video_config.resize_factor}.pkl")
        
        # Get tracks and add positions
        tracks = self.tracker.get_object_tracks(
            frames,
            read_from_stub=True,
            stub_path=stub_path
        )
        self.tracker.add_position_to_tracks(tracks)
        
        return tracks

    def detect_pitch_lines(self, frames: List[np.ndarray]) -> Tuple[List, List]:
        """Analyze pitch lines and camera movement."""
        print("... Analyzing pitch")
        
        pitch_lines = []
        calibration_params = []
        
        for frame in frames:
            if isinstance(self.pitch_detector, NoBellsJustWhistles):
                # Network-based detection with calibration
                kp_dict = self.pitch_detector.inference(frame)
                pitch_lines.append(kp_dict)
                # calibration_params.append(params)
            else:
                # Basic line detection
                lines_mask = self.pitch_detector.create_lines_mask(frame)
                pitch_lines.append(lines_mask)
                
        return pitch_lines, calibration_params

    def estimate_camera(self, frames: List[np.ndarray]) -> Dict:
        """Estimate camera movement between frames."""
        print("Estimating camera movement...")
        return self.camera_estimator.get_camera_movement(frames)

    def create_visualizations(self, 
                            frames: List[np.ndarray], 
                            results: Dict,
                            video_name: str):
        """Generate debug visualizations if enabled."""
        if not self.config['camera_movement']['save_visualization']:
            return
            
        print("Generating visualizations...")
        
        # Pitch lines visualization
        pitch_frames = []
        for frame, mask in zip(frames, results['pitch_lines']):
            combined = self.debug.create_side_by_side(
                frame, mask,
                titles=("Original", "Pitch Lines")
            )
            pitch_frames.append(combined)
            
        VideoIO.save_media(
            pitch_frames,
            self.output_dir / f"{video_name}_pitch_lines",
            fps=30,
            duration=100,
            formats=['gif']
        )
        
        # Camera movement visualization
        movement_frames = self.camera_estimator.draw_camera_movement(
            frames, 
            results['camera_movement']
        )
        
        camera_frames = []
        for frame, mov_frame in zip(frames, movement_frames):
            combined = self.debug.create_side_by_side(
                frame, mov_frame,
                titles=("Original", "Camera Movement")
            )
            camera_frames.append(combined)
            
        VideoIO.save_media(
            camera_frames,
            self.output_dir / f"{video_name}_camera_movement",
            fps=30,
            duration=100,
            formats=['gif']
        )

    def process_video(self, video_path: str) -> Dict:
        """Main processing pipeline for soccer video analysis."""
        video_name = Path(video_path).stem
        
        # 1. Load and preprocess video
        frames, props = self.load_video(video_path)
        frames = self.preprocess_frames(frames)
        
        # 2. Initialize models
        self.setup_models(frames[0], video_name)
        
        # 3. Process frames
        results = {}
        
        # 4. Object detection and tracking
        results['tracks'] = self.detect_objects(frames, video_name)
        
        # 5. Pitch analysis
        pitch_lines, calibration = self.detect_pitch_lines(frames)
        results['pitch_lines'] = pitch_lines
        if calibration:
            results['calibration'] = calibration
            
        # 6. Camera movement estimation
        # results['camera_movement'] = self.estimate_camera(frames)
        
        # 7. Generate visualizations
        # self.create_visualizations(frames, results, video_name)
        
        return results

def main():
    # Initialize and run pipeline
    processor = SoccerVideoProcessor('config/config.yaml')
    results = processor.process_video('input_videos/iniesta_sample.mp4')
    print("Processing complete! Check debug_output/ for visualizations.")
        # len(results['tracks']['players']) --> N frames
        # len(results['pitch_lines'])       --> N frames

if __name__ == "__main__":
    main()