import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.configs_loader import ConfigurationLoader
from src.trackers import Tracker, PitchLineDetector, CameraMovementEstimator
from src.utils import VideoIO
from tests.debug_visualizer import DebugVisualizer

class SoccerAnalysisPipeline:
    """
    Main pipeline for soccer video analysis combining player tracking,
    pitch line detection, and camera movement estimation.
    """
    def __init__(self, config_path: str):
        """
        Initialize the analysis pipeline.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        # Load configuration
        self.config = ConfigurationLoader(config_path).load_config()
        
        # Initialize components
        self.tracker = None
        self.pitch_detector = None
        self.camera_estimator = None
        
        # Debug visualization
        self.debug_visualizer = DebugVisualizer()
        self.output_dir = Path("debug_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def initialize_components(self, first_frame: np.ndarray) -> None:
        """
        Initialize analysis components with the first frame.
        
        Args:
            first_frame: First frame of the video
        """
        # Initialize object tracker
        self.tracker = Tracker(
            model_path='models/best.pt',
            stub_path=f'stubs/{self.video_name}_track_stubs.pkl'
        )
        
        # Initialize pitch line detector
        self.pitch_detector = PitchLineDetector(
            config=self.config['pitch_line_detector']
        )
        
        # Initialize camera movement estimator with first frame
        self.camera_estimator = CameraMovementEstimator(
            frame=first_frame,
            video_filename=self.video_name,
            config=self.config['camera_movement'], 
        )
        
        # Get features mask from pitch detector to help with tracking
        first_frame_mask = self.pitch_detector.get_features_mask(first_frame)
        self.camera_estimator.features['mask'] = first_frame_mask

    def process_video(self, video_path: str) -> Dict:
        """
        Process a soccer video through the complete analysis pipeline.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Dictionary containing analysis results
        """
        self.video_name = Path(video_path).stem
        
        # Read video frames with specified frame rate reduction
        print("Reading video frames...")
        frames = VideoIO.read_video(
            video_path,
            frame_rate_reduction=self.config['input_video']['frame_rate_reduction'],
            resize_factor=self.config['input_video']['resize_factor']
        )
        
        # Initialize components with first frame
        print("Initializing analysis components...")
        self.initialize_components(frames[0])
        
        # Process each frame
        print("Processing frames...")
        results = self._process_frames(frames)
        
        # Generate debug visualizations
        if self.config['camera_movement']['save_visualization']:
            print("... Generating debug visualizations...")
            self._generate_debug_visualizations(frames, results)
        
        return results
    
    def _process_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        Process all frames through each analysis component.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary containing results from each component
        """
        # Get object tracks (players, ball, referees)
        tracks = self.tracker.get_object_tracks(
            frames,
            read_from_stub=True,
            stub_path=f'stubs/{self.video_name}_track_stubs.pkl'
        )
        self.tracker.add_position_to_tracks(tracks)
        
        # Process pitch lines for each frame
        # pitch_lines = []
        # for frame in frames:
        #     line_mask = self.pitch_detector.create_line_mask(frame)
        #     pitch_lines.append(line_mask)
            
        # Estimate camera movement
        camera_movement = self.camera_estimator.get_camera_movement(frames)
        
        return {
            'tracks': tracks,
            # 'pitch_lines': pitch_lines,
            'pitch_lines': [],
            'camera_movement': camera_movement
        }
    
    def _generate_debug_visualizations(
        self,
        frames: List[np.ndarray],
        results: Dict
    ) -> None:
        """
        Generate and save debug visualizations.
        
        Args:
            frames: Original video frames
            results: Analysis results dictionary
        """
        # Generate pitch line visualization
        pitch_vis_frames = []
        for frame, line_mask in zip(frames, results['pitch_lines']):
            combined = self.debug_visualizer.create_side_by_side(
                frame, 
                line_mask,
                titles=("Original Frame", "Detected Lines")
            )
            pitch_vis_frames.append(combined)
            
        # Save pitch line visualization
        VideoIO.save_gif(
            pitch_vis_frames,
            self.output_dir / f"{self.video_name}_pitch_lines.gif",
            duration=100
        )
        VideoIO.save_video(
            pitch_vis_frames,
            self.output_dir / f"{self.video_name}_pitch_lines.mp4",
            fps=30
        )
        
        # Generate camera movement visualization
        movement_frames = self.camera_estimator.draw_camera_movement(
            frames, 
            results['camera_movement']
        )
        
        camera_vis_frames = []
        for frame, mov_frame in zip(frames, movement_frames):
            combined = self.debug_visualizer.create_side_by_side(
                frame,
                mov_frame,
                titles=("Original Frame", "Camera Movement")
            )
            camera_vis_frames.append(combined)
            
        # Save camera movement visualization
        VideoIO.save_gif(
            camera_vis_frames,
            self.output_dir / f"{self.video_name}_camera_movement.gif",
            duration=100
        )
        VideoIO.save_video(
            camera_vis_frames,
            self.output_dir / f"{self.video_name}_camera_movement.mp4",
            fps=30
        )

def main():
    """Main entry point for the soccer analysis pipeline."""
    # Initialize and run pipeline
    pipeline = SoccerAnalysisPipeline('config/config.yaml')
    
    # Process video
    video_path = 'input_videos/08fd33_4.mp4'
    results = pipeline.process_video(video_path)
    
    print("Analysis complete! Debug visualizations saved to debug_output/")

if __name__ == "__main__":
    main()