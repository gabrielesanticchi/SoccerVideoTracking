import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigurationLoader:
    """
    A simple YAML configuration loader that reads and structures configuration data.
    Each component in the system will handle its own default parameters and validation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration loader with a path to the YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the YAML file. The method performs basic validation
        to ensure the file exists and contains valid YAML, but leaves parameter
        validation to individual components.
        
        Returns:
            Dictionary containing the structured configuration data
        
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        # Check if config file exists
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )
        
        # Load and parse YAML
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Return empty dict if file is empty
            return config if config is not None else {}
                
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Error parsing YAML configuration: {str(e)}"
            )
            
def create_example_config(output_path: str) -> None:
    """
    Create an example configuration file with comments explaining each section.
    
    Args:
        output_path: Path where the example configuration will be saved
    """
    example_config = {
        'input_video': {
            'frame_rate_reduction': 20,  # Reduce frame rate by this factor
            'resize_factor': 0.5         # Resize factor for input video
        },
        'camera_movement': {
            'read_from_stub': True,      # Read cached results instead of computing new ones
            'visualization': {
                'save_visualization': True  # Save visualization results
            }
        },
        'pitch_lines_detector': {
            'canny_low': 50,             # Canny edge detection low threshold
            'canny_high': 150,           # Canny edge detection high threshold
            'hough_threshold': 50,       # Hough transform threshold
            'min_lines_length': 100,      # Minimum line length for Hough transform
            'max_lines_gap': 10,          # Maximum gap between lines for Hough transform
            'binary_threshold': 200      # Binary threshold for white line detection
        }
    }
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write configuration with comments
    with open(output_path, 'w') as file:
        yaml.dump(
            example_config,
            file,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=80
        )