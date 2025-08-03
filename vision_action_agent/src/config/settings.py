import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
from loguru import logger

@dataclass
class ScreenCaptureConfig:
    enable_failsafe: bool = True
    default_screenshot_format: str = "png"
    screenshot_quality: int = 95
    capture_cursor: bool = False
    
@dataclass 
class OCRConfig:
    engine: str = "easyocr"  # "easyocr" or "tesseract"
    languages: list = field(default_factory=lambda: ["en"])
    confidence_threshold: float = 0.3
    preprocessing: bool = True
    
@dataclass
class MouseConfig:
    enable_failsafe: bool = True
    default_pause: float = 0.5
    movement_speed: float = 1.0
    human_like_movement: bool = True
    click_delay: float = 0.1
    
@dataclass
class KeyboardConfig:
    default_pause: float = 0.1
    typing_speed: float = 0.05
    human_like_typing: bool = True
    
@dataclass
class ElementDetectionConfig:
    button_confidence: float = 0.7
    text_field_confidence: float = 0.6
    checkbox_confidence: float = 0.6
    dropdown_confidence: float = 0.7
    image_min_size: int = 1000
    overlap_threshold: float = 0.5
    
@dataclass
class WorkflowConfig:
    default_timeout: float = 30.0
    default_retry_count: int = 3
    step_pause: float = 0.5
    continue_on_failure: bool = False
    save_screenshots_on_failure: bool = True
    
@dataclass
class PACSConfig:
    application_name: str = "PACS"
    window_title_contains: str = "PACS"
    default_patient_search_timeout: float = 10.0
    image_load_timeout: float = 30.0
    report_generation_timeout: float = 60.0
    study_viewer_region: Optional[Dict[str, int]] = None
    common_ui_elements: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/vision_action.log"
    max_log_file_size: str = "10MB"
    backup_count: int = 5
    
@dataclass
class AgentConfig:
    screen_capture: ScreenCaptureConfig = field(default_factory=ScreenCaptureConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    element_detection: ElementDetectionConfig = field(default_factory=ElementDetectionConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    pacs: PACSConfig = field(default_factory=PACSConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default locations.
        """
        self.config = AgentConfig()
        self.config_path = config_path or self._find_config_file()
        
        # Load configuration if file exists
        if self.config_path and os.path.exists(self.config_path):
            self.load_config()
        else:
            logger.info("No configuration file found, using defaults")
            
        # Setup PACS default elements
        self._setup_default_pacs_config()
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations."""
        possible_paths = [
            "config.yaml",
            "config.yml", 
            "config.json",
            "configs/config.yaml",
            "configs/config.yml",
            "configs/config.json",
            os.path.expanduser("~/.vision_action/config.yaml"),
            os.path.expanduser("~/.vision_action/config.yml"),
            os.path.expanduser("~/.vision_action/config.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found configuration file: {path}")
                return path
        
        return None
    
    def _setup_default_pacs_config(self):
        """Setup default PACS UI elements configuration."""
        default_elements = {
            "patient_search": {
                "type": "text_field",
                "text_contains": ["Patient", "Search", "ID"],
                "region": None
            },
            "search_button": {
                "type": "button", 
                "text_contains": ["Search", "Find", "Go"],
                "region": None
            },
            "patient_list": {
                "type": "list",
                "text_contains": ["Patient List", "Results"],
                "region": None
            },
            "open_study": {
                "type": "button",
                "text_contains": ["Open", "View", "Study"],
                "region": None
            },
            "study_viewer": {
                "type": "image",
                "region": self.config.pacs.study_viewer_region,
                "min_size": 50000
            },
            "report_area": {
                "type": "text_field",
                "text_contains": ["Report", "Notes", "Findings"],
                "region": None
            },
            "save_report": {
                "type": "button",
                "text_contains": ["Save", "Submit", "Finalize"],
                "region": None
            }
        }
        
        if not self.config.pacs.common_ui_elements:
            self.config.pacs.common_ui_elements = default_elements
    
    def load_config(self, config_path: Optional[str] = None) -> bool:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file. If None, uses stored path.
        """
        path = config_path or self.config_path
        if not path or not os.path.exists(path):
            logger.error(f"Configuration file not found: {path}")
            return False
        
        try:
            with open(path, 'r') as f:
                if path.endswith('.json'):
                    config_data = json.load(f)
                else:  # Assume YAML
                    config_data = yaml.safe_load(f)
            
            # Update configuration with loaded data
            self._update_config_from_dict(config_data)
            
            logger.info(f"Configuration loaded from: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration. If None, uses stored path.
        """
        path = config_path or self.config_path or "config.yaml"
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            config_dict = asdict(self.config)
            
            with open(path, 'w') as f:
                if path.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                else:  # Save as YAML
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.config_path = path
            logger.info(f"Configuration saved to: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration object from dictionary."""
        if 'screen_capture' in config_data:
            self._update_dataclass(self.config.screen_capture, config_data['screen_capture'])
        
        if 'ocr' in config_data:
            self._update_dataclass(self.config.ocr, config_data['ocr'])
            
        if 'mouse' in config_data:
            self._update_dataclass(self.config.mouse, config_data['mouse'])
            
        if 'keyboard' in config_data:
            self._update_dataclass(self.config.keyboard, config_data['keyboard'])
            
        if 'element_detection' in config_data:
            self._update_dataclass(self.config.element_detection, config_data['element_detection'])
            
        if 'workflow' in config_data:
            self._update_dataclass(self.config.workflow, config_data['workflow'])
            
        if 'pacs' in config_data:
            self._update_dataclass(self.config.pacs, config_data['pacs'])
            
        if 'logging' in config_data:
            self._update_dataclass(self.config.logging, config_data['logging'])
    
    def _update_dataclass(self, dataclass_obj, update_dict: Dict[str, Any]):
        """Update dataclass object with values from dictionary."""
        for key, value in update_dict.items():
            if hasattr(dataclass_obj, key):
                setattr(dataclass_obj, key, value)
    
    def get_config(self) -> AgentConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, section: str, updates: Dict[str, Any]):
        """
        Update specific configuration section.
        
        Args:
            section: Configuration section name (e.g., 'mouse', 'ocr')
            updates: Dictionary of updates to apply
        """
        if hasattr(self.config, section):
            section_obj = getattr(self.config, section)
            self._update_dataclass(section_obj, updates)
            logger.info(f"Updated {section} configuration")
        else:
            logger.error(f"Unknown configuration section: {section}")
    
    def create_default_config_file(self, path: str = "config.yaml"):
        """Create a default configuration file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            
            # Save current (default) configuration
            self.save_config(path)
            
            logger.info(f"Default configuration file created: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create default configuration file: {e}")
            return False
    
    def validate_config(self) -> bool:
        """Validate current configuration."""
        try:
            # Validate OCR engine
            if self.config.ocr.engine not in ["easyocr", "tesseract"]:
                logger.error(f"Invalid OCR engine: {self.config.ocr.engine}")
                return False
            
            # Validate confidence thresholds
            if not 0.0 <= self.config.ocr.confidence_threshold <= 1.0:
                logger.error("OCR confidence threshold must be between 0.0 and 1.0")
                return False
                
            # Validate mouse speed
            if not 0.1 <= self.config.mouse.movement_speed <= 2.0:
                logger.error("Mouse movement speed must be between 0.1 and 2.0")
                return False
            
            # Validate timeouts
            if self.config.workflow.default_timeout <= 0:
                logger.error("Workflow timeout must be positive")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_pacs_element_config(self, element_name: str) -> Optional[Dict[str, Any]]:
        """Get PACS UI element configuration."""
        return self.config.pacs.common_ui_elements.get(element_name)
    
    def add_pacs_element_config(self, element_name: str, element_config: Dict[str, Any]):
        """Add new PACS UI element configuration."""
        self.config.pacs.common_ui_elements[element_name] = element_config
        logger.info(f"Added PACS element configuration: {element_name}")
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        from loguru import logger
        
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            lambda msg: print(msg, end=""),
            level=self.config.logging.level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Add file handler if enabled
        if self.config.logging.log_to_file:
            log_dir = os.path.dirname(self.config.logging.log_file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            logger.add(
                self.config.logging.log_file_path,
                level=self.config.logging.level,
                rotation=self.config.logging.max_log_file_size,
                retention=self.config.logging.backup_count,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
            )

# Global configuration manager instance
config_manager = ConfigManager()