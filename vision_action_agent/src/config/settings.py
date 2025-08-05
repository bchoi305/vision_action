import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
from loguru import logger

@dataclass
class GeneralConfig:
    default_timeout: float = 30.0
    default_retry_count: int = 3

@dataclass
class ScreenCaptureConfig:
    enable_failsafe: bool = True
    default_screenshot_format: str = "png"
    screenshot_quality: int = 95
    capture_cursor: bool = False

@dataclass
class OCRConfig:
    engine: str = "easyocr"
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
    step_pause: float = 0.5
    continue_on_failure: bool = False
    save_screenshots_on_failure: bool = True

@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/vision_action.log"
    max_log_file_size: str = "10MB"
    backup_count: int = 5

@dataclass
class ApplicationConfig:
    name: str
    window_title_contains: str
    general: Optional[GeneralConfig] = None
    screen_capture: Optional[ScreenCaptureConfig] = None
    ocr: Optional[OCRConfig] = None
    mouse: Optional[MouseConfig] = None
    keyboard: Optional[KeyboardConfig] = None
    element_detection: Optional[ElementDetectionConfig] = None
    workflow: Optional[WorkflowConfig] = None
    logging: Optional[LoggingConfig] = None

@dataclass
class AgentConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    screen_capture: ScreenCaptureConfig = field(default_factory=ScreenCaptureConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    element_detection: ElementDetectionConfig = field(default_factory=ElementDetectionConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    applications: List[ApplicationConfig] = field(default_factory=list)

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config = AgentConfig()
        self.config_path = config_path or self._find_config_file()
        if self.config_path and os.path.exists(self.config_path):
            self.load_config()
        else:
            logger.info("No configuration file found, using defaults")

    def _find_config_file(self) -> Optional[str]:
        possible_paths = [
            "config.yaml",
            "config.yml",
            "config.json",
            os.path.expanduser("~/.vision_action/config.yaml"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found configuration file: {path}")
                return path
        return None

    def load_config(self, config_path: Optional[str] = None) -> bool:
        path = config_path or self.config_path
        if not path or not os.path.exists(path):
            logger.error(f"Configuration file not found: {path}")
            return False
        try:
            with open(path, 'r') as f:
                config_data = yaml.safe_load(f)
            self._update_config_from_dict(config_data)
            logger.info(f"Configuration loaded from: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        for section, section_config in config_data.items():
            if hasattr(self.config, section):
                self._update_dataclass(getattr(self.config, section), section_config)

    def _update_dataclass(self, dataclass_obj, update_dict: Dict[str, Any]):
        for key, value in update_dict.items():
            if hasattr(dataclass_obj, key):
                setattr(dataclass_obj, key, value)

    def get_config(self) -> AgentConfig:
        return self.config

    def get_app_config(self, app_name: str) -> Optional[ApplicationConfig]:
        for app_config in self.config.applications:
            if app_config.name == app_name:
                return app_config
        return None

    def validate_config(self) -> bool:
        try:
            # Validate general settings
            if not 0 <= self.config.general.default_timeout <= 300:
                logger.error("Default timeout must be between 0 and 300 seconds.")
                return False
            if not 0 <= self.config.general.default_retry_count <= 10:
                logger.error("Default retry count must be between 0 and 10.")
                return False

            # Validate OCR settings
            if self.config.ocr.engine not in ["easyocr", "tesseract"]:
                logger.error(f"Invalid OCR engine: {self.config.ocr.engine}")
                return False
            if not 0.0 <= self.config.ocr.confidence_threshold <= 1.0:
                logger.error("OCR confidence threshold must be between 0.0 and 1.0")
                return False

            # Validate mouse settings
            if not 0.1 <= self.config.mouse.movement_speed <= 2.0:
                logger.error("Mouse movement speed must be between 0.1 and 2.0")
                return False

            logger.info("Configuration validation passed")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

config_manager = ConfigManager()
