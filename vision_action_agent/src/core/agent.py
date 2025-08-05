import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from loguru import logger

from ..config.settings import ConfigManager, config_manager
from ..vision.screen_capture import ScreenCapture, CaptureRegion
from ..vision.ocr_engine import OCRTextDetector, OCREngine
from ..vision.element_detector import ElementDetector, UIElement, ElementType
from ..actions.mouse_controller import MouseController, ClickType
from ..actions.keyboard_controller import KeyboardController
from ..actions.workflow_executor import WorkflowExecutor, WorkflowStep, ActionType, WorkflowResult
from ..utils.error_handler import ErrorHandler, ErrorCategory

class VisionActionAgent:
    """
    Main agent class that coordinates all vision and action components.
    Designed for general-purpose UI automation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Vision Action Agent.
        
        Args:
            config_path: Path to configuration file. If None, uses default configuration.
        """
        # Initialize configuration
        if config_path:
            self.config_manager = ConfigManager(config_path)
        else:
            self.config_manager = config_manager
        
        self.config = self.config_manager.get_config()
        
        # Setup logging
        self.config_manager.setup_logging()
        
        logger.info("Initializing Vision Action Agent")
        
        # Initialize core components
        self._initialize_components()
        
        # General state
        self.last_screenshot: Optional[np.ndarray] = None
        self.last_action: Optional[str] = None
        self.current_application_context: Optional[str] = None
        
        # Error handler
        self.error_handler = ErrorHandler()
        
        logger.info("Vision Action Agent initialized successfully")
    
    def _initialize_components(self):
        """Initialize all agent components."""
        try:
            # Screen capture
            self.screen_capture = ScreenCapture(
                enable_failsafe=self.config.screen_capture.enable_failsafe
            )
            
            # OCR engine
            ocr_engine = OCREngine.EASYOCR if self.config.ocr.engine == "easyocr" else OCREngine.TESSERACT
            self.ocr_detector = OCRTextDetector(
                engine=ocr_engine,
                languages=self.config.ocr.languages
            )
            
            # Element detector
            self.element_detector = ElementDetector(self.ocr_detector, template_dir="./templates")
            
            # Mouse controller
            self.mouse = MouseController(
                enable_failsafe=self.config.mouse.enable_failsafe,
                default_pause=self.config.mouse.default_pause,
                movement_speed=self.config.mouse.movement_speed
            )
            
            # Keyboard controller
            self.keyboard = KeyboardController(
                default_pause=self.config.keyboard.default_pause,
                typing_speed=self.config.keyboard.typing_speed
            )
            
            # Workflow executor
            self.workflow_executor = WorkflowExecutor()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def capture_screen(self, region: Optional[CaptureRegion] = None) -> np.ndarray:
        """
        Capture the screen or a specific region.
        
        Args:
            region: Optional region to capture. If None, captures full screen.
            
        Returns:
            Screenshot as numpy array
        """
        try:
            if region:
                screenshot = self.screen_capture.capture_region(region)
            else:
                screenshot = self.screen_capture.capture_full_screen()
            
            self.last_screenshot = screenshot
            return screenshot
            
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.SCREEN_CAPTURE, context={"method": "capture_screen"})
            raise
    
    def find_element(self, text: Optional[str] = None, element_type: Optional[ElementType] = None,
                    image_template: Optional[str] = None, region: Optional[CaptureRegion] = None) -> List[UIElement]:
        """
        Find UI elements by text, type, or image template.
        
        Args:
            text: Optional text to search for within elements.
            element_type: Optional element type filter.
            image_template: Optional name of a pre-loaded image template to search for.
            region: Optional region to search in.
            
        Returns:
            List of found UI elements.
        """
        try:
            screenshot = self.capture_screen(region)
            
            if image_template:
                elements = self.element_detector.find_element_by_image(screenshot, image_template)
            else:
                elements = self.element_detector.detect_all_elements(screenshot).elements
            
            filtered_elements = []
            for element in elements:
                text_match = True
                type_match = True

                if text is not None:
                    text_match = element.text and text.lower() in element.text.lower()
                
                if element_type is not None:
                    type_match = element.element_type == element_type
                
                if text_match and type_match:
                    filtered_elements.append(element)
            
            logger.info(f"Found {len(filtered_elements)} elements matching criteria (text='{text}', type='{element_type}', image_template='{image_template}')")
            return filtered_elements
            
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.ELEMENT_DETECTION, context={"method": "find_element", "text": text, "element_type": element_type, "image_template": image_template})
            return []
    
    def click_element(self, text: Optional[str] = None, element_type: Optional[ElementType] = None,
                     click_type: ClickType = ClickType.LEFT, 
                     region: Optional[CaptureRegion] = None) -> bool:
        """
        Find and click on a UI element.
        
        Args:
            text: Optional text of element to click.
            element_type: Optional element type filter.
            click_type: Type of click to perform.
            region: Optional region to search in.
            
        Returns:
            True if element was found and clicked, False otherwise.
        """
        try:
            elements = self.find_element(text, element_type, region)
            
            if not elements:
                logger.error(f"Element (text='{text}', type='{element_type}') not found")
                return False
            
            element = elements[0]  # Use first match
            success = self.mouse.click(element.center[0], element.center[1], click_type)
            
            if success:
                logger.info(f"Clicked element '{element.text or element.element_type.value}' at {element.center}")
            
            return success
            
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.MOUSE_ACTION, context={"method": "click_element", "text": text, "element_type": element_type})
            return False
    
    def type_text(self, text: str, clear_first: bool = True) -> bool:
        """
        Type text using keyboard controller.
        
        Args:
            text: Text to type
            clear_first: Whether to clear existing text first
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if clear_first:
                self.keyboard.select_all()
                time.sleep(0.1)
            
            return self.keyboard.type_text(text, self.config.keyboard.human_like_typing)
            
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.KEYBOARD_ACTION, context={"method": "type_text"})
            return False

    def type_text_in_element(self, text: str, element_text: Optional[str] = None, 
                             element_type: Optional[ElementType] = None, 
                             clear_first: bool = True, 
                             region: Optional[CaptureRegion] = None) -> bool:
        """
        Find a UI element and type text into it.
        
        Args:
            text: Text to type.
            element_text: Optional text of the element to type into.
            element_type: Optional element type filter.
            clear_first: Whether to clear existing text first.
            region: Optional region to search in.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            elements = self.find_element(element_text, element_type, region)
            
            if not elements:
                logger.error(f"Element (text='{element_text}', type='{element_type}') not found for typing")
                return False
            
            element = elements[0]  # Use first match
            
            # Click the element to focus it before typing
            self.mouse.click(element.center[0], element.center[1])
            time.sleep(0.1) # Give time for focus
            
            if clear_first:
                self.keyboard.select_all()
                time.sleep(0.1)
            
            return self.keyboard.type_text(text, self.config.keyboard.human_like_typing)
            
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.KEYBOARD_ACTION, context={"method": "type_text_in_element", "element_text": element_text, "element_type": element_type})
            return False
    
    def wait_for_element(self, text: Optional[str] = None, element_type: Optional[ElementType] = None, 
                        timeout: float = 10.0, region: Optional[CaptureRegion] = None) -> bool:
        """
        Wait for an element to appear on screen.
        
        Args:
            text: Optional text of element to wait for.
            element_type: Optional element type filter.
            timeout: Maximum time to wait in seconds.
            region: Optional region to search in.
            
        Returns:
            True if element appeared, False if timeout.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            elements = self.find_element(text, element_type, region)
            
            if elements:
                logger.info(f"Element (text='{text}', type='{element_type}') appeared after {time.time() - start_time:.2f}s")
                return True
            
            time.sleep(0.5)
        
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.ELEMENT_DETECTION, context={"method": "wait_for_element", "text": text, "element_type": element_type})
            return False

    def wait_for_element_to_disappear(self, text: Optional[str] = None, element_type: Optional[ElementType] = None,
                                      timeout: float = 10.0, region: Optional[CaptureRegion] = None) -> bool:
        """
        Wait for an element to disappear from screen.
        
        Args:
            text: Optional text of element to wait for disappearance.
            element_type: Optional element type filter.
            timeout: Maximum time to wait in seconds.
            region: Optional region to search in.
            
        Returns:
            True if element disappeared, False if timeout.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            elements = self.find_element(text, element_type, region)
            
            if not elements:
                logger.info(f"Element (text='{text}', type='{element_type}') disappeared after {time.time() - start_time:.2f}s")
                return True
            
            time.sleep(0.5)
        
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.ELEMENT_DETECTION, context={"method": "wait_for_element_to_disappear", "text": text, "element_type": element_type})
            return False
    
    def verify_text_present(self, text: str, region: Optional[CaptureRegion] = None) -> bool:
        """
        Verify that specific text is present on screen.
        
        Args:
            text: Text to verify
            region: Optional region to search in
            
        Returns:
            True if text is found, False otherwise
        """
        try:
            screenshot = self.capture_screen(region)
            ocr_result = self.ocr_detector.extract_text(screenshot)
            
            found = text.lower() in ocr_result.raw_text.lower()
            
            if found:
                logger.info(f"Text '{text}' verified successfully")
            else:
                logger.warning(f"Text '{text}' not found on screen")
            
            return found
            
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.OCR, context={"method": "verify_text_present", "text": text})
            return False
    
    def save_screenshot(self, filepath: str, region: Optional[CaptureRegion] = None) -> bool:
        """
        Save a screenshot to file.
        
        Args:
            filepath: Path to save screenshot
            region: Optional region to capture
            
        Returns:
            True if successful, False otherwise
        """
        try:
            screenshot = self.capture_screen(region)
            return self.screen_capture.save_screenshot(screenshot, filepath)
            
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.SCREEN_CAPTURE, context={"method": "save_screenshot", "filepath": filepath})
            return False
    
    def execute_workflow(self, workflow_steps: List[WorkflowStep>, 
                        workflow_id: Optional[str] = None) -> WorkflowResult:
        """
        Execute a workflow using the workflow executor.
        
        Args:
            workflow_steps: List of workflow steps to execute
            workflow_id: Optional workflow identifier
            
        Returns:
            Workflow execution result
        """
        try:
            return self.workflow_executor.execute_workflow(workflow_steps, workflow_id)
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.WORKFLOW, context={"method": "execute_workflow", "workflow_id": workflow_id})
            raise
    
    def load_workflow_from_file(self, filepath: str) -> List[WorkflowStep]:
        """
        Load workflow from YAML file.
        
        Args:
            filepath: Path to workflow file
            
        Returns:
            List of workflow steps
        """
        try:
            return self.workflow_executor.load_workflow_from_yaml(filepath)
        except Exception as e:
            self.error_handler.handle_error(e, ErrorCategory.WORKFLOW, context={"method": "load_workflow_from_file", "filepath": filepath})
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.
        """
        return {
            "last_screenshot_time": time.time() if self.last_screenshot is not None else None,
            "last_action": self.last_action,
            "current_application_context": self.current_application_context,
            "config_loaded": self.config_manager.config_path is not None,
            "components_initialized": True,
            "error_statistics": self.error_handler.get_error_statistics()
        }
    
    def cleanup(self):
        """
        Cleanup agent resources.
        """
        logger.info("Cleaning up Vision Action Agent")
        
        # Clear action histories
        if hasattr(self, 'mouse'):
            self.mouse.clear_history()
        
        if hasattr(self, 'keyboard'):
            self.keyboard.clear_history()
        
        # Reset state
        self.last_screenshot = None
        self.last_action = None
        self.current_application_context = None
        
        # Clear error history
        self.error_handler.clear_history()
        
        logger.info("Agent cleanup completed")