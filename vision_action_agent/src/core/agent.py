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

class VisionActionAgent:
    """
    Main agent class that coordinates all vision and action components.
    Designed specifically for PACS interaction and medical imaging workflows.
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
        
        # PACS-specific state
        self.current_patient_id: Optional[str] = None
        self.current_study_open: bool = False
        self.last_screenshot: Optional[np.ndarray] = None
        
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
            self.element_detector = ElementDetector(self.ocr_detector)
            
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
            logger.error(f"Failed to capture screen: {e}")
            raise
    
    def find_element(self, text: str, element_type: Optional[ElementType] = None, 
                    region: Optional[CaptureRegion] = None) -> List[UIElement]:
        """
        Find UI elements by text.
        
        Args:
            text: Text to search for
            element_type: Optional element type filter
            region: Optional region to search in
            
        Returns:
            List of found UI elements
        """
        try:
            screenshot = self.capture_screen(region)
            elements = self.element_detector.find_element_by_text(screenshot, text, element_type)
            
            logger.info(f"Found {len(elements)} elements matching '{text}'")
            return elements
            
        except Exception as e:
            logger.error(f"Failed to find element '{text}': {e}")
            return []
    
    def click_element(self, text: str, element_type: Optional[ElementType] = None,
                     click_type: ClickType = ClickType.LEFT, 
                     region: Optional[CaptureRegion] = None) -> bool:
        """
        Find and click on a UI element.
        
        Args:
            text: Text of element to click
            element_type: Optional element type filter
            click_type: Type of click to perform
            region: Optional region to search in
            
        Returns:
            True if element was found and clicked, False otherwise
        """
        try:
            elements = self.find_element(text, element_type, region)
            
            if not elements:
                logger.error(f"Element '{text}' not found")
                return False
            
            element = elements[0]  # Use first match
            success = self.mouse.click(element.center[0], element.center[1], click_type)
            
            if success:
                logger.info(f"Clicked element '{text}' at {element.center}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to click element '{text}': {e}")
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
            logger.error(f"Failed to type text: {e}")
            return False
    
    def wait_for_element(self, text: str, timeout: float = 10.0, 
                        element_type: Optional[ElementType] = None) -> bool:
        """
        Wait for an element to appear on screen.
        
        Args:
            text: Text of element to wait for
            timeout: Maximum time to wait in seconds
            element_type: Optional element type filter
            
        Returns:
            True if element appeared, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            elements = self.find_element(text, element_type)
            
            if elements:
                logger.info(f"Element '{text}' appeared after {time.time() - start_time:.2f}s")
                return True
            
            time.sleep(0.5)
        
        logger.warning(f"Element '{text}' did not appear within {timeout}s")
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
            logger.error(f"Failed to verify text '{text}': {e}")
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
            logger.error(f"Failed to save screenshot: {e}")
            return False
    
    def execute_workflow(self, workflow_steps: List[WorkflowStep], 
                        workflow_id: Optional[str] = None) -> WorkflowResult:
        """
        Execute a workflow using the workflow executor.
        
        Args:
            workflow_steps: List of workflow steps to execute
            workflow_id: Optional workflow identifier
            
        Returns:
            Workflow execution result
        """
        return self.workflow_executor.execute_workflow(workflow_steps, workflow_id)
    
    def load_workflow_from_file(self, filepath: str) -> List[WorkflowStep]:
        """
        Load workflow from YAML file.
        
        Args:
            filepath: Path to workflow file
            
        Returns:
            List of workflow steps
        """
        return self.workflow_executor.load_workflow_from_yaml(filepath)
    
    # PACS-specific methods
    
    def open_pacs_application(self, window_title: Optional[str] = None) -> bool:
        """
        Open PACS application and verify it's loaded.
        
        Args:
            window_title: Optional specific window title to look for
            
        Returns:
            True if PACS application is opened successfully
        """
        try:
            title = window_title or self.config.pacs.window_title_contains
            
            # Try to find PACS window
            windows = self.screen_capture.find_application_windows(title)
            
            if windows:
                # PACS is already open, bring to front
                window = windows[0]
                if window['is_minimized']:
                    # Restore window (this is simplified - might need more complex logic)
                    self.mouse.click(100, 100)  # Click on taskbar area
                
                logger.info("PACS application is already open")
                return True
            else:
                # Try to open PACS
                logger.info("Attempting to open PACS application")
                
                # Look for PACS icon or menu item
                if self.click_element("PACS"):
                    return self.wait_for_element("Patient", timeout=30.0)
                
                logger.error("Could not open PACS application")
                return False
                
        except Exception as e:
            logger.error(f"Failed to open PACS application: {e}")
            return False
    
    def search_patient(self, patient_id: str) -> bool:
        """
        Search for a patient in PACS.
        
        Args:
            patient_id: Patient ID to search for
            
        Returns:
            True if patient search was successful
        """
        try:
            # Find patient search field
            search_elements = self.find_element("Patient")
            if not search_elements:
                search_elements = self.find_element("Search")
            
            if not search_elements:
                logger.error("Could not find patient search field")
                return False
            
            # Click on search field
            search_element = search_elements[0]
            self.mouse.click(search_element.center[0], search_element.center[1])
            time.sleep(0.5)
            
            # Enter patient ID
            if self.type_text(patient_id, clear_first=True):
                # Press Enter or click Search button
                self.keyboard.press_key('enter')
                
                # Wait for results
                if self.wait_for_element("Results", timeout=10.0) or \
                   self.wait_for_element("Patient List", timeout=10.0):
                    self.current_patient_id = patient_id
                    logger.info(f"Patient search successful for ID: {patient_id}")
                    return True
            
            logger.error(f"Patient search failed for ID: {patient_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to search patient: {e}")
            return False
    
    def open_patient_study(self, study_description: Optional[str] = None) -> bool:
        """
        Open a patient study in PACS.
        
        Args:
            study_description: Optional study description to look for
            
        Returns:
            True if study was opened successfully
        """
        try:
            # Look for "Open" or "View" button
            if self.click_element("Open"):
                pass
            elif self.click_element("View"):
                pass
            elif self.click_element("Study"):
                pass
            else:
                # Try double-clicking on first study in list
                screenshot = self.capture_screen()
                elements = self.element_detector.detect_all_elements(screenshot).elements
                
                # Find likely study entry (large text element)
                for element in elements:
                    if (element.element_type == ElementType.BUTTON and 
                        element.text and len(element.text) > 10):
                        self.mouse.click(element.center[0], element.center[1], ClickType.DOUBLE)
                        break
                else:
                    logger.error("Could not find study to open")
                    return False
            
            # Wait for study to load
            if self.wait_for_element("Image", timeout=30.0) or \
               self.wait_for_element("View", timeout=30.0):
                self.current_study_open = True
                logger.info("Study opened successfully")
                return True
            
            logger.error("Study did not load properly")
            return False
            
        except Exception as e:
            logger.error(f"Failed to open study: {e}")
            return False
    
    def capture_study_images(self, save_path: Optional[str] = None) -> List[str]:
        """
        Capture all visible medical images in the current study.
        
        Args:
            save_path: Optional directory path to save images
            
        Returns:
            List of saved image file paths
        """
        try:
            if not self.current_study_open:
                logger.error("No study is currently open")
                return []
            
            screenshot = self.capture_screen()
            
            # Detect image elements (medical images)
            image_elements = []
            all_elements = self.element_detector.detect_all_elements(screenshot).elements
            
            for element in all_elements:
                if element.element_type == ElementType.IMAGE:
                    # Filter for large images (likely medical images)
                    area = element.bbox[2] * element.bbox[3]
                    if area > self.config.element_detection.image_min_size:
                        image_elements.append(element)
            
            saved_files = []
            
            for i, element in enumerate(image_elements):
                # Extract image region
                x, y, w, h = element.bbox
                region = CaptureRegion(x, y, w, h)
                image = self.screen_capture.capture_region(region)
                
                # Save image
                if save_path:
                    filename = f"{save_path}/study_image_{i+1}.png"
                else:
                    filename = f"study_image_{i+1}.png"
                
                if self.screen_capture.save_screenshot(image, filename):
                    saved_files.append(filename)
            
            logger.info(f"Captured {len(saved_files)} study images")
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to capture study images: {e}")
            return []
    
    def create_pacs_workflow(self, patient_id: str, 
                           custom_steps: Optional[List[WorkflowStep]] = None) -> List[WorkflowStep]:
        """
        Create a complete PACS workflow for patient study processing.
        
        Args:
            patient_id: Patient ID to process
            custom_steps: Optional custom steps to add to workflow
            
        Returns:
            Complete workflow steps
        """
        steps = [
            WorkflowStep(
                id="open_pacs",
                action_type=ActionType.CUSTOM,
                parameters={
                    "action_name": "open_pacs",
                    "action_params": {}
                },
                description="Open PACS application"
            ),
            WorkflowStep(
                id="search_patient",
                action_type=ActionType.CUSTOM,
                parameters={
                    "action_name": "search_patient",
                    "action_params": {"patient_id": patient_id}
                },
                description=f"Search for patient {patient_id}"
            ),
            WorkflowStep(
                id="open_study",
                action_type=ActionType.CUSTOM,
                parameters={
                    "action_name": "open_study",
                    "action_params": {}
                },
                description="Open patient study"
            ),
            WorkflowStep(
                id="capture_images",
                action_type=ActionType.CUSTOM,
                parameters={
                    "action_name": "capture_images",
                    "action_params": {"save_path": f"patient_{patient_id}"}
                },
                description="Capture study images"
            ),
            WorkflowStep(
                id="take_screenshot",
                action_type=ActionType.SCREENSHOT,
                parameters={"filepath": f"patient_{patient_id}_full_screen.png"},
                description="Take full screen screenshot"
            )
        ]
        
        # Add custom steps if provided
        if custom_steps:
            steps.extend(custom_steps)
        
        # Register custom actions
        self.workflow_executor.register_custom_action("open_pacs", self._custom_open_pacs)
        self.workflow_executor.register_custom_action("search_patient", self._custom_search_patient)
        self.workflow_executor.register_custom_action("open_study", self._custom_open_study)
        self.workflow_executor.register_custom_action("capture_images", self._custom_capture_images)
        
        return steps
    
    # Custom action implementations for workflow executor
    
    def _custom_open_pacs(self, params: Dict[str, Any], executor) -> bool:
        """Custom action to open PACS."""
        return self.open_pacs_application()
    
    def _custom_search_patient(self, params: Dict[str, Any], executor) -> bool:
        """Custom action to search patient."""
        patient_id = params.get("patient_id")
        return self.search_patient(patient_id) if patient_id else False
    
    def _custom_open_study(self, params: Dict[str, Any], executor) -> bool:
        """Custom action to open study."""
        return self.open_patient_study()
    
    def _custom_capture_images(self, params: Dict[str, Any], executor) -> bool:
        """Custom action to capture images."""
        save_path = params.get("save_path")
        images = self.capture_study_images(save_path)
        return len(images) > 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "current_patient_id": self.current_patient_id,
            "current_study_open": self.current_study_open,
            "last_screenshot_time": time.time() if self.last_screenshot is not None else None,
            "config_loaded": self.config_manager.config_path is not None,
            "components_initialized": True
        }
    
    def cleanup(self):
        """Cleanup agent resources."""
        logger.info("Cleaning up Vision Action Agent")
        
        # Clear action histories
        if hasattr(self, 'mouse'):
            self.mouse.clear_history()
        
        if hasattr(self, 'keyboard'):
            self.keyboard.clear_history()
        
        # Reset state
        self.current_patient_id = None
        self.current_study_open = False
        self.last_screenshot = None
        
        logger.info("Agent cleanup completed")