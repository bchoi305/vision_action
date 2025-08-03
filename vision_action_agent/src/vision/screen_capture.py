import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageGrab
from typing import Optional, Tuple, Union
import time
from dataclasses import dataclass
from loguru import logger

@dataclass
class CaptureRegion:
    x: int
    y: int
    width: int
    height: int

class ScreenCapture:
    def __init__(self, enable_failsafe: bool = True):
        if enable_failsafe:
            pyautogui.FAILSAFE = True
        else:
            pyautogui.FAILSAFE = False
        
        self.screen_size = pyautogui.size()
        logger.info(f"Screen size detected: {self.screen_size}")
    
    def capture_full_screen(self) -> np.ndarray:
        """Capture the entire screen and return as numpy array."""
        try:
            screenshot = ImageGrab.grab()
            return np.array(screenshot)
        except Exception as e:
            logger.error(f"Failed to capture full screen: {e}")
            raise
    
    def capture_region(self, region: CaptureRegion) -> np.ndarray:
        """Capture a specific region of the screen."""
        try:
            bbox = (region.x, region.y, region.x + region.width, region.y + region.height)
            screenshot = ImageGrab.grab(bbox=bbox)
            return np.array(screenshot)
        except Exception as e:
            logger.error(f"Failed to capture region {region}: {e}")
            raise
    
    def capture_window(self, window_title: str) -> Optional[np.ndarray]:
        """Capture a specific window by title."""
        try:
            windows = pyautogui.getWindowsWithTitle(window_title)
            if not windows:
                logger.warning(f"No window found with title: {window_title}")
                return None
            
            window = windows[0]
            if window.isMinimized:
                window.restore()
            
            window.activate()
            time.sleep(0.5)  # Wait for window to come to front
            
            region = CaptureRegion(
                x=window.left,
                y=window.top,
                width=window.width,
                height=window.height
            )
            
            return self.capture_region(region)
        except Exception as e:
            logger.error(f"Failed to capture window '{window_title}': {e}")
            return None
    
    def save_screenshot(self, image: np.ndarray, filepath: str) -> bool:
        """Save screenshot to file."""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb)
            pil_image.save(filepath)
            logger.info(f"Screenshot saved to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            return False
    
    def get_screen_dimensions(self) -> Tuple[int, int]:
        """Get screen width and height."""
        return self.screen_size
    
    def find_application_windows(self, app_name: str) -> list:
        """Find all windows containing the application name."""
        try:
            all_windows = pyautogui.getAllWindows()
            matching_windows = []
            
            for window in all_windows:
                if app_name.lower() in window.title.lower():
                    matching_windows.append({
                        'title': window.title,
                        'left': window.left,
                        'top': window.top,
                        'width': window.width,
                        'height': window.height,
                        'is_minimized': window.isMinimized
                    })
            
            return matching_windows
        except Exception as e:
            logger.error(f"Failed to find application windows: {e}")
            return []
    
    def monitor_region(self, region: CaptureRegion, callback, interval: float = 1.0):
        """Monitor a specific region and call callback when changes are detected."""
        previous_image = None
        
        try:
            while True:
                current_image = self.capture_region(region)
                
                if previous_image is not None:
                    # Calculate difference between images
                    diff = cv2.absdiff(current_image, previous_image)
                    diff_sum = np.sum(diff)
                    
                    if diff_sum > 1000:  # Threshold for change detection
                        callback(current_image, diff)
                
                previous_image = current_image.copy()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")