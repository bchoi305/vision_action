import pyautogui
import time
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import random

class ClickType(Enum):
    LEFT = "left"
    RIGHT = "right"
    DOUBLE = "double"
    MIDDLE = "middle"

class DragDirection(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

@dataclass
class MouseAction:
    action_type: str
    x: int
    y: int
    duration: float = 0.0
    button: str = "left"
    clicks: int = 1
    success: bool = False
    timestamp: float = 0.0

class MouseController:
    def __init__(self, 
                 enable_failsafe: bool = True,
                 default_pause: float = 0.5,
                 movement_speed: float = 1.0):
        """
        Initialize mouse controller.
        
        Args:
            enable_failsafe: Enable PyAutoGUI failsafe (move mouse to corner to stop)
            default_pause: Default pause between actions
            movement_speed: Speed multiplier for mouse movements (0.1-2.0)
        """
        pyautogui.FAILSAFE = enable_failsafe
        pyautogui.PAUSE = default_pause
        self.movement_speed = max(0.1, min(2.0, movement_speed))
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        logger.info(f"Mouse controller initialized. Screen: {self.screen_width}x{self.screen_height}")
        
        # Action history for debugging
        self.action_history: List[MouseAction] = []
    
    def _add_to_history(self, action: MouseAction):
        """Add action to history for debugging."""
        action.timestamp = time.time()
        self.action_history.append(action)
        
        # Keep only last 100 actions
        if len(self.action_history) > 100:
            self.action_history.pop(0)
    
    def _validate_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Validate and clamp coordinates to screen bounds."""
        x = max(0, min(self.screen_width - 1, x))
        y = max(0, min(self.screen_height - 1, y))
        return x, y
    
    def _human_like_movement(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5):
        """Move mouse in a human-like curved path."""
        if duration <= 0:
            pyautogui.moveTo(end_x, end_y)
            return
        
        # Calculate control points for bezier curve
        distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
        
        # Add some randomness to the curve
        mid_x = (start_x + end_x) / 2 + random.randint(-50, 50)
        mid_y = (start_y + end_y) / 2 + random.randint(-50, 50)
        
        # Generate curve points
        steps = max(10, int(distance / 20))
        for i in range(steps + 1):
            t = i / steps
            
            # Quadratic bezier curve
            x = (1 - t) ** 2 * start_x + 2 * (1 - t) * t * mid_x + t ** 2 * end_x
            y = (1 - t) ** 2 * start_y + 2 * (1 - t) * t * mid_y + t ** 2 * end_y
            
            pyautogui.moveTo(int(x), int(y))
            time.sleep(duration / steps)
    
    def get_current_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        return pyautogui.position()
    
    def move_to(self, x: int, y: int, duration: float = 0.5, human_like: bool = True) -> bool:
        """
        Move mouse to specific coordinates.
        
        Args:
            x, y: Target coordinates
            duration: Movement duration in seconds
            human_like: Use human-like curved movement
        """
        try:
            x, y = self._validate_coordinates(x, y)
            start_x, start_y = self.get_current_position()
            
            if human_like and duration > 0:
                self._human_like_movement(start_x, start_y, x, y, duration * self.movement_speed)
            else:
                pyautogui.moveTo(x, y, duration * self.movement_speed)
            
            action = MouseAction("move", x, y, duration, success=True)
            self._add_to_history(action)
            
            logger.debug(f"Moved mouse to ({x}, {y})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move mouse to ({x}, {y}): {e}")
            action = MouseAction("move", x, y, duration, success=False)
            self._add_to_history(action)
            return False
    
    def click(self, x: Optional[int] = None, y: Optional[int] = None, 
              click_type: ClickType = ClickType.LEFT, clicks: int = 1, 
              interval: float = 0.1) -> bool:
        """
        Perform mouse click.
        
        Args:
            x, y: Click coordinates (None for current position)
            click_type: Type of click (left, right, double, middle)
            clicks: Number of clicks
            interval: Interval between multiple clicks
        """
        try:
            if x is not None and y is not None:
                x, y = self._validate_coordinates(x, y)
                self.move_to(x, y, duration=0.2)
            else:
                x, y = self.get_current_position()
            
            if click_type == ClickType.DOUBLE:
                pyautogui.doubleClick(x, y, interval=interval)
            elif click_type == ClickType.RIGHT:
                pyautogui.rightClick(x, y)
            elif click_type == ClickType.MIDDLE:
                pyautogui.middleClick(x, y)
            else:  # LEFT click
                pyautogui.click(x, y, clicks=clicks, interval=interval)
            
            action = MouseAction("click", x, y, button=click_type.value, clicks=clicks, success=True)
            self._add_to_history(action)
            
            logger.debug(f"Clicked at ({x}, {y}) with {click_type.value} button, {clicks} times")
            return True
            
        except Exception as e:
            logger.error(f"Failed to click at ({x}, {y}): {e}")
            action = MouseAction("click", x or 0, y or 0, button=click_type.value, success=False)
            self._add_to_history(action)
            return False
    
    def drag_and_drop(self, start_x: int, start_y: int, end_x: int, end_y: int, 
                      duration: float = 1.0, button: str = "left") -> bool:
        """
        Perform drag and drop operation.
        
        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            duration: Drag duration in seconds
            button: Mouse button to use for dragging
        """
        try:
            start_x, start_y = self._validate_coordinates(start_x, start_y)
            end_x, end_y = self._validate_coordinates(end_x, end_y)
            
            # Move to start position
            self.move_to(start_x, start_y)
            time.sleep(0.1)
            
            # Perform drag
            pyautogui.dragTo(end_x, end_y, duration * self.movement_speed, button=button)
            
            action = MouseAction("drag", end_x, end_y, duration, button=button, success=True)
            self._add_to_history(action)
            
            logger.debug(f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drag from ({start_x}, {start_y}) to ({end_x}, {end_y}): {e}")
            action = MouseAction("drag", end_x, end_y, duration, button=button, success=False)
            self._add_to_history(action)
            return False
    
    def scroll(self, x: int, y: int, direction: DragDirection, amount: int = 3) -> bool:
        """
        Scroll at specific coordinates.
        
        Args:
            x, y: Scroll position
            direction: Scroll direction (UP, DOWN, LEFT, RIGHT)
            amount: Scroll amount (positive integer)
        """
        try:
            x, y = self._validate_coordinates(x, y)
            self.move_to(x, y)
            
            if direction == DragDirection.UP:
                pyautogui.scroll(amount, x, y)
            elif direction == DragDirection.DOWN:
                pyautogui.scroll(-amount, x, y)
            elif direction == DragDirection.LEFT:
                pyautogui.hscroll(-amount, x, y)
            elif direction == DragDirection.RIGHT:
                pyautogui.hscroll(amount, x, y)
            
            action = MouseAction("scroll", x, y, success=True)
            self._add_to_history(action)
            
            logger.debug(f"Scrolled {direction.value} at ({x}, {y})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scroll at ({x}, {y}): {e}")
            action = MouseAction("scroll", x, y, success=False)
            self._add_to_history(action)
            return False
    
    def hover(self, x: int, y: int, duration: float = 1.0) -> bool:
        """
        Hover over coordinates for specified duration.
        
        Args:
            x, y: Hover coordinates
            duration: Hover duration in seconds
        """
        try:
            x, y = self._validate_coordinates(x, y)
            self.move_to(x, y)
            time.sleep(duration)
            
            action = MouseAction("hover", x, y, duration, success=True)
            self._add_to_history(action)
            
            logger.debug(f"Hovered at ({x}, {y}) for {duration}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to hover at ({x}, {y}): {e}")
            action = MouseAction("hover", x, y, duration, success=False)
            self._add_to_history(action)
            return False
    
    def wait_for_element_to_appear(self, template_image: np.ndarray, 
                                   timeout: float = 10.0, 
                                   confidence: float = 0.8) -> Optional[Tuple[int, int]]:
        """
        Wait for a visual element to appear on screen.
        
        Args:
            template_image: Template image to look for
            timeout: Maximum wait time in seconds
            confidence: Match confidence threshold (0.0-1.0)
            
        Returns:
            Center coordinates of found element or None if not found
        """
        import cv2
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Capture current screen
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
            
            # Convert template to grayscale if needed
            if len(template_image.shape) == 3:
                template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template_image
            
            # Template matching
            result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= confidence:
                # Calculate center of found template
                h, w = template_gray.shape
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                
                logger.debug(f"Element found at ({center_x}, {center_y}) with confidence {max_val}")
                return center_x, center_y
            
            time.sleep(0.5)
        
        logger.warning(f"Element not found within {timeout}s timeout")
        return None
    
    def click_image(self, template_image: np.ndarray, 
                    confidence: float = 0.8, 
                    timeout: float = 5.0,
                    click_type: ClickType = ClickType.LEFT) -> bool:
        """
        Find and click on a visual element.
        
        Args:
            template_image: Template image to find and click
            confidence: Match confidence threshold
            timeout: Maximum search time
            click_type: Type of click to perform
        """
        position = self.wait_for_element_to_appear(template_image, timeout, confidence)
        
        if position:
            return self.click(position[0], position[1], click_type)
        else:
            logger.error("Could not find element to click")
            return False
    
    def get_action_history(self, limit: int = 10) -> List[MouseAction]:
        """Get recent mouse action history."""
        return self.action_history[-limit:]
    
    def clear_history(self):
        """Clear action history."""
        self.action_history.clear()
        logger.debug("Mouse action history cleared")
    
    def set_speed(self, speed: float):
        """Set movement speed multiplier (0.1-2.0)."""
        self.movement_speed = max(0.1, min(2.0, speed))
        logger.debug(f"Mouse speed set to {self.movement_speed}")
    
    def emergency_stop(self):
        """Emergency stop - move mouse to corner to trigger failsafe."""
        pyautogui.moveTo(0, 0)
        logger.warning("Emergency stop triggered - mouse moved to corner")