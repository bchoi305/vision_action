import pyautogui
import time
from typing import List, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import random

class KeyAction(Enum):
    PRESS = "press"
    HOLD = "hold"
    RELEASE = "release"
    TYPE = "type"
    HOTKEY = "hotkey"

@dataclass
class KeyboardAction:
    action_type: KeyAction
    keys: Union[str, List[str]]
    duration: float = 0.0
    success: bool = False
    timestamp: float = 0.0

class KeyboardController:
    def __init__(self, 
                 default_pause: float = 0.1,
                 typing_speed: float = 0.05):
        """
        Initialize keyboard controller.
        
        Args:
            default_pause: Default pause between key actions
            typing_speed: Delay between individual characters when typing
        """
        self.default_pause = default_pause
        self.typing_speed = typing_speed
        
        # Action history for debugging
        self.action_history: List[KeyboardAction] = []
        
        # Common key mappings
        self.special_keys = {
            'enter': 'enter',
            'return': 'enter',
            'tab': 'tab',
            'space': 'space',
            'backspace': 'backspace',
            'delete': 'delete',
            'escape': 'esc',
            'esc': 'esc',
            'shift': 'shift',
            'ctrl': 'ctrl',
            'alt': 'alt',
            'win': 'win',
            'cmd': 'cmd',
            'up': 'up',
            'down': 'down',
            'left': 'left',
            'right': 'right',
            'home': 'home',
            'end': 'end',
            'pageup': 'pageup',
            'pagedown': 'pagedown',
            'f1': 'f1', 'f2': 'f2', 'f3': 'f3', 'f4': 'f4',
            'f5': 'f5', 'f6': 'f6', 'f7': 'f7', 'f8': 'f8',
            'f9': 'f9', 'f10': 'f10', 'f11': 'f11', 'f12': 'f12'
        }
        
        logger.info("Keyboard controller initialized")
    
    def _add_to_history(self, action: KeyboardAction):
        """Add action to history for debugging."""
        action.timestamp = time.time()
        self.action_history.append(action)
        
        # Keep only last 100 actions
        if len(self.action_history) > 100:
            self.action_history.pop(0)
    
    def _normalize_key(self, key: str) -> str:
        """Normalize key name to PyAutoGUI format."""
        key_lower = key.lower().strip()
        return self.special_keys.get(key_lower, key)
    
    def _human_like_typing_delay(self) -> float:
        """Generate human-like typing delay with slight randomness."""
        base_delay = self.typing_speed
        variation = random.uniform(-0.02, 0.02)
        return max(0.01, base_delay + variation)
    
    def press_key(self, key: str, presses: int = 1, interval: float = 0.1) -> bool:
        """
        Press a key one or more times.
        
        Args:
            key: Key to press (e.g., 'enter', 'space', 'a')
            presses: Number of times to press the key
            interval: Interval between presses
        """
        try:
            normalized_key = self._normalize_key(key)
            
            for _ in range(presses):
                pyautogui.press(normalized_key)
                if presses > 1:
                    time.sleep(interval)
            
            action = KeyboardAction(KeyAction.PRESS, normalized_key, success=True)
            self._add_to_history(action)
            
            logger.debug(f"Pressed key '{normalized_key}' {presses} times")
            return True
            
        except Exception as e:
            logger.error(f"Failed to press key '{key}': {e}")
            action = KeyboardAction(KeyAction.PRESS, key, success=False)
            self._add_to_history(action)
            return False
    
    def hold_key(self, key: str, duration: float = 1.0) -> bool:
        """
        Hold a key for specified duration.
        
        Args:
            key: Key to hold
            duration: How long to hold the key in seconds
        """
        try:
            normalized_key = self._normalize_key(key)
            
            pyautogui.keyDown(normalized_key)
            time.sleep(duration)
            pyautogui.keyUp(normalized_key)
            
            action = KeyboardAction(KeyAction.HOLD, normalized_key, duration, success=True)
            self._add_to_history(action)
            
            logger.debug(f"Held key '{normalized_key}' for {duration}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to hold key '{key}': {e}")
            action = KeyboardAction(KeyAction.HOLD, key, duration, success=False)
            self._add_to_history(action)
            return False
    
    def type_text(self, text: str, human_like: bool = True) -> bool:
        """
        Type text with optional human-like timing.
        
        Args:
            text: Text to type
            human_like: Use human-like typing delays
        """
        try:
            if human_like:
                for char in text:
                    pyautogui.write(char)
                    time.sleep(self._human_like_typing_delay())
            else:
                pyautogui.write(text, interval=self.typing_speed)
            
            action = KeyboardAction(KeyAction.TYPE, text, success=True)
            self._add_to_history(action)
            
            logger.debug(f"Typed text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            action = KeyboardAction(KeyAction.TYPE, text, success=False)
            self._add_to_history(action)
            return False
    
    def hotkey(self, *keys) -> bool:
        """
        Press a combination of keys simultaneously.
        
        Args:
            *keys: Keys to press together (e.g., 'ctrl', 'c')
        """
        try:
            normalized_keys = [self._normalize_key(key) for key in keys]
            pyautogui.hotkey(*normalized_keys)
            
            action = KeyboardAction(KeyAction.HOTKEY, list(normalized_keys), success=True)
            self._add_to_history(action)
            
            logger.debug(f"Pressed hotkey: {' + '.join(normalized_keys)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to press hotkey {keys}: {e}")
            action = KeyboardAction(KeyAction.HOTKEY, list(keys), success=False)
            self._add_to_history(action)
            return False
    
    def copy_text(self) -> bool:
        """Copy selected text to clipboard (Ctrl+C)."""
        return self.hotkey('ctrl', 'c')
    
    def paste_text(self) -> bool:
        """Paste text from clipboard (Ctrl+V)."""
        return self.hotkey('ctrl', 'v')
    
    def cut_text(self) -> bool:
        """Cut selected text to clipboard (Ctrl+X)."""
        return self.hotkey('ctrl', 'x')
    
    def select_all(self) -> bool:
        """Select all text (Ctrl+A)."""
        return self.hotkey('ctrl', 'a')
    
    def undo(self) -> bool:
        """Undo last action (Ctrl+Z)."""
        return self.hotkey('ctrl', 'z')
    
    def redo(self) -> bool:
        """Redo last undone action (Ctrl+Y)."""
        return self.hotkey('ctrl', 'y')
    
    def save(self) -> bool:
        """Save document (Ctrl+S)."""
        return self.hotkey('ctrl', 's')
    
    def open_file(self) -> bool:
        """Open file dialog (Ctrl+O)."""
        return self.hotkey('ctrl', 'o')
    
    def new_document(self) -> bool:
        """Create new document (Ctrl+N)."""
        return self.hotkey('ctrl', 'n')
    
    def find(self) -> bool:
        """Open find dialog (Ctrl+F)."""
        return self.hotkey('ctrl', 'f')
    
    def print_document(self) -> bool:
        """Print document (Ctrl+P)."""
        return self.hotkey('ctrl', 'p')
    
    def close_window(self) -> bool:
        """Close current window (Alt+F4)."""
        return self.hotkey('alt', 'f4')
    
    def switch_application(self) -> bool:
        """Switch between applications (Alt+Tab)."""
        return self.hotkey('alt', 'tab')
    
    def minimize_window(self) -> bool:
        """Minimize current window (Win+Down)."""
        return self.hotkey('win', 'down')
    
    def maximize_window(self) -> bool:
        """Maximize current window (Win+Up)."""
        return self.hotkey('win', 'up')
    
    def navigate_menu(self, menu_path: List[str]) -> bool:
        """
        Navigate through menu items using keyboard.
        
        Args:
            menu_path: List of menu items to navigate (e.g., ['File', 'Open'])
        """
        try:
            # Press Alt to activate menu bar
            self.press_key('alt')
            time.sleep(0.2)
            
            for menu_item in menu_path:
                # Type the first letter of the menu item
                if menu_item:
                    first_letter = menu_item[0].lower()
                    self.press_key(first_letter)
                    time.sleep(0.3)
            
            # Press Enter to activate the final menu item
            self.press_key('enter')
            
            logger.debug(f"Navigated menu path: {' -> '.join(menu_path)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to navigate menu path {menu_path}: {e}")
            return False
    
    def fill_form_field(self, text: str, clear_first: bool = True) -> bool:
        """
        Fill a form field with text.
        
        Args:
            text: Text to enter
            clear_first: Whether to clear the field first
        """
        try:
            if clear_first:
                self.select_all()
                time.sleep(0.1)
            
            self.type_text(text)
            
            logger.debug(f"Filled form field with: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fill form field: {e}")
            return False
    
    def navigate_with_arrows(self, direction: str, steps: int = 1) -> bool:
        """
        Navigate using arrow keys.
        
        Args:
            direction: Direction to navigate ('up', 'down', 'left', 'right')
            steps: Number of steps to take
        """
        try:
            direction_lower = direction.lower()
            if direction_lower not in ['up', 'down', 'left', 'right']:
                raise ValueError(f"Invalid direction: {direction}")
            
            for _ in range(steps):
                self.press_key(direction_lower)
                time.sleep(0.1)
            
            logger.debug(f"Navigated {direction} {steps} steps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to navigate with arrows: {e}")
            return False
    
    def type_with_autocomplete(self, text: str, wait_for_suggestions: float = 1.0) -> bool:
        """
        Type text and wait for autocomplete suggestions.
        
        Args:
            text: Text to type
            wait_for_suggestions: Time to wait for suggestions to appear
        """
        try:
            self.type_text(text)
            time.sleep(wait_for_suggestions)
            
            # Press down arrow to select first suggestion, then enter
            self.press_key('down')
            time.sleep(0.2)
            self.press_key('enter')
            
            logger.debug(f"Typed with autocomplete: '{text}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to type with autocomplete: {e}")
            return False
    
    def handle_dialog(self, action: str = "ok") -> bool:
        """
        Handle common dialog boxes.
        
        Args:
            action: Action to take ('ok', 'cancel', 'yes', 'no')
        """
        try:
            action_lower = action.lower()
            
            if action_lower == "ok":
                self.press_key('enter')
            elif action_lower == "cancel":
                self.press_key('esc')
            elif action_lower == "yes":
                self.press_key('y')
            elif action_lower == "no":
                self.press_key('n')
            else:
                raise ValueError(f"Unknown dialog action: {action}")
            
            logger.debug(f"Handled dialog with action: {action}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle dialog: {e}")
            return False
    
    def get_action_history(self, limit: int = 10) -> List[KeyboardAction]:
        """Get recent keyboard action history."""
        return self.action_history[-limit:]
    
    def clear_history(self):
        """Clear action history."""
        self.action_history.clear()
        logger.debug("Keyboard action history cleared")
    
    def set_typing_speed(self, speed: float):
        """Set typing speed (delay between characters)."""
        self.typing_speed = max(0.01, speed)
        logger.debug(f"Typing speed set to {self.typing_speed}s per character")