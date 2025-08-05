import traceback
import time
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import functools

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    SCREEN_CAPTURE = "screen_capture"
    OCR = "ocr"
    ELEMENT_DETECTION = "element_detection"
    MOUSE_ACTION = "mouse_action"
    KEYBOARD_ACTION = "keyboard_action"
    WORKFLOW = "workflow"
    
    CONFIGURATION = "configuration"
    SYSTEM = "system"

@dataclass
class ErrorInfo:
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    timestamp: float = 0.0
    context: Optional[Dict[str, Any]] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False

class ErrorHandler:
    def __init__(self, max_errors: int = 100):
        """
        Initialize error handler.
        
        Args:
            max_errors: Maximum number of errors to keep in history
        """
        self.error_history: List[ErrorInfo] = []
        self.max_errors = max_errors
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default error recovery strategies."""
        self.recovery_strategies = {
            ErrorCategory.SCREEN_CAPTURE: [
                self._retry_with_delay,
                self._fallback_screenshot_method
            ],
            ErrorCategory.OCR: [
                self._retry_with_preprocessing,
                self._try_alternative_ocr_engine
            ],
            ErrorCategory.ELEMENT_DETECTION: [
                self._retry_with_delay,
                self._expand_search_region,
                self._try_alternative_detection_method
            ],
            ErrorCategory.MOUSE_ACTION: [
                self._retry_mouse_action,
                self._adjust_coordinates,
                self._check_screen_resolution
            ],
            ErrorCategory.KEYBOARD_ACTION: [
                self._retry_with_delay,
                self._check_keyboard_focus
            ],
            ErrorCategory.WORKFLOW: [
                self._retry_failed_step,
                self._skip_optional_step
            ],
            
        }
    
    def handle_error(self, error: Exception, category: ErrorCategory, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None,
                    attempt_recovery: bool = True) -> bool:
        """
        Handle an error with optional recovery attempts.
        
        Args:
            error: The exception that occurred
            category: Error category
            severity: Error severity level
            context: Additional context information
            attempt_recovery: Whether to attempt error recovery
            
        Returns:
            True if error was recovered from, False otherwise
        """
        error_info = ErrorInfo(
            category=category,
            severity=severity,
            message=str(error),
            exception=error,
            timestamp=time.time(),
            context=context or {}
        )
        
        # Log the error
        self._log_error(error_info)
        
        # Add to history
        self._add_to_history(error_info)
        
        # Attempt recovery if requested and strategies are available
        if attempt_recovery and category in self.recovery_strategies:
            error_info.recovery_attempted = True
            
            for strategy in self.recovery_strategies[category]:
                try:
                    if strategy(error_info):
                        error_info.recovery_successful = True
                        logger.info(f"Error recovery successful using strategy: {strategy.__name__}")
                        return True
                except Exception as recovery_error:
                    logger.warning(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
        
        return False
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error information."""
        log_message = f"[{error_info.category.value.upper()}] {error_info.message}"
        
        if error_info.context:
            log_message += f" | Context: {error_info.context}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log full traceback for debugging
        if error_info.exception:
            logger.debug(f"Full traceback: {traceback.format_exc()}")
    
    def _add_to_history(self, error_info: ErrorInfo):
        """Add error to history."""
        self.error_history.append(error_info)
        
        # Maintain maximum history size
        if len(self.error_history) > self.max_errors:
            self.error_history.pop(0)
    
    # Recovery strategies
    
    def _retry_with_delay(self, error_info: ErrorInfo, delay: float = 1.0, max_retries: int = 3) -> bool:
        """Generic retry with delay strategy."""
        context = error_info.context or {}
        retry_count = context.get('retry_count', 0)
        
        if retry_count < max_retries:
            logger.info(f"Retrying in {delay}s (attempt {retry_count + 1}/{max_retries})")
            time.sleep(delay)
            context['retry_count'] = retry_count + 1
            return True
        
        return False
    
    def _fallback_screenshot_method(self, error_info: ErrorInfo) -> bool:
        """Try alternative screenshot method."""
        try:
            import pyautogui
            # Try basic pyautogui screenshot as fallback
            screenshot = pyautogui.screenshot()
            logger.info("Fallback screenshot method successful")
            return True
        except Exception as e:
            logger.warning(f"Fallback screenshot method failed: {e}")
            return False
    
    def _retry_with_preprocessing(self, error_info: ErrorInfo) -> bool:
        """Retry OCR with additional preprocessing."""
        context = error_info.context or {}
        
        if not context.get('preprocessing_attempted'):
            context['preprocessing_attempted'] = True
            context['use_enhanced_preprocessing'] = True
            logger.info("Retrying OCR with enhanced preprocessing")
            return True
        
        return False
    
    def _try_alternative_ocr_engine(self, error_info: ErrorInfo) -> bool:
        """Try alternative OCR engine."""
        context = error_info.context or {}
        current_engine = context.get('ocr_engine', 'easyocr')
        
        if current_engine == 'easyocr' and not context.get('tesseract_attempted'):
            context['ocr_engine'] = 'tesseract'
            context['tesseract_attempted'] = True
            logger.info("Switching to Tesseract OCR engine")
            return True
        elif current_engine == 'tesseract' and not context.get('easyocr_attempted'):
            context['ocr_engine'] = 'easyocr'
            context['easyocr_attempted'] = True
            logger.info("Switching to EasyOCR engine")
            return True
        
        return False
    
    def _expand_search_region(self, error_info: ErrorInfo) -> bool:
        """Expand search region for element detection."""
        context = error_info.context or {}
        
        if not context.get('region_expanded'):
            # Expand search region by 50%
            region = context.get('search_region')
            if region:
                expanded_region = {
                    'x': max(0, region['x'] - region['width'] * 0.25),
                    'y': max(0, region['y'] - region['height'] * 0.25),
                    'width': region['width'] * 1.5,
                    'height': region['height'] * 1.5
                }
                context['search_region'] = expanded_region
                context['region_expanded'] = True
                logger.info("Expanded search region for element detection")
                return True
        
        return False
    
    def _try_alternative_detection_method(self, error_info: ErrorInfo) -> bool:
        """Try alternative element detection method."""
        context = error_info.context or {}
        
        if not context.get('alternative_method_attempted'):
            context['alternative_method_attempted'] = True
            context['use_template_matching'] = True
            logger.info("Trying template matching for element detection")
            return True
        
        return False
    
    def _retry_mouse_action(self, error_info: ErrorInfo) -> bool:
        """Retry mouse action with slight coordinate adjustment."""
        context = error_info.context or {}
        retry_count = context.get('mouse_retry_count', 0)
        
        if retry_count < 2:
            # Add small random offset to coordinates
            import random
            offset_x = random.randint(-5, 5)
            offset_y = random.randint(-5, 5)
            
            if 'target_x' in context and 'target_y' in context:
                context['target_x'] += offset_x
                context['target_y'] += offset_y
                context['mouse_retry_count'] = retry_count + 1
                
                logger.info(f"Retrying mouse action with offset ({offset_x}, {offset_y})")
                time.sleep(0.5)
                return True
        
        return False
    
    def _adjust_coordinates(self, error_info: ErrorInfo) -> bool:
        """Adjust coordinates for different screen resolution."""
        context = error_info.context or {}
        
        if not context.get('coordinates_adjusted'):
            # Try adjusting for potential DPI scaling
            if 'target_x' in context and 'target_y' in context:
                scale_factor = context.get('dpi_scale', 1.25)  # Common Windows scaling
                context['target_x'] = int(context['target_x'] * scale_factor)
                context['target_y'] = int(context['target_y'] * scale_factor)
                context['coordinates_adjusted'] = True
                
                logger.info(f"Adjusted coordinates for DPI scaling ({scale_factor})")
                return True
        
        return False
    
    def _check_screen_resolution(self, error_info: ErrorInfo) -> bool:
        """Check and log current screen resolution."""
        try:
            import pyautogui
            width, height = pyautogui.size()
            logger.info(f"Current screen resolution: {width}x{height}")
            
            context = error_info.context or {}
            context['screen_width'] = width
            context['screen_height'] = height
            
            return False  # This is informational, doesn't fix the error
        except Exception as e:
            logger.warning(f"Could not get screen resolution: {e}")
            return False
    
    def _check_keyboard_focus(self, error_info: ErrorInfo) -> bool:
        """Check if correct window has keyboard focus."""
        try:
            import pyautogui
            # Try clicking to ensure focus, then retry
            current_pos = pyautogui.position()
            pyautogui.click(current_pos[0], current_pos[1])
            time.sleep(0.2)
            
            logger.info("Attempted to restore keyboard focus")
            return True
        except Exception as e:
            logger.warning(f"Could not restore keyboard focus: {e}")
            return False
    
    def _retry_failed_step(self, error_info: ErrorInfo) -> bool:
        """Retry a failed workflow step."""
        context = error_info.context or {}
        step_retry_count = context.get('step_retry_count', 0)
        max_step_retries = context.get('max_step_retries', 2)
        
        if step_retry_count < max_step_retries:
            context['step_retry_count'] = step_retry_count + 1
            logger.info(f"Retrying workflow step (attempt {step_retry_count + 1})")
            time.sleep(1.0)
            return True
        
        return False
    
    def _skip_optional_step(self, error_info: ErrorInfo) -> bool:
        """Skip optional workflow step."""
        context = error_info.context or {}
        
        if context.get('step_optional', False):
            logger.info("Skipping optional workflow step")
            context['step_skipped'] = True
            return True
        
        return False
    
    
    
    def register_recovery_strategy(self, category: ErrorCategory, strategy: Callable):
        """Register a custom recovery strategy."""
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        
        self.recovery_strategies[category].append(strategy)
        logger.info(f"Registered recovery strategy for {category.value}: {strategy.__name__}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        total_errors = len(self.error_history)
        errors_by_category = {}
        errors_by_severity = {}
        recovery_success_rate = 0
        
        for error in self.error_history:
            # Count by category
            category = error.category.value
            errors_by_category[category] = errors_by_category.get(category, 0) + 1
            
            # Count by severity
            severity = error.severity.value
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1
        
        # Calculate recovery success rate
        recovery_attempts = sum(1 for e in self.error_history if e.recovery_attempted)
        if recovery_attempts > 0:
            successful_recoveries = sum(1 for e in self.error_history if e.recovery_successful)
            recovery_success_rate = successful_recoveries / recovery_attempts
        
        return {
            "total_errors": total_errors,
            "errors_by_category": errors_by_category,
            "errors_by_severity": errors_by_severity,
            "recovery_success_rate": recovery_success_rate,
            "recent_errors": [
                {
                    "category": e.category.value,
                    "severity": e.severity.value,
                    "message": e.message,
                    "timestamp": e.timestamp,
                    "recovered": e.recovery_successful
                }
                for e in self.error_history[-10:]  # Last 10 errors
            ]
        }
    
    def clear_history(self):
        """Clear error history."""
        self.error_history.clear()
        logger.info("Error history cleared")

# Decorator for automatic error handling
def with_error_handling(category: ErrorCategory, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """
    Decorator to automatically handle errors in functions.
    
    Args:
        category: Error category
        severity: Error severity level
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Try to find error handler in instance
                error_handler = None
                if args and hasattr(args[0], 'error_handler'):
                    error_handler = args[0].error_handler
                else:
                    # Use global error handler
                    error_handler = global_error_handler
                
                if error_handler:
                    context = {
                        "function_name": func.__name__,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                    
                    recovered = error_handler.handle_error(e, category, severity, context)
                    if not recovered:
                        raise  # Re-raise if not recovered
                    
                    # If recovered, try calling function again
                    return func(*args, **kwargs)
                else:
                    raise
        
        return wrapper
    return decorator

# Global error handler instance
global_error_handler = ErrorHandler()