import time
import json
import yaml
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger
import numpy as np

from .mouse_controller import MouseController, ClickType
from .keyboard_controller import KeyboardController
from ..vision.screen_capture import ScreenCapture, CaptureRegion
from ..vision.ocr_engine import OCRTextDetector
from ..vision.element_detector import ElementDetector, UIElement, ElementType

class ActionType(Enum):
    CLICK = "click"
    TYPE = "type"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    FIND_ELEMENT = "find_element"
    VERIFY_TEXT = "verify_text"
    DRAG_DROP = "drag_drop"
    SCROLL = "scroll"
    HOTKEY = "hotkey"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    CUSTOM = "custom"

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowStep:
    id: str
    action_type: ActionType
    parameters: Dict[str, Any]
    description: Optional[str] = None
    timeout: float = 30.0
    retry_count: int = 3
    continue_on_failure: bool = False
    status: ExecutionStatus = ExecutionStatus.PENDING
    execution_time: float = 0.0
    error_message: Optional[str] = None

@dataclass
class WorkflowResult:
    workflow_id: str
    total_steps: int
    successful_steps: int
    failed_steps: int
    execution_time: float
    status: ExecutionStatus
    step_results: List[Dict[str, Any]]
    error_message: Optional[str] = None

class WorkflowExecutor:
    def __init__(self):
        """Initialize workflow executor with all necessary controllers."""
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.screen_capture = ScreenCapture()
        self.ocr_detector = OCRTextDetector()
        self.element_detector = ElementDetector(self.ocr_detector)
        
        # Workflow state
        self.current_workflow: Optional[str] = None
        self.workflow_variables: Dict[str, Any] = {}
        self.custom_actions: Dict[str, Callable] = {}
        
        logger.info("Workflow executor initialized")
    
    def register_custom_action(self, name: str, action_func: Callable):
        """Register a custom action function."""
        self.custom_actions[name] = action_func
        logger.info(f"Custom action '{name}' registered")
    
    def set_workflow_variable(self, name: str, value: Any):
        """Set a workflow variable."""
        self.workflow_variables[name] = value
    
    def get_workflow_variable(self, name: str, default: Any = None) -> Any:
        """Get a workflow variable."""
        return self.workflow_variables.get(name, default)
    
    def execute_click_action(self, step: WorkflowStep) -> bool:
        """Execute click action."""
        params = step.parameters
        
        # Get coordinates
        x = params.get('x')
        y = params.get('y')
        element_text = params.get('element_text')
        element_type = params.get('element_type')
        click_type = ClickType(params.get('click_type', 'left'))
        
        if element_text:
            # Find element by text
            screenshot = self.screen_capture.capture_full_screen()
            elements = self.element_detector.find_element_by_text(
                screenshot, element_text, 
                ElementType(element_type) if element_type else None
            )
            
            if elements:
                element = elements[0]  # Use first match
                x, y = element.center
                logger.info(f"Found element '{element_text}' at ({x}, {y})")
            else:
                logger.error(f"Element '{element_text}' not found")
                return False
        
        if x is None or y is None:
            logger.error("No coordinates specified for click action")
            return False
        
        return self.mouse.click(x, y, click_type)
    
    def execute_type_action(self, step: WorkflowStep) -> bool:
        """Execute type action."""
        params = step.parameters
        text = params.get('text', '')
        clear_first = params.get('clear_first', True)
        human_like = params.get('human_like', True)
        
        if clear_first:
            self.keyboard.select_all()
            time.sleep(0.1)
        
        return self.keyboard.type_text(text, human_like)
    
    def execute_wait_action(self, step: WorkflowStep) -> bool:
        """Execute wait action."""
        params = step.parameters
        duration = params.get('duration', 1.0)
        wait_for_element = params.get('wait_for_element')
        
        if wait_for_element:
            # Wait for specific element to appear
            element_text = wait_for_element.get('text')
            timeout = wait_for_element.get('timeout', step.timeout)
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                screenshot = self.screen_capture.capture_full_screen()
                elements = self.element_detector.find_element_by_text(screenshot, element_text)
                
                if elements:
                    logger.info(f"Element '{element_text}' appeared")
                    return True
                
                time.sleep(0.5)
            
            logger.error(f"Element '{element_text}' did not appear within {timeout}s")
            return False
        else:
            time.sleep(duration)
            return True
    
    def execute_screenshot_action(self, step: WorkflowStep) -> bool:
        """Execute screenshot action."""
        params = step.parameters
        filepath = params.get('filepath', f'screenshot_{int(time.time())}.png')
        region = params.get('region')
        
        try:
            if region:
                capture_region = CaptureRegion(**region)
                screenshot = self.screen_capture.capture_region(capture_region)
            else:
                screenshot = self.screen_capture.capture_full_screen()
            
            success = self.screen_capture.save_screenshot(screenshot, filepath)
            if success:
                logger.info(f"Screenshot saved to {filepath}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return False
    
    def execute_find_element_action(self, step: WorkflowStep) -> bool:
        """Execute find element action."""
        params = step.parameters
        element_text = params.get('text')
        element_type = params.get('type')
        store_in_variable = params.get('store_in_variable')
        
        screenshot = self.screen_capture.capture_full_screen()
        elements = self.element_detector.find_element_by_text(
            screenshot, element_text,
            ElementType(element_type) if element_type else None
        )
        
        if elements:
            element = elements[0]
            logger.info(f"Found element '{element_text}' at {element.center}")
            
            if store_in_variable:
                self.set_workflow_variable(store_in_variable, {
                    'center': element.center,
                    'bbox': element.bbox,
                    'text': element.text,
                    'type': element.element_type.value
                })
            
            return True
        else:
            logger.error(f"Element '{element_text}' not found")
            return False
    
    def execute_verify_text_action(self, step: WorkflowStep) -> bool:
        """Execute verify text action."""
        params = step.parameters
        expected_text = params.get('text')
        region = params.get('region')
        case_sensitive = params.get('case_sensitive', False)
        
        try:
            if region:
                capture_region = CaptureRegion(**region)
                screenshot = self.screen_capture.capture_region(capture_region)
            else:
                screenshot = self.screen_capture.capture_full_screen()
            
            ocr_result = self.ocr_detector.extract_text(screenshot)
            found_text = ocr_result.raw_text
            
            if not case_sensitive:
                expected_text = expected_text.lower()
                found_text = found_text.lower()
            
            if expected_text in found_text:
                logger.info(f"Text '{expected_text}' verified successfully")
                return True
            else:
                logger.error(f"Text '{expected_text}' not found. Found: '{found_text[:100]}'")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify text: {e}")
            return False
    
    def execute_hotkey_action(self, step: WorkflowStep) -> bool:
        """Execute hotkey action."""
        params = step.parameters
        keys = params.get('keys', [])
        
        if isinstance(keys, str):
            keys = [keys]
        
        return self.keyboard.hotkey(*keys)
    
    def execute_drag_drop_action(self, step: WorkflowStep) -> bool:
        """Execute drag and drop action."""
        params = step.parameters
        start_x = params.get('start_x')
        start_y = params.get('start_y')
        end_x = params.get('end_x')
        end_y = params.get('end_y')
        duration = params.get('duration', 1.0)
        
        if None in [start_x, start_y, end_x, end_y]:
            logger.error("Missing coordinates for drag and drop action")
            return False
        
        return self.mouse.drag_and_drop(start_x, start_y, end_x, end_y, duration)
    
    def execute_scroll_action(self, step: WorkflowStep) -> bool:
        """Execute scroll action."""
        params = step.parameters
        x = params.get('x', 500)
        y = params.get('y', 500)
        direction = params.get('direction', 'down')
        amount = params.get('amount', 3)
        
        from .mouse_controller import DragDirection
        scroll_direction = DragDirection(direction.upper())
        
        return self.mouse.scroll(x, y, scroll_direction, amount)
    
    def execute_loop_action(self, step: WorkflowStep) -> bool:
        """
        Execute a loop action.
        
        Args:
            step: The WorkflowStep object containing loop parameters.
            
        Returns:
            True if the loop completed successfully, False otherwise.
        """
        params = step.parameters
        loop_type = params.get('loop_type')
        loop_params = params.get('loop_params', {})
        loop_steps_data = params.get('steps', [])

        if not loop_steps_data:
            logger.warning(f"Loop step '{step.id}' has no sub-steps defined.")
            return True

        loop_steps = [WorkflowStep(**sd) for sd in loop_steps_data]

        if loop_type == 'for_range':
            start = loop_params.get('start', 0)
            end = loop_params.get('end', 0)
            step_val = loop_params.get('step', 1)
            variable_name = loop_params.get('variable_name', 'i')

            for i in range(start, end, step_val):
                logger.info(f"Executing loop '{step.id}': iteration {i}")
                self.set_workflow_variable(variable_name, i)
                for sub_step in loop_steps:
                    if not self.execute_step(sub_step):
                        logger.error(f"Loop '{step.id}' failed at iteration {i}, sub-step '{sub_step.id}'.")
                        return False
            return True

        elif loop_type == 'while_condition':
            condition_type = loop_params.get('condition_type')
            condition_params = loop_params.get('condition_params', {})
            max_iterations = loop_params.get('max_iterations', 100)
            iteration_count = 0

            while iteration_count < max_iterations:
                iteration_count += 1
                logger.info(f"Executing loop '{step.id}': iteration {iteration_count}")
                
                # Evaluate condition
                condition_result = False
                if condition_type == 'element_exists':
                    element_text = condition_params.get('text')
                    screenshot = self.screen_capture.capture_full_screen()
                    elements = self.element_detector.find_element_by_text(screenshot, element_text)
                    condition_result = len(elements) > 0
                elif condition_type == 'text_exists':
                    text = condition_params.get('text')
                    region = condition_params.get('region')
                    if region:
                        capture_region = CaptureRegion(**region)
                        screenshot = self.screen_capture.capture_region(capture_region)
                    else:
                        screenshot = self.screen_capture.capture_full_screen()
                    ocr_result = self.ocr_detector.extract_text(screenshot)
                    if text is not None:
                        condition_result = text.lower() in ocr_result.raw_text.lower()
                    else:
                        condition_result = False
                elif condition_type == 'variable_equals':
                    var_name = condition_params.get('variable')
                    expected_value = condition_params.get('value')
                    actual_value = self.get_workflow_variable(var_name)
                    condition_result = actual_value == expected_value
                
                if not condition_result:
                    logger.info(f"Loop '{step.id}' condition met, exiting loop.")
                    break

                for sub_step in loop_steps:
                    if not self.execute_step(sub_step):
                        logger.error(f"Loop '{step.id}' failed at iteration {iteration_count}, sub-step '{sub_step.id}'.")
                        return False
            else:
                logger.warning(f"Loop '{step.id}' reached max iterations ({max_iterations}) without condition met.")
                return False # Loop failed because max iterations reached
            return True

        else:
            logger.error(f"Unknown loop type: {loop_type}")
            return False

    def execute_conditional_action(self, step: WorkflowStep) -> bool:
        """Execute conditional action."""
        params = step.parameters
        condition_type = params.get('condition_type')
        condition_params = params.get('condition_params', {})
        true_steps = params.get('true_steps', [])
        false_steps = params.get('false_steps', [])
        
        # Evaluate condition
        condition_result = False
        
        if condition_type == 'element_exists':
            element_text = condition_params.get('text')
            screenshot = self.screen_capture.capture_full_screen()
            elements = self.element_detector.find_element_by_text(screenshot, element_text)
            condition_result = len(elements) > 0
        
        elif condition_type == 'text_exists':
            text = condition_params.get('text')
            region = condition_params.get('region')
            
            if region:
                capture_region = CaptureRegion(**region)
                screenshot = self.screen_capture.capture_region(capture_region)
            else:
                screenshot = self.screen_capture.capture_full_screen()
            
            ocr_result = self.ocr_detector.extract_text(screenshot)
            condition_result = text.lower() in ocr_result.raw_text.lower()
        
        elif condition_type == 'variable_equals':
            var_name = condition_params.get('variable')
            expected_value = condition_params.get('value')
            actual_value = self.get_workflow_variable(var_name)
            condition_result = actual_value == expected_value
        
        # Execute appropriate steps
        steps_to_execute = true_steps if condition_result else false_steps
        
        for step_data in steps_to_execute:
            sub_step = WorkflowStep(**step_data)
            if not self.execute_step(sub_step):
                return False
        
        return True
    
    def execute_custom_action(self, step: WorkflowStep) -> bool:
        """Execute custom action."""
        params = step.parameters
        action_name = params.get('action_name')
        action_params = params.get('action_params', {})
        
        if action_name not in self.custom_actions:
            logger.error(f"Custom action '{action_name}' not registered")
            return False
        
        try:
            return self.custom_actions[action_name](action_params, self)
        except Exception as e:
            logger.error(f"Custom action '{action_name}' failed: {e}")
            return False
    
    def execute_step(self, step: WorkflowStep) -> bool:
        """Execute a single workflow step."""
        logger.info(f"Executing step {step.id}: {step.action_type.value}")
        step.status = ExecutionStatus.RUNNING
        start_time = time.time()
        
        try:
            # Execute based on action type
            if step.action_type == ActionType.CLICK:
                success = self.execute_click_action(step)
            elif step.action_type == ActionType.TYPE:
                success = self.execute_type_action(step)
            elif step.action_type == ActionType.WAIT:
                success = self.execute_wait_action(step)
            elif step.action_type == ActionType.SCREENSHOT:
                success = self.execute_screenshot_action(step)
            elif step.action_type == ActionType.FIND_ELEMENT:
                success = self.execute_find_element_action(step)
            elif step.action_type == ActionType.VERIFY_TEXT:
                success = self.execute_verify_text_action(step)
            elif step.action_type == ActionType.HOTKEY:
                success = self.execute_hotkey_action(step)
            elif step.action_type == ActionType.DRAG_DROP:
                success = self.execute_drag_drop_action(step)
            elif step.action_type == ActionType.SCROLL:
                success = self.execute_scroll_action(step)
            elif step.action_type == ActionType.CONDITIONAL:
                success = self.execute_conditional_action(step)
            elif step.action_type == ActionType.LOOP:
                success = self.execute_loop_action(step)
            elif step.action_type == ActionType.CUSTOM:
                success = self.execute_custom_action(step)
            else:
                logger.error(f"Unknown action type: {step.action_type}")
                success = False
            
            step.execution_time = time.time() - start_time
            
            if success:
                step.status = ExecutionStatus.SUCCESS
                logger.info(f"Step {step.id} completed successfully in {step.execution_time:.2f}s")
            else:
                step.status = ExecutionStatus.FAILED
                logger.error(f"Step {step.id} failed after {step.execution_time:.2f}s")
            
            return success
            
        except Exception as e:
            step.execution_time = time.time() - start_time
            step.status = ExecutionStatus.FAILED
            step.error_message = str(e)
            logger.error(f"Step {step.id} failed with exception: {e}")
            return False
    
    def execute_workflow(self, steps: List[WorkflowStep], workflow_id: Optional[str] = None) -> WorkflowResult:
        """Execute a complete workflow."""
        workflow_id = workflow_id or f"workflow_{int(time.time())}"
        self.current_workflow = workflow_id
        
        logger.info(f"Starting workflow execution: {workflow_id}")
        start_time = time.time()
        
        successful_steps = 0
        failed_steps = 0
        step_results = []
        
        for step in steps:
            step_result = {
                'step_id': step.id,
                'action_type': step.action_type.value,
                'status': step.status.value,
                'execution_time': 0.0,
                'error_message': step.error_message
            }
            
            # Execute step with retry logic
            success = False
            for attempt in range(step.retry_count + 1):
                if attempt > 0:
                    logger.info(f"Retrying step {step.id} (attempt {attempt + 1})")
                    time.sleep(1.0)  # Brief pause before retry
                
                success = self.execute_step(step)
                if success:
                    break
            
            step_result.update({
                'status': step.status.value,
                'execution_time': step.execution_time,
                'error_message': step.error_message
            })
            step_results.append(step_result)
            
            if success:
                successful_steps += 1
            else:
                failed_steps += 1
                if not step.continue_on_failure:
                    logger.error(f"Workflow stopped due to step {step.id} failure")
                    break
        
        total_execution_time = time.time() - start_time
        
        # Determine overall workflow status
        if failed_steps == 0:
            workflow_status = ExecutionStatus.SUCCESS
        elif successful_steps > 0:
            workflow_status = ExecutionStatus.FAILED  # Partial success is still failure
        else:
            workflow_status = ExecutionStatus.FAILED
        
        result = WorkflowResult(
            workflow_id=workflow_id,
            total_steps=len(steps),
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            execution_time=total_execution_time,
            status=workflow_status,
            step_results=step_results
        )
        
        logger.info(f"Workflow {workflow_id} completed: {successful_steps}/{len(steps)} steps successful")
        return result
    
    def load_workflow_from_yaml(self, filepath: str) -> List[WorkflowStep]:
        """Load workflow from YAML file.

        Logs and skips steps with unknown action_type values to make issues apparent.
        """
        try:
            with open(filepath, 'r') as file:
                workflow_data = yaml.safe_load(file)

            steps: List[WorkflowStep] = []
            invalid_count = 0

            # Accept a few common synonyms
            synonym_map = {
                'type_text': 'type',
            }

            for step_data in workflow_data.get('steps', []):
                try:
                    raw_type = step_data.get('action_type')
                    mapped_type = synonym_map.get(str(raw_type).lower(), raw_type)
                    action_type = ActionType(mapped_type)
                except Exception:
                    invalid_count += 1
                    logger.error(
                        f"Unknown action_type in step '{step_data.get('id','unknown')}': {step_data.get('action_type')}"
                    )
                    continue

                step = WorkflowStep(
                    id=step_data['id'],
                    action_type=action_type,
                    parameters=step_data.get('parameters', {}),
                    description=step_data.get('description'),
                    timeout=step_data.get('timeout', 30.0),
                    retry_count=step_data.get('retry_count', 3),
                    continue_on_failure=step_data.get('continue_on_failure', False)
                )
                steps.append(step)

            logger.info(f"Loaded {len(steps)} steps from {filepath}")
            if invalid_count:
                logger.error(f"Skipped {invalid_count} invalid step(s) due to unknown action_type.")
            return steps

        except Exception as e:
            logger.error(f"Failed to load workflow from {filepath}: {e}")
            return []
    
    def save_workflow_to_yaml(self, steps: List[WorkflowStep], filepath: str):
        """Save workflow to YAML file."""
        try:
            workflow_data = {
                'steps': [asdict(step) for step in steps]
            }
            
            with open(filepath, 'w') as file:
                yaml.dump(workflow_data, file, default_flow_style=False)
            
            logger.info(f"Workflow saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save workflow to {filepath}: {e}")
    
    
