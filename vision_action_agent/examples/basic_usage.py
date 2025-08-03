#!/usr/bin/env python3
"""
Basic usage example for Vision Action Agent.
Demonstrates core functionality for screen interaction.
"""

import sys
import os
import time

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_action_agent import VisionActionAgent
from vision.element_detector import ElementType
from actions.workflow_executor import WorkflowStep, ActionType

def basic_screen_interaction():
    """Demonstrate basic screen interaction capabilities."""
    print("=== Basic Screen Interaction Demo ===")
    
    # Initialize the agent
    agent = VisionActionAgent()
    
    try:
        # Capture and save a screenshot
        print("1. Capturing screenshot...")
        if agent.save_screenshot("demo_screenshot.png"):
            print("   ✓ Screenshot saved successfully")
        else:
            print("   ✗ Failed to save screenshot")
        
        # Find elements on screen
        print("\n2. Finding UI elements...")
        elements = agent.find_element("Start", ElementType.BUTTON)
        print(f"   Found {len(elements)} button elements containing 'Start'")
        
        # Demonstrate text detection
        print("\n3. Checking for text on screen...")
        if agent.verify_text_present("Desktop"):
            print("   ✓ Found 'Desktop' text on screen")
        else:
            print("   ✗ 'Desktop' text not found")
        
        # Get current agent status
        print("\n4. Agent status:")
        status = agent.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"Error during demo: {e}")
    finally:
        agent.cleanup()

def workflow_demo():
    """Demonstrate workflow execution."""
    print("\n=== Workflow Execution Demo ===")
    
    agent = VisionActionAgent()
    
    try:
        # Create a simple workflow
        workflow_steps = [
            WorkflowStep(
                id="screenshot_1",
                action_type=ActionType.SCREENSHOT,
                parameters={"filepath": "workflow_step1.png"},
                description="Take initial screenshot"
            ),
            WorkflowStep(
                id="wait_1",
                action_type=ActionType.WAIT,
                parameters={"duration": 1.0},
                description="Wait 1 second"
            ),
            WorkflowStep(
                id="find_element_1",
                action_type=ActionType.FIND_ELEMENT,
                parameters={"text": "Start", "store_in_variable": "start_button"},
                description="Find Start button"
            ),
            WorkflowStep(
                id="screenshot_2",
                action_type=ActionType.SCREENSHOT,
                parameters={"filepath": "workflow_step2.png"},
                description="Take final screenshot"
            )
        ]
        
        print("Executing workflow with 4 steps...")
        result = agent.execute_workflow(workflow_steps, "demo_workflow")
        
        print(f"\nWorkflow Results:")
        print(f"   Status: {result.status.value}")
        print(f"   Total steps: {result.total_steps}")
        print(f"   Successful: {result.successful_steps}")
        print(f"   Failed: {result.failed_steps}")
        print(f"   Execution time: {result.execution_time:.2f}s")
        
        # Print step details
        print("\n   Step details:")
        for step_result in result.step_results:
            status_icon = "✓" if step_result['status'] == 'success' else "✗"
            print(f"   {status_icon} {step_result['step_id']}: {step_result['status']} ({step_result['execution_time']:.2f}s)")
            
    except Exception as e:
        print(f"Error during workflow demo: {e}")
    finally:
        agent.cleanup()

def pacs_workflow_demo():
    """Demonstrate PACS-specific workflow (simulation)."""
    print("\n=== PACS Workflow Demo (Simulation) ===")
    
    agent = VisionActionAgent()
    
    try:
        # Create a PACS workflow template
        patient_id = "12345"
        pacs_workflow = agent.create_pacs_workflow(patient_id)
        
        print(f"Created PACS workflow for patient {patient_id}")
        print(f"Workflow contains {len(pacs_workflow)} steps:")
        
        for i, step in enumerate(pacs_workflow, 1):
            print(f"   {i}. {step.description}")
        
        # Note: This would only work with actual PACS software running
        print("\n   Note: This workflow would execute with real PACS software")
        print("   For demonstration, we're just showing the workflow structure")
        
    except Exception as e:
        print(f"Error during PACS demo: {e}")
    finally:
        agent.cleanup()

def configuration_demo():
    """Demonstrate configuration management."""
    print("\n=== Configuration Management Demo ===")
    
    try:
        agent = VisionActionAgent()
        
        # Show current configuration
        print("Current configuration:")
        config = agent.config_manager.get_config()
        
        print(f"   OCR Engine: {config.ocr.engine}")
        print(f"   OCR Languages: {config.ocr.languages}")
        print(f"   Mouse Speed: {config.mouse.movement_speed}")
        print(f"   Typing Speed: {config.keyboard.typing_speed}")
        print(f"   PACS Application: {config.pacs.application_name}")
        
        # Update configuration
        print("\nUpdating mouse speed...")
        agent.config_manager.update_config('mouse', {'movement_speed': 0.8})
        
        updated_config = agent.config_manager.get_config()
        print(f"   Updated mouse speed: {updated_config.mouse.movement_speed}")
        
        # Create default config file
        print("\nCreating default configuration file...")
        if agent.config_manager.create_default_config_file("example_config.yaml"):
            print("   ✓ Configuration file created: example_config.yaml")
        
        agent.cleanup()
        
    except Exception as e:
        print(f"Error during configuration demo: {e}")

def main():
    """Run all demonstrations."""
    print("Vision Action Agent - Usage Examples")
    print("=" * 40)
    
    try:
        # Run basic demo
        basic_screen_interaction()
        
        # Run workflow demo
        workflow_demo()
        
        # Run PACS demo
        pacs_workflow_demo()
        
        # Run configuration demo
        configuration_demo()
        
        print("\n" + "=" * 40)
        print("All demonstrations completed!")
        print("\nNext steps:")
        print("1. Install required dependencies: pip install -r requirements.txt")
        print("2. Configure your PACS-specific settings in config.yaml")
        print("3. Test with your actual PACS software")
        print("4. Create custom workflows for your specific use cases")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()