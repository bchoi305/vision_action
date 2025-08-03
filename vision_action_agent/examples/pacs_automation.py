#!/usr/bin/env python3
"""
PACS automation example for medical imaging workflows.
This example demonstrates how to automate chest X-ray reading workflows.
"""

import sys
import os
import time
from typing import List, Dict, Any

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_action_agent import VisionActionAgent
from actions.workflow_executor import WorkflowStep, ActionType
from config.settings import ConfigManager

class PACSAutomation:
    def __init__(self, config_path: str = None):
        """Initialize PACS automation system."""
        self.agent = VisionActionAgent(config_path)
        self.processed_patients: List[str] = []
        self.workflow_results: List[Dict[str, Any]] = []
        
    def setup_pacs_configuration(self):
        """Setup PACS-specific configuration."""
        print("Setting up PACS configuration...")
        
        # Update PACS settings
        pacs_config = {
            'application_name': 'PACS Viewer',
            'window_title_contains': 'PACS',
            'default_patient_search_timeout': 15.0,
            'image_load_timeout': 45.0,
            'report_generation_timeout': 120.0
        }
        
        self.agent.config_manager.update_config('pacs', pacs_config)
        
        # Add custom UI elements for your specific PACS
        self.agent.config_manager.add_pacs_element_config(
            'patient_id_field',
            {
                'type': 'text_field',
                'text_contains': ['Patient ID', 'MRN', 'Medical Record'],
                'region': None  # Auto-detect
            }
        )
        
        self.agent.config_manager.add_pacs_element_config(
            'chest_xray_study',
            {
                'type': 'button',
                'text_contains': ['Chest', 'CXR', 'X-Ray', 'Radiograph'],
                'region': None
            }
        )
        
        print("✓ PACS configuration updated")
    
    def create_chest_xray_workflow(self, patient_id: str) -> List[WorkflowStep]:
        """Create a complete chest X-ray reading workflow."""
        
        workflow_steps = [
            # Step 1: Ensure PACS is open
            WorkflowStep(
                id="ensure_pacs_open",
                action_type=ActionType.CUSTOM,
                parameters={
                    "action_name": "open_pacs",
                    "action_params": {}
                },
                description="Ensure PACS application is open",
                timeout=30.0
            ),
            
            # Step 2: Search for patient
            WorkflowStep(
                id="search_patient",
                action_type=ActionType.CUSTOM,
                parameters={
                    "action_name": "search_patient",
                    "action_params": {"patient_id": patient_id}
                },
                description=f"Search for patient {patient_id}",
                timeout=15.0,
                retry_count=2
            ),
            
            # Step 3: Select chest X-ray study
            WorkflowStep(
                id="select_chest_study",
                action_type=ActionType.CLICK,
                parameters={
                    "element_text": "Chest",
                    "element_type": "button"
                },
                description="Select chest X-ray study",
                timeout=10.0
            ),
            
            # Step 4: Open study in viewer
            WorkflowStep(
                id="open_study",
                action_type=ActionType.CUSTOM,
                parameters={
                    "action_name": "open_study",
                    "action_params": {}
                },
                description="Open study in viewer",
                timeout=45.0
            ),
            
            # Step 5: Wait for images to load
            WorkflowStep(
                id="wait_for_images",
                action_type=ActionType.WAIT,
                parameters={
                    "wait_for_element": {
                        "text": "Image",
                        "timeout": 30.0
                    }
                },
                description="Wait for images to load completely"
            ),
            
            # Step 6: Capture frontal view
            WorkflowStep(
                id="capture_frontal_view",
                action_type=ActionType.SCREENSHOT,
                parameters={
                    "filepath": f"patient_{patient_id}_frontal.png",
                    "region": None  # Full screen for now
                },
                description="Capture frontal chest X-ray view"
            ),
            
            # Step 7: Switch to lateral view (if available)
            WorkflowStep(
                id="switch_to_lateral",
                action_type=ActionType.CLICK,
                parameters={
                    "element_text": "Lateral",
                    "element_type": "button"
                },
                description="Switch to lateral view",
                continue_on_failure=True  # Continue if lateral view not available
            ),
            
            # Step 8: Capture lateral view
            WorkflowStep(
                id="capture_lateral_view",
                action_type=ActionType.SCREENSHOT,
                parameters={
                    "filepath": f"patient_{patient_id}_lateral.png"
                },
                description="Capture lateral chest X-ray view",
                continue_on_failure=True
            ),
            
            # Step 9: Open reporting interface
            WorkflowStep(
                id="open_reporting",
                action_type=ActionType.CLICK,
                parameters={
                    "element_text": "Report",
                    "element_type": "button"
                },
                description="Open reporting interface",
                timeout=15.0
            ),
            
            # Step 10: Wait for report template
            WorkflowStep(
                id="wait_for_report_template",
                action_type=ActionType.WAIT,
                parameters={
                    "wait_for_element": {
                        "text": "Findings",
                        "timeout": 10.0
                    }
                },
                description="Wait for report template to load"
            ),
            
            # Step 11: Take screenshot of report interface
            WorkflowStep(
                id="capture_report_interface",
                action_type=ActionType.SCREENSHOT,
                parameters={
                    "filepath": f"patient_{patient_id}_report_interface.png"
                },
                description="Capture report interface for reference"
            ),
            
            # Step 12: Mark as ready for AI analysis
            WorkflowStep(
                id="mark_ready_for_ai",
                action_type=ActionType.CUSTOM,
                parameters={
                    "action_name": "mark_for_ai_analysis",
                    "action_params": {
                        "patient_id": patient_id,
                        "study_type": "chest_xray"
                    }
                },
                description="Mark study as ready for AI analysis"
            )
        ]
        
        return workflow_steps
    
    def process_patient_list(self, patient_ids: List[str]) -> Dict[str, Any]:
        """Process a list of patients automatically."""
        print(f"Starting automated processing of {len(patient_ids)} patients...")
        
        results = {
            "total_patients": len(patient_ids),
            "successful": 0,
            "failed": 0,
            "patient_results": {},
            "processing_time": 0.0
        }
        
        start_time = time.time()
        
        for i, patient_id in enumerate(patient_ids, 1):
            print(f"\n--- Processing Patient {i}/{len(patient_ids)}: {patient_id} ---")
            
            try:
                # Create workflow for this patient
                workflow = self.create_chest_xray_workflow(patient_id)
                
                # Execute workflow
                workflow_result = self.agent.execute_workflow(
                    workflow, 
                    f"chest_xray_{patient_id}"
                )
                
                # Store results
                patient_result = {
                    "status": workflow_result.status.value,
                    "successful_steps": workflow_result.successful_steps,
                    "total_steps": workflow_result.total_steps,
                    "execution_time": workflow_result.execution_time,
                    "images_captured": self._count_captured_images(patient_id),
                    "ready_for_ai": workflow_result.status.value == "success"
                }
                
                results["patient_results"][patient_id] = patient_result
                
                if workflow_result.status.value == "success":
                    results["successful"] += 1
                    self.processed_patients.append(patient_id)
                    print(f"✓ Patient {patient_id} processed successfully")
                else:
                    results["failed"] += 1
                    print(f"✗ Patient {patient_id} processing failed")
                
                # Brief pause between patients
                time.sleep(2.0)
                
            except Exception as e:
                print(f"✗ Error processing patient {patient_id}: {e}")
                results["failed"] += 1
                results["patient_results"][patient_id] = {
                    "status": "error",
                    "error": str(e),
                    "ready_for_ai": False
                }
        
        results["processing_time"] = time.time() - start_time
        
        print(f"\n=== Processing Complete ===")
        print(f"Total time: {results['processing_time']:.1f}s")
        print(f"Successful: {results['successful']}/{results['total_patients']}")
        print(f"Failed: {results['failed']}/{results['total_patients']}")
        
        return results
    
    def _count_captured_images(self, patient_id: str) -> int:
        """Count how many images were captured for a patient."""
        import glob
        pattern = f"patient_{patient_id}_*.png"
        return len(glob.glob(pattern))
    
    def generate_processing_report(self, results: Dict[str, Any]) -> str:
        """Generate a summary report of the processing session."""
        report_lines = [
            "PACS Automation Processing Report",
            "=" * 40,
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Patients: {results['total_patients']}",
            f"Successful: {results['successful']}",
            f"Failed: {results['failed']}",
            f"Success Rate: {(results['successful']/results['total_patients']*100):.1f}%",
            f"Total Processing Time: {results['processing_time']:.1f}s",
            f"Average Time per Patient: {(results['processing_time']/results['total_patients']):.1f}s",
            "",
            "Patient Details:",
            "-" * 20
        ]
        
        for patient_id, result in results["patient_results"].items():
            status_icon = "✓" if result["status"] == "success" else "✗"
            report_lines.append(
                f"{status_icon} {patient_id}: {result['status']} "
                f"({result.get('execution_time', 0):.1f}s)"
            )
            
            if result.get("images_captured", 0) > 0:
                report_lines.append(f"    Images captured: {result['images_captured']}")
        
        report_content = "\n".join(report_lines)
        
        # Save report to file
        report_filename = f"pacs_processing_report_{int(time.time())}.txt"
        with open(report_filename, 'w') as f:
            f.write(report_content)
        
        print(f"\n📄 Report saved to: {report_filename}")
        return report_content
    
    def setup_ai_integration_placeholder(self):
        """Placeholder for future AI integration setup."""
        print("\n🤖 AI Integration Setup (Future Implementation)")
        print("This is where you would:")
        print("1. Load your vision-language model")
        print("2. Setup model inference pipeline")
        print("3. Configure report generation templates")
        print("4. Setup integration with your reporting system")
        
        # Register custom action for AI processing
        def process_with_ai(params, executor):
            patient_id = params.get('patient_id')
            study_type = params.get('study_type')
            
            print(f"🔬 AI Analysis placeholder for {patient_id} ({study_type})")
            print("   - Would analyze captured images")
            print("   - Generate preliminary findings")
            print("   - Populate report template")
            
            # Simulate AI processing time
            time.sleep(1.0)
            return True
        
        self.agent.workflow_executor.register_custom_action(
            "mark_for_ai_analysis", 
            process_with_ai
        )
    
    def cleanup(self):
        """Cleanup resources."""
        self.agent.cleanup()

def main():
    """Main execution function."""
    print("PACS Automation for Chest X-Ray Reading")
    print("=" * 40)
    
    # Initialize automation system
    pacs_automation = PACSAutomation()
    
    try:
        # Setup configuration
        pacs_automation.setup_pacs_configuration()
        
        # Setup AI integration placeholder
        pacs_automation.setup_ai_integration_placeholder()
        
        # Example patient list (replace with your actual patient IDs)
        patient_ids = [
            "CXR001",
            "CXR002", 
            "CXR003"
        ]
        
        print(f"\n🏥 Ready to process {len(patient_ids)} patients")
        print("Patient IDs:", ", ".join(patient_ids))
        
        # Ask for confirmation before proceeding
        response = input("\nProceed with automated processing? (y/N): ")
        
        if response.lower() == 'y':
            # Process patients
            results = pacs_automation.process_patient_list(patient_ids)
            
            # Generate report
            report = pacs_automation.generate_processing_report(results)
            
            print("\n" + "=" * 40)
            print("Automation completed successfully!")
            print("\nNext steps:")
            print("1. Review captured images")
            print("2. Integrate with your vision-language model")
            print("3. Setup automated report generation")
            print("4. Configure quality assurance workflows")
            
        else:
            print("Processing cancelled by user")
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during automation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pacs_automation.cleanup()

if __name__ == "__main__":
    main()