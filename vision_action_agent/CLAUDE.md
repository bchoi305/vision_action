# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -r requirements.txt[dev]
```

### Development Commands
```bash
# Run basic functionality test
cd examples && python basic_usage.py

# Run PACS automation example
cd examples && python pacs_automation.py

# Format code (if black is installed)
black src/

# Type checking (if mypy is installed)
mypy src/

# Run tests (if pytest is installed)
pytest tests/
```

### Testing Individual Components
```bash
# Test screen capture
python -c "from src.vision.screen_capture import ScreenCapture; sc = ScreenCapture(); print('Screen capture working')"

# Test OCR
python -c "from src.vision.ocr_engine import OCRTextDetector; ocr = OCRTextDetector(); print('OCR initialized')"

# Test configuration
python -c "from src.config.settings import ConfigManager; cm = ConfigManager(); print('Config loaded')"
```

## Architecture Overview

### Core Architecture Pattern
The system follows a **layered architecture** with clear separation of concerns:

1. **Vision Layer** (`src/vision/`): Screen capture, OCR, and UI element detection
2. **Actions Layer** (`src/actions/`): Mouse/keyboard control and workflow execution  
3. **Core Layer** (`src/core/`): Main agent orchestration
4. **Configuration Layer** (`src/config/`): Settings and PACS customization
5. **Utils Layer** (`src/utils/`): Error handling and recovery

### Key Integration Points

**VisionActionAgent** (`src/core/agent.py`) is the main orchestrator that:
- Initializes all components through `_initialize_components()`
- Manages PACS-specific state (`current_patient_id`, `current_study_open`)
- Provides high-level methods that combine vision + action capabilities
- Registers custom actions for workflow execution

**WorkflowExecutor** (`src/actions/workflow_executor.py`) is the automation engine that:
- Executes YAML-defined workflows through `execute_workflow()`
- Supports 12+ action types (CLICK, TYPE, SCREENSHOT, etc.)
- Manages workflow variables and custom action registration
- Handles retry logic and error recovery per step

**Configuration System** (`src/config/settings.py`) provides:
- Dataclass-based configuration with `AgentConfig`
- PACS-specific UI element definitions in `common_ui_elements`
- Runtime configuration updates through `ConfigManager`
- YAML/JSON configuration file support

### PACS-Specific Design

The system is specifically designed for medical imaging workflows:

- **PACS State Management**: Tracks patient ID and study status
- **Medical Image Detection**: Specialized detection for radiology images
- **Workflow Templates**: Pre-built workflows for chest X-rays and other studies
- **AI Integration Ready**: Designed for vision-language model integration

### Component Communication Flow

1. **Agent** receives high-level requests (e.g., `click_element("Open Study")`)
2. **Vision components** capture screen and detect UI elements
3. **Action components** execute mouse/keyboard actions
4. **Workflow executor** orchestrates multi-step automation sequences
5. **Error handler** provides automatic retry and recovery mechanisms

### Key Configuration Areas

- `config.yaml`: Main configuration file with PACS-specific settings
- `common_ui_elements`: Defines expected PACS interface elements
- OCR engine selection: EasyOCR (default) vs Tesseract
- Automation speeds and human-like behavior settings

### Extension Points

- **Custom Actions**: Register via `workflow_executor.register_custom_action()`
- **PACS Elements**: Add via `config_manager.add_pacs_element_config()`
- **Error Recovery**: Add strategies via `error_handler.register_recovery_strategy()`
- **Workflow Steps**: Define in YAML with 12+ supported action types

### Important Files for Modification

- `src/core/agent.py`: Main agent API and PACS methods
- `src/actions/workflow_executor.py`: Workflow action implementations
- `src/config/settings.py`: Configuration schema and PACS elements
- `examples/config.yaml`: Default configuration template
- `examples/pacs_automation.py`: PACS workflow examples

### Development Notes

- The system uses `loguru` for logging with configurable levels
- All components are designed to work with screenshots as numpy arrays
- OCR results include bounding boxes and confidence scores
- Workflow steps support retry counts, timeouts, and continue-on-failure options
- PACS workflows are designed to be template-driven and customizable