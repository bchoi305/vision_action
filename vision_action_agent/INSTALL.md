# Installation Guide

This guide will help you install and set up the Vision Action Agent for PACS automation.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended for large workflows)
- **Disk Space**: At least 2GB free space

### Required Software
1. **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
2. **Tesseract OCR** (optional, for Tesseract engine):
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

## Installation Steps

### 1. Clone or Download the Repository

```bash
# If using git
git clone https://github.com/yourusername/vision-action-agent.git
cd vision-action-agent

# Or download and extract the ZIP file
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# For development (optional)
pip install -r requirements.txt[dev]
```

### 4. Install EasyOCR (Primary OCR Engine)

EasyOCR will be installed automatically with the requirements, but first-time setup may take a few minutes as it downloads language models.

```python
# Test EasyOCR installation
python -c "import easyocr; reader = easyocr.Reader(['en']); print('EasyOCR installed successfully')"
```

### 5. Configure Tesseract (Optional)

If you want to use Tesseract OCR as an alternative:

**Windows:**
1. Download Tesseract installer
2. Install to default location (usually `C:\Program Files\Tesseract-OCR\`)
3. Add to PATH or set environment variable:
   ```bash
   # Add this to your environment variables
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

**macOS/Linux:**
```bash
# Test Tesseract installation
tesseract --version
```

### 6. Verify Installation

Run the basic verification script:

```bash
# From the project root directory
cd examples
python basic_usage.py
```

If everything is installed correctly, you should see:
- Screenshot capture working
- OCR engine initialized
- UI element detection functioning

## Configuration

### 1. Create Configuration File

Copy the example configuration:

```bash
cp examples/config.yaml config.yaml
```

### 2. Customize PACS Settings

Edit `config.yaml` to match your PACS system:

```yaml
pacs:
  application_name: "Your PACS Name"
  window_title_contains: "PACS Window Title"
  default_patient_search_timeout: 10.0
  # ... other settings
```

### 3. Test PACS Integration

```bash
cd examples
python pacs_automation.py
```

## Common Installation Issues

### Issue: ImportError for cv2 (OpenCV)

**Solution:**
```bash
pip uninstall opencv-python
pip install opencv-python==4.8.1.78
```

### Issue: EasyOCR CUDA/GPU Issues

**Solution (CPU-only):**
```bash
pip uninstall easyocr
pip install easyocr --no-deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: PyAutoGUI Permission Errors (macOS)

**Solution:**
1. Go to System Preferences → Security & Privacy → Privacy
2. Add Terminal/Python to "Accessibility" and "Screen Recording"

### Issue: Tesseract Command Not Found

**Solution:**

**Windows:**
```bash
# Set environment variable
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

### Issue: Permission Denied Errors (Linux)

**Solution:**
```bash
# Add user to input group for mouse/keyboard control
sudo usermod -a -G input $USER
# Log out and log back in
```

## Development Installation

For development and contributing:

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt[dev]

# Run tests
pytest tests/

# Format code
black src/

# Type checking
mypy src/
```

## Docker Installation (Alternative)

If you prefer using Docker:

```bash
# Build Docker image
docker build -t vision-action-agent .

# Run container
docker run -it --rm \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  vision-action-agent
```

## Performance Optimization

### 1. GPU Acceleration (Optional)

For faster OCR processing:

```bash
# Install CUDA-enabled PyTorch (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Memory Optimization

For large-scale processing:

```python
# In your config.yaml
workflow:
  save_screenshots_on_failure: false  # Reduces disk usage
  
ocr:
  confidence_threshold: 0.5  # Higher threshold = faster processing
```

## Troubleshooting

### 1. Enable Debug Logging

```python
# In your script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Test Individual Components

```python
# Test screen capture
from vision_action_agent import ScreenCapture
capture = ScreenCapture()
image = capture.capture_full_screen()
print(f"Captured image shape: {image.shape}")

# Test OCR
from vision_action_agent import OCRTextDetector
ocr = OCRTextDetector()
result = ocr.extract_text(image)
print(f"Detected text: {result.raw_text}")
```

### 3. Check System Resources

```bash
# Monitor CPU/Memory usage during processing
htop  # Linux/macOS
# or Task Manager on Windows
```

## Next Steps

After successful installation:

1. **Read the Documentation**: Check `README.md` for usage examples
2. **Run Examples**: Try the PACS automation example
3. **Customize Workflows**: Create your own workflow files
4. **Integration**: Connect with your vision-language model
5. **Production Setup**: Configure logging and monitoring

## Getting Help

- **Documentation**: Check the `docs/` directory
- **Examples**: Look at files in `examples/` directory
- **Issues**: Report bugs on GitHub Issues
- **Community**: Join our Discord/Slack community

## Security Considerations

1. **Screen Access**: The agent needs screen recording permissions
2. **Keyboard/Mouse**: Requires accessibility permissions
3. **PACS Access**: Ensure compliance with healthcare regulations
4. **Data Privacy**: Configure secure storage for captured images

Remember to follow your organization's IT security policies when deploying this system.