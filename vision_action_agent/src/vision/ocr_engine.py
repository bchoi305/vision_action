import cv2
import numpy as np
import pytesseract
import easyocr
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import re

class OCREngine(Enum):
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"

@dataclass
class TextElement:
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    center: Tuple[int, int]

@dataclass
class OCRResult:
    elements: List[TextElement]
    raw_text: str
    processing_time: float

class OCRTextDetector:
    def __init__(self, engine: OCREngine = OCREngine.EASYOCR, languages: List[str] = ['en']):
        self.engine = engine
        self.languages = languages
        
        if engine == OCREngine.EASYOCR:
            try:
                self.reader = easyocr.Reader(languages)
                logger.info(f"EasyOCR initialized with languages: {languages}")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                raise
        elif engine == OCREngine.TESSERACT:
            # Test if Tesseract is available
            try:
                pytesseract.get_tesseract_version()
                logger.info("Tesseract OCR initialized")
            except Exception as e:
                logger.error(f"Tesseract not found: {e}")
                raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Remove noise
        denoised = cv2.medianBlur(thresh, 3)
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_tesseract(self, image: np.ndarray) -> OCRResult:
        """Extract text using Tesseract OCR."""
        import time
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Get detailed data including bounding boxes
            data = pytesseract.image_to_data(
                processed_image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'
            )
            
            elements = []
            raw_text = ""
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and int(data['conf'][i]) > 30:  # Confidence threshold
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    center = (x + w // 2, y + h // 2)
                    
                    element = TextElement(
                        text=text,
                        bbox=(x, y, w, h),
                        confidence=float(data['conf'][i]) / 100.0,
                        center=center
                    )
                    elements.append(element)
                    raw_text += text + " "
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                elements=elements,
                raw_text=raw_text.strip(),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return OCRResult(elements=[], raw_text="", processing_time=0)
    
    def extract_text_easyocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using EasyOCR."""
        import time
        start_time = time.time()
        
        try:
            # EasyOCR works better with original image
            results = self.reader.readtext(image)
            
            elements = []
            raw_text = ""
            
            for bbox, text, confidence in results:
                if confidence > 0.3:  # Confidence threshold
                    # Convert bbox to standard format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    w = int(max(x_coords) - min(x_coords))
                    h = int(max(y_coords) - min(y_coords))
                    
                    center = (x + w // 2, y + h // 2)
                    
                    element = TextElement(
                        text=text.strip(),
                        bbox=(x, y, w, h),
                        confidence=confidence,
                        center=center
                    )
                    elements.append(element)
                    raw_text += text + " "
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                elements=elements,
                raw_text=raw_text.strip(),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return OCRResult(elements=[], raw_text="", processing_time=0)
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extract text using the configured OCR engine."""
        if self.engine == OCREngine.TESSERACT:
            return self.extract_text_tesseract(image)
        elif self.engine == OCREngine.EASYOCR:
            return self.extract_text_easyocr(image)
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine}")
    
    def find_text(self, image: np.ndarray, target_text: str, case_sensitive: bool = False) -> List[TextElement]:
        """Find specific text in the image."""
        result = self.extract_text(image)
        matches = []
        
        search_text = target_text if case_sensitive else target_text.lower()
        
        for element in result.elements:
            element_text = element.text if case_sensitive else element.text.lower()
            
            if search_text in element_text:
                matches.append(element)
        
        return matches
    
    def find_text_regex(self, image: np.ndarray, pattern: str) -> List[TextElement]:
        """Find text matching a regex pattern."""
        result = self.extract_text(image)
        matches = []
        
        compiled_pattern = re.compile(pattern)
        
        for element in result.elements:
            if compiled_pattern.search(element.text):
                matches.append(element)
        
        return matches
    
    def get_text_near_coordinates(self, image: np.ndarray, x: int, y: int, radius: int = 50) -> List[TextElement]:
        """Find text elements near specific coordinates."""
        result = self.extract_text(image)
        nearby_elements = []
        
        for element in result.elements:
            distance = ((element.center[0] - x) ** 2 + (element.center[1] - y) ** 2) ** 0.5
            if distance <= radius:
                nearby_elements.append(element)
        
        return nearby_elements
    
    def visualize_ocr_results(self, image: np.ndarray, result: OCRResult) -> np.ndarray:
        """Draw bounding boxes and text on the image for visualization."""
        vis_image = image.copy()
        
        for element in result.elements:
            x, y, w, h = element.bbox
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw text and confidence
            label = f"{element.text} ({element.confidence:.2f})"
            cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis_image