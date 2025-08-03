import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from .ocr_engine import OCRTextDetector, TextElement

class ElementType(Enum):
    BUTTON = "button"
    TEXT_FIELD = "text_field"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    IMAGE = "image"
    ICON = "icon"
    MENU_ITEM = "menu_item"
    TAB = "tab"
    SCROLLBAR = "scrollbar"
    UNKNOWN = "unknown"

@dataclass
class UIElement:
    element_type: ElementType
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[int, int]
    confidence: float
    text: Optional[str] = None
    attributes: Optional[Dict] = None

@dataclass
class DetectionResult:
    elements: List[UIElement]
    processing_time: float

class ElementDetector:
    def __init__(self, ocr_detector: Optional[OCRTextDetector] = None):
        self.ocr_detector = ocr_detector or OCRTextDetector()
        
        # Template images for common UI elements (can be loaded from files)
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load template images for UI element detection."""
        # In a real implementation, you would load these from files
        # For now, we'll use programmatically generated templates
        
        # Button template (rounded rectangle)
        button_template = np.zeros((30, 100, 3), dtype=np.uint8)
        cv2.rectangle(button_template, (5, 5), (95, 25), (200, 200, 200), -1)
        cv2.rectangle(button_template, (5, 5), (95, 25), (100, 100, 100), 2)
        self.templates[ElementType.BUTTON] = button_template
        
        # Checkbox template
        checkbox_template = np.zeros((20, 20, 3), dtype=np.uint8)
        cv2.rectangle(checkbox_template, (2, 2), (18, 18), (255, 255, 255), -1)
        cv2.rectangle(checkbox_template, (2, 2), (18, 18), (0, 0, 0), 2)
        self.templates[ElementType.CHECKBOX] = checkbox_template
    
    def detect_buttons(self, image: np.ndarray) -> List[UIElement]:
        """Detect button elements in the image."""
        buttons = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter contours by area and aspect ratio
            area = cv2.contourArea(contour)
            if area < 500 or area > 50000:  # Size limits for buttons
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check if it looks like a button (reasonable aspect ratio)
            if 0.3 <= aspect_ratio <= 10:
                center = (x + w // 2, y + h // 2)
                
                # Extract text from this region using OCR
                button_region = image[y:y+h, x:x+w]
                text_elements = self.ocr_detector.extract_text(button_region).elements
                button_text = " ".join([elem.text for elem in text_elements]) if text_elements else None
                
                button = UIElement(
                    element_type=ElementType.BUTTON,
                    bbox=(x, y, w, h),
                    center=center,
                    confidence=0.7,  # Base confidence
                    text=button_text,
                    attributes={"area": area, "aspect_ratio": aspect_ratio}
                )
                buttons.append(button)
        
        return buttons
    
    def detect_text_fields(self, image: np.ndarray) -> List[UIElement]:
        """Detect text input fields."""
        text_fields = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for rectangular shapes that could be text fields
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200 or area > 20000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Text fields are typically wide and short
            if aspect_ratio > 2 and h < 50:
                center = (x + w // 2, y + h // 2)
                
                # Check if the region looks like a text field (light background)
                roi = gray[y:y+h, x:x+w]
                mean_intensity = np.mean(roi)
                
                if mean_intensity > 200:  # Light background typical of text fields
                    text_field = UIElement(
                        element_type=ElementType.TEXT_FIELD,
                        bbox=(x, y, w, h),
                        center=center,
                        confidence=0.6,
                        attributes={"mean_intensity": mean_intensity}
                    )
                    text_fields.append(text_field)
        
        return text_fields
    
    def detect_checkboxes(self, image: np.ndarray) -> List[UIElement]:
        """Detect checkbox elements."""
        checkboxes = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Template matching for checkbox detection
        if ElementType.CHECKBOX in self.templates:
            template = cv2.cvtColor(self.templates[ElementType.CHECKBOX], cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            
            threshold = 0.6
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                x, y = pt
                w, h = template.shape[::-1]
                center = (x + w // 2, y + h // 2)
                
                checkbox = UIElement(
                    element_type=ElementType.CHECKBOX,
                    bbox=(x, y, w, h),
                    center=center,
                    confidence=result[y, x],
                    attributes={"template_match": True}
                )
                checkboxes.append(checkbox)
        
        return checkboxes
    
    def detect_dropdown_menus(self, image: np.ndarray) -> List[UIElement]:
        """Detect dropdown menu elements."""
        dropdowns = []
        
        # Look for dropdown arrow patterns
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create a simple dropdown arrow template
        arrow_template = np.zeros((10, 10), dtype=np.uint8)
        cv2.fillPoly(arrow_template, [np.array([[2, 3], [8, 3], [5, 7]])], 255)
        
        result = cv2.matchTemplate(gray, arrow_template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        locations = np.where(result >= threshold)
        
        for pt in zip(*locations[::-1]):
            x, y = pt
            
            # Expand to include likely dropdown area
            dropdown_x = max(0, x - 100)
            dropdown_y = y
            dropdown_w = min(150, image.shape[1] - dropdown_x)
            dropdown_h = 25
            
            center = (dropdown_x + dropdown_w // 2, dropdown_y + dropdown_h // 2)
            
            dropdown = UIElement(
                element_type=ElementType.DROPDOWN,
                bbox=(dropdown_x, dropdown_y, dropdown_w, dropdown_h),
                center=center,
                confidence=result[y, x],
                attributes={"arrow_detected": True}
            )
            dropdowns.append(dropdown)
        
        return dropdowns
    
    def detect_images(self, image: np.ndarray, min_size: int = 1000) -> List[UIElement]:
        """Detect image elements (like medical images in PACS)."""
        images = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive threshold to find image boundaries
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_size:  # Filter small regions
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Images typically have reasonable aspect ratios
            if 0.1 <= aspect_ratio <= 10:
                center = (x + w // 2, y + h // 2)
                
                # Check if region has image-like characteristics
                roi = image[y:y+h, x:x+w]
                std_dev = np.std(roi)
                
                # Images typically have higher standard deviation than UI elements
                if std_dev > 20:
                    img_element = UIElement(
                        element_type=ElementType.IMAGE,
                        bbox=(x, y, w, h),
                        center=center,
                        confidence=0.8,
                        attributes={
                            "area": area,
                            "std_dev": std_dev,
                            "aspect_ratio": aspect_ratio
                        }
                    )
                    images.append(img_element)
        
        return images
    
    def detect_all_elements(self, image: np.ndarray) -> DetectionResult:
        """Detect all types of UI elements in the image."""
        import time
        start_time = time.time()
        
        all_elements = []
        
        try:
            # Detect different types of elements
            buttons = self.detect_buttons(image)
            text_fields = self.detect_text_fields(image)
            checkboxes = self.detect_checkboxes(image)
            dropdowns = self.detect_dropdown_menus(image)
            images = self.detect_images(image)
            
            all_elements.extend(buttons)
            all_elements.extend(text_fields)
            all_elements.extend(checkboxes)
            all_elements.extend(dropdowns)
            all_elements.extend(images)
            
            # Remove overlapping elements (keep higher confidence)
            all_elements = self._remove_overlapping_elements(all_elements)
            
        except Exception as e:
            logger.error(f"Error during element detection: {e}")
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            elements=all_elements,
            processing_time=processing_time
        )
    
    def _remove_overlapping_elements(self, elements: List[UIElement], overlap_threshold: float = 0.5) -> List[UIElement]:
        """Remove overlapping elements, keeping the one with higher confidence."""
        if not elements:
            return elements
        
        # Sort by confidence (descending)
        sorted_elements = sorted(elements, key=lambda x: x.confidence, reverse=True)
        filtered_elements = []
        
        for element in sorted_elements:
            is_overlapping = False
            
            for existing_element in filtered_elements:
                if self._calculate_overlap(element.bbox, existing_element.bbox) > overlap_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_elements.append(element)
        
        return filtered_elements
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if right <= left or bottom <= top:
            return 0.0
        
        intersection_area = (right - left) * (bottom - top)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def find_element_by_text(self, image: np.ndarray, text: str, element_type: Optional[ElementType] = None) -> List[UIElement]:
        """Find UI elements containing specific text."""
        elements = self.detect_all_elements(image).elements
        matches = []
        
        for element in elements:
            if element_type and element.element_type != element_type:
                continue
                
            if element.text and text.lower() in element.text.lower():
                matches.append(element)
        
        return matches
    
    def visualize_elements(self, image: np.ndarray, elements: List[UIElement]) -> np.ndarray:
        """Draw detected elements on the image for visualization."""
        vis_image = image.copy()
        
        # Color mapping for different element types
        colors = {
            ElementType.BUTTON: (0, 255, 0),      # Green
            ElementType.TEXT_FIELD: (255, 0, 0),  # Blue
            ElementType.CHECKBOX: (0, 0, 255),    # Red
            ElementType.DROPDOWN: (255, 255, 0),  # Cyan
            ElementType.IMAGE: (255, 0, 255),     # Magenta
            ElementType.UNKNOWN: (128, 128, 128)  # Gray
        }
        
        for element in elements:
            x, y, w, h = element.bbox
            color = colors.get(element.element_type, colors[ElementType.UNKNOWN])
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{element.element_type.value}"
            if element.text:
                label += f": {element.text[:20]}"
            label += f" ({element.confidence:.2f})"
            
            cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image