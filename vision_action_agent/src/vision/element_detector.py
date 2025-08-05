import os
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
    def __init__(self, ocr_detector: Optional[OCRTextDetector] = None, template_dir: str = "templates"):
        self.ocr_detector = ocr_detector or OCRTextDetector()
        self.templates = {}
        self._load_templates(template_dir)
    
    def _load_templates(self, template_dir: str = "templates"):
        """Load template images for UI element detection from a directory."""
        self.templates = {}
        if not os.path.exists(template_dir):
            logger.warning(f"Template directory not found: {template_dir}")
            return

        for filename in os.listdir(template_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                filepath = os.path.join(template_dir, filename)
                try:
                    template_image = cv2.imread(filepath, cv2.IMREAD_COLOR)
                    if template_image is not None:
                        # Use filename (without extension) as the template name
                        template_name = os.path.splitext(filename)[0]
                        self.templates[template_name] = template_image
                        logger.info(f"Loaded template: {template_name} from {filepath}")
                    else:
                        logger.warning(f"Could not load image file: {filepath}")
                except Exception as e:
                    logger.error(f"Error loading template {filepath}: {e}")
    
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
            
            # Check if it looks like a button (reasonable aspect ratio and color uniformity)
            if 0.3 <= aspect_ratio <= 10:
                # Check color uniformity (low standard deviation in color channels)
                roi_color = image[y:y+h, x:x+w]
                if roi_color.size > 0:
                    std_dev_color = np.std(roi_color)
                    if std_dev_color < 40: # Threshold for color uniformity
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
                            attributes={"area": area, "aspect_ratio": aspect_ratio, "std_dev_color": std_dev_color}
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
            
            # Text fields are typically wide and short with a clear background or border
            if aspect_ratio > 2 and h < 50:
                center = (x + w // 2, y + h // 2)
                
                # Check for clear background (high mean intensity) or a distinct border
                roi = gray[y:y+h, x:x+w]
                mean_intensity = np.mean(roi)
                
                # Simple border detection: check intensity variance near edges
                border_variance = np.var(gray[y:y+h, x:x+5]) + np.var(gray[y:y+h, x+w-5:x+w]) + \
                                  np.var(gray[y:y+5, x:x+w]) + np.var(gray[y+h-5:y+h, x:x+w])

                if mean_intensity > 200 or border_variance > 500:  # Light background or significant border
                    text_field = UIElement(
                        element_type=ElementType.TEXT_FIELD,
                        bbox=(x, y, w, h),
                        center=center,
                        confidence=0.6,
                        attributes={"mean_intensity": mean_intensity, "border_variance": border_variance}
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
        """Detect image elements."""
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

    def detect_unknown_elements(self, image: np.ndarray) -> List[UIElement]:
        """
        Detect general rectangular regions that might be UI elements but don't fit other categories.
        These are potential candidates for further classification or interaction.
        """
        unknown_elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use a combination of thresholding and contour detection
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 100000:  # Filter by reasonable size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                # Filter by aspect ratio (avoid very thin or very wide elements)
                if 0.1 < aspect_ratio < 10:
                    center = (x + w // 2, y + h // 2)
                    roi = image[y:y+h, x:x+w]
                    
                    # Check for some visual complexity (not just a solid block)
                    if roi.size > 0:
                        std_dev_color = np.std(roi)
                        if std_dev_color > 10: # Ensure it's not just a uniform background patch
                            unknown_elements.append(UIElement(
                                element_type=ElementType.UNKNOWN,
                                bbox=(x, y, w, h),
                                center=center,
                                confidence=0.5, # Default confidence
                                attributes={
                                    "area": area,
                                    "aspect_ratio": aspect_ratio,
                                    "std_dev_color": std_dev_color
                                }
                            ))
        return unknown_elements
    
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
            all_elements.extend(self.detect_unknown_elements(image))
            
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

    def find_element_by_image(self, image: np.ndarray, template_name: str, confidence_threshold: float = 0.8) -> List[UIElement]:
        """
        Find UI elements by matching a template image.
        
        Args:
            image: The screenshot (numpy array) to search within.
            template_name: The name of the pre-loaded template image (e.g., 'login_button_icon').
            confidence_threshold: The minimum confidence score to consider a match.
            
        Returns:
            A list of UIElement objects found, or an empty list if none are found.
        """
        if template_name not in self.templates:
            logger.warning(f"Template '{template_name}' not found in loaded templates.")
            return []

        template = self.templates[template_name]
        
        # Convert images to grayscale for template matching
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= confidence_threshold)

        found_elements = []
        for pt in zip(*loc[::-1]):  # Swap x and y for correct coordinates
            x, y = pt[0], pt[1]
            w, h = template.shape[1], template.shape[0]
            center = (x + w // 2, y + h // 2)
            confidence = res[pt[1], pt[0]]

            # Create a UIElement for the found template
            found_elements.append(UIElement(
                element_type=ElementType.ICON,  # Assuming image templates are icons
                bbox=(x, y, w, h),
                center=center,
                confidence=confidence,
                attributes={"template_name": template_name}
            ))
        
        # Remove overlapping detections
        found_elements = self._remove_overlapping_elements(found_elements, overlap_threshold=0.1) # Lower overlap for icons

        if found_elements:
            logger.info(f"Found {len(found_elements)} instances of template '{template_name}' with confidence >= {confidence_threshold:.2f}")
        else:
            logger.debug(f"No instances of template '{template_name}' found with confidence >= {confidence_threshold:.2f}")

        return found_elements

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
            ElementType.UNKNOWN: (128, 128, 128), # Gray
            ElementType.ICON: (0, 255, 255)       # Yellow
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

    def analyze_layout(self, elements: List[UIElement], proximity_threshold: int = 20) -> Dict[str, Any]:
        """
        Analyzes the spatial relationships between UI elements to infer layout structure.
        Groups elements that are close to each other horizontally or vertically.
        
        Args:
            elements: A list of UIElement objects.
            proximity_threshold: Maximum distance (in pixels) to consider elements as proximate.
            
        Returns:
            A dictionary describing the detected layout groups.
        """
        if not elements:
            return {"groups": []}

        # Sort elements by their top-left corner for consistent processing
        sorted_elements = sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0]))

        groups = []
        used_indices = set()

        for i, elem1 in enumerate(sorted_elements):
            if i in used_indices:
                continue

            current_group = [elem1]
            used_indices.add(i)

            for j, elem2 in enumerate(sorted_elements):
                if i == j or j in used_indices:
                    continue

                # Check for horizontal proximity and vertical alignment
                # Elements are horizontally proximate if their horizontal distance is within threshold
                # And their vertical centers are aligned (within threshold)
                horizontal_distance = abs((elem1.bbox[0] + elem1.bbox[2]) - elem2.bbox[0])
                vertical_alignment = abs(elem1.center[1] - elem2.center[1])

                if horizontal_distance < proximity_threshold and vertical_alignment < proximity_threshold:
                    current_group.append(elem2)
                    used_indices.add(j)
                    continue

                # Check for vertical proximity and horizontal alignment
                # Elements are vertically proximate if their vertical distance is within threshold
                # And their horizontal centers are aligned (within threshold)
                vertical_distance = abs((elem1.bbox[1] + elem1.bbox[3]) - elem2.bbox[1])
                horizontal_alignment = abs(elem1.center[0] - elem2.center[0])

                if vertical_distance < proximity_threshold and horizontal_alignment < proximity_threshold:
                    current_group.append(elem2)
                    used_indices.add(j)

            if len(current_group) > 1:
                # Calculate bounding box for the group
                min_x = min(e.bbox[0] for e in current_group)
                min_y = min(e.bbox[1] for e in current_group)
                max_x = max(e.bbox[0] + e.bbox[2] for e in current_group)
                max_y = max(e.bbox[1] + e.bbox[3] for e in current_group)
                
                groups.append({
                    "type": "group",
                    "elements": [e.text or e.element_type.value for e in current_group],
                    "bbox": (min_x, min_y, max_x - min_x, max_y - min_y)
                })
        
        logger.info(f"Analyzed layout: Found {len(groups)} element groups.")
        return {"groups": groups}