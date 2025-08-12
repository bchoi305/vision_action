import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger

logger.info("element_detector.py loaded and logger initialized.")

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
        """Detect button elements in the image using color and shape heuristics."""
        buttons = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive thresholding to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter contours by area to exclude very small or very large regions
            area = cv2.contourArea(contour)
            print(f"Contour area: {area}")
            if area < 100 or area > 100000:  # Adjusted size limits for more general buttons
                print("Area condition not met.")
                continue

            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # Check if the contour is rectangular (4 vertices)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h

                # Filter by aspect ratio typical for buttons
                if 0.2 <= aspect_ratio <= 5.0:  # More flexible aspect ratio
                    # Check color uniformity within the potential button region
                    roi_color = image[y:y+h, x:x+w]
                    if roi_color.size > 0:
                        std_dev_color = np.std(roi_color)
                        if std_dev_color < 30:  # Lower threshold for stricter uniformity
                            center = (x + w // 2, y + h // 2)

                            # Attempt to extract text from this region using OCR
                            button_region_img = image[y:y+h, x:x+w]
                            text_elements = self.ocr_detector.extract_text(button_region_img).elements
                            button_text = " ".join([elem.text for elem in text_elements]) if text_elements else None

                            buttons.append(UIElement(
                                element_type=ElementType.BUTTON,
                                bbox=(x, y, w, h),
                                center=center,
                                confidence=0.7,  # Base confidence
                                text=button_text,
                                attributes={
                                    "area": area,
                                    "aspect_ratio": aspect_ratio,
                                    "std_dev_color": std_dev_color
                                }
                            ))
        return buttons
    
    def detect_text_fields(self, image: np.ndarray) -> List[UIElement]:
        """Detect text input fields using shape and color heuristics."""
        text_fields = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection to find strong edges
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours in the edged image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area to exclude very small or very large regions
            if area < 150 or area > 50000:
                continue

            # Approximate the contour to a rectangle
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Text fields are typically rectangular (4 vertices)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h

                # Text fields are usually wide and relatively short
                if aspect_ratio > 2.0 and h > 15 and h < 60:  # Refined height range
                    center = (x + w // 2, y + h // 2)

                    # Analyze the region of interest (ROI) for text field characteristics
                    roi = image[y:y+h, x:x+w]
                    if roi.size == 0: # Skip empty regions
                        continue

                    # Check for a relatively uniform background (low standard deviation)
                    std_dev_color = np.std(roi)
                    if std_dev_color < 35:  # Adjusted threshold for background uniformity
                        # Check for presence of a distinct border or background color
                        # This can be done by analyzing pixel intensity near the edges
                        # For simplicity, we'll check if the average intensity is high (light background)
                        # or if there's a significant intensity change at the borders.
                        mean_intensity = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

                        # A simple check for a light background or a clear border
                        if mean_intensity > 180 or self._has_distinct_border(gray, (x, y, w, h)): # New helper for border
                            text_field = UIElement(
                                element_type=ElementType.TEXT_FIELD,
                                bbox=(x, y, w, h),
                                center=center,
                                confidence=0.65, # Slightly increased confidence
                                attributes={
                                    "mean_intensity": mean_intensity,
                                    "std_dev_color": std_dev_color
                                }
                            )
                            text_fields.append(text_field)
        return text_fields

    def _has_distinct_border(self, gray_image: np.ndarray, bbox: Tuple[int, int, int, int], border_width: int = 3, intensity_diff_threshold: int = 40) -> bool:
        """Helper to check for a distinct border around a bounding box."""
        x, y, w, h = bbox
        # Ensure border region is within image bounds
        if w <= 2 * border_width or h <= 2 * border_width:
            return False

        # Extract inner and outer regions
        inner_region = gray_image[y + border_width : y + h - border_width,
                                  x + border_width : x + w - border_width]
        outer_border_horizontal = np.concatenate([
            gray_image[y : y + border_width, x : x + w],
            gray_image[y + h - border_width : y + h, x : x + w]
        ])
        outer_border_vertical = np.concatenate([
            gray_image[y : y + h, x : x + border_width],
            gray_image[y : y + h, x + w - border_width : x + w]
        ])

        if inner_region.size == 0 or outer_border_horizontal.size == 0 or outer_border_vertical.size == 0:
            return False

        # Compare average intensity of inner region with border regions
        avg_inner = np.mean(inner_region)
        avg_outer_h = np.mean(outer_border_horizontal)
        avg_outer_v = np.mean(outer_border_vertical)

        # Check if there's a significant difference between inner and outer regions
        if abs(avg_inner - avg_outer_h) > intensity_diff_threshold or \
           abs(avg_inner - avg_outer_v) > intensity_diff_threshold:
            return True
        return False
    
    def detect_checkboxes(self, image: np.ndarray) -> List[UIElement]:
        """Detect checkbox elements using shape and color heuristics."""
        checkboxes = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Use adaptive thresholding to find potential checkbox shapes
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area to find square-like elements
            if area < 50 or area > 1000:  # Typical size range for checkboxes
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # Check if the contour is square-like (4 vertices and aspect ratio close to 1)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h

                if 0.8 <= aspect_ratio <= 1.2:  # Aspect ratio for squares
                    center = (x + w // 2, y + h // 2)

                    # Further check: color uniformity within the checkbox area
                    roi_color = image[y:y+h, x:x+w]
                    if roi_color.size > 0:
                        std_dev_color = np.std(roi_color)
                        if std_dev_color < 30:  # Low standard deviation for uniform color
                            checkboxes.append(UIElement(
                                element_type=ElementType.CHECKBOX,
                                bbox=(x, y, w, h),
                                center=center,
                                confidence=0.8,  # High confidence for strong square match
                                attributes={
                                    "area": area,
                                    "aspect_ratio": aspect_ratio,
                                    "std_dev_color": std_dev_color
                                }
                            ))
        return checkboxes
    
    def detect_dropdown_menus(self, image: np.ndarray) -> List[UIElement]:
        """Detect dropdown menu elements using shape and text heuristics."""
        dropdowns = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Look for rectangular shapes that might contain a dropdown arrow or text
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200 or area > 50000:  # Filter by reasonable size
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4:  # Look for rectangular shapes
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h

                # Dropdowns are typically wide and relatively short
                if aspect_ratio > 2.0 and h > 20 and h < 70:  # Adjusted height range
                    center = (x + w // 2, y + h // 2)

                    # Check for a uniform background and potential text/arrow inside
                    roi_color = image[y:y+h, x:x+w]
                    if roi_color.size == 0:
                        continue

                    std_dev_color = np.std(roi_color)
                    if std_dev_color < 40:  # Relatively uniform background
                        # Attempt to extract text from this region using OCR
                        dropdown_region_img = image[y:y+h, x:x+w]
                        text_elements = self.ocr_detector.extract_text(dropdown_region_img).elements
                        dropdown_text = " ".join([elem.text for elem in text_elements]) if text_elements else None

                        # Check for common dropdown indicators (e.g., a small triangle/arrow)
                        # This is a simple heuristic, can be improved with template matching for arrows
                        has_arrow_indicator = False
                        # A very basic check for a dark triangle shape (common for dropdowns)
                        arrow_roi = gray[y:y+h, x + w - 30 : x + w] # Look at the right side for an arrow
                        if arrow_roi.size > 0:
                            _, arrow_thresh = cv2.threshold(arrow_roi, 150, 255, cv2.THRESH_BINARY_INV)
                            arrow_contours, _ = cv2.findContours(arrow_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for arrow_cnt in arrow_contours:
                                if cv2.contourArea(arrow_cnt) > 10 and cv2.contourArea(arrow_cnt) < 100: # Small arrow size
                                    has_arrow_indicator = True
                                    break

                        if dropdown_text or has_arrow_indicator:
                            dropdown = UIElement(
                                element_type=ElementType.DROPDOWN,
                                bbox=(x, y, w, h),
                                center=center,
                                confidence=0.7,  # Base confidence
                                text=dropdown_text,
                                attributes={
                                    "area": area,
                                    "aspect_ratio": aspect_ratio,
                                    "std_dev_color": std_dev_color,
                                    "has_arrow": has_arrow_indicator
                                }
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

    def detect_icons_from_tf_hub(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[UIElement]:
        """
        Detects icons in an image using a pre-trained model from TensorFlow Hub.

        Args:
            image: The image to process.
            confidence_threshold: The minimum confidence score to consider a detection.

        Returns:
            A list of UIElement objects representing the detected icons.
        """
        # Import TensorFlow and TF Hub lazily to avoid import-time failures when not needed
        try:
            import tensorflow as tf  # type: ignore
            import tensorflow_hub as hub  # type: ignore
        except Exception as e:
            logger.error(f"TensorFlow/TF Hub not available: {e}")
            return []

        # Load the model from TensorFlow Hub
        model_url = "https://tfhub.dev/google/ssd/mobilenet_v2/2"
        logger.debug(f"Attempting to load TensorFlow Hub model from: {model_url}")
        detector = hub.load(model_url)
        logger.info("TensorFlow Hub model loaded successfully.")
        logger.debug("TensorFlow Hub model loaded and ready for inference.")

        # Preprocess the image for the model
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize pixel values to [0, 1]
        normalized_image = rgb_image / 255.0
        # Convert to a TensorFlow tensor and add batch dimension
        input_tensor = tf.convert_to_tensor(normalized_image, dtype=tf.float32)
        input_tensor = input_tensor[tf.newaxis, ...]
        logger.debug(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

        # Run the model
        result = detector(input_tensor)

        # Process the results
        result = {key: value.numpy() for key, value in result.items()}
        boxes = result["detection_boxes"][0]
        scores = result["detection_scores"][0]
        classes = result["detection_classes"][0]

        found_elements = []
        for i in range(boxes.shape[0]):
            if scores[i] >= confidence_threshold:
                ymin, xmin, ymax, xmax = tuple(boxes[i])
                im_height, im_width, _ = image.shape
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)
                
                x, y, w, h = int(left), int(top), int(right - left), int(bottom - top)
                center = (x + w // 2, y + h // 2)
                
                # Create a UIElement for the found icon
                found_elements.append(UIElement(
                    element_type=ElementType.ICON,
                    bbox=(x, y, w, h),
                    center=center,
                    confidence=scores[i],
                    attributes={"class_id": classes[i]}
                ))

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
