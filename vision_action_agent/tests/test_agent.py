import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from vision_action_agent.src.core.agent import VisionActionAgent, UIElement, ElementType

class TestVisionActionAgent(unittest.TestCase):

    def setUp(self):
        """Set up a mock environment for the agent."""
        with patch('vision_action_agent.src.core.agent.config_manager'):
            with patch('vision_action_agent.src.core.agent.ScreenCapture'):
                with patch('vision_action_agent.src.core.agent.OCRTextDetector'):
                    with patch('vision_action_agent.src.core.agent.ElementDetector') as mock_element_detector:
                        with patch('vision_action_agent.src.core.agent.MouseController'):
                            with patch('vision_action_agent.src.core.agent.KeyboardController'):
                                with patch('vision_action_agent.src.core.agent.WorkflowExecutor'):
                                    self.agent = VisionActionAgent()
                                    self.agent.element_detector = mock_element_detector()

    def test_find_element_by_image_template(self):
        """Test that find_element correctly calls find_element_by_image when an image_template is provided."""
        # Arrange
        mock_screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        self.agent.capture_screen = MagicMock(return_value=mock_screenshot)
        
        expected_elements = [UIElement(element_type=ElementType.ICON, bbox=(10, 10, 20, 20), center=(20, 20), confidence=0.9, attributes={'template_name': 'test_icon'})]
        self.agent.element_detector.find_element_by_image = MagicMock(return_value=expected_elements)

        # Act
        found_elements = self.agent.find_element(image_template='test_icon')

        # Assert
        self.agent.element_detector.find_element_by_image.assert_called_once_with(mock_screenshot, 'test_icon')
        self.assertEqual(found_elements, expected_elements)

    def test_find_element_by_icon_detection(self):
        """Test that find_element correctly calls detect_icons_from_tf_hub when the element_type is ICON."""
        # Arrange
        mock_screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        self.agent.capture_screen = MagicMock(return_value=mock_screenshot)
        
        expected_elements = [UIElement(element_type=ElementType.ICON, bbox=(10, 10, 20, 20), center=(20, 20), confidence=0.9, attributes={'class_id': 1})]
        self.agent.element_detector.detect_icons_from_tf_hub = MagicMock(return_value=expected_elements)

        # Act
        found_elements = self.agent.find_element(element_type=ElementType.ICON)

        # Assert
        self.agent.element_detector.detect_icons_from_tf_hub.assert_called_once_with(mock_screenshot)
        self.assertEqual(found_elements, expected_elements)

if __name__ == '__main__':
    unittest.main()