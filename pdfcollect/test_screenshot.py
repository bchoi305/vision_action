import pyautogui
import time
import os

print("Attempting to take screenshot...")
try:
    # Ensure the screenshots directory exists
    if not os.path.exists("test_screenshots"):
        os.makedirs("test_screenshots")

    screenshot = pyautogui.screenshot()
    screenshot.save("test_screenshots/test_screenshot.png")
    print("Screenshot taken successfully: test_screenshots/test_screenshot.png")
except Exception as e:
    print(f"Failed to take screenshot: {e}")
