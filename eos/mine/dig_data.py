"""
Extracts pure text information from image files
"""

import cv2
from pytesseract import image_to_string

class OCRMiner:
    """
    Converts an image into a data structure acceptable to analytics functions
    """
    def __init__(self, img_path: str):
        self.img_path = img_path

    @staticmethod
    def img_to_txt(img_path: str) -> str:
        img = cv2.imread(img_path)
        raw_str: str = image_to_string(img)
        return raw_str
