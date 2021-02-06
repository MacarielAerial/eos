"""
Extracts pure text information from image files
"""

import cv2
from numpy import ndarray
from pytesseract import image_to_data


class OCRMiner:
    """
    Converts an image into a data structure acceptable to analytics functions
    """

    def __init__(self, img_path: str) -> None:
        self.img_path = img_path

    def _load_img(self) -> None:
        # Load image with cv2
        self.img_cv2: ndarray = cv2.imread(self.img_path)
        # Convert cv2's default BGR scheme to RGB scheme
        self.img_raw = cv2.cvtColor(self.img_cv2, cv2.COLOR_BGR2RGB)
        # Logging
        print(f"OCRMiner: Raw image loaded from {self.img_path}")

    def _preprocess_img(self) -> None:
        self.img_preprocessed: ndarray = ImagePreprocess.remove_noise(img=self.img_raw)
        # Logging
        print("OCRMiner: Image preprocessed")

    def _img_to_data(self, output_type: str = "dict") -> None:
        # Extract data from image
        self.img_data: str = image_to_data(
            image=self.img_preprocessed, output_type=output_type
        )
        # Logging
        print(f"OCRMiner: Image data extracted in the form of {output_type}")

    def orchestrate(self) -> None:
        self._load_img()
        self._preprocess_img()
        self._img_to_data()

    @property
    def input_path(self):
        return self.img_path

    @property
    def data(self):
        return self.img_data


class ImagePreprocess:
    """
    A collection of image preprocessing functions ready to be applied
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def remove_noise(img: ndarray) -> ndarray:
        """noise removal"""
        img_blurred: ndarray = cv2.medianBlur(img, 5)
        return img_blurred
