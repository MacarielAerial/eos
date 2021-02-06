"""
Test the end to end capability for OCRMiner
"""
from eos.mine.dig_data import OCRMiner

img_path: str = "tests/data/integration/ocr_miner/tess_sample_receipt_1.jpeg"

def test_ocr_miner():
    ocr_miner_obj = OCRMiner(img_path = img_path)
    ocr_miner_obj.orchestrate()
    assert True
