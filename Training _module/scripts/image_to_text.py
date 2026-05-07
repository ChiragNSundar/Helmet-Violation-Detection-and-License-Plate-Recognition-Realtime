import easyocr
import csv
import os
import numpy as np
import cv2
import re


def predict_number_plate(img, ocr):
    """Predict number plate text using EasyOCR.
    
    Args:
        img: Cropped number plate image (numpy array)
        ocr: EasyOCR Reader instance
    
    Returns:
        tuple: (cleaned_text, confidence) or (None, None)
    """
    try:
        result = ocr.readtext(img)
        
        if not result:
            return None, None
        
        text = result[0][1]
        score = result[0][2]
        
        cleaned_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
        
        if (score * 100) >= 70 and len(cleaned_text) == 10:
            return cleaned_text, score
        
        return None, None
    
    except Exception as e:
        print(f"Number plate prediction error: {e}")
        return None, None
