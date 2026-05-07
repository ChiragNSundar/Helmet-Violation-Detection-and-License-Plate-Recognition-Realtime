import re
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email import encoders
import datetime
import cv2
from app.db import read_violations_by_vehicle
from email.mime.base import MIMEBase

def _correct_ocr_plate(text):
    """Apply OCR error correction for Indian number plate format.
    
    Indian format: XX00XX0000 (2 letters, 2 digits, 2 letters, 4 digits)
    Common OCR confusions: 0↔O↔Q, 1↔I↔L, 5↔S, 8↔B, 4↔A
    """
    if len(text) != 10:
        return text
    
    # Position rules: L=letter expected, D=digit expected
    pattern = "LLDDLLDDDD"
    
    # Character correction maps
    to_letter = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
    to_digit = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'A': '4', 'D': '0'}
    
    corrected = list(text)
    for i, (char, expected) in enumerate(zip(text, pattern)):
        if expected == 'L' and char.isdigit():
            corrected[i] = to_letter.get(char, char)
        elif expected == 'D' and char.isalpha():
            corrected[i] = to_digit.get(char.upper(), char)
    
    return ''.join(corrected)


def is_valid_indian_number_plate(vehicle_number):
    """Validates Indian number plate format."""
    pattern = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
    return re.match(pattern, vehicle_number) is not None


def check_daily_violation(vehicle_number):
    """Check if violation for this vehicle has already been logged today."""
    try:
        violations = read_violations_by_vehicle(vehicle_number)
        today = datetime.date.today()
        daily_violations = [v for v in violations if datetime.datetime.strptime(v['timestamp'], "%Y-%m-%d %H:%M:%S").date() == today]
        return len(daily_violations) > 0
    except Exception as e:
        print(f"Error checking daily violations: {e}")
        return False

def save_violation_image(frame, vehicle_number):
    """Save violation image with timestamp."""
    violation_dir = "violation_images"
    os.makedirs(violation_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{violation_dir}/{vehicle_number}_{timestamp}.jpg"
    
    cv2.imwrite(filename, frame)
    return filename

def predict_number_plate(img, ocr):
    """Predict and clean vehicle number plate using EasyOCR.
    
    Args:
        img: Cropped number plate image (numpy array)
        ocr: EasyOCR Reader instance
    
    Returns:
        tuple: (cleaned_text, confidence) or (None, None)
    """
    try:
        # Upscale small crops for better OCR accuracy
        h, w = img.shape[:2]
        min_height = 100
        if h < min_height:
            scale = min_height / h
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # EasyOCR readtext returns list of (bbox, text, confidence)
        result = ocr.readtext(gray)
        
        if not result:
            print(f"[OCR] No text detected in crop ({img.shape})")
            return None, None
        
        # Combine all detected text segments
        all_texts = [(r[1], r[2]) for r in result]
        combined_text = ''.join([t[0] for t in all_texts])
        avg_score = sum([t[1] for t in all_texts]) / len(all_texts)
        
        # Clean: keep only uppercase alphanumeric chars
        cleaned_text = ''.join(re.findall(r'[A-Z0-9]', combined_text.upper()))
        
        print(f"[OCR] Raw: '{combined_text}' -> Cleaned: '{cleaned_text}' (len={len(cleaned_text)}, conf={avg_score:.2f})")
        
        # Indian plates are 10 chars, but allow some flexibility (9-11)
        if avg_score >= 0.3 and 9 <= len(cleaned_text) <= 11:
            # Try to extract exactly 10 chars if possible
            if len(cleaned_text) > 10:
                cleaned_text = cleaned_text[:10]
            
            # Return raw cleaned text — correction is done post-voting
            return cleaned_text, avg_score
        
        return None, None
    
    except Exception as e:
        print(f"Number plate prediction error: {e}")
        return None, None


def send_violation_email(vehicle_number, violation_image):
    """Send email notification for helmet violation."""
    sender_email = "sender-email"
    sender_password = ""  # Use App Password for Gmail
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = "receiver-email"
    msg['Subject'] = "Helmet Violation Detected"
    
    body = f"""
    Helmet Violation Detected!
    
    Vehicle Number: {vehicle_number}
    Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Violation Type: No Helmet
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with open(violation_image, 'rb') as file:
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(file.read())
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(violation_image)}"')
        msg.attach(attachment)
    except Exception as e:
        print(f"Error attaching image: {e}")
        return

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("Violation email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")