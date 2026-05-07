import cv2
import os
import csv
import torch
from datetime import datetime
import easyocr
from ultralytics import YOLO
import cvzone
from app.utils import (
    predict_number_plate, 
    send_violation_email, 
    is_valid_indian_number_plate, 
    check_daily_violation, 
    save_violation_image
)
from app.db import log_violation

# Project root directory (where main.py is run from)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE_PATH = os.path.join(PROJECT_ROOT, "number_plates.csv")

# ─── Device Detection ───────────────────────────────────────────────
# CUDA (NVIDIA) > DirectML (AMD/Intel on Windows) > CPU
USE_GPU = False
DEVICE_STR = "cpu"

if torch.cuda.is_available():
    USE_GPU = True
    DEVICE_STR = "cuda"
    print(f"[INIT] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    try:
        import torch_directml
        dml_device = torch_directml.device()
        gpu_name = torch_directml.device_name(0)
        # Note: YOLO/Ultralytics doesn't support DirectML, so YOLO stays on CPU
        # But DirectML is available for custom tensor ops
        print(f"[INIT] AMD/Intel GPU detected via DirectML: {gpu_name}")
        print(f"[INIT] Note: YOLO runs on CPU (Ultralytics doesn't support DirectML)")
    except ImportError:
        pass
    print(f"[INIT] Using device: CPU")

device = torch.device("cpu")  # Tensor ops device (for box calculations)

# ─── Constants ───────────────────────────────────────────────────────
classNames = ["with helmet", "without helmet", "rider", "number plate"]

# ─── Initialize OCR (EasyOCR — PaddleOCR has OneDNN bug on Windows) ─
print("[INIT] Loading EasyOCR reader...")
ocr = easyocr.Reader(['en'], gpu=USE_GPU)
print("[INIT] EasyOCR reader loaded.")

# ─── Initialize YOLO ─────────────────────────────────────────────────
model = YOLO("app/models/yolov8_best.pt")

# Track already-logged plates to avoid spamming per-frame
_logged_plates = set()


def save_to_csv(vehicle_number, conf):
    """Save violation to number_plates.csv for backward compatibility."""
    try:
        file_exists = os.path.exists(CSV_FILE_PATH)
        existing_entries = set()
        
        if file_exists:
            try:
                with open(CSV_FILE_PATH, 'r') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    existing_entries = {row[0] for row in reader if row}
            except Exception:
                pass
        
        if vehicle_number not in existing_entries:
            with open(CSV_FILE_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(CSV_FILE_PATH) == 0:
                    writer.writerow(['Vehicle Number', 'Confidence', 'Timestamp', 'Helmet Violation'])
                writer.writerow([
                    vehicle_number,
                    round(conf * 100, 2),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Yes"
                ])
            print(f"[CSV] New entry added: {vehicle_number}")
            return True
        else:
            return False
    except Exception as e:
        print(f"[CSV] Error saving to CSV: {e}")
        return False


def _bbox_overlap(box1, box2):
    """Calculate what fraction of box1's area overlaps with box2."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    
    if box1_area == 0:
        return 0.0
    
    return intersection / box1_area


def process_frame(img):
    """Processes a single frame for helmet detection and number plate extraction.
    
    Two-pass approach:
    1. Collect all detections by class from YOLO output
    2. Associate detections to riders using overlap-based matching
    """
    # Keep a clean copy for OCR cropping (before any labels are drawn)
    clean_img = img.copy()
    
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device="cpu")

    for r in results:
        boxes = r.boxes
        if len(boxes) == 0:
            continue
            
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        
        # Collect all detections by class
        riders = []
        helmets = []     # "without helmet" detections
        plates = []      # "number plate" detections
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, xy[i].tolist())
            conf = float(confidences[i])
            cls = int(classes[i])
            
            if conf < 0.25:
                continue
                
            if cls == 2:    # rider
                riders.append((x1, y1, x2, y2, conf))
            elif cls == 1:  # without helmet
                helmets.append((x1, y1, x2, y2, conf))
            elif cls == 3:  # number plate
                plates.append((x1, y1, x2, y2, conf))
        
        # For each rider, find associated without-helmet and number-plate
        for rider_box in riders:
            rx1, ry1, rx2, ry2, rconf = rider_box
            rw, rh = rx2 - rx1, ry2 - ry1
            
            # Draw rider bounding box
            cvzone.cornerRect(img, (rx1, ry1, rw, rh), l=15, rt=5, colorR=(255, 0, 0))
            cvzone.putTextRect(img, "RIDER", (rx1 + 10, ry1 - 10), scale=1.5,
                               offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
            
            # Find "without helmet" associated with this rider (overlap > 30%)
            has_no_helmet = False
            for hbox in helmets:
                overlap = _bbox_overlap(hbox[:4], rider_box[:4])
                if overlap > 0.3:
                    has_no_helmet = True
                    hx1, hy1, hx2, hy2 = hbox[:4]
                    hw, hh = hx2 - hx1, hy2 - hy1
                    cvzone.cornerRect(img, (hx1, hy1, hw, hh), l=15, rt=5, colorR=(255, 0, 0))
                    cvzone.putTextRect(img, "WITHOUT HELMET", (hx1 + 10, hy1 - 10), scale=1.5,
                                       offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                    break
            
            if not has_no_helmet:
                continue
            
            # Find number plate associated with this rider
            for pbox in plates:
                overlap = _bbox_overlap(pbox[:4], rider_box[:4])
                if overlap > 0.3:
                    px1, py1, px2, py2, pconf = pbox
                    pw, ph = px2 - px1, py2 - py1
                    
                    # Crop from CLEAN image (no labels drawn) with padding
                    img_h, img_w = clean_img.shape[:2]
                    pad_y = max(ph, 20)
                    pad_x = max(pw // 4, 10)
                    crop_y1 = max(0, py1 - pad_y)
                    crop_y2 = min(img_h, py2 + pad_y)
                    crop_x1 = max(0, px1 - pad_x)
                    crop_x2 = min(img_w, px2 + pad_x)
                    crop = clean_img[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    # Draw plate label on display image
                    cvzone.cornerRect(img, (px1, py1, pw, ph), l=15, rt=5, colorR=(255, 0, 0))
                    cvzone.putTextRect(img, "NUMBER PLATE", (px1 + 10, py1 - 10), scale=1.5,
                                       offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                    
                    if crop.size == 0:
                        continue
                    
                    try:
                        vehicle_number, ocr_conf = predict_number_plate(crop, ocr)
                        if vehicle_number and ocr_conf:
                            cvzone.putTextRect(img, f"{vehicle_number} {round(ocr_conf * 100, 2)}%",
                                               (px1, py1 - 50), scale=1.5, offset=10,
                                               thickness=2, colorT=(39, 40, 41), colorR=(105, 255, 255))
                            
                            if is_valid_indian_number_plate(vehicle_number):
                                # Avoid spamming the same plate every frame
                                if vehicle_number not in _logged_plates:
                                    print(f"[VIOLATION] Detected: {vehicle_number}")
                                    _logged_plates.add(vehicle_number)
                                    
                                    # Save to CSV
                                    save_to_csv(vehicle_number, ocr_conf)
                                    
                                    # Log to violations.json (once per day)
                                    if not check_daily_violation(vehicle_number):
                                        violation_image = save_violation_image(img, vehicle_number)
                                        result = log_violation(vehicle_number, "Without Helmet", violation_image)
                                        if result:
                                            print(f"[JSON] Violation logged: {vehicle_number}")
                                        else:
                                            print(f"[JSON] Failed to log: {vehicle_number}")
                                        
                                        # Send email (best-effort)
                                        try:
                                            send_violation_email(vehicle_number, violation_image)
                                        except Exception as email_err:
                                            print(f"[EMAIL] Failed (non-critical): {email_err}")
                            else:
                                print(f"[PLATE] Invalid format: {vehicle_number}")
                    except Exception as e:
                        print(f"[ERROR] Number plate processing: {e}")
                    break  # One plate per rider
    return img

def process_video(video_path):
    """Detects helmet violations from a video file."""
    global _logged_plates
    _logged_plates = set()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame)
        cv2.imshow("Helmet Detection - Video", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()     