import cv2
import os
import csv
import torch
from datetime import datetime
from collections import Counter
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
USE_GPU = False
DEVICE_STR = "cpu"

if torch.cuda.is_available():
    USE_GPU = True
    DEVICE_STR = "cuda"
    print(f"[INIT] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    try:
        import torch_directml
        gpu_name = torch_directml.device_name(0)
        print(f"[INIT] AMD/Intel GPU detected via DirectML: {gpu_name}")
        print(f"[INIT] Note: YOLO runs on CPU (Ultralytics doesn't support DirectML)")
    except ImportError:
        pass
    print(f"[INIT] Using device: CPU")

# ─── Constants ───────────────────────────────────────────────────────
classNames = ["with helmet", "without helmet", "rider", "number plate"]

# ─── Initialize OCR ─────────────────────────────────────────────────
print("[INIT] Loading EasyOCR reader...")
ocr = easyocr.Reader(['en'], gpu=USE_GPU)
print("[INIT] EasyOCR reader loaded.")

# ─── Initialize YOLO ─────────────────────────────────────────────────
model = YOLO("app/models/yolov8_best.pt")


def save_to_csv(vehicle_number, conf):
    """Save violation to number_plates.csv."""
    try:
        file_exists = os.path.exists(CSV_FILE_PATH)
        existing_entries = set()
        
        if file_exists:
            try:
                with open(CSV_FILE_PATH, 'r') as f:
                    reader = csv.reader(f)
                    next(reader, None)
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
            print(f"[CSV] Logged: {vehicle_number}")
            return True
        return False
    except Exception as e:
        print(f"[CSV] Error: {e}")
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


def _plate_distance(a, b):
    """Count character differences between two plate strings."""
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)


def _group_and_vote(plate_readings):
    """Group similar plate readings and return the best candidate per group.
    
    Groups plates that differ by ≤ 2 characters, then picks the most
    frequent reading in each group (ties broken by highest confidence).
    
    Args:
        plate_readings: list of (plate_text, confidence, frame_img)
    
    Returns:
        list of (best_plate, best_confidence, best_frame) — one per unique vehicle
    """
    if not plate_readings:
        return []
    
    # Sort by frequency first — most common readings are more likely correct
    plate_counts = Counter(p[0] for p in plate_readings)
    
    # Build groups of similar plates
    groups = []  # list of lists of (plate, conf, img)
    used = set()
    
    # Process most frequent plates first
    for plate, _count in plate_counts.most_common():
        if plate in used:
            continue
        
        group = []
        for p_text, p_conf, p_img in plate_readings:
            if p_text not in used and _plate_distance(plate, p_text) <= 2:
                group.append((p_text, p_conf, p_img))
                used.add(p_text)
        
        if group:
            groups.append(group)
    
    # For each group, pick the best candidate
    results = []
    for group in groups:
        # Count occurrences of each reading in the group
        reading_counts = Counter(p[0] for p in group)
        # Find the most common reading
        best_plate = reading_counts.most_common(1)[0][0]
        # Get the highest confidence instance of that reading
        best_conf = max(p[1] for p in group if p[0] == best_plate)
        best_img = next(p[2] for p in group if p[0] == best_plate and p[1] == best_conf)
        
        # Only accept if seen at least 2 times (across all similar readings)
        total_sightings = len(group)
        if total_sightings >= 2 and best_conf >= 0.40:
            results.append((best_plate, best_conf, best_img, total_sightings))
            print(f"[VOTE] Plate '{best_plate}' — {total_sightings} sightings, "
                  f"best conf={best_conf:.2f}, {len(reading_counts)} variants")
        else:
            print(f"[VOTE] Rejected '{best_plate}' — only {total_sightings} sighting(s), "
                  f"conf={best_conf:.2f}")
    
    return results


def process_frame(img, plate_accumulator):
    """Process a single frame: detect riders, helmets, plates.
    
    Instead of logging immediately, appends valid plate readings to
    plate_accumulator for later consensus voting.
    
    Args:
        img: BGR frame from video
        plate_accumulator: list to append (plate_text, confidence, frame_img) to
    
    Returns:
        Annotated frame for display
    """
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
        
        riders = []
        helmets = []
        plates = []
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, xy[i].tolist())
            conf = float(confidences[i])
            cls = int(classes[i])
            
            if conf < 0.25:
                continue
                
            if cls == 2:
                riders.append((x1, y1, x2, y2, conf))
            elif cls == 1:
                helmets.append((x1, y1, x2, y2, conf))
            elif cls == 3:
                plates.append((x1, y1, x2, y2, conf))
        
        for rider_box in riders:
            rx1, ry1, rx2, ry2, rconf = rider_box
            rw, rh = rx2 - rx1, ry2 - ry1
            
            cvzone.cornerRect(img, (rx1, ry1, rw, rh), l=15, rt=5, colorR=(255, 0, 0))
            cvzone.putTextRect(img, "RIDER", (rx1 + 10, ry1 - 10), scale=1.5,
                               offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
            
            # Find "without helmet" for this rider
            has_no_helmet = False
            for hbox in helmets:
                overlap = _bbox_overlap(hbox[:4], rider_box[:4])
                if overlap > 0.3:
                    has_no_helmet = True
                    hx1, hy1, hx2, hy2 = hbox[:4]
                    hw, hh = hx2 - hx1, hy2 - hy1
                    cvzone.cornerRect(img, (hx1, hy1, hw, hh), l=15, rt=5, colorR=(255, 0, 0))
                    cvzone.putTextRect(img, "NO HELMET", (hx1 + 10, hy1 - 10), scale=1.5,
                                       offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                    break
            
            if not has_no_helmet:
                continue
            
            # Find plate for this rider
            for pbox in plates:
                overlap = _bbox_overlap(pbox[:4], rider_box[:4])
                if overlap > 0.3:
                    px1, py1, px2, py2, pconf = pbox
                    pw, ph = px2 - px1, py2 - py1
                    
                    # Crop from CLEAN image with padding
                    img_h, img_w = clean_img.shape[:2]
                    pad_y = max(ph, 20)
                    pad_x = max(pw // 4, 10)
                    cy1 = max(0, py1 - pad_y)
                    cy2 = min(img_h, py2 + pad_y)
                    cx1 = max(0, px1 - pad_x)
                    cx2 = min(img_w, px2 + pad_x)
                    crop = clean_img[cy1:cy2, cx1:cx2].copy()
                    
                    # Draw plate box
                    cvzone.cornerRect(img, (px1, py1, pw, ph), l=15, rt=5, colorR=(255, 0, 0))
                    cvzone.putTextRect(img, "PLATE", (px1 + 10, py1 - 10), scale=1.5,
                                       offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                    
                    if crop.size == 0:
                        continue
                    
                    try:
                        vehicle_number, ocr_conf = predict_number_plate(crop, ocr)
                        if vehicle_number and ocr_conf:
                            cvzone.putTextRect(img, f"{vehicle_number}", (px1, py1 - 50),
                                               scale=1.5, offset=10, thickness=2,
                                               colorT=(39, 40, 41), colorR=(105, 255, 255))
                            
                            if is_valid_indian_number_plate(vehicle_number):
                                # Don't log yet — accumulate for voting
                                plate_accumulator.append((vehicle_number, ocr_conf, img.copy()))
                    except Exception as e:
                        pass  # OCR errors are non-critical
                    break
    return img


def _finalize_violations(plate_accumulator):
    """After video is fully processed, vote on plates and log violations."""
    if not plate_accumulator:
        print("[FINAL] No plate readings accumulated.")
        return
    
    print(f"[FINAL] Processing {len(plate_accumulator)} total plate readings...")
    
    # Group and vote
    winners = _group_and_vote(plate_accumulator)
    
    if not winners:
        print("[FINAL] No plates passed the voting threshold.")
        return
    
    for plate, conf, frame_img, sightings in winners:
        print(f"[VIOLATION] Confirmed: {plate} (conf={conf:.2f}, seen {sightings}x)")
        
        # Save to CSV
        save_to_csv(plate, conf)
        
        # Log to violations.json
        if not check_daily_violation(plate):
            violation_image = save_violation_image(frame_img, plate)
            result = log_violation(plate, "Without Helmet", violation_image)
            if result:
                print(f"[JSON] Logged: {plate}")
            
            # Email (best-effort)
            try:
                send_violation_email(plate, violation_image)
            except Exception:
                pass
    
    print(f"[FINAL] Done. {len(winners)} violation(s) logged.")


def process_video_web(video_path):
    """Process video for the web app — no GUI, with consensus voting.
    
    This is the main entry point for the FastAPI route. It:
    1. Processes every frame, accumulating plate readings
    2. After the video ends, runs consensus voting
    3. Logs only the confirmed violations
    """
    plate_accumulator = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[VIDEO] Processing: {total_frames} frames @ {fps:.1f} FPS")
    
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        # Process every frame for detection
        process_frame(frame, plate_accumulator)
        
        # Progress update every 20 frames
        if frame_count % 20 == 0:
            print(f"[VIDEO] Frame {frame_count}/{total_frames} "
                  f"({frame_count/total_frames*100:.0f}%) — "
                  f"{len(plate_accumulator)} plate readings so far")
    
    cap.release()
    print(f"[VIDEO] Finished processing {frame_count} frames.")
    
    # NOW finalize: vote and log
    _finalize_violations(plate_accumulator)


def process_video(video_path):
    """Process video with GUI display (standalone mode)."""
    plate_accumulator = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame, plate_accumulator)
        cv2.imshow("Helmet Detection - Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Finalize after video ends
    _finalize_violations(plate_accumulator)