import cv2
import os
import torch
from datetime import datetime
import csv
import re
import easyocr
from ultralytics import YOLO
import cvzone
import math
from image_to_text import predict_number_plate
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from app.utils import apply_weather_resilience

# ─── Auto-detect GPU/CPU ────────────────────────────────────────────
USE_GPU = torch.cuda.is_available()
DEVICE_STR = "cuda" if USE_GPU else "cpu"
device = torch.device(DEVICE_STR)
print(f"[INIT] Using device: {DEVICE_STR} (CUDA available: {USE_GPU})")

classNames = ["with helmet", "without helmet", "rider", "number plate"]
ocr = easyocr.Reader(['en'], gpu=USE_GPU)  # Initialize EasyOCR

# YOLO Model
model = YOLO(os.path.join(os.path.dirname(__file__), "../runs/detect/train7/weights/best.pt"))  # Replace with the actual path to the YOLO model

# Helper Functions
def is_valid_indian_number_plate(number_plate):
    """Validates Indian number plate format."""
    pattern = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
    return re.match(pattern, number_plate) is not None


def extract_and_store_number_plate(vehicle_number, conf, without_helmet_detected,
                                   csv_file_path=os.path.join(os.path.dirname(__file__), '../results/number_plates.csv')):
    """Extracts and stores valid number plate details."""
    print(
        f"Debug: Received values - Vehicle Number: {vehicle_number}, Confidence: {conf}, Without Helmet: {without_helmet_detected}")

    if vehicle_number and conf and without_helmet_detected and is_valid_indian_number_plate(vehicle_number):
        print(f"Debug: Passed validation checks for {vehicle_number}")
        existing_entries = set()

        try:
            # Safely read the CSV file
            with open(csv_file_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader, None)  # Skip the header row if present
                if header is not None:
                    print(f"Debug: CSV Header - {header}")
                    existing_entries = {row[0] for row in reader if row}  # Only non-empty rows
                print(f"Debug: Existing entries in CSV: {existing_entries}")
        except FileNotFoundError:
            print(f"Debug: CSV file not found. A new file will be created.")
        except Exception as e:
            print(f"Debug: Error reading CSV file - {e}")

        # Write the number plate information if not duplicate
        if vehicle_number not in existing_entries:
            try:
                with open(csv_file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    if file.tell() == 0:  # Add header if file is empty
                        writer.writerow(['Vehicle Number', 'Confidence', 'Timestamp', 'Helmet Violation'])
                    writer.writerow([vehicle_number, round(conf * 100, 2),
                                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                     "Yes"])
                print(f"New entry added: {vehicle_number}")
                return True
            except Exception as e:
                print(f"Debug: Error writing to CSV file - {e}")
        else:
            print(f"Debug: Duplicate entry not added: {vehicle_number}")
    else:
        print(
            f"Debug: Failed validation - Vehicle Number: {vehicle_number}, Confidence: {conf}, Without Helmet: {without_helmet_detected}")
    return False


def process_frame(img):
    """Processes a single frame for helmet detection and number plate extraction."""
    if os.getenv("ENABLE_WEATHER_RESILIENCE", "true").lower() == "true":
        img = apply_weather_resilience(img)
        
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device=DEVICE_STR)
    li = dict()
    rider_box = []

    for r in results:
        boxes = r.boxes
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)
        new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]

        try:
            indices = torch.where(new_boxes[:, -1] == 2)  # Rider detection
            rows = new_boxes[indices]
            for box in rows:
                x1, y1, x2, y2 = map(int, box[:4])
                rider_box.append((x1, y1, x2, y2))
        except:
            pass

        for i, box in enumerate(new_boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            conf = round(float(box[4]) * 100) / 100
            cls = int(box[5])

            if classNames[cls] in ["without helmet", "rider", "number plate"] and conf >= 0.5:
                if classNames[cls] == "rider":
                    rider_box.append((x1, y1, x2, y2))

                for j, rider in enumerate(rider_box):
                    if x1 + 10 >= rider[0] and y1 + 10 >= rider[1] and x2 <= rider[2] and y2 <= rider[3]:
                        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                        cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                           offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                        li.setdefault(f"rider{j}", []).append(classNames[cls])

                        if classNames[cls] == "number plate":
                            crop = img[y1:y1 + h, x1:x1 + w]
                            if len(set(li[f"rider{j}"])) == 3:
                                try:
                                    vechicle_number, conf = predict_number_plate(crop, ocr)
                                    if vechicle_number and conf:
                                        cvzone.putTextRect(img, f"{vechicle_number} {round(conf * 100, 2)}%",
                                                           (x1, y1 - 50), scale=1.5, offset=10,
                                                           thickness=2, colorT=(39, 40, 41), colorR=(105, 255, 255))
                                        without_helmet_detected = any(
                                            'without helmet' in rider_classes for rider_classes in li.values())
                                        extract_and_store_number_plate(vechicle_number, conf, without_helmet_detected)
                                except Exception as e:
                                    print(e)
    return img


def detect_realtime_camera():
    """Detects helmet violations in real-time using webcam."""
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        frame = process_frame(frame)
        cv2.imshow("Helmet Detection - Real-Time", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_from_video(video_path):
    """Detects helmet violations from a video file."""
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


if __name__ == "__main__":
    choice = input("Enter '1' for real-time camera detection or '2' to process a video file: ")
    if choice == '1':
        detect_realtime_camera()
    elif choice == '2':
        video_path = os.path.join(os.path.dirname(__file__), "../videos/22.mp4")
        detect_from_video(video_path)
    else:
        print("Invalid input. Exiting program.")