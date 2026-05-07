from ultralytics import YOLO
import math
import cv2
import os
import cvzone
import torch
from image_to_text import predict_number_plate
import easyocr
import csv
from datetime import datetime
import re

# ─── Auto-detect GPU/CPU ────────────────────────────────────────────
USE_GPU = torch.cuda.is_available()
DEVICE_STR = "cuda" if USE_GPU else "cpu"
device = torch.device(DEVICE_STR)
print(f"[INIT] Using device: {DEVICE_STR} (CUDA available: {USE_GPU})")

cap = cv2.VideoCapture(os.path.join(os.path.dirname(__file__), "../videos/22.mp4"))  # For videos

model = YOLO(os.path.join(os.path.dirname(__file__), "../runs/detect/train7/weights/best.pt")) # after training update the location of best.pt

classNames = ["with helmet", "without helmet", "rider", "number plate"]
num = 0
old_npconf = 0

# grab the width, height, and fps of the frames in the video stream.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter(os.path.join(os.path.dirname(__file__), '../results/output.mp4'), fourcc, fps, (frame_width, frame_height))

# Initialize EasyOCR (PaddleOCR has OneDNN crash on Windows)
ocr = easyocr.Reader(['en'], gpu=USE_GPU)


def is_valid_indian_number_plate(number_plate):

    # Regular expression pattern for Indian number plate format
    pattern = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
    return re.match(pattern, number_plate) is not None


def extract_and_store_number_plate(vehicle_number, conf, without_helmet_detected, 
                                   csv_file_path=os.path.join(os.path.dirname(__file__), '../results/number_plates.csv')):
    # Check if the number plate is valid, in Indian format, and a person without helmet is detected
    if vehicle_number and conf and without_helmet_detected and is_valid_indian_number_plate(vehicle_number):
        # Read existing entries to check for duplicates
        existing_entries = set()
        try:
            with open(csv_file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                existing_entries = set(row[0] for row in reader)
        except FileNotFoundError:
            pass  # File doesn't exist yet, which is fine

        # If the vehicle number is not a duplicate, add it to the CSV
        if vehicle_number not in existing_entries:
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:  # File is empty, write header
                    writer.writerow(['Vehicle Number', 'Confidence', 'Timestamp', 'Helmet Violation'])
                writer.writerow([vehicle_number, round(conf*100, 2),
                                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 "Yes"])  # Always "Yes" since we only record when without_helmet_detected is True
            print(f"New entry added: {vehicle_number}")
            return True
        else:
            print(f"Duplicate entry not added: {vehicle_number}")
    else:
        if not is_valid_indian_number_plate(vehicle_number):
            print(f"Invalid Indian number plate format: {vehicle_number}")
    return False

while True:
    success, img = cap.read()
    # Check if the frame was read successfully
    if not success:
        break
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device=DEVICE_STR)
    for r in results:
        boxes = r.boxes
        li = dict()
        rider_box = list()
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)
        try:
            new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
            # Get the indices of the rows where the value in column 1 is equal to 5.
            indices = torch.where(new_boxes[:, -1] == 2)
            # Select the rows where the mask is True.
            rows = new_boxes[indices]
            # Add rider details in the list
            for box in rows:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                rider_box.append((x1, y1, x2, y2))
        except:
            pass
        for i, box in enumerate(new_boxes):
            # Bounding box
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box[4] * 100)) / 100
            # Class Name
            cls = int(box[5])
            if classNames[cls] == "without helmet" and conf >= 0.5 or classNames[cls] == "rider" and conf >= 0.45 or \
                    classNames[cls] == "number plate" and conf >= 0.5:
                if classNames[cls] == "rider":
                    rider_box.append((x1, y1, x2, y2))
                if rider_box:
                    for j, rider in enumerate(rider_box):
                        if x1 + 10 >= rider_box[j][0] and y1 + 10 >= rider_box[j][1] and x2 <= rider_box[j][2] and \
                                y2 <= rider_box[j][3]:
                            # highlight or outline objects detected by object detection models
                            cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                            cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                               offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                            li.setdefault(f"rider{j}", [])
                            li[f"rider{j}"].append(classNames[cls])
                            if classNames[cls] == "number plate":
                                npx, npy, npw, nph, npconf = x1, y1, w, h, conf
                                crop = img[npy:npy + h, npx:npx + w]
                        if li:
                            for key, value in li.items():
                                if key == f"rider{j}":
                                    if len(list(set(li[f"rider{j}"]))) == 3:
                                        try:
                                            # crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) # for easy ocr
                                            vechicle_number, conf = predict_number_plate(crop, ocr)
                                            if vechicle_number and conf:
                                                cvzone.putTextRect(img, f"{vechicle_number} {round(conf*100, 2)}%",
                                                                   (x1, y1 - 50), scale=1.5, offset=10,
                                                                   thickness=2, colorT=(39, 40, 41),
                                                                   colorR=(105, 255, 255))
                                                # Check if anyone is detected without a helmet
                                                without_helmet_detected = any(
                                                    'without helmet' in rider_classes for rider_classes in li.values())

                                                # Extract and store the number plate information
                                                if extract_and_store_number_plate(vechicle_number, conf,
                                                                                  without_helmet_detected):
                                                    print(f"Number plate {vechicle_number} stored in CSV.")

                                        except Exception as e:
                                            print(e)
        # Display the frame
        output.write(img)
        cv2.imshow('Video', img)
        li = list()
        rider_box = list()

        # Exit the program if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            output.release()
            break
