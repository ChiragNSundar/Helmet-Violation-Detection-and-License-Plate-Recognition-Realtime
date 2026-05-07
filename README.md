# 🏍️ Helmet Violation Detection & License Plate Recognition Realtime

This project is an **AI-powered helmet violation detection system** that uses **YOLOv8** for real-time object detection and **PaddleOCR** for license plate recognition. It automatically identifies riders without helmets, crops their number plates, extracts the registration number, and logs the violation with a timestamp and visual evidence.

---

## ⚡ Project Overview  

This project automates traffic rule enforcement by:  
✅ **Detecting Riders Without Helmets**  
✅ **Recognizing License Plates (OCR)**  
✅ **Validating Number Plate Formats (Indian Standard)**  
✅ **Storing Violations in a Database**  
✅ **Sending Email Alerts to Authorities**  
✅ **FastAPI-powered Web Dashboard**
---

## 🛠️ Tech Stack  

| Component | Technology |  
|-----------|------------|  
| **Backend** | FastAPI |  
| **Object Detection** | YOLOv8 (Ultralytics) |  
| **OCR** | PaddleOCR |  
| **Logic** | Python 3.10, OpenCV, cvzone, Torch |  
| **Database** | JSON / CSV Logging |  
| **Web Interface** | Jinja2, HTML, CSS |  

---

## 🚀 Installation & Setup

### 1. Prerequisites

- **Python 3.10** (Recommended for compatibility)
- Windows / Linux / macOS

### 2. Clone the Repository

```bash
git clone https://github.com/ChiragNSundar/Helmet-Violation-Detection-and-License-Plate-Recognition-Realtime.git
cd Helmet-Violation-Detection-and-License-Plate-Recognition-Realtime
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

### Running the Web Dashboard

To start the FastAPI web interface, run:

```bash
python -m app.main
```

Once started, access the dashboard at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### Running the Training Module Scripts

If you want to run the standalone detection scripts inside the `Training _module`:

```bash
python "Training _module/main.py"
```

*Note: This will process the sample video in `Training _module/videos/22.mp4` and generate an `output.mp4`.*

---

## 🧠 Training & Model Development

### Dataset

The model was trained on a comprehensive dataset containing annotated images of riders, helmets, and number plates.

- **Total Classes:** 4 (`with helmet`, `without helmet`, `rider`, `number plate`)
- **Source:** Kaggle Dataset

### How to Train

1. **Prepare Data:** Place your images and labels in the `Training _module/archive/` folder.
2. **Configure YAML:** Update `Training _module/coco128.yaml` with the correct paths.
3. **Run Training:**

   ```bash
   python "Training _module/training.py"
   ```

4. **Update Weights:** Once training is complete, copy the `best.pt` file from the `runs/` directory to `app/models/yolov8_best.pt`.

---

## 🏗️ System Architecture

### 1. Object Detection (YOLOv8)

The system uses YOLOv8 to detect four classes. It implements **Spatial Association Logic** to ensure that a helmet (or lack thereof) and a number plate are correctly linked to a specific rider by checking for bounding box overlaps.

### 2. Optical Character Recognition (PaddleOCR)

When a violation (no helmet) is confirmed, the system crops the detected number plate area and passes it to **PaddleOCR**.

### 3. Validation Logic

The extracted text is passed through a **Regular Expression** filter to validate it against Indian number plate formats:
`pattern = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'`

---

## 📂 Project Structure

```text
├── app/
│   ├── main.py              # FastAPI Entry Point
│   ├── video_processing.py   # Core Detection & Logic
│   ├── routes.py            # API Routes
│   ├── models/              # Pre-trained YOLO weights
│   └── templates/           # Web Interface
├── Training _module/
│   ├── training.py          # Script to train YOLO
│   ├── main.py              # Standalone detection script
│   ├── test.py              # Testing script for camera/video
│   └── archive/             # Dataset folder
├── requirements.txt         # Project dependencies
└── README.md                # Project Documentation
```

---

## 🎥 Demo

![Demo](Training%20_module/bike.gif)

## ⭐️ Support

If you find this project useful, please give it a star! ⭐️

## 📧 Contact

For more information or dataset access, contact me on **[LinkedIn](https://www.linkedin.com/in/chirag-n-sundar/)**.
