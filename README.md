# 🏍️ RoadWatch: Helmet Violation Detection & License Plate Recognition Realtime

An AI-powered traffic enforcement system designed for real-time monitoring of motorcycle helmet violations. This system utilizes deep learning to identify riders without helmets, recognizes their license plates via advanced OCR consensus logic, and logs violations with visual evidence.

---

## 🚀 Key Features

- **Real-Time Detection**: Powered by **YOLOv8** for high-speed identification of riders, helmets, and number plates.
- **Advanced OCR Consensus**: Implements a unique **Position-Level Voting System** that processes multiple frames to ensure 99%+ accuracy in plate recognition, effectively resolving character confusions like `D` vs `Q`.
- **Smart Filtering**: Automatically ignores duplicate readings and applies format-aware correction based on **Indian Standard License Plates** (XX00XX0000).
- **Automated Alerts**: Logs violations to JSON/CSV databases and supports **Email Notifications** with attached visual evidence.
- **Web Dashboard**: A modern, interactive dashboard built with **FastAPI** for easy video upload and violation monitoring.
- **Hardware Acceleration**: Optimized support for **NVIDIA (CUDA)** and **AMD (DirectML)** GPUs.

---

## 🛠️ Tech Stack

| Component | Technology | Version |
| :--- | :--- | :--- |
| **Language** | Python | **3.10.x** (Required) |
| **Backend** | FastAPI | 0.136.1 |
| **Object Detection** | YOLOv8 | 8.4.47 |
| **OCR Engine** | EasyOCR / PaddleOCR | 1.7.2 / 2.9.1 |
| **Deep Learning** | PyTorch | 2.4.1 |
| **Web UI** | Jinja2 / Starlette | 3.1.6 / 1.0.0 |
| **Hardware Access** | OpenCV / DirectML | 4.11 / 0.2.5 |

---

## ⚙️ Installation & Setup

### 1. Prerequisites

- **Python 3.10.x** is strictly required for compatibility with OCR engines and hardware acceleration layers.
- Git installed on your system.

### 2. Clone the Repository

```bash
git clone https://github.com/ChiragNSundar/Helmet-Violation-Detection-and-License-Plate-Recognition-Realtime.git
cd Helmet-Violation-Detection-and-License-Plate-Recognition-Realtime
```

### 3. Install Dependencies

```bash
py -3.10 -m pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory to configure your server and email settings:

```env
# Server
HOST=127.0.0.1
PORT=8000

# Email Alerts
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password
RECEIVER_EMAIL=authority-email@gmail.com

# Thresholds
CONFIDENCE_THRESHOLD=0.40
OCR_CONFIDENCE_THRESHOLD=0.30
```

---

## 🖥️ Running the Application

### Launching the Dashboard

To start the FastAPI web interface:

```bash
py -3.10 -m app.main
```

Access the system at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### Using the Standalone Module

For local testing of the detection logic via CLI:

```bash
py -3.10 "Training _module/main.py"
```

---

## 🧠 Core Intelligence

### 1. Spatial Association Logic

The system doesn't just detect objects; it associates them. It only triggers a violation if a `number plate` and a `no helmet` detection both overlap with a detected `rider` bounding box by at least 30%.

### 2. Consensus Voting System

To combat OCR noise and frame-by-frame variation, the system:

1. Accumulates all plate readings across the entire video.
2. Groups similar readings using string-distance heuristics.
3. Performs **Position-Level Voting** for each character in the plate.
4. Filters out "ghost" detections seen less than twice or with low confidence.

---

## 📂 Project Structure

```text
├── app/
│   ├── main.py              # FastAPI Application Entry
│   ├── video_processing.py   # Detection "Brain" & Consensus Logic
│   ├── routes.py            # API Endpoint Management
│   ├── models/              # Pre-trained YOLOv8 Weights
│   └── utils.py             # OCR Correction & Validation
├── Training _module/
│   ├── scripts/             # CLI Detection & Training scripts
│   ├── config/              # Dataset (YAML) & Class configurations
│   ├── docs/                # Module documentation & Assets
│   ├── results/             # Output videos & processed CSVs
│   ├── archive/             # Dataset Management
│   └── yolo-weights/        # Base YOLO models
├── requirements.txt         # Dependency Manifest
└── README.md                # Documentation
```

---

## ⭐️ Support & Contribution

If you find this project useful for your research or implementation, please give it a star! ⭐️

## 📧 Contact

For dataset access or professional inquiries, reach out on **[LinkedIn](https://www.linkedin.com/in/chirag-n-sundar/)**.

<https://www.loom.com/share/67e87037e4884eab8ab88ee5439de8d4>
