# 🤖 Agent Context & Memory (agents.md)

This file serves as a persistent context guide for AI coding assistants (Claude, Gemini, etc.) to understand the codebase, architectural decisions, and known constraints of the **Helmet Violation Detection & License Plate Recognition** project.

---

## 🎯 Project Purpose
An AI system designed to detect motorcycle riders without helmets in real-time, recognize their license plates using OCR, and log violations for enforcement.

## 🛠️ Core Tech Stack
- **Framework:** FastAPI (Backend), Jinja2 (Templates)
- **Object Detection:** YOLOv8 (Ultralytics)
- **OCR:** PaddleOCR (for number plate extraction)
- **Language:** Python 3.10 (Required for specific dependency stability)
- **Libraries:** OpenCV, cvzone, Torch, Starlette

---

## 📂 Key Directory Map
- `/app`: The main FastAPI application.
    - `main.py`: Entry point, server configuration, and root routing.
    - `video_processing.py`: The "Brain" of the real-time detection logic.
    - `routes.py`: API endpoints for video upload and processing.
    - `models/`: Stores the primary YOLO weights (`yolov8_best.pt`).
- `/Training _module`: Standalone environment for model development.
    - `training.py`: Script to fine-tune YOLO on the custom dataset.
    - `main.py` / `test.py`: Standalone CLI-based detection scripts for local testing.
    - `coco128.yaml`: Dataset configuration.

---

## 🧠 Critical Logic Patterns

### 1. Spatial Association Logic
In `app/video_processing.py`, objects are not just detected; they are associated. 
- A detection is valid only if a `helmet`/`no helmet` and a `number plate` bounding box fall **inside** the `rider` bounding box coordinates.
- This prevents "cross-bike" false positives.

### 2. OCR Validation
Number plates are extracted via PaddleOCR but must pass a **Regex validation** (`r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'`) to be logged. This ensures only valid Indian standard plates are stored.

### 3. Template Response (Starlette 1.0.0 Fix)
**Crucial:** The environment uses Starlette 1.0.0. The `TemplateResponse` signature **must** use keyword arguments:
`templates.TemplateResponse(request=request, name="index.html")`
Using positional arguments will cause a `TypeError` due to a signature change in this version.

---

## 🚀 How to Run (Quick Ref)
- **Main Web App:** `python -m app.main` (Accessible at :8000)
- **Local Test Script:** `python "Training _module/main.py"`
- **Training Model:** `python "Training _module/training.py"`

---

## ⚠️ Known Constraints & Environment
- **Python Version:** Always use **Python 3.10**. Newer versions may have conflicts with PaddleOCR or specific Torch builds.
- **Paths:** Always use `os.path.join(os.path.dirname(__file__), ...)` for file paths to ensure compatibility between the root and sub-modules.
- **GUI:** The training scripts use `cv2.imshow()`. In headless/server environments, these calls must be commented out or handled.

---

## 📝 Recent Architectural Changes
- Consolidated `Training _module` logic into the root documentation.
- Standardized relative pathing across all scripts.
- Updated `app/main.py` to be compatible with Starlette 1.0.0.
