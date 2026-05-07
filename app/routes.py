import os
import json
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app.video_processing import process_video_web
from app.db import read_violations_by_vehicle, VIOLATIONS_FILE
import shutil

router = APIRouter()

@router.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process video and detect violations
        process_video_web(file_path)
        return {"message": "Video processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get-violations/")
async def get_violations(vehicle_number: str = Form(...)):
    try:
        violations = read_violations_by_vehicle(vehicle_number)
        return JSONResponse(content=violations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/violations")
async def list_all_violations():
    """Return all violations from violations.json."""
    try:
        if os.path.exists(VIOLATIONS_FILE):
            with open(VIOLATIONS_FILE, 'r') as f:
                violations = json.load(f)
            return JSONResponse(content=violations)
        return JSONResponse(content=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))