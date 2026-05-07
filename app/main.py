import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil

from app.routes import router as api_router
from app.video_processing import process_video

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/violation_images", StaticFiles(directory="violation_images"), name="violation_images")
templates = Jinja2Templates(directory="app/templates")

# Include router from routes
app.include_router(api_router)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)