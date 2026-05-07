import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.routes import router as api_router

# Ensure necessary directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("violation_images", exist_ok=True)

app = FastAPI(title="RoadWatch: Smart Traffic Monitoring System")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/violation_images", StaticFiles(directory="violation_images"), name="violation_images")
templates = Jinja2Templates(directory="app/templates")

# Include router from routes
app.include_router(api_router)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request=request, name="main.html")

@app.get("/violations", response_class=HTMLResponse)
async def read_violations(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)