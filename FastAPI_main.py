from fastapi import FastAPI, File, UploadFile
import torch
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np

app = FastAPI()
try:
    model = YOLO("checkpoints/best.pt") 
except:
    model = YOLO("yolov8n-cls.pt")

CLASSES = ["Cargo", "F-15", "F-16", "F-22", "F-35", "J-20", "Mirage", "Passenger", "Sukhoi SU-27", "Sukhoi SU-30"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
    
    results = model(img)

    probs = results[0].probs
    top1_idx = probs.top1
    confidence = probs.top1conf.item()
    
    return {
        "class": CLASSES[top1_idx],
        "confidence": f"{confidence:.2%}"
    }