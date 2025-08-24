from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from pathlib import Path
import shutil
import tempfile
from ultralytics import YOLO
from PIL import Image
import io
import os

# Lazy load model (loaded on first request)
_model = None
MODEL_PATH = Path("runs/layout_v8m/weights/best.pt")

app = FastAPI(title="Layout Detection API", version="0.1.0")


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    model = get_model()
    images = []
    file_names = []
    for uf in files:
        data = await uf.read()
        try:
            Image.open(io.BytesIO(data)).verify()
            images.append(data)
            file_names.append(uf.filename or "image")
        except Exception:
            raise HTTPException(status_code=400, detail=f"File {uf.filename} is not a valid image")

    # Run predictions
    results = model.predict(images, verbose=False)
    response = []
    for fname, r in zip(file_names, results):
        img_h, img_w = r.orig_shape
        anns = []
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            anns.append({
                "bbox": [x1, y1, w, h],
                "category_id": cls_id,
                "category_name": r.names.get(cls_id, str(cls_id)),
                "confidence": conf
            })
        response.append({
            "file_name": fname,
            "width": img_w,
            "height": img_h,
            "annotations": anns
        })
    return JSONResponse(content={"results": response})


@app.get("/health")
async def health():
    return {"status": "ok"}

# To run: uvicorn Tools.server:app --host 0.0.0.0 --port 8000
