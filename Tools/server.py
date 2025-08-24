from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import io
import os

# Added for PDF page rendering (PyMuPDF). Imported lazily so non-PDF users are unaffected.
try:  # noqa: SIM105
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - handled at runtime if PDF used
    fitz = None

# Lazy load model (loaded on first request)
_model = None
MODEL_PATH = Path("runs/layout_v8m2/weights/best.pt")

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
    """Accept image files (PNG/JPG, etc.) and/or PDF files.

    For a PDF, each page is converted to an image and treated like an individual image input.
    Output schema remains the same; PDF pages are given synthetic file names: <original>.pdf_page<N>.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    model = get_model()
    # Collect PIL.Image objects (Ultralytics does not accept raw bytes list)
    images = []
    file_names: List[str] = []

    for uf in files:
        data = await uf.read()
        fname = uf.filename or "file"
        lower_name = fname.lower()
        content_type = (uf.content_type or "").lower()

        is_pdf = lower_name.endswith(".pdf") or content_type == "application/pdf"
        if is_pdf:
            if fitz is None:  # Library not available
                raise HTTPException(status_code=400, detail="PDF support requires PyMuPDF (install 'PyMuPDF')")
            try:
                # Open from bytes
                doc = fitz.open(stream=data, filetype="pdf")
            except Exception as e:  # Invalid PDF
                raise HTTPException(status_code=400, detail=f"Invalid PDF '{fname}': {e}") from e
            if doc.page_count == 0:
                continue  # skip empty PDF
            for page_index, page in enumerate(doc):
                # Render page to image (default ~72 DPI). Increase quality by scaling matrix if needed later.
                pix = page.get_pixmap(alpha=False)
                img_bytes = pix.tobytes("png")
                try:
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as e:  # pragma: no cover
                    raise HTTPException(status_code=500, detail=f"Failed to render PDF page {page_index+1} of {fname}: {e}") from e
                images.append(pil_img)
                file_names.append(f"{fname}_page{page_index+1}")
            continue  # next uploaded file

        # Standard image handling
        try:
            pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail=f"File {fname} is not a valid image or PDF")
        images.append(pil_img)
        file_names.append(fname)

    if not images:
        raise HTTPException(status_code=400, detail="No valid image pages found in upload")

    # Run predictions (batch)
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
                "confidence": conf,
            })
        response.append({
            "file_name": fname,
            "width": img_w,
            "height": img_h,
            "annotations": anns,
        })
    return JSONResponse(content={"results": response})


@app.get("/health")
async def health():
    return {"status": "ok"}

# To run: uvicorn Tools.server:app --host 0.0.0.0 --port 8000
