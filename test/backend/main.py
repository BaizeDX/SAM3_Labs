"""
SAM 3 Three-Mode Segmentation Backend
======================================
A: Pure text (simple structures like walls/roofs)
B: Box + text, multiple masks (complex components)
C: Box + text, single best mask (precision targeting)

Usage:
  cd Test/backend && python main.py
  http://localhost:8501
"""

import os, sys, json, base64, warnings, logging
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope, Receive, Send
from pydantic import BaseModel


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles that adds no-cache headers."""
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            async def send_no_cache(message):
                if message["type"] == "http.response.start":
                    headers = message.get("headers", [])
                    headers.extend([
                        (b"cache-control", b"no-cache, no-store, must-revalidate"),
                        (b"pragma", b"no-cache"),
                        (b"expires", b"0"),
                    ])
                    message["headers"] = headers
                await send(message)
            await super().__call__(scope, receive, send_no_cache)
        else:
            await super().__call__(scope, receive, send)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# Path detection
# =====================================================================
_cwd = Path.cwd().resolve()
for p in [_cwd, _cwd.parent, _cwd.parent.parent]:
    if (p / "sam3").is_dir():
        REPO_ROOT = p
        break
else:
    raise RuntimeError(f"Cannot find repo root from {_cwd}")

CKPT_PATH = REPO_ROOT / "sam3.pt"
INPUT_DIR = REPO_ROOT / "Inputs" / "RawImages"
OUTPUT_DIR = REPO_ROOT / "Outputs" / "LabABC_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROMPTS_PATH = REPO_ROOT / "config" / "batch_prompts.txt"

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

# Default A-mode prompts
mode_a_prompts = [
    "red brick wall",
    "white stone plinth",
    "red tile roof",
    "stone pavement",
    "inner courtyard",
]

# Try load from config file
if DEFAULT_PROMPTS_PATH.exists():
    loaded = []
    with open(DEFAULT_PROMPTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                loaded.append(line)
    if loaded:
        mode_a_prompts = loaded

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
image_files = sorted([f for f in os.listdir(INPUT_DIR)
                       if Path(f).suffix.lower() in IMAGE_EXTENSIONS])

logger.info(f"Repo root: {REPO_ROOT}")
logger.info(f"Images: {len(image_files)}")
logger.info(f"Mode-A prompts: {len(mode_a_prompts)}")

# =====================================================================
# Global model
# =====================================================================
model = None
processor = None
device = "cpu"


def load_sam3():
    global model, processor, device
    logger.info("Loading SAM 3...")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    model = build_sam3_image_model(
        checkpoint_path=str(CKPT_PATH), load_from_HF=False,
        device=device, eval_mode=True,
    )
    processor = Sam3Processor(model, device=device)
    logger.info(f"SAM 3 loaded | {sum(p.numel() for p in model.parameters())/1e6:.0f}M")


def mask_to_b64(mask_np):
    """numpy mask (H,W) -> base64 PNG data URI"""
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    buf = BytesIO()
    mask_pil.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def text_to_key(text):
    """Normalize text for use in instance IDs"""
    return text.replace(" ", "_").replace("-", "_").replace("'", "").replace('"', "")


# =====================================================================
# Pydantic models
# =====================================================================

class SegmentARequest(BaseModel):
    image_id: str
    text: str

class SegmentBRequest(BaseModel):
    image_id: str
    text: str
    box: List[float]  # [x1, y1, x2, y2] pixel coords

class SegmentCRequest(BaseModel):
    image_id: str
    text: str
    box: List[float]

class MaskItem(BaseModel):
    instance_id: str
    score: float
    mask_data: str

class ExportRequest(BaseModel):
    image_id: str
    masks: List[MaskItem]

# =====================================================================
# SAM 3 inference core
# =====================================================================

def run_pure_text(img, text):
    """Mode A: pure text-prompt segmentation."""
    state = processor.set_image(img)
    state = processor.set_text_prompt(prompt=text, state=state)
    masks = state["masks"]
    scores = state["scores"]
    results = []
    for i in range(len(masks)):
        mask_np = masks[i].squeeze().cpu().numpy()
        score = float(scores[i].item())
        results.append({"mask": mask_np, "score": score})
    return results


def run_box_text_multi(img, text, box_px):
    """Mode B: box + text, return all masks."""
    w, h = img.size
    cx = ((box_px[0] + box_px[2]) / 2) / w
    cy = ((box_px[1] + box_px[3]) / 2) / h
    bw = (box_px[2] - box_px[0]) / w
    bh = (box_px[3] - box_px[1]) / h
    state = processor.set_image(img)
    state = processor.set_text_prompt(prompt=text, state=state)
    state = processor.add_geometric_prompt(box=[cx, cy, bw, bh], label=True, state=state)
    masks = state["masks"]
    scores = state["scores"]
    results = []
    for i in range(len(masks)):
        mask_np = masks[i].squeeze().cpu().numpy()
        score = float(scores[i].item())
        results.append({"mask": mask_np, "score": score})
    return results


def run_box_text_single(img, text, box_px):
    """Mode C: box + text, return only the best mask."""
    w, h = img.size
    cx = ((box_px[0] + box_px[2]) / 2) / w
    cy = ((box_px[1] + box_px[3]) / 2) / h
    bw = (box_px[2] - box_px[0]) / w
    bh = (box_px[3] - box_px[1]) / h
    state = processor.set_image(img)
    state = processor.set_text_prompt(prompt=text, state=state)
    state = processor.add_geometric_prompt(box=[cx, cy, bw, bh], label=True, state=state)
    masks = state["masks"]
    scores = state["scores"]
    # Pick the best mask
    best_idx = int(scores.argmax().item())
    return [{
        "mask": masks[best_idx].squeeze().cpu().numpy(),
        "score": float(scores[best_idx].item()),
    }]


# =====================================================================
# FastAPI app
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_sam3()
    yield

app = FastAPI(title="SAM 3 Three-Mode Segmentation", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ---- Image endpoints ----

@app.get("/api/images")
async def list_images():
    return {"images": [
        {"id": Path(f).stem} for f in image_files
    ]}

@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    for f in image_files:
        if Path(f).stem == image_id:
            img_path = INPUT_DIR / f
            img = Image.open(img_path).convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return {"image_id": image_id, "width": img.width, "height": img.height,
                    "data_url": f"data:image/png;base64,{b64}"}
    raise HTTPException(404, f"Image {image_id} not found")

# ---- A-mode prompts config ----

@app.get("/api/config-prompts")
async def get_prompts():
    return {"prompts": mode_a_prompts}

# ---- Segmentation endpoints ----

@app.post("/api/segment-a")
async def segment_a(req: SegmentARequest):
    if processor is None:
        raise HTTPException(500, "Model not loaded")
    try:
        img_path = INPUT_DIR / f"{req.image_id}_RawImage.png"
        if not img_path.exists():
            # Try to find by scanning all files
            found = False
            for f in image_files:
                if Path(f).stem == req.image_id:
                    img_path = INPUT_DIR / f
                    found = True
                    break
            if not found:
                raise HTTPException(404, f"Image {req.image_id} not found")
        img = Image.open(img_path).convert("RGB")
        results = run_pure_text(img, req.text)
        return {"masks": [
            {"score": r["score"], "mask_data": mask_to_b64(r["mask"])}
            for r in results
        ]}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Segment A failed")
        raise HTTPException(500, str(e))

@app.post("/api/segment-b")
async def segment_b(req: SegmentBRequest):
    if processor is None:
        raise HTTPException(500, "Model not loaded")
    try:
        img_path = INPUT_DIR / f"{req.image_id}_RawImage.png"
        if not img_path.exists():
            for f in image_files:
                if Path(f).stem == req.image_id:
                    img_path = INPUT_DIR / f
                    break
        img = Image.open(img_path).convert("RGB")
        results = run_box_text_multi(img, req.text, req.box)
        return {"masks": [
            {"score": r["score"], "mask_data": mask_to_b64(r["mask"])}
            for r in results
        ]}
    except Exception as e:
        logger.exception("Segment B failed")
        raise HTTPException(500, str(e))

@app.post("/api/segment-c")
async def segment_c(req: SegmentCRequest):
    if processor is None:
        raise HTTPException(500, "Model not loaded")
    try:
        img_path = INPUT_DIR / f"{req.image_id}_RawImage.png"
        if not img_path.exists():
            for f in image_files:
                if Path(f).stem == req.image_id:
                    img_path = INPUT_DIR / f
                    break
        img = Image.open(img_path).convert("RGB")
        results = run_box_text_single(img, req.text, req.box)
        return {"masks": [
            {"score": r["score"], "mask_data": mask_to_b64(r["mask"])}
            for r in results
        ]}
    except Exception as e:
        logger.exception("Segment C failed")
        raise HTTPException(500, str(e))

# ---- Export ----

@app.post("/api/export")
async def export_masks(req: ExportRequest):
    ts = datetime.now().strftime("%H%M%S")
    save_dir = OUTPUT_DIR / f"{req.image_id}_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_list = []
    for item in req.masks:
        b64_data = item.mask_data.split(",")[1] if "," in item.mask_data else item.mask_data
        mask_bytes = base64.b64decode(b64_data)
        mask_pil = Image.open(BytesIO(mask_bytes)).convert("L")
        fname = f"{item.instance_id.replace('#','_').replace('/','_')}_{item.score:.3f}.png"
        mask_pil.save(save_dir / fname)
        meta_list.append({"instance_id": item.instance_id, "score": item.score, "file": fname})
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2, ensure_ascii=False)
    return {"saved_to": str(save_dir), "count": len(req.masks)}

# ---- Frontend ----

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", NoCacheStaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting on http://localhost:8501")
    uvicorn.run(app, host="0.0.0.0", port=8501)
