"""
SAM 3 Interactive Segmentation Backend
=======================================
FastAPI server that loads SAM 3 model and provides REST API for:
  - Image browsing
  - Box + text segmentation
  - Point refinement
  - Export results

Usage:
  cd Test/backend
  pip install -r requirements.txt
  python main.py

Then open http://localhost:8501 in browser.
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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# Path detection (same logic as notebooks)
# =====================================================================
_cwd = Path.cwd().resolve()
if (_cwd / "sam3").is_dir():
    REPO_ROOT = _cwd
elif (_cwd.parent / "sam3").is_dir():
    REPO_ROOT = _cwd.parent
elif (_cwd.parent.parent / "sam3").is_dir():
    REPO_ROOT = _cwd.parent.parent
else:
    raise RuntimeError(f"Cannot find repo root (sam3/ not found from {_cwd})")

CKPT_PATH = REPO_ROOT / "sam3.pt"
INPUT_DIR = REPO_ROOT / "Inputs" / "RawImages"
CSV_PATH = INPUT_DIR / "试标注图像清单.csv"
OUTPUT_DIR = REPO_ROOT / "Outputs" / "Lab5_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"SAM 3 checkpoint not found: {CKPT_PATH}")

# Load image notes
image_notes = {}
if CSV_PATH.exists():
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            image_notes[row["编号"]] = row

# =====================================================================
# SUPPORTED IMAGE EXTENSIONS
# =====================================================================
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

# List available images - support multiple extensions
image_files = []
image_ids = []
for f in os.listdir(INPUT_DIR):
    # Check if file has a supported extension
    ext = Path(f).suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        # Check if it's a raw image (ends with _RawImage or just any image)
        # We'll use the filename without extension as ID
        base_name = Path(f).stem
        # If it ends with _RawImage, remove that suffix
        if base_name.endswith("_RawImage"):
            img_id = base_name.replace("_RawImage", "")
        else:
            img_id = base_name
        image_files.append(f)
        image_ids.append(img_id)

# Sort for consistent ordering
image_files = sorted(image_files, key=lambda x: Path(x).stem)
image_ids = sorted(image_ids)

logger.info(f"Repo root: {REPO_ROOT}")
logger.info(f"Checkpoint: {CKPT_PATH}")
logger.info(f"Images found: {len(image_files)}")
for f in image_files:
    logger.info(f"  - {f}")

# =====================================================================
# Global model reference (loaded at startup)
# =====================================================================
model = None
processor = None
device = "cpu"

# =====================================================================
# Pydantic models
# =====================================================================

class SegmentRequest(BaseModel):
    image_id: str
    text: str
    box: List[float]  # [x1, y1, x2, y2] in pixel coords
    instance_counter: Optional[int] = 0  # for generating instance IDs

class SegmentResponse(BaseModel):
    masks: list  # list of {id, instance_id, score, mask_data (base64 PNG)}

class RefineRequest(BaseModel):
    image_id: str
    text: str
    box: List[float]  # original box [x1, y1, x2, y2]
    pts_pos: List[List[float]]  # positive points [[x, y], ...]
    pts_neg: List[List[float]]  # negative points [[x, y], ...]
    neg_boxes: Optional[List[List[float]]] = None  # exclusion boxes

class RefineResponse(BaseModel):
    mask_id: int
    instance_id: str
    score: float
    mask_data: str  # base64 PNG
    candidates: list = []  # other candidate masks [{score, mask_data}]

class ExportRequest(BaseModel):
    image_id: str
    masks: list  # [{instance_id, score, mask_data(optional), ...}]
    # masks is sent from frontend for direct saving

# =====================================================================
# SAM 3 helper functions
# =====================================================================

def load_sam3():
    """Load SAM 3 model and processor (called once at startup)."""
    global model, processor, device
    logger.info("Loading SAM 3 model...")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    model = build_sam3_image_model(
        checkpoint_path=str(CKPT_PATH),
        load_from_HF=False,
        device=device,
        eval_mode=True,
    )
    processor = Sam3Processor(model, device=device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"SAM 3 loaded | Parameters: {param_count:.0f}M")


def get_image_path(image_id: str) -> Optional[Path]:
    """
    Find the image file for a given image_id.
    Supports multiple extensions.
    """
    # Try to find any file that starts with the image_id
    # First try: {image_id}_RawImage.*
    for ext in SUPPORTED_EXTENSIONS:
        test_path = INPUT_DIR / f"{image_id}_RawImage{ext}"
        if test_path.exists():
            return test_path
    
    # Second try: {image_id}.*
    for ext in SUPPORTED_EXTENSIONS:
        test_path = INPUT_DIR / f"{image_id}{ext}"
        if test_path.exists():
            return test_path
    
    # Third try: any file that starts with image_id_ or image_id
    for f in os.listdir(INPUT_DIR):
        if f.startswith(image_id):
            ext = Path(f).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                return INPUT_DIR / f
    
    return None


def add_point_prompt(state, points_norm, labels_val):
    """Add point prompts to SAM 3 state (same as Lab3/Lab5)."""
    pts_tensor = torch.tensor(points_norm, device=device, dtype=torch.float32).view(-1, 1, 2)
    lbl_tensor = torch.tensor(labels_val, device=device, dtype=torch.long).view(-1, 1)
    state["geometric_prompt"].append_points(pts_tensor, lbl_tensor)
    return state


def mask_to_base64(mask_np):
    """Convert numpy mask (H,W) to base64 PNG data URI."""
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    buf = BytesIO()
    mask_pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def run_segmentation_pipeline(img, text_prompt, box_px,
                              pos_points=None, neg_points=None, neg_boxes=None):
    """
    Core SAM 3 inference pipeline.
    
    Args:
        img: PIL Image
        text_prompt: str
        box_px: [x1, y1, x2, y2] in pixel coords (positive box)
        pos_points: list of [x, y] in pixel coords (positive)
        neg_points: list of [x, y] in pixel coords (negative)
        neg_boxes: list of [x1, y1, x2, y2] in pixel coords (negative/exclusion)
    
    Returns:
        List of dicts with mask, score
    """
    w, h = img.size
    cx = ((box_px[0] + box_px[2]) / 2) / w
    cy = ((box_px[1] + box_px[3]) / 2) / h
    bw = (box_px[2] - box_px[0]) / w
    bh = (box_px[3] - box_px[1]) / h

    state = processor.set_image(img)
    state = processor.set_text_prompt(prompt=text_prompt, state=state)
    state = processor.add_geometric_prompt(box=[cx, cy, bw, bh], label=True, state=state)

    # Add refinement points if any
    if pos_points:
        pts_norm = [[p[0] / w, p[1] / h] for p in pos_points]
        labels_val = [1] * len(pos_points)
        state = add_point_prompt(state, pts_norm, labels_val)
    if neg_points:
        pts_norm = [[p[0] / w, p[1] / h] for p in neg_points]
        labels_val = [0] * len(neg_points)
        state = add_point_prompt(state, pts_norm, labels_val)

    # Re-run forward pass if any refinement was added (points only)
    if pos_points or neg_points:
        state = processor._forward_grounding(state)

    results = []
    for i in range(len(state["masks"])):
        mask_t = state["masks"][i].squeeze().cpu().numpy()
        score = float(state["scores"][i].item())
        results.append({"mask": mask_t, "score": score})

    # Post-process: zero out negative box regions (only on the results, not SAM 3 state)
    if neg_boxes:
        for i, r in enumerate(results):
            mask_t = r["mask"]
            for nb in neg_boxes:
                x1_i, y1_i, x2_i, y2_i = [int(v) for v in nb]
                x1_i = max(0, min(w-1, x1_i))
                y1_i = max(0, min(h-1, y1_i))
                x2_i = max(0, min(w-1, x2_i))
                y2_i = max(0, min(h-1, y2_i))
                if x2_i > x1_i and y2_i > y1_i:
                    mask_t[y1_i:y2_i, x1_i:x2_i] = 0.0
            results[i]["mask"] = mask_t

    return results


# =====================================================================
# FastAPI app
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load SAM 3 model on startup."""
    load_sam3()
    yield

app = FastAPI(title="SAM 3 Segmentation", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# API Routes
# =====================================================================

@app.get("/api/images")
async def list_images():
    """Return list of available image IDs with metadata."""
    items = []
    for img_id in image_ids:
        notes = image_notes.get(img_id, {})
        # Find the actual file extension for this image
        img_path = get_image_path(img_id)
        ext = img_path.suffix if img_path else ""
        items.append({
            "id": img_id,
            "extension": ext,
            "notes": {
                "主要构件": notes.get("主要构件", ""),
                "场景特点": notes.get("场景特点", ""),
                "干扰项": notes.get("干扰项", ""),
            }
        })
    return {"images": items}


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    """Return image as base64 data URI."""
    img_path = get_image_path(image_id)
    if not img_path or not img_path.exists():
        raise HTTPException(404, f"Image {image_id} not found")
    
    try:
        img = Image.open(img_path).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {
            "image_id": image_id,
            "width": img.width,
            "height": img.height,
            "extension": img_path.suffix,
            "data_url": f"data:image/png;base64,{b64}"
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to load image: {str(e)}")


@app.post("/api/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    """Run box + text segmentation."""
    if processor is None:
        raise HTTPException(500, "SAM 3 model not loaded")

    img_path = get_image_path(req.image_id)
    if not img_path or not img_path.exists():
        raise HTTPException(404, f"Image {req.image_id} not found")

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(500, f"Failed to load image: {str(e)}")

    try:
        results = run_segmentation_pipeline(img, req.text, req.box)
    except Exception as e:
        logger.exception("Segmentation failed")
        raise HTTPException(500, str(e))

    counters = {}
    masks_out = []
    for i, r in enumerate(results):
        text_key = req.text.replace(" ", "_")
        counters[text_key] = counters.get(text_key, req.instance_counter) + 1
        instance_id = f"{text_key}#{counters[text_key]}"

        masks_out.append({
            "id": i,
            "instance_id": instance_id,
            "score": r["score"],
            "mask_data": mask_to_base64(r["mask"]),
            "width": img.width,
            "height": img.height,
        })

    return SegmentResponse(masks=masks_out)


@app.post("/api/refine")
async def refine(req: RefineRequest):
    """Refine a mask with positive/negative points."""
    if processor is None:
        raise HTTPException(500, "SAM 3 model not loaded")

    img_path = get_image_path(req.image_id)
    if not img_path or not img_path.exists():
        raise HTTPException(404, f"Image {req.image_id} not found")

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(500, f"Failed to load image: {str(e)}")

    # Deduplicate points (remove points that appear in both)
    pos_set = list(dict.fromkeys(tuple(p) for p in req.pts_pos))
    neg_set = list(dict.fromkeys(tuple(p) for p in req.pts_neg))
    overlap = set(pos_set) & set(neg_set)
    pos_set = [list(p) for p in pos_set if p not in overlap]
    neg_set = [list(p) for p in neg_set if p not in overlap]

    try:
        results = run_segmentation_pipeline(img, req.text, req.box,
                                            pos_set, neg_set, req.neg_boxes)
    except Exception as e:
        logger.exception("Refinement failed")
        raise HTTPException(500, str(e))

    if not results:
        return RefineResponse(mask_id=0, instance_id="", score=0.0, mask_data="")

    # Build candidates list (skip best=index 0)
    candidates_out = []
    for j, r_cand in enumerate(results):
        if j == 0:
            continue
        candidates_out.append({
            "score": r_cand["score"],
            "mask_data": mask_to_base64(r_cand["mask"]),
        })

    r = results[0]
    return RefineResponse(
        mask_id=0,
        instance_id="refined",
        score=r["score"],
        mask_data=mask_to_base64(r["mask"]),
        candidates=candidates_out,
    )


@app.post("/api/export")
async def export_masks(req: ExportRequest):
    """Export masks: save PNGs + composite + metadata."""
    ts = datetime.now().strftime("%H%M%S")
    save_dir = OUTPUT_DIR / f"{req.image_id}_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    meta_list = []
    for item in req.masks:
        instance_id = item.get("instance_id", f"mask_{item['id']}")
        score = item.get("score", 0.0)

        # Decode base64 mask
        data_url = item["mask_data"]
        b64_data = data_url.split(",")[1] if "," in data_url else data_url
        mask_bytes = base64.b64decode(b64_data)
        mask_pil = Image.open(BytesIO(mask_bytes)).convert("L")

        fname = f"mask_{instance_id.replace('#','_')}_{score:.3f}.png"
        mask_pil.save(save_dir / fname)

        meta_list.append({
            "instance_id": instance_id,
            "score": score,
            "file": fname,
        })

    # Composite overlay
    img_path = get_image_path(req.image_id)
    if img_path and img_path.exists():
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            h, w = img_np.shape[:2]
            overlay = np.zeros((h, w, 4), dtype=np.float32)
            cmap = plt.cm.tab10

            for i, item in enumerate(req.masks):
                data_url = item["mask_data"]
                b64_data = data_url.split(",")[1] if "," in data_url else data_url
                mask_bytes = base64.b64decode(b64_data)
                mask_pil = Image.open(BytesIO(mask_bytes)).convert("L")
                mask_np = np.array(mask_pil, dtype=bool)

                color = cmap(i % 10)
                overlay[mask_np] = list(color[:3]) + [0.5]

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(img_np)
            ax.imshow(overlay)
            ax.set_title(f"Lab5 - {req.image_id}")
            ax.axis("off")
            plt.tight_layout()
            fig.savefig(save_dir / "composite.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create composite: {e}")

    # Metadata JSON
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2, ensure_ascii=False)

    return {"saved_to": str(save_dir), "count": len(req.masks)}


# =====================================================================
# Static file serving for frontend
# =====================================================================
from fastapi.responses import FileResponse

frontend_dir = Path(__file__).parent.parent / "frontend"
index_path = frontend_dir / "index.html"


@app.get("/")
async def serve_frontend():
    """Serve the SPA frontend."""
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"error": "Frontend not found"}


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting SAM 3 backend on http://localhost:8501")
    uvicorn.run(app, host="0.0.0.0", port=8501)