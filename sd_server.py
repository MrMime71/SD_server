import os
# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import io
import gc
import time
import uuid
import base64
import asyncio
import inspect
import traceback
from typing import Optional, Dict, Any, Tuple, List

import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel
from diffusers import Flux2KleinPipeline

app = FastAPI(title="Local FLUX.2 OpenAI-compatible Image API")

# Configuration Defaults
HOST = os.getenv("SD_HOST", "127.0.0.1")
PORT = int(os.getenv("SD_PORT", "8000"))
OUTPUT_DIR = os.getenv("SD_OUTPUT_DIR", "./output_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Checks local downloaded directory first, falls back to HF hub repository
LOCAL_MODEL_PATH = "./models/FLUX.2-klein-4B"
HF_MODEL_PATH = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_MODEL = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else HF_MODEL_PATH

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

_LOADED_PIPE = None
_LOCK = asyncio.Lock()


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: int = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"
    negative_prompt: Optional[str] = None
    guidance_scale: Optional[float] = 3.5
    num_inference_steps: Optional[int] = 20
    seed: Optional[int] = None


class ImageEditJsonRequest(BaseModel):
    prompt: str
    image: str
    mask: Optional[str] = None
    model: Optional[str] = None
    n: int = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"
    negative_prompt: Optional[str] = None
    guidance_scale: Optional[float] = 3.5
    num_inference_steps: Optional[int] = 20
    strength: Optional[float] = 0.75
    seed: Optional[int] = None


def clean_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_size(size: Optional[str]) -> Tuple[int, int]:
    if not size:
        return 1024, 1024
    try:
        w, h = size.lower().replace(" ", "").split("x", 1)
        width, height = int(w), int(h)
    except Exception:
        width, height = 1024, 1024
    # FLUX expects dimensions divisible by 16
    width = max(256, min(2048, (width // 16) * 16))
    height = max(256, min(2048, (height // 16) * 16))
    return width, height


def make_generator(seed: Optional[int]):
    if seed is None:
        return None
    return torch.Generator(device=DEVICE).manual_seed(int(seed))


def decode_base64_image(value: str, mode: str = "RGB") -> Image.Image:
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(value))).convert(mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode base64 image: {exc}")


async def upload_to_image(upload: UploadFile, mode: str = "RGB") -> Image.Image:
    try:
        raw = await upload.read()
        return Image.open(io.BytesIO(raw)).convert(mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {exc}")


def prepare_image(img: Image.Image, size: Tuple[int, int], mode: str = "RGB") -> Image.Image:
    return ImageOps.exif_transpose(img).convert(mode).resize(size, Image.LANCZOS)


def image_to_b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_image(image: Image.Image) -> str:
    path = os.path.abspath(os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.png"))
    image.save(path, format="PNG")
    return path


def openai_image_response(images: List[Image.Image], response_format: str = "b64_json"):
    data = []
    for image in images:
        path = save_image(image)
        if response_format == "url":
            data.append({"url": path})
        else:
            data.append({"b64_json": image_to_b64_png(image)})
    return {"created": int(time.time()), "data": data}


def run_pipeline_safely(pipe, kwargs: Dict[str, Any]):
    """Dynamically filters out kwargs not supported by the loaded pipeline call method."""
    sig = inspect.signature(pipe.__call__)
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if not has_var_kwargs:
        valid_params = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    else:
        filtered_kwargs = kwargs

    return pipe(**filtered_kwargs)


async def get_pipeline():
    global _LOADED_PIPE
    async with _LOCK:
        if _LOADED_PIPE is not None:
            return _LOADED_PIPE

        print(f"Loading FLUX.2 pipeline from: {DEFAULT_MODEL}")
        kwargs = {"torch_dtype": DTYPE}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        try:
            pipe = Flux2KleinPipeline.from_pretrained(DEFAULT_MODEL, **kwargs)
        except Exception:
            from diffusers import AutoPipelineForText2Image
            pipe = AutoPipelineForText2Image.from_pretrained(DEFAULT_MODEL, **kwargs)

        pipe = pipe.to(DEVICE)
        _LOADED_PIPE = pipe
        return _LOADED_PIPE


@app.get("/")
async def root():
    return {"status": "ok", "device": DEVICE, "model": "FLUX.2-klein-4B"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "black-forest-labs/FLUX.2-klein-4B", "object": "model", "owned_by": "local"},
            {"id": "FLUX.2-klein-4B", "object": "model", "owned_by": "local"}
        ]
    }


# Optional progress route to prevent frontends throwing 404 warnings
@app.get("/v1/images/progress/{task_id}")
@app.get("/v1/images/progress")
async def image_progress(task_id: Optional[str] = None):
    return {"progress": 1.0, "status": "completed"}


@app.post("/v1/images/generations")
async def image_generations(req: ImageGenerationRequest):
    width, height = parse_size(req.size)
    n = max(1, min(int(req.n or 1), 4))
    pipe = await get_pipeline()

    kwargs = {
        "prompt": req.prompt,
        "width": width,
        "height": height,
        "num_images_per_prompt": n,
        "num_inference_steps": int(req.num_inference_steps or 20),
        "guidance_scale": float(req.guidance_scale if req.guidance_scale is not None else 3.5),
        "generator": make_generator(req.seed),
    }

    try:
        result = run_pipeline_safely(pipe, kwargs)
        clean_gpu()
        return openai_image_response(result.images, req.response_format or "b64_json")
    except torch.cuda.OutOfMemoryError:
        clean_gpu()
        raise HTTPException(status_code=507, detail="CUDA out of memory.")
    except Exception as exc:
        clean_gpu()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/images/edits")
async def image_edits(request: Request):
    content_type = request.headers.get("content-type", "").lower()

    # Handle Multipart Form Uploads
    if "multipart/form-data" in content_type:
        form = await request.form()
        prompt = str(form.get("prompt") or "")
        upload = form.get("image")
        init_image = await upload_to_image(upload, "RGB") if upload else None
        size = str(form.get("size") or "1024x1024")
        response_format = str(form.get("response_format") or "b64_json")
        steps = int(form.get("num_inference_steps") or 20)
        guidance = float(form.get("guidance_scale") or 3.5)
        strength = float(form.get("strength") or 0.75)
        seed = int(form.get("seed")) if form.get("seed") else None
        n = int(form.get("n") or 1)
    # Handle JSON Base64 Payloads
    else:
        payload = await request.json()
        req = ImageEditJsonRequest(**payload)
        prompt = req.prompt
        init_image = decode_base64_image(req.image, "RGB")
        size = req.size
        response_format = req.response_format or "b64_json"
        steps = int(req.num_inference_steps or 20)
        guidance = float(req.guidance_scale if req.guidance_scale is not None else 3.5)
        strength = float(req.strength if req.strength is not None else 0.75)
        seed = req.seed
        n = req.n

    if init_image is None:
        raise HTTPException(status_code=400, detail="No base image provided for editing.")

    width, height = parse_size(size)
    init_image = prepare_image(init_image, (width, height), "RGB")

    pipe = await get_pipeline()

    kwargs = {
        "prompt": prompt,
        "image": init_image,
        "width": width,
        "height": height,
        "num_images_per_prompt": max(1, min(int(n or 1), 4)),
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "strength": strength,
        "generator": make_generator(seed),
    }

    try:
        result = run_pipeline_safely(pipe, kwargs)
        clean_gpu()
        return openai_image_response(result.images, response_format)
    except torch.cuda.OutOfMemoryError:
        clean_gpu()
        raise HTTPException(status_code=507, detail="CUDA out of memory during edit.")
    except Exception as exc:
        clean_gpu()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    print("Starting FLUX.2 API backend...")
    print(f"Device: {DEVICE}, Precision: {DTYPE}")
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=600)