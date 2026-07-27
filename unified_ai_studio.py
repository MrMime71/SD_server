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
import tempfile
import traceback
from typing import Optional, Dict, Any, Tuple, List, Union

import torch
import numpy as np
import imageio
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

# Diffusers Imports
from diffusers import (
    Flux2KleinPipeline,
    WanPipeline,
    WanImageToVideoPipeline,
)
from diffusers.utils import load_image

app = FastAPI(title="Unified FLUX.2 & Wan2.2 Local API Server")

# ============================================================================
# Configuration Defaults
# ============================================================================

HOST = os.getenv("SD_HOST", "127.0.0.1")
PORT = int(os.getenv("SD_PORT", "8000"))
OUTPUT_DIR = os.getenv("SD_OUTPUT_DIR", "./output_media")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Models Path Configurations
LOCAL_FLUX_PATH = "./models/FLUX.2-klein-4B"
HF_FLUX_PATH = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_FLUX_MODEL = LOCAL_FLUX_PATH if os.path.exists(LOCAL_FLUX_PATH) else HF_FLUX_PATH

LOCAL_WAN_PATH = "./models/Wan2.2-TI2V-5B-Diffusers"
HF_WAN_PATH = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_WAN_MODEL = LOCAL_WAN_PATH if os.path.exists(LOCAL_WAN_PATH) else HF_WAN_PATH

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Global Pipeline Caches & Async Lock
_LOADED_FLUX_PIPE = None
_LOADED_WAN_T2V = None
_LOADED_WAN_I2V = None
_LOCK = asyncio.Lock()


# ============================================================================
# Request / Response Schemas
# ============================================================================

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


class VideoGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    size: Optional[str] = "1280x704"
    frames: Optional[int] = 81
    fps: Optional[int] = 24
    response_format: Optional[str] = "b64_json"
    negative_prompt: Optional[str] = None
    guidance_scale: Optional[float] = 6.0
    num_inference_steps: Optional[int] = 30
    seed: Optional[int] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


# ============================================================================
# Utility Functions
# ============================================================================

def clean_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_size(size: Optional[str], default_w: int = 1024, default_h: int = 1024) -> Tuple[int, int]:
    if not size:
        return default_w, default_h
    try:
        w, h = size.lower().replace(" ", "").split("x", 1)
        width, height = int(w), int(h)
    except Exception:
        width, height = default_w, default_h

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


def prepare_image(img: Image.Image, size: Tuple[int, int], mode: str = "RGB") -> Image.Image:
    return ImageOps.exif_transpose(img).convert(mode).resize(size, Image.LANCZOS)


def image_to_b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def file_to_b64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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


def openai_video_response(video_path: str, response_format: str = "b64_json"):
    if response_format == "url":
        return {"created": int(time.time()), "data": [{"url": video_path}]}
    else:
        return {"created": int(time.time()), "data": [{"b64_json": file_to_b64(video_path)}]}


def extract_prompt_and_image_from_chat(messages: List[ChatMessage]) -> Tuple[str, Optional[Any]]:
    prompt_text = ""
    init_image = None

    for msg in reversed(messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                prompt_text = msg.content
            elif isinstance(msg.content, list):
                text_parts = []
                for item in msg.content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            img_obj = item.get("image_url", {})
                            init_image = img_obj.get("url") if isinstance(img_obj, dict) else img_obj
                prompt_text = " ".join(text_parts)
            break

    return prompt_text.strip(), init_image


def run_pipeline_safely(pipe, kwargs: Dict[str, Any]):
    sig = inspect.signature(pipe.__call__)
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if not has_var_kwargs:
        valid_params = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    else:
        filtered_kwargs = kwargs

    return pipe(**filtered_kwargs)


def process_video_export(result_frames, fps: int) -> str:
    """Robust video export function using imageio directly to prevent OpenCV errors."""
    output_filename = os.path.abspath(os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.mp4"))
    
    formatted_frames = []
    for frame in result_frames:
        if isinstance(frame, Image.Image):
            formatted_frames.append(np.array(frame))
        elif isinstance(frame, torch.Tensor):
            formatted_frames.append((frame.cpu().numpy() * 255).astype(np.uint8))
        else:
            formatted_frames.append(np.array(frame))

    writer = imageio.get_writer(output_filename, fps=fps, codec='libx264', quality=8)
    for frame in formatted_frames:
        writer.append_data(frame)
    writer.close()

    return output_filename


# ============================================================================
# Dynamic VRAM-Aware Pipeline Loaders
# ============================================================================

def unload_active_pipelines():
    global _LOADED_FLUX_PIPE, _LOADED_WAN_T2V, _LOADED_WAN_I2V
    
    if _LOADED_FLUX_PIPE is not None:
        _LOADED_FLUX_PIPE.to("cpu")
        _LOADED_FLUX_PIPE = None

    if _LOADED_WAN_T2V is not None:
        _LOADED_WAN_T2V.to("cpu")
        _LOADED_WAN_T2V = None

    if _LOADED_WAN_I2V is not None:
        _LOADED_WAN_I2V.to("cpu")
        _LOADED_WAN_I2V = None

    clean_gpu()


async def get_flux_pipeline():
    global _LOADED_FLUX_PIPE
    async with _LOCK:
        if _LOADED_FLUX_PIPE is not None:
            return _LOADED_FLUX_PIPE

        unload_active_pipelines()
        print(f"Loading FLUX.2 pipeline from: {DEFAULT_FLUX_MODEL}")
        kwargs = {"torch_dtype": DTYPE}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        try:
            pipe = Flux2KleinPipeline.from_pretrained(DEFAULT_FLUX_MODEL, **kwargs)
        except Exception:
            from diffusers import AutoPipelineForText2Image
            pipe = AutoPipelineForText2Image.from_pretrained(DEFAULT_FLUX_MODEL, **kwargs)

        pipe = pipe.to(DEVICE)
        _LOADED_FLUX_PIPE = pipe
        return _LOADED_FLUX_PIPE


async def get_wan_t2v_pipeline():
    global _LOADED_WAN_T2V
    async with _LOCK:
        if _LOADED_WAN_T2V is not None:
            return _LOADED_WAN_T2V

        unload_active_pipelines()
        print(f"Loading Wan2.2 Text-to-Video pipeline from: {DEFAULT_WAN_MODEL}")
        kwargs = {"torch_dtype": DTYPE}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        pipe = WanPipeline.from_pretrained(DEFAULT_WAN_MODEL, **kwargs).to(DEVICE)
        _LOADED_WAN_T2V = pipe
        return _LOADED_WAN_T2V


async def get_wan_i2v_pipeline():
    global _LOADED_WAN_I2V
    async with _LOCK:
        if _LOADED_WAN_I2V is not None:
            return _LOADED_WAN_I2V

        unload_active_pipelines()
        print(f"Loading Wan2.2 Image-to-Video pipeline from: {DEFAULT_WAN_MODEL}")
        kwargs = {"torch_dtype": DTYPE}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        pipe = WanImageToVideoPipeline.from_pretrained(DEFAULT_WAN_MODEL, **kwargs).to(DEVICE)
        _LOADED_WAN_I2V = pipe
        return _LOADED_WAN_I2V


# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
async def root():
    return {
        "status": "ok", 
        "device": DEVICE, 
        "image_model": "FLUX.2-klein-4B",
        "video_model": "Wan2.2-TI2V-5B"
    }


@app.get("/v1/models")
@app.get("/api/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "black-forest-labs/FLUX.2-klein-4B", "object": "model", "owned_by": "local"},
            {"id": "FLUX.2-klein-4B", "object": "model", "owned_by": "local"},
            {"id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "object": "model", "owned_by": "local"},
            {"id": "Wan2.2-TI2V-5B-Diffusers", "object": "model", "owned_by": "local"},
            {"id": "Wan2.2-TI2V-5B", "object": "model", "owned_by": "local"}
        ]
    }


@app.get("/v1/images/progress/{task_id}")
@app.get("/v1/images/progress")
@app.get("/v1/videos/progress/{task_id}")
@app.get("/v1/videos/progress")
async def media_progress(task_id: Optional[str] = None):
    return {"progress": 1.0, "status": "completed"}


@app.post("/v1/images/generations")
@app.post("/api/v1/images/generations")
async def image_generations(req: ImageGenerationRequest):
    width, height = parse_size(req.size, 1024, 1024)
    n = max(1, min(int(req.n or 1), 4))
    pipe = await get_flux_pipeline()

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
        result = await run_in_threadpool(run_pipeline_safely, pipe, kwargs)
        clean_gpu()
        return openai_image_response(result.images, req.response_format or "b64_json")
    except torch.cuda.OutOfMemoryError:
        clean_gpu()
        raise HTTPException(status_code=507, detail="CUDA out of memory.")
    except Exception as exc:
        clean_gpu()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/videos/generations")
@app.post("/api/v1/videos/generations")
async def video_generations(req: VideoGenerationRequest):
    width, height = parse_size(req.size, 1280, 704)
    pipe = await get_wan_t2v_pipeline()

    kwargs = {
        "prompt": req.prompt,
        "height": height,
        "width": width,
        "num_frames": int(req.frames or 81),
        "num_inference_steps": int(req.num_inference_steps or 30),
        "guidance_scale": float(req.guidance_scale if req.guidance_scale is not None else 6.0),
        "generator": make_generator(req.seed),
    }

    try:
        result = await run_in_threadpool(run_pipeline_safely, pipe, kwargs)
        output_filename = await run_in_threadpool(process_video_export, result.frames[0], int(req.fps or 24))
        
        clean_gpu()
        return openai_video_response(output_filename, req.response_format or "b64_json")
    except torch.cuda.OutOfMemoryError:
        clean_gpu()
        raise HTTPException(status_code=507, detail="CUDA out of memory.")
    except Exception as exc:
        clean_gpu()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/chat/completions")
@app.post("/api/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    prompt, image_input = extract_prompt_and_image_from_chat(req.messages)
    model_lower = req.model.lower()

    try:
        # Task 1: Video Generation (Wan2.2)
        if "wan" in model_lower or "video" in model_lower or "ti2v" in model_lower:
            if image_input:
                if str(image_input).startswith("data:") or "base64" in str(image_input):
                    pil_img = decode_base64_image(str(image_input))
                else:
                    pil_img = load_image(str(image_input)).convert("RGB")

                width, height = parse_size("1280x704", 1280, 704)
                pil_img = prepare_image(pil_img, (width, height), "RGB")

                pipe = await get_wan_i2v_pipeline()
                kwargs = {
                    "prompt": prompt or "Animate this image",
                    "image": pil_img,
                    "height": height,
                    "width": width,
                    "num_frames": 81,
                    "guidance_scale": 6.0,
                    "num_inference_steps": 30,
                }
            else:
                pipe = await get_wan_t2v_pipeline()
                width, height = parse_size("1280x704", 1280, 704)
                kwargs = {
                    "prompt": prompt,
                    "height": height,
                    "width": width,
                    "num_frames": 81,
                    "guidance_scale": 6.0,
                    "num_inference_steps": 30,
                }

            result = await run_in_threadpool(run_pipeline_safely, pipe, kwargs)
            output_filename = await run_in_threadpool(process_video_export, result.frames[0], 24)
            clean_gpu()

            response_content = f"Here is your generated video:\n\n![Generated Video]({output_filename})"

        # Task 2: Image Generation (FLUX.2)
        else:
            pipe = await get_flux_pipeline()
            width, height = parse_size("1024x1024", 1024, 1024)
            
            kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": 20,
                "guidance_scale": 3.5,
            }
            result = await run_in_threadpool(run_pipeline_safely, pipe, kwargs)
            
            path = save_image(result.images[0])
            clean_gpu()

            response_content = f"Here is your generated image:\n\n![Generated Image]({path})"

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
    except Exception as exc:
        clean_gpu()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    print("Starting Unified Image & Video API Backend...")
    print(f"Device: {DEVICE}, Precision: {DTYPE}")
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=600)