import os
# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import io
import gc
import json
import time
import uuid
import base64
import asyncio
import inspect
import traceback
from typing import Optional, Dict, Any, Tuple, List, Union

import torch
import numpy as np
import imageio
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
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
    frames: Optional[int] = 49
    fps: Optional[int] = 24
    response_format: Optional[str] = "b64_json"
    negative_prompt: Optional[str] = None
    guidance_scale: Optional[float] = 6.0
    num_inference_steps: Optional[int] = 20
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


def load_any_image(img_input: Any) -> Image.Image:
    """Loads an image regardless of whether it's Base64, a URL, or a local file path."""
    str_input = str(img_input).strip()

    # Base64 string
    if str_input.startswith("data:") or len(str_input) > 500:
        return decode_base64_image(str_input)
    
    # HTTP / HTTPS URL
    if str_input.startswith("http://") or str_input.startswith("https://"):
        return load_image(str_input).convert("RGB")
    
    # Local disk file path
    if os.path.exists(str_input):
        return Image.open(str_input).convert("RGB")

    # Fallback to base64 decode attempt
    return decode_base64_image(str_input)


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
                        # Extract prompt text
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        
                        # Flexible image_url parsing for various OpenAI spec implementations
                        elif item.get("type") == "image_url":
                            img_obj = item.get("image_url")
                            if isinstance(img_obj, dict):
                                init_image = img_obj.get("url") or img_obj.get("path")
                            elif isinstance(img_obj, str):
                                init_image = img_obj
                        
                        elif "image_url" in item:
                            init_image = item["image_url"]
                        elif "url" in item:
                            init_image = item["url"]

                prompt_text = " ".join(text_parts)

            if prompt_text or init_image:
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
    output_filename = os.path.abspath(os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.mp4"))
    
    formatted_frames = []
    for frame in result_frames:
        if isinstance(frame, Image.Image):
            formatted_frames.append(np.array(frame, dtype=np.uint8))
        elif isinstance(frame, torch.Tensor):
            tensor_np = frame.cpu().numpy()
            if tensor_np.dtype == np.float32 or tensor_np.dtype == np.float64:
                tensor_np = (np.clip(tensor_np, 0, 1) * 255).astype(np.uint8)
            formatted_frames.append(tensor_np)
        elif isinstance(frame, np.ndarray):
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            formatted_frames.append(frame)
        else:
            formatted_frames.append(np.array(frame, dtype=np.uint8))

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


@app.post("/v1/images/edits")
@app.post("/api/v1/images/edits")
@app.post("/v1/images/harmonize")
@app.post("/api/v1/images/harmonize")
async def image_edits(
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    mask: Optional[UploadFile] = File(None),
    model: Optional[str] = Form(None),
    n: int = Form(1),
    size: Optional[str] = Form("1024x1024"),
    response_format: Optional[str] = Form("b64_json"),
    guidance_scale: Optional[float] = Form(3.5),
    num_inference_steps: Optional[int] = Form(20),
    seed: Optional[int] = Form(None)
):
    width, height = parse_size(size, 1024, 1024)
    init_img = None
    if image is not None:
        image_bytes = await image.read()
        init_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        init_img = prepare_image(init_img, (width, height), "RGB")

    pipe = await get_flux_pipeline()

    kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": int(num_inference_steps or 20),
        "guidance_scale": float(guidance_scale if guidance_scale is not None else 3.5),
        "generator": make_generator(seed),
    }

    if init_img is not None and "image" in inspect.signature(pipe.__call__).parameters:
        kwargs["image"] = init_img

    try:
        result = await run_in_threadpool(run_pipeline_safely, pipe, kwargs)
        clean_gpu()
        return openai_image_response(result.images, response_format or "b64_json")
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
        "num_frames": int(req.frames or 49),
        "num_inference_steps": int(req.num_inference_steps or 20),
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
    msg_id = f"chatcmpl-{uuid.uuid4()}"

    async def generate_chat_stream():
        initial_chunk = {
            "id": msg_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        try:
            if "wan" in model_lower or "video" in model_lower or "ti2v" in model_lower:
                if image_input:
                    print("Detected Image-to-Video request! Loading template image...")
                    pil_img = load_any_image(image_input)
                    width, height = parse_size("1280x704", 1280, 704)
                    pil_img = prepare_image(pil_img, (width, height), "RGB")

                    pipe = await get_wan_i2v_pipeline()
                    kwargs = {
                        "prompt": prompt or "Animate this image",
                        "image": pil_img,
                        "height": height,
                        "width": width,
                        "num_frames": 49,
                        "guidance_scale": 6.0,
                        "num_inference_steps": 20,
                    }
                else:
                    print("No image detected in payload. Defaulting to Text-to-Video pipeline...")
                    pipe = await get_wan_t2v_pipeline()
                    width, height = parse_size("1280x704", 1280, 704)
                    kwargs = {
                        "prompt": prompt,
                        "height": height,
                        "width": width,
                        "num_frames": 49,
                        "guidance_scale": 6.0,
                        "num_inference_steps": 20,
                    }

                pipe_task = asyncio.create_task(run_in_threadpool(run_pipeline_safely, pipe, kwargs))

                while not pipe_task.done():
                    await asyncio.sleep(5)
                    keep_alive_chunk = {
                        "id": msg_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": " "}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(keep_alive_chunk)}\n\n"

                result = await pipe_task
                output_filename = await run_in_threadpool(process_video_export, result.frames[0], 24)
                clean_gpu()

                response_text = f"\n\nHere is your generated video:\n\n![Generated Video]({output_filename})"

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

                pipe_task = asyncio.create_task(run_in_threadpool(run_pipeline_safely, pipe, kwargs))
                while not pipe_task.done():
                    await asyncio.sleep(2)
                    keep_alive_chunk = {
                        "id": msg_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": " "}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(keep_alive_chunk)}\n\n"

                result = await pipe_task
                path = save_image(result.images[0])
                clean_gpu()

                response_text = f"\n\nHere is your generated image:\n\n![Generated Image]({path})"

            final_content_chunk = {
                "id": msg_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": response_text}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_content_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            clean_gpu()
            traceback.print_exc()
            err_chunk = {
                "id": msg_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": f"\n\nError: {str(exc)}"}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(err_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate_chat_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    print("Starting Unified Image & Video API Backend...")
    print(f"Device: {DEVICE}, Precision: {DTYPE}")
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=1200)