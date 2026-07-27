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
    FluxFillPipeline,
    WanPipeline,
    WanImageToVideoPipeline,
    WanVideoToVideoPipeline,
)
from diffusers.utils import load_image, load_video

app = FastAPI(title="Unified Multimodal Media Production Server")

# ============================================================================
# Environment Configurations
# ============================================================================

HOST = os.getenv("SD_HOST", "127.0.0.1")
PORT = int(os.getenv("SD_PORT", "8000"))
OUTPUT_DIR = os.getenv("SD_OUTPUT_DIR", "./output_media")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path definitions
FLUX_MODEL = os.getenv("FLUX_MODEL_PATH", "black-forest-labs/FLUX.2-klein-4B")
FLUX_FILL_MODEL = os.getenv("FLUX_FILL_PATH", "black-forest-labs/FLUX.1-Fill-dev")
WAN_MODEL = os.getenv("WAN_MODEL_PATH", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# VRAM Cache Management
_PIPELINE_CACHE: Dict[str, Any] = {}
_LOCK = asyncio.Lock()


# ============================================================================
# API Schemas
# ============================================================================

class ImageGenReq(BaseModel):
    prompt: str
    image: Optional[str] = None  # Base64 or URL for Image-to-Image
    mask: Optional[str] = None   # Base64 or URL for Inpainting/Harmonization
    size: Optional[str] = "1024x1024"
    n: int = 1
    response_format: Optional[str] = "b64_json"
    guidance_scale: Optional[float] = 3.5
    strength: Optional[float] = 0.75
    num_inference_steps: Optional[int] = 20
    seed: Optional[int] = None


class VideoGenReq(BaseModel):
    prompt: str
    image: Optional[str] = None  # Base64 or URL for Image-to-Video
    video: Optional[str] = None  # Base64, URL, or file path for Video-to-Video
    size: Optional[str] = "1280x704"
    frames: Optional[int] = 49
    fps: Optional[int] = 24
    response_format: Optional[str] = "b64_json"
    guidance_scale: Optional[float] = 6.0
    strength: Optional[float] = 0.75  # For V2V
    num_inference_steps: Optional[int] = 20
    seed: Optional[int] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatReq(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = True


# ============================================================================
# Helper & Utility Functions
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


def decode_media_input(input_src: Any) -> Any:
    """Decodes raw input (Base64, URL, local path) into appropriate memory objects."""
    if not input_src:
        return None
    
    str_src = str(input_src).strip()
    
    # Base64 Image
    if str_src.startswith("data:image") or (len(str_src) > 500 and "data:video" not in str_src):
        if "," in str_src:
            str_src = str_src.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(str_src))).convert("RGB")
    
    # Remote URL
    if str_src.startswith("http://") or str_src.startswith("https://"):
        return str_src
        
    # Local disk file path
    if os.path.exists(str_src):
        if str_src.endswith((".png", ".jpg", ".jpeg", ".webp")):
            return Image.open(str_src).convert("RGB")
        return str_src
        
    return str_src


def load_video_frames(video_input: Any) -> List[Image.Image]:
    """Loads input video frames into PIL images for Video-to-Video processing."""
    if isinstance(video_input, list):
        return video_input
    
    src = decode_media_input(video_input)
    if isinstance(src, str):
        reader = imageio.get_reader(src)
        frames = [Image.fromarray(frame).convert("RGB") for frame in reader]
        reader.close()
        return frames
    raise HTTPException(status_code=400, detail="Could not resolve video source.")


def prepare_image(img: Image.Image, size: Tuple[int, int], mode: str = "RGB") -> Image.Image:
    return ImageOps.exif_transpose(img).convert(mode).resize(size, Image.LANCZOS)


def save_image(image: Image.Image) -> str:
    path = os.path.abspath(os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.png"))
    image.save(path, format="PNG")
    return path


def process_video_export(result_frames, fps: int) -> str:
    output_filename = os.path.abspath(os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.mp4"))
    formatted_frames = []
    
    for frame in result_frames:
        if isinstance(frame, Image.Image):
            formatted_frames.append(np.array(frame, dtype=np.uint8))
        elif isinstance(frame, torch.Tensor):
            tensor_np = frame.cpu().numpy()
            if tensor_np.dtype in (np.float32, np.float64):
                tensor_np = (np.clip(tensor_np, 0, 1) * 255).astype(np.uint8)
            formatted_frames.append(tensor_np)
        else:
            formatted_frames.append(np.array(frame, dtype=np.uint8))

    writer = imageio.get_writer(output_filename, fps=fps, codec='libx264', quality=8)
    for frame in formatted_frames:
        writer.append_data(frame)
    writer.close()
    return output_filename


def extract_chat_payload(messages: List[ChatMessage]) -> Tuple[str, Optional[Any], Optional[Any]]:
    """Extracts prompt, attached image, and attached video from chat context."""
    prompt_text, init_img, init_vid = "", None, None

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
                            img_obj = item.get("image_url")
                            init_img = img_obj.get("url") if isinstance(img_obj, dict) else img_obj
                        elif "video" in item or item.get("type") == "video_url":
                            init_vid = item.get("video") or item.get("video_url")
                prompt_text = " ".join(text_parts)

            if prompt_text or init_img or init_vid:
                break

    return prompt_text.strip(), init_img, init_vid


def run_pipeline_safely(pipe, kwargs: Dict[str, Any]):
    sig = inspect.signature(pipe.__call__)
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    filtered_kwargs = kwargs if has_var_kwargs else {k: v for k, v in kwargs.items() if k in sig.parameters}
    return pipe(**filtered_kwargs)


# ============================================================================
# Dynamic Model Pipeline Router & VRAM Management
# ============================================================================

async def get_pipeline(pipeline_type: str):
    """Loads requested pipeline dynamically and unloads inactive models from GPU."""
    async with _LOCK:
        if pipeline_type in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[pipeline_type]

        # Offload existing models to keep VRAM clean
        for key in list(_PIPELINE_CACHE.keys()):
            _PIPELINE_CACHE[key].to("cpu")
            del _PIPELINE_CACHE[key]
        clean_gpu()

        print(f"--> [Model Manager] Loading pipeline: {pipeline_type}")
        kwargs = {"torch_dtype": DTYPE}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        if pipeline_type == "flux_t2i":
            pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, **kwargs)
        elif pipeline_type == "flux_fill":
            pipe = FluxFillPipeline.from_pretrained(FLUX_FILL_MODEL, **kwargs)
        elif pipeline_type == "wan_t2v":
            pipe = WanPipeline.from_pretrained(WAN_MODEL, **kwargs)
        elif pipeline_type == "wan_i2v":
            pipe = WanImageToVideoPipeline.from_pretrained(WAN_MODEL, **kwargs)
        elif pipeline_type == "wan_v2v":
            pipe = WanVideoToVideoPipeline.from_pretrained(WAN_MODEL, **kwargs)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

        pipe = pipe.to(DEVICE)
        _PIPELINE_CACHE[pipeline_type] = pipe
        return pipe


# ============================================================================
# Core API Routes
# ============================================================================

@app.get("/")
async def health_check():
    return {
        "status": "ready",
        "device": DEVICE,
        "capabilities": [
            "text-to-image", "image-to-image", "harmonization-inpainting",
            "text-to-video", "image-to-video", "video-to-video"
        ]
    }


# --- Unified Image Endpoint (T2I, I2I, Harmonization/Inpainting) ---
@app.post("/v1/images/generations")
@app.post("/v1/images/edits")
@app.post("/v1/images/harmonize")
async def generate_or_edit_image(req: ImageGenReq):
    width, height = parse_size(req.size, 1024, 1024)
    init_img = decode_media_input(req.image) if req.image else None
    mask_img = decode_media_input(req.mask) if req.mask else None

    # Router logic
    if init_img and mask_img:
        pipe_type = "flux_fill"
        pipe = await get_pipeline(pipe_type)
        kwargs = {
            "prompt": req.prompt,
            "image": prepare_image(init_img, (width, height)),
            "mask_image": prepare_image(mask_img, (width, height), "L"),
            "num_inference_steps": req.num_inference_steps,
            "guidance_scale": req.guidance_scale,
        }
    else:
        pipe_type = "flux_t2i"
        pipe = await get_pipeline(pipe_type)
        kwargs = {
            "prompt": req.prompt,
            "width": width,
            "height": height,
            "num_inference_steps": req.num_inference_steps,
            "guidance_scale": req.guidance_scale,
        }
        if init_img:
            kwargs["image"] = prepare_image(init_img, (width, height))
            kwargs["strength"] = req.strength

    try:
        result = await run_in_threadpool(run_pipeline_safely, pipe, kwargs)
        path = save_image(result.images[0])
        clean_gpu()
        
        b64_data = base64.b64encode(open(path, "rb").read()).decode("utf-8")
        return {"created": int(time.time()), "data": [{"b64_json": b64_data, "url": path}]}
    except Exception as exc:
        clean_gpu()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# --- Unified Video Endpoint (T2V, I2V, V2V) ---
@app.post("/v1/videos/generations")
@app.post("/v1/videos/edits")
async def generate_or_edit_video(req: VideoGenReq):
    width, height = parse_size(req.size, 1280, 704)
    init_img = decode_media_input(req.image) if req.image else None
    init_vid = req.video

    # Router logic
    if init_vid:
        print("--> Processing Video-to-Video Task...")
        pipe = await get_pipeline("wan_v2v")
        frames = load_video_frames(init_vid)
        resized_frames = [prepare_image(f, (width, height)) for f in frames]
        kwargs = {
            "prompt": req.prompt,
            "video": resized_frames,
            "strength": req.strength or 0.75,
            "guidance_scale": req.guidance_scale,
            "num_inference_steps": req.num_inference_steps,
        }
    elif init_img:
        print("--> Processing Image-to-Video Task...")
        pipe = await get_pipeline("wan_i2v")
        kwargs = {
            "prompt": req.prompt or "Animate this picture seamlessly",
            "image": prepare_image(init_img, (width, height)),
            "height": height,
            "width": width,
            "num_frames": req.frames,
            "guidance_scale": req.guidance_scale,
            "num_inference_steps": req.num_inference_steps,
        }
    else:
        print("--> Processing Text-to-Video Task...")
        pipe = await get_pipeline("wan_t2v")
        kwargs = {
            "prompt": req.prompt,
            "height": height,
            "width": width,
            "num_frames": req.frames,
            "guidance_scale": req.guidance_scale,
            "num_inference_steps": req.num_inference_steps,
        }

    try:
        result = await run_in_threadpool(run_pipeline_safely, pipe, kwargs)
        video_path = await run_in_threadpool(process_video_export, result.frames[0], req.fps or 24)
        clean_gpu()

        b64_data = base64.b64encode(open(video_path, "rb").read()).decode("utf-8")
        return {"created": int(time.time()), "data": [{"b64_json": b64_data, "url": video_path}]}
    except Exception as exc:
        clean_gpu()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# --- OpenAI-Compatible Chat Router (/v1/chat/completions) ---
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatReq):
    prompt, raw_img, raw_vid = extract_chat_payload(req.messages)
    msg_id = f"chatcmpl-{uuid.uuid4()}"
    model_name = req.model.lower()

    async def chat_stream():
        # Signal stream opening
        yield f"data: {json.dumps({'id': msg_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}}]})}\n\n"

        try:
            # Detect Media Mode
            is_video_task = any(k in model_name for k in ["wan", "video", "t2v", "i2v", "v2v"]) or raw_vid is not None

            if is_video_task:
                if raw_vid:
                    pipe = await get_pipeline("wan_v2v")
                    frames = load_video_frames(raw_vid)
                    kwargs = {"prompt": prompt, "video": frames, "num_inference_steps": 20}
                elif raw_img:
                    pipe = await get_pipeline("wan_i2v")
                    pil_img = prepare_image(decode_media_input(raw_img), (1280, 704))
                    kwargs = {"prompt": prompt or "Animate image", "image": pil_img, "height": 704, "width": 1280, "num_frames": 49}
                else:
                    pipe = await get_pipeline("wan_t2v")
                    kwargs = {"prompt": prompt, "height": 704, "width": 1280, "num_frames": 49}
            else:
                pipe = await get_pipeline("flux_t2i")
                kwargs = {"prompt": prompt, "width": 1024, "height": 1024}

            # Asynchronous Execution with SSE Ping Keeps Alive
            pipe_task = asyncio.create_task(run_in_threadpool(run_pipeline_safely, pipe, kwargs))
            while not pipe_task.done():
                await asyncio.sleep(4)
                yield f"data: {json.dumps({'id': msg_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': ' '}}]})}\n\n"

            result = await pipe_task
            clean_gpu()

            if is_video_task:
                out_path = await run_in_threadpool(process_video_export, result.frames[0], 24)
                md_response = f"\n\nHere is your generated video:\n\n![Generated Video]({out_path})"
            else:
                out_path = save_image(result.images[0])
                md_response = f"\n\nHere is your generated image:\n\n![Generated Image]({out_path})"

            yield f"data: {json.dumps({'id': msg_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': md_response}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            clean_gpu()
            traceback.print_exc()
            yield f"data: {json.dumps({'id': msg_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': f'\n\nError: {str(exc)}'}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(chat_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    print("Starting All-in-One Multimodal Generation Server...")
    print(f"Device: {DEVICE}, Precision: {DTYPE}")
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=600)