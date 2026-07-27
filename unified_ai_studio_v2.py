import os
import io
import gc
import re
import time
import uuid
import base64
import asyncio
from typing import Optional, Dict, Any, Tuple, List
from urllib.parse import urlparse, unquote

import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict
from diffusers import DiffusionPipeline

try:
    from diffusers import Flux2KleinPipeline
except Exception:
    Flux2KleinPipeline = None

app = FastAPI(title="Unified Multimodal AI Studio Backend RTX 5090")

HOST = os.getenv("SD_HOST", "127.0.0.1")
PORT = int(os.getenv("SD_PORT", "8000"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.getenv("SD_OUTPUT_DIR", os.path.join(BASE_DIR, "output_media"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

_LOADED_MODELS: Dict[str, Any] = {}
_MODEL_LOCK = asyncio.Lock()
INFERENCE_LOCK = asyncio.Lock()
LAST_REQUEST_DEBUG: Dict[str, Any] = {}


def get_hf_cache_path(repo_folder_name: str) -> Optional[str]:
    cache_dir = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
        repo_folder_name,
        "snapshots",
    )

    if not os.path.exists(cache_dir):
        return None

    snapshots = [
        os.path.join(cache_dir, d)
        for d in os.listdir(cache_dir)
        if os.path.isdir(os.path.join(cache_dir, d))
    ]

    if not snapshots:
        return None

    return max(snapshots, key=os.path.getmtime)


WAN_CACHE_PATH = get_hf_cache_path("models--Wan-AI--Wan2.2-TI2V-5B-Diffusers")
WAN_LOCAL_PATH = os.path.join(MODELS_DIR, "Wan2.2-TI2V-5B-Diffusers")

if WAN_CACHE_PATH:
    DEFAULT_WAN_MODEL = WAN_CACHE_PATH
    print(f"--> Found cached Wan 2.2 model at: {DEFAULT_WAN_MODEL}")
elif os.path.exists(WAN_LOCAL_PATH):
    DEFAULT_WAN_MODEL = WAN_LOCAL_PATH
    print(f"--> Found local Wan 2.2 model at: {DEFAULT_WAN_MODEL}")
else:
    DEFAULT_WAN_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

FLUX_CACHE_PATH = get_hf_cache_path("models--black-forest-labs--FLUX.2-klein-4B")
FLUX_LOCAL_PATH = os.path.join(MODELS_DIR, "FLUX.2-klein-4B")

if FLUX_CACHE_PATH:
    DEFAULT_FLUX_MODEL = FLUX_CACHE_PATH
    print(f"--> Found cached FLUX model at: {DEFAULT_FLUX_MODEL}")
elif os.path.exists(FLUX_LOCAL_PATH):
    DEFAULT_FLUX_MODEL = FLUX_LOCAL_PATH
    print(f"--> Found local FLUX model at: {DEFAULT_FLUX_MODEL}")
else:
    DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.2-klein-4B"


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: int = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"
    negative_prompt: Optional[str] = None
    guidance_scale: Optional[float] = 1.0
    num_inference_steps: Optional[int] = 4
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
    guidance_scale: Optional[float] = 1.0
    num_inference_steps: Optional[int] = 4
    seed: Optional[int] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "flux"
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    files: Optional[List[Any]] = None
    images: Optional[List[Any]] = None
    attachments: Optional[List[Any]] = None
    prompt_images: Optional[List[Any]] = None
    model_config = ConfigDict(extra="allow")


def clean_gpu() -> None:
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
        raw = base64.b64decode(value, validate=False)
        return Image.open(io.BytesIO(raw)).convert(mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode base64 image: {exc}")


async def upload_to_image(upload: UploadFile, mode: str = "RGB") -> Image.Image:
    try:
        raw = await upload.read()
        return Image.open(io.BytesIO(raw)).convert(mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {exc}")


def prepare_image(img: Image.Image, size: Tuple[int, int], mode: str = "RGB") -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = img.convert(mode)
    img = img.resize(size, Image.LANCZOS)
    return img


def image_to_b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_image(image: Image.Image) -> str:
    path = os.path.abspath(os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.png"))
    image.save(path, format="PNG")
    return path


def openai_image_response(images: List[Image.Image], response_format: str = "b64_json") -> Dict[str, Any]:
    data = []
    for image in images:
        path = save_image(image)
        if response_format == "url":
            data.append({"url": path})
        else:
            data.append({"b64_json": image_to_b64_png(image)})
    return {"created": int(time.time()), "data": data}


def looks_like_base64_image(value: str) -> bool:
    return len(value) >= 200 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", value) is not None


def normalize_path_candidate(value: str) -> str:
    value = value.strip().strip('"').strip("'")
    value = unquote(value)
    if value.startswith("file://"):
        parsed = urlparse(value)
        return unquote(parsed.path)
    return value


def candidate_file_ids_from_string(value: str) -> List[str]:
    s = value.strip()
    candidates: List[str] = []
    patterns = [
        r"/api/v1/files/([^/]+)/content",
        r"/api/files/([^/]+)/content",
        r"/files/([^/]+)/content",
        r"/api/v1/files/([^/?#]+)",
        r"/api/files/([^/?#]+)",
        r"/files/([^/?#]+)",
        r"file[-_]?id=([^&#]+)",
        r"file_id=([^&#]+)",
        r"id=([^&#]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            candidates.append(unquote(match.group(1)))
    parsed = urlparse(s)
    if parsed.path:
        parts = [p for p in parsed.path.split("/") if p]
        for part in reversed(parts):
            if part.lower() not in {"content", "download", "view", "file", "files", "image", "images"}:
                candidates.append(unquote(part))
                break
    last = s.split("/")[-1].split("?")[0].split("#")[0]
    if last:
        candidates.append(unquote(last))
    bad = {"content", "download", "view", "file", "files", "image", "images", "api", "v1"}
    cleaned: List[str] = []
    for item in candidates:
        item = item.strip().strip('"').strip("'")
        if item and item.lower() not in bad and len(item) >= 4 and item not in cleaned:
            cleaned.append(item)
    return cleaned


def image_extensions() -> Tuple[str, ...]:
    return ("", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


def search_roots() -> List[str]:
    roots = [
        BASE_DIR,
        os.path.dirname(BASE_DIR),
        os.path.expanduser("~"),
        os.getenv("ODYSSEUS_UPLOAD_DIR", ""),
        os.getenv("OPEN_WEBUI_DATA_DIR", ""),
        os.getenv("OPEN_WEBUI_UPLOAD_DIR", ""),
        os.getenv("SD_UPLOAD_DIR", ""),
        os.getenv("TEMP", ""),
        os.getenv("TMP", ""),
    ]
    result = []
    for root in roots:
        if root and os.path.exists(root) and root not in result:
            result.append(root)
    return result


def search_sub_dirs() -> List[str]:
    return [
        "",
        "uploads",
        "files",
        "data",
        "data/uploads",
        "backend/data",
        "backend/data/uploads",
        ".open-webui",
        ".open-webui/uploads",
        ".open-webui/data",
        ".open-webui/data/uploads",
        "AppData/Local/Temp",
        "AppData/Roaming",
    ]


def find_file_id_on_disk(file_id: str) -> Optional[str]:
    if not file_id:
        return None
    file_id = file_id.strip().strip('"').strip("'")
    roots = search_roots()
    sub_dirs = search_sub_dirs()
    extensions = image_extensions()
    for root in roots:
        for sub in sub_dirs:
            base = os.path.join(root, sub)
            if not os.path.exists(base):
                continue
            for ext in extensions:
                direct_path = os.path.join(base, file_id + ext)
                if os.path.isfile(direct_path):
                    return direct_path
            possible_path = os.path.join(base, file_id)
            if os.path.isfile(possible_path):
                return possible_path
    max_files_checked = 5000
    checked = 0
    file_id_lower = file_id.lower()
    for root in roots:
        for sub in sub_dirs:
            base = os.path.join(root, sub)
            if not os.path.exists(base):
                continue
            try:
                for dirpath, _, filenames in os.walk(base):
                    for filename in filenames:
                        checked += 1
                        if checked > max_files_checked:
                            return None
                        filename_lower = filename.lower()
                        if filename_lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")):
                            if file_id_lower in filename_lower:
                                return os.path.join(dirpath, filename)
            except Exception:
                continue
    return None


def try_open_image_path(path: str, indent: str = "") -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        print(f"{indent}--> [IMAGE] opened image file: {path}, size={img.size}")
        return img
    except Exception as exc:
        print(f"{indent}--> [WARN] failed opening image file '{path}': {exc}")
        return None


def try_load_image_from_val(val: Any, depth: int = 0) -> Optional[Image.Image]:
    indent = "  " * depth
    if val is None:
        return None
    if isinstance(val, Image.Image):
        return val.convert("RGB")
    if isinstance(val, list):
        for item in val:
            img = try_load_image_from_val(item, depth + 1)
            if img:
                return img
        return None
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str:
            return None
        if val_str.startswith("data:image"):
            print(f"{indent}--> [IMAGE] data:image base64 detected")
            return decode_base64_image(val_str, "RGB")
        if looks_like_base64_image(val_str):
            try:
                print(f"{indent}--> [IMAGE] raw base64 candidate detected")
                return decode_base64_image(val_str, "RGB")
            except Exception:
                pass
        path_candidate = normalize_path_candidate(val_str)
        if os.path.isfile(path_candidate):
            print(f"{indent}--> [IMAGE] direct path match: {path_candidate}")
            return try_open_image_path(path_candidate, indent)
        for file_id in candidate_file_ids_from_string(val_str):
            resolved = find_file_id_on_disk(file_id)
            if resolved:
                print(f"{indent}--> [IMAGE] resolved file id '{file_id}' to: {resolved}")
                return try_open_image_path(resolved, indent)
        return None
    if isinstance(val, dict):
        if "image_url" in val:
            image_url_val = val.get("image_url")
            if isinstance(image_url_val, dict):
                img = try_load_image_from_val(image_url_val.get("url"), depth + 1)
                if img:
                    return img
            img = try_load_image_from_val(image_url_val, depth + 1)
            if img:
                return img
        priority_keys = [
            "url", "path", "file_path", "filepath", "full_path", "filename", "name",
            "b64_json", "base64", "data", "id", "file_id", "fileId", "file", "meta", "metadata",
        ]
        for key in priority_keys:
            if key in val and val.get(key):
                img = try_load_image_from_val(val.get(key), depth + 1)
                if img:
                    return img
        for key, sub_val in val.items():
            if key in priority_keys:
                continue
            img = try_load_image_from_val(sub_val, depth + 1)
            if img:
                return img
    return None


def extract_image_from_request(req: ChatCompletionRequest) -> Tuple[str, Optional[Image.Image]]:
    req_dict = req.model_dump() if hasattr(req, "model_dump") else req.dict()

    import json

    print("=" * 80)
    print("FILES:")
    print(json.dumps(req_dict.get("files"), indent=2, default=str))

    print("=" * 80)
    print("IMAGES:")
    print(json.dumps(req_dict.get("images"), indent=2, default=str))

    print("=" * 80)
    print("ATTACHMENTS:")
    print(json.dumps(req_dict.get("attachments"), indent=2, default=str))

    print("=" * 80)
    print("PROMPT_IMAGES:")
    print(json.dumps(req_dict.get("prompt_images"), indent=2, default=str))

    print("=" * 80)

    prompt = ""
    input_image: Optional[Image.Image] = None
    print("--> [DEBUG] request top-level keys:", list(req_dict.keys()))
    messages = req.messages or []
    for msg in reversed(messages):
        content = msg.get("content", "")
        if isinstance(content, str):
            if not prompt:
                prompt = content
            for match in re.finditer(r"!\[.*?\]\((.*?)\)", content):
                img_ref = match.group(1).strip()
                input_image = try_load_image_from_val(img_ref)
                if input_image:
                    prompt = re.sub(r"!\[.*?\]\((.*?)\)", "", prompt).strip()
                    break
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    if not prompt:
                        prompt = part
                    continue
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in ("text", "input_text") and not prompt:
                    prompt = part.get("text", "") or part.get("content", "")
                if part_type in ("image_url", "input_image", "image") and not input_image:
                    input_image = try_load_image_from_val(part)
                if not input_image:
                    input_image = try_load_image_from_val(part)
                if input_image:
                    break
        if not input_image:
            for msg_key in ["files", "images", "attachments", "prompt_images"]:
                input_image = try_load_image_from_val(msg.get(msg_key))
                if input_image:
                    break
        if input_image:
            break
    if not input_image:
        for key in ["files", "images", "attachments", "prompt_images"]:
            input_image = try_load_image_from_val(req_dict.get(key))
            if input_image:
                break
    global LAST_REQUEST_DEBUG
    LAST_REQUEST_DEBUG = {
        "top_level_keys": list(req_dict.keys()),
        "message_count": len(messages),
        "prompt_preview": prompt[:500] if prompt else "",
        "prompt_length": len(prompt or ""),
        "image_found": input_image is not None,
        "image_size": input_image.size if input_image else None,
        "image_mode": input_image.mode if input_image else None,
        "files": req_dict.get("files"),
        "images": req_dict.get("images"),
        "attachments": req_dict.get("attachments"),
        "prompt_images": req_dict.get("prompt_images"),
    }
    print(f"--> [DEBUG] prompt length={len(prompt or '')}, image_found={input_image is not None}")
    if input_image:
        print(f"--> [DEBUG] image size={input_image.size}, mode={input_image.mode}")
    return prompt, input_image


async def get_pipeline(model_name: Optional[str], image_mode: bool = False):
    async with _MODEL_LOCK:
        model_name = model_name or "flux"
        is_wan = "wan" in model_name.lower()
        if is_wan and image_mode:
            model_key = "wan_i2v"
        elif is_wan:
            model_key = "wan_t2v"
        else:
            model_key = "flux"
        if model_key in _LOADED_MODELS:
            return _LOADED_MODELS[model_key], "wan" if is_wan else "flux"
        clean_gpu()
        kwargs = {"torch_dtype": DTYPE}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        if is_wan:
            if image_mode:
                print(f"--> [Model Manager] Loading Wan image-to-video pipeline: {DEFAULT_WAN_MODEL}")
                try:
                    from diffusers import WanImageToVideoPipeline
                    pipe = WanImageToVideoPipeline.from_pretrained(DEFAULT_WAN_MODEL, **kwargs)
                except Exception as exc:
                    print(f"--> [WARN] WanImageToVideoPipeline failed: {exc}")
                    pipe = DiffusionPipeline.from_pretrained(DEFAULT_WAN_MODEL, **kwargs)
            else:
                print(f"--> [Model Manager] Loading Wan text-to-video pipeline: {DEFAULT_WAN_MODEL}")
                try:
                    from diffusers import WanPipeline
                    pipe = WanPipeline.from_pretrained(DEFAULT_WAN_MODEL, **kwargs)
                except Exception as exc:
                    print(f"--> [WARN] WanPipeline failed: {exc}")
                    pipe = DiffusionPipeline.from_pretrained(DEFAULT_WAN_MODEL, **kwargs)
            pipe = pipe.to(DEVICE)
            _LOADED_MODELS[model_key] = pipe
            return pipe, "wan"
        print(f"--> [Model Manager] Loading FLUX pipeline: {DEFAULT_FLUX_MODEL}")
        if Flux2KleinPipeline is not None:
            try:
                pipe = Flux2KleinPipeline.from_pretrained(DEFAULT_FLUX_MODEL, **kwargs)
            except Exception as exc:
                print(f"--> [WARN] Flux2KleinPipeline failed: {exc}")
                pipe = DiffusionPipeline.from_pretrained(DEFAULT_FLUX_MODEL, **kwargs)
        else:
            pipe = DiffusionPipeline.from_pretrained(DEFAULT_FLUX_MODEL, **kwargs)
        pipe = pipe.to(DEVICE)
        _LOADED_MODELS[model_key] = pipe
        return pipe, "flux"


def run_pipeline_safely(pipe, kwargs):
    if hasattr(pipe, "scheduler") and hasattr(pipe.scheduler, "_step_index"):
        pipe.scheduler._step_index = None
    return pipe(**kwargs)


def get_extra_value(req: BaseModel, key: str, default: Any = None) -> Any:
    try:
        if hasattr(req, "model_extra") and req.model_extra:
            return req.model_extra.get(key, default)
    except Exception:
        pass
    return default


@app.get("/")
async def root():
    return {
        "status": "ok",
        "device": DEVICE,
        "precision": str(DTYPE),
        "base_dir": BASE_DIR,
        "output_dir": OUTPUT_DIR,
        "wan_model": DEFAULT_WAN_MODEL,
        "flux_model": DEFAULT_FLUX_MODEL,
    }


@app.get("/debug/last-request")
async def debug_last_request():
    return LAST_REQUEST_DEBUG


@app.get("/v1/models")
@app.get("/api/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "black-forest-labs/FLUX.2-klein-4B", "object": "model", "owned_by": "local"},
            {"id": "FLUX.2-klein-4B", "object": "model", "owned_by": "local"},
            {"id": "wan_t2v", "object": "model", "owned_by": "local"},
            {"id": "wan_i2v", "object": "model", "owned_by": "local"},
            {"id": "Wan2.2-TI2V-5B-Diffusers", "object": "model", "owned_by": "local"},
        ],
    }


@app.get("/v1/images/progress/{job_id}")
async def get_progress(job_id: str):
    return {"status": "processing", "progress": 50, "id": job_id}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    prompt, input_image = extract_image_from_request(req)
    requested_model = req.model or "flux"
    print("--> [DEBUG] requested_model:", requested_model)
    print("--> [DEBUG] prompt preview:", repr((prompt or "")[:300]))
    print("--> [DEBUG] input_image:", input_image.size if input_image else None)
    pipe, model_type = await get_pipeline(requested_model, image_mode=input_image is not None)
    try:
        async with INFERENCE_LOCK:
            loop = asyncio.get_running_loop()
            if model_type == "wan":
                wan_width = int(get_extra_value(req, "width", 832) or 832)
                wan_height = int(get_extra_value(req, "height", 480) or 480)
                wan_frames = int(get_extra_value(req, "num_frames", 81) or 81)
                wan_steps = int(get_extra_value(req, "num_inference_steps", 50) or 50)
                wan_guidance = float(get_extra_value(req, "guidance_scale", 5.0) or 5.0)
                negative_prompt = get_extra_value(req, "negative_prompt", None)
                wan_width = max(256, min(2048, (wan_width // 16) * 16))
                wan_height = max(256, min(2048, (wan_height // 16) * 16))
                kwargs = {
                    "prompt": prompt or "Animate the provided image naturally.",
                    "height": wan_height,
                    "width": wan_width,
                    "num_frames": wan_frames,
                    "num_inference_steps": wan_steps,
                    "guidance_scale": wan_guidance,
                    "output_type": "pil",
                }
                if negative_prompt:
                    kwargs["negative_prompt"] = negative_prompt
                if input_image:
                    print("--> [Wan 2.2] Image input detected. Running image-to-video.")
                    kwargs["image"] = prepare_image(input_image, (wan_width, wan_height))
                else:
                    print("--> [Wan 2.2] No image detected. Running text-to-video.")
                output = await loop.run_in_executor(None, run_pipeline_safely, pipe, kwargs)
                video_frames = getattr(output, "frames", None)
                if video_frames is not None:
                    if isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], list):
                        video_frames = video_frames[0]
                    save_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.mp4"))
                    print(f"--> Exporting video to: {save_path}")
                    from diffusers.utils import export_to_video
                    export_to_video(video_frames, save_path, fps=15)
                    content = f"Video generated successfully: `{save_path}`"
                else:
                    content = "Video generation completed, but no video frames were returned."
            else:
                width = int(get_extra_value(req, "width", 1024) or 1024)
                height = int(get_extra_value(req, "height", 1024) or 1024)
                steps = int(get_extra_value(req, "num_inference_steps", 4) or 4)
                guidance = float(get_extra_value(req, "guidance_scale", 1.0) or 1.0)
                seed = get_extra_value(req, "seed", None)
                width = max(256, min(2048, (width // 16) * 16))
                height = max(256, min(2048, (height // 16) * 16))
                kwargs = {
                    "prompt": prompt or "Generate an image.",
                    "height": height,
                    "width": width,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                    "generator": make_generator(seed),
                }
                if input_image:
                    print("--> [FLUX] Image input detected. Passing image to pipeline.")
                    kwargs["image"] = prepare_image(input_image, (width, height))
                result = await loop.run_in_executor(None, run_pipeline_safely, pipe, kwargs)
                if not hasattr(result, "images") or not result.images:
                    raise RuntimeError("Pipeline did not return images.")
                b64_img = image_to_b64_png(result.images[0])
                content = f"data:image/png;base64,{b64_img}"
        clean_gpu()
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }
    except torch.cuda.OutOfMemoryError:
        print("--> Generation failed: CUDA out of memory.")
        clean_gpu()
        raise HTTPException(status_code=507, detail="CUDA out of memory.")
    except Exception as exc:
        print(f"--> Generation failed with error: {exc}")
        clean_gpu()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/images/generations")
async def image_generations(req: ImageGenerationRequest):
    width, height = parse_size(req.size)
    n = max(1, min(int(req.n or 1), 4))
    pipe, model_type = await get_pipeline(req.model or "flux", image_mode=False)
    kwargs = {
        "prompt": req.prompt,
        "num_inference_steps": int(req.num_inference_steps or 4),
        "guidance_scale": float(req.guidance_scale if req.guidance_scale is not None else 1.0),
        "generator": make_generator(req.seed),
    }
    if model_type == "wan":
        kwargs.update({"height": height, "width": width, "num_frames": 81, "output_type": "pil"})
    else:
        kwargs.update({"width": width, "height": height, "num_images_per_prompt": n})
    try:
        async with INFERENCE_LOCK:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, run_pipeline_safely, pipe, kwargs)
        clean_gpu()
        if model_type == "wan":
            video_frames = getattr(result, "frames", None)
            if video_frames is None:
                raise RuntimeError("Wan pipeline did not return video frames.")
            if isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], list):
                video_frames = video_frames[0]
            save_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.mp4"))
            from diffusers.utils import export_to_video
            export_to_video(video_frames, save_path, fps=15)
            return {"created": int(time.time()), "data": [{"url": save_path}]}
        return openai_image_response(result.images, req.response_format or "b64_json")
    except torch.cuda.OutOfMemoryError:
        clean_gpu()
        raise HTTPException(status_code=507, detail="CUDA out of memory.")
    except Exception as exc:
        clean_gpu()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/images/edits")
async def image_edits(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        prompt = str(form.get("prompt") or "")
        upload = form.get("image")
        init_image = await upload_to_image(upload, "RGB") if upload else None
        size = str(form.get("size") or "1024x1024")
        response_format = str(form.get("response_format") or "b64_json")
        steps = int(form.get("num_inference_steps") or 4)
        seed = int(form.get("seed")) if form.get("seed") else None
        n = int(form.get("n") or 1)
        model = str(form.get("model") or "flux")
    else:
        payload = await request.json()
        req = ImageEditJsonRequest(**payload)
        prompt = req.prompt
        init_image = decode_base64_image(req.image, "RGB")
        size = req.size
        response_format = req.response_format or "b64_json"
        steps = int(req.num_inference_steps or 4)
        seed = req.seed
        n = req.n
        model = req.model or "flux"
    if init_image is None:
        raise HTTPException(status_code=400, detail="No base image provided for editing.")
    width, height = parse_size(size)
    init_image = prepare_image(init_image, (width, height), "RGB")
    pipe, model_type = await get_pipeline(model, image_mode=True)
    if model_type == "wan":
        kwargs = {
            "prompt": prompt or "Animate the provided image naturally.",
            "image": prepare_image(init_image, (832, 480)),
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "num_inference_steps": max(10, steps),
            "guidance_scale": 5.0,
            "output_type": "pil",
        }
    else:
        kwargs = {
            "prompt": prompt,
            "image": init_image,
            "height": height,
            "width": width,
            "num_images_per_prompt": max(1, min(int(n or 1), 4)),
            "num_inference_steps": steps,
            "guidance_scale": 1.0,
            "generator": make_generator(seed),
        }
    try:
        async with INFERENCE_LOCK:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, run_pipeline_safely, pipe, kwargs)
        clean_gpu()
        if model_type == "wan":
            video_frames = getattr(result, "frames", None)
            if video_frames is None:
                raise RuntimeError("Wan image-to-video pipeline did not return video frames.")
            if isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], list):
                video_frames = video_frames[0]
            save_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.mp4"))
            from diffusers.utils import export_to_video
            export_to_video(video_frames, save_path, fps=15)
            return {"created": int(time.time()), "data": [{"url": save_path}]}
        return openai_image_response(result.images, response_format)
    except torch.cuda.OutOfMemoryError:
        clean_gpu()
        raise HTTPException(status_code=507, detail="CUDA out of memory during edit.")
    except Exception as exc:
        clean_gpu()
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    print("Starting All-in-One Multimodal Generation Server...")
    print(f"Device: {DEVICE}, Precision: {DTYPE}")
    print(f"Base dir: {BASE_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Wan model: {DEFAULT_WAN_MODEL}")
    print(f"Flux model: {DEFAULT_FLUX_MODEL}")
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=600)
