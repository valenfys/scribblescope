"""FLUX ControlNet img2img + IP-Adapter style transfer for single frames, run on fal.ai."""

import io
from dataclasses import dataclass, field

import cv2
import fal_client
import httpx
import numpy as np
from PIL import Image

ENDPOINT = "fal-ai/flux-general/image-to-image"

# fal bills $0.075 per megapixel, rounded *up*, so stay just under 1MP per frame.
MAX_PIXELS = 1_000_000

CONTROLNET_UNION = "InstantX/FLUX.1-dev-Controlnet-Union"
IP_ADAPTER = {
    "path": "XLabs-AI/flux-ip-adapter",
    "weight_name": "ip_adapter.safetensors",
    "image_encoder_path": "openai/clip-vit-large-patch14",
}

# canny: edge map computed locally. tile: the (downscaled) source frame itself.
CONTROLS = ["canny", "tile"]


@dataclass
class Stylizer:
    controls: list[str]
    lora: str | None = None
    lora_scale: float = 1.0
    # Uploaded URLs keyed by local image identity, so each style image is uploaded once.
    _uploads: dict = field(default_factory=dict)

    def upload_style(self, img: Image.Image) -> str:
        if id(img) not in self._uploads:
            self._uploads[id(img)] = fal_client.upload_image(img, "png")
        return self._uploads[id(img)]


def build_pipeline(controls: list[str], lora: str | None = None, lora_scale: float = 1.0) -> Stylizer:
    return Stylizer(controls, lora, lora_scale)


def _fit(img: Image.Image, width: int) -> Image.Image:
    """Resize to `width` (capped so the frame stays under MAX_PIXELS), sides a multiple of 16."""
    scale = min(width / img.width, (MAX_PIXELS / (img.width * img.height)) ** 0.5)
    w = int(img.width * scale) // 16 * 16
    h = int(img.height * scale) // 16 * 16
    return img.resize((w, h), Image.LANCZOS)


def _canny(img: Image.Image) -> Image.Image:
    edges = cv2.Canny(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY), 100, 200)
    return Image.fromarray(edges).convert("RGB")


def stylize_frame(
    s: Stylizer,
    frame: Image.Image,
    style_image: Image.Image | None,
    prompt: str,
    width: int = 768,
    strength: float = 0.75,
    control_scale: float = 0.7,
    ip_scale: float = 0.7,
    steps: int = 28,
    guidance: float = 3.5,
    seed: int = 0,
) -> Image.Image:
    src = _fit(frame.convert("RGB"), width)
    src_url = fal_client.upload_image(src, "png")

    controls = []
    for c in s.controls:
        control_url = fal_client.upload_image(_canny(src), "png") if c == "canny" else src_url
        controls.append({"control_image_url": control_url, "control_mode": c, "conditioning_scale": control_scale})

    args = {
        "prompt": prompt,
        "image_url": src_url,
        "strength": strength,
        "image_size": {"width": src.width, "height": src.height},
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
        "output_format": "png",
    }
    if controls:
        # The endpoint supports one union ControlNet, which takes several control modes.
        args["controlnet_unions"] = [{"path": CONTROLNET_UNION, "controls": controls}]
    if style_image is not None and ip_scale > 0:
        args["ip_adapters"] = [IP_ADAPTER | {"image_url": s.upload_style(style_image), "scale": ip_scale}]
    if s.lora:
        args["loras"] = [{"path": s.lora, "scale": s.lora_scale}]

    result = fal_client.subscribe(ENDPOINT, arguments=args)
    return fetch_image(result["images"][0]["url"]).resize(frame.size, Image.LANCZOS)


def fetch_image(url: str) -> Image.Image:
    resp = httpx.get(url, timeout=120, follow_redirects=True)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")
