"""SD1.5 ControlNet img2img + IP-Adapter style transfer for single frames."""

from dataclasses import dataclass

import torch
from controlnet_aux import CannyDetector, LineartDetector, MidasDetector
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

CONTROLNETS = {
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
}

# SD1.5 up_blocks 1-3 hold the attention layers; block_1 is the lowest-resolution one, which mostly
# carries style/texture rather than layout (the SD1.5 analogue of InstantStyle's SDXL "up.block_0").
IP_MODES = {
    "style": lambda s: {"up": {"block_1": s}},
    "full": lambda s: s,
}

NEGATIVE_PROMPT = "blurry, low quality, deformed, watermark, text"


def _detectors(controls: list[str]) -> dict:
    detectors = {}
    if "lineart" in controls:
        detectors["lineart"] = LineartDetector.from_pretrained("lllyasviel/Annotators")
    if "canny" in controls:
        detectors["canny"] = CannyDetector()
    if "depth" in controls:
        detectors["depth"] = MidasDetector.from_pretrained("lllyasviel/Annotators")
    return detectors


@dataclass
class Stylizer:
    pipe: StableDiffusionControlNetImg2ImgPipeline
    controls: list[str]
    detectors: dict
    has_ip_adapter: bool


def build_pipeline(
    model: str,
    controls: list[str],
    ip_adapter: str | None = "ip-adapter-plus_sd15.safetensors",
    lora: str | None = None,
) -> Stylizer:
    controlnets = [ControlNetModel.from_pretrained(CONTROLNETS[c], torch_dtype=DTYPE) for c in controls]
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model,
        controlnet=controlnets if len(controlnets) > 1 else controlnets[0],
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )
    if ip_adapter:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=ip_adapter)
    if lora:
        pipe.load_lora_weights(lora)
    # No enable_attention_slicing(): it swaps out IP-Adapter's attention processors and breaks it.
    pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return Stylizer(pipe, controls, _detectors(controls), ip_adapter is not None)


def _fit(img: Image.Image, width: int) -> Image.Image:
    """Resize to `width`, keeping aspect ratio, with both sides a multiple of 8."""
    w = width // 8 * 8
    h = round(img.height * width / img.width / 8) * 8
    return img.resize((w, h), Image.LANCZOS)


def stylize_frame(
    s: Stylizer,
    frame: Image.Image,
    style_images: list[Image.Image],
    prompt: str,
    width: int = 640,
    strength: float = 0.65,
    control_scale: float = 0.8,
    ip_scale: float = 0.8,
    ip_mode: str = "style",
    steps: int = 25,
    guidance: float = 6.0,
    seed: int = 0,
) -> Image.Image:
    src = _fit(frame.convert("RGB"), width)
    size = src.size
    control_images = [
        s.detectors[c](src, detect_resolution=min(size), image_resolution=min(size)).resize(size)
        for c in s.controls
    ]

    kwargs = {}
    if s.has_ip_adapter and style_images:
        s.pipe.set_ip_adapter_scale(IP_MODES[ip_mode](ip_scale))
        kwargs["ip_adapter_image"] = [style_images]

    out = s.pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=src,
        control_image=control_images if len(control_images) > 1 else control_images[0],
        controlnet_conditioning_scale=[control_scale] * len(control_images) if len(control_images) > 1 else control_scale,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator("cpu").manual_seed(seed),
        width=size[0],
        height=size[1],
        **kwargs,
    ).images[0]
    return out.resize(frame.size, Image.LANCZOS)
