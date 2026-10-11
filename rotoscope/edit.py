"""Instruction-based style transfer for single frames with an image-editing model on fal.ai.

The frame and a few style references are sent together with an instruction to redraw the frame in the
references' style. There is no strength/ControlNet: composition is held by the instruction alone.
"""

import fal_client
from PIL import Image

from rotoscope.stylize import fetch_image

DEFAULT_MODEL = "fal-ai/nano-banana-2/edit"

# Nano Banana 2 edit: $0.08 per image at 1K; 0.5K is 0.75x, 2K 1.5x, 4K 2x.
USD_PER_IMAGE = {"0.5K": 0.06, "1K": 0.08, "2K": 0.12, "4K": 0.16}

ASPECT_RATIOS = ["21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"]

INSTRUCTION = (
    "Redraw the first image in the artistic style of the other images: match their medium, mark-making, "
    "palette, texture and level of detail. Keep the first image's exact composition, framing, camera angle, "
    "and the position, pose, size and proportions of every person and object. Do not add, remove or move "
    "anything, and do not copy any content from the style images."
)


def _aspect_ratio(size: tuple[int, int]) -> str:
    w, h = size
    return min(ASPECT_RATIOS, key=lambda r: abs(int(r.split(":")[0]) / int(r.split(":")[1]) - w / h))


class EditStylizer:
    def __init__(self, model: str = DEFAULT_MODEL, resolution: str = "1K"):
        self.model = model
        self.resolution = resolution
        self._uploads: dict[int, str] = {}

    def upload_style(self, img: Image.Image) -> str:
        if id(img) not in self._uploads:
            self._uploads[id(img)] = fal_client.upload_image(img, "png")
        return self._uploads[id(img)]

    def stylize_frame(
        self,
        frame: Image.Image,
        style_images: list[Image.Image],
        prompt: str,
        seed: int = 0,
    ) -> Image.Image:
        frame = frame.convert("RGB")
        instruction = INSTRUCTION + (f" Style notes: {prompt}" if prompt else "")
        args = {
            "prompt": instruction,
            # Frames are sent as JPEG to keep uploads small; the model re-renders the image anyway.
            "image_urls": [fal_client.upload_image(frame, "jpeg")] + [self.upload_style(s) for s in style_images],
            "aspect_ratio": _aspect_ratio(frame.size),
            "resolution": self.resolution,
            "seed": seed,
            "output_format": "png",
        }
        result = fal_client.subscribe(self.model, arguments=args)
        return fetch_image(result["images"][0]["url"]).resize(frame.size, Image.LANCZOS)
