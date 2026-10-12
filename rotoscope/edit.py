"""Instruction-based style transfer for single frames with an image-editing model on fal.ai.

The frame and a few style references are sent together with an instruction to redraw the frame in the
references' style. There is no strength/ControlNet: composition is held by the instruction alone.
"""

import fal_client
from PIL import Image

from rotoscope.stylize import fetch_image

# Per-model endpoint, price per output image, and how many style references fit alongside the frame.
# Seedream models take an explicit size with a minimum area (`min_pixels`); Seedream 5 has no seed.
# Seedream 5 Pro also charges per extra input image (`usd_per_ref`).
MODELS = {
    # $0.08 per image at 1K; 0.5K is 0.75x, 2K 1.5x, 4K 2x.
    "nano-banana-2": {"endpoint": "fal-ai/nano-banana-2/edit", "max_refs": 13,
                      "usd": {"0.5K": 0.06, "1K": 0.08, "2K": 0.12, "4K": 0.16}},
    "seedream-4": {"endpoint": "fal-ai/bytedance/seedream/v4/edit", "max_refs": 9, "usd": 0.03,
                   "min_pixels": 921_600},
    "seedream-4.5": {"endpoint": "fal-ai/bytedance/seedream/v4.5/edit", "max_refs": 9, "usd": 0.04,
                     "min_pixels": 2560 * 1440},
    "seedream-5-lite": {"endpoint": "fal-ai/bytedance/seedream/v5/lite/edit", "max_refs": 9, "usd": 0.035,
                        "min_pixels": 2560 * 1440, "seed": False},
    "seedream-5-flash": {"endpoint": "bytedance/seedream/v5/flash/edit", "max_refs": 9, "usd": 0.027,
                         "min_pixels": 1024 * 1024, "seed": False},
    # $0.0675 up to 1536x1536 output (double above that); min_pixels keeps us under it.
    "seedream-5-pro": {"endpoint": "bytedance/seedream/v5/pro/edit", "max_refs": 9, "usd": 0.0675,
                       "usd_per_ref": 0.0045, "min_pixels": 1024 * 1024, "seed": False},
}
DEFAULT_MODEL = "seedream-4"
RESOLUTIONS = ["0.5K", "1K", "2K", "4K"]


def usd_per_image(model: str, resolution: str, refs: int = 0) -> float:
    usd = MODELS[model]["usd"]
    base = usd[resolution] if isinstance(usd, dict) else usd
    return base + MODELS[model].get("usd_per_ref", 0) * min(refs, MODELS[model]["max_refs"])


ASPECT_RATIOS = ["21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"]

INSTRUCTION = (
    "Redraw the first image in the artistic style of the other images: match their medium, mark-making, "
    "palette, texture and level of detail. Keep the first image's exact composition, framing, camera angle, "
    "and the position, pose, size and proportions of every person and object. Do not add, remove or move "
    "anything, and do not copy any content from the style images."
)


# Structure maps sent in place of the photo, so its colours and textures can't leak into the result.
# Both preprocessors are free on fal.ai.
STRUCTURE_ENDPOINTS = {
    "lineart": "fal-ai/image-preprocessors/lineart",
    "depth": "fal-ai/image-preprocessors/depth-anything/v2",
}
STRUCTURE_INSTRUCTION = {
    "lineart": "The first image is a line drawing traced from a video frame. Colour it in as a drawing in the "
               "artistic style of the other images: match their medium, mark-making, palette, texture and level of "
               "detail. Follow the line drawing exactly: every person, object, building and edge must stay in exactly "
               "the place, size and outline shown by its lines. Do not add anything that has no lines in the first "
               "image (no extra people, lamps, vehicles, buildings or props), do not remove, move or resize anything, "
               "and keep the person as one separate shape that does not merge with other objects. Take only the "
               "style from the other images, never their subject matter or content.",
    "depth": "The first image is a depth map of a video frame (near is bright, far is dark). Draw the scene it "
             "describes, keeping the position, shape and size of every person and object, in the artistic style "
             "of the other images: match their medium, mark-making, palette, texture and level of detail. Take "
             "only the style from the other images, never their subject matter or content.",
}


def instruction(structure: str, look: str = "") -> str:
    """The built-in instruction for `structure`, followed by the style preset's `look` for it."""
    base = INSTRUCTION if structure == "photo" else STRUCTURE_INSTRUCTION[structure]
    return f"{base} {look}" if look else base


def _structure_map_url(frame: Image.Image, structure: str) -> str:
    result = fal_client.subscribe(STRUCTURE_ENDPOINTS[structure],
                                  arguments={"image_url": fal_client.upload_image(frame, "jpeg")})
    return result["image"]["url"]


def _aspect_ratio(size: tuple[int, int]) -> str:
    w, h = size
    return min(ASPECT_RATIOS, key=lambda r: abs(int(r.split(":")[0]) / int(r.split(":")[1]) - w / h))


def _seedream_size(size: tuple[int, int], min_pixels: int) -> dict:
    """Smallest frame-shaped size (multiple of 16) that meets Seedream's minimum area."""
    w, h = size
    scale = (min_pixels / (w * h)) ** 0.5
    return {"width": -(-int(w * scale) // 16) * 16, "height": -(-int(h * scale) // 16) * 16}


class EditStylizer:
    def __init__(self, model: str = DEFAULT_MODEL, resolution: str = "1K", structure: str = "photo",
                 look: str = ""):
        self.model = model
        self.endpoint = MODELS[model]["endpoint"]
        self.max_refs = MODELS[model]["max_refs"]
        self.min_pixels = MODELS[model].get("min_pixels")
        self.takes_seed = MODELS[model].get("seed", True)
        self.resolution = resolution
        self.structure = structure
        self.look = look
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
        text = instruction(self.structure, self.look) + (f" Style notes: {prompt}" if prompt else "")
        if self.structure == "photo":
            # Frames are sent as JPEG to keep uploads small; the model re-renders the image anyway.
            first = fal_client.upload_image(frame, "jpeg")
        else:
            first = _structure_map_url(frame, self.structure)
        args = {
            "prompt": text,
            "image_urls": [first]
            + [self.upload_style(s) for s in style_images[: self.max_refs]],
        }
        if self.takes_seed:
            args["seed"] = seed
        if self.min_pixels:
            args["image_size"] = _seedream_size(frame.size, self.min_pixels)
        else:
            args |= {"aspect_ratio": _aspect_ratio(frame.size), "resolution": self.resolution, "output_format": "png"}
        result = fal_client.subscribe(self.endpoint, arguments=args)
        return fetch_image(result["images"][0]["url"]).resize(frame.size, Image.LANCZOS)
