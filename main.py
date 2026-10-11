"""Rotoscope a video: restyle every frame with SD1.5 + ControlNet + IP-Adapter, then reassemble.

Example:
    uv run python main.py clip.mp4 --style-dir styles/ --prompt "watercolour painting" --preview 3
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from rotoscope import frames
from rotoscope.stylize import CONTROLNETS, IP_MODES, build_pipeline, stylize_frame

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--style-dir", type=Path, help="folder of reference images in the target style")
    p.add_argument("--prompt", default="", help="text description of the style/content")
    p.add_argument("--out", type=Path, help="output video (default: <video>_roto.mp4)")
    p.add_argument("--work-dir", type=Path, default=Path("work"))
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--lora", help="path or hub id of a style LoRA")
    p.add_argument("--control", nargs="+", choices=CONTROLNETS, default=["lineart"])
    p.add_argument("--control-scale", type=float, default=0.8, help="how tightly structure is held")
    p.add_argument("--strength", type=float, default=0.65, help="0 = original frame, 1 = fully regenerated")
    p.add_argument("--ip-scale", type=float, default=0.8, help="strength of the reference-image style")
    p.add_argument("--ip-mode", choices=IP_MODES, default="style",
                   help="'style' = style-only blocks, 'full' = also borrows content from references")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-mode", choices=["random", "fixed"], default="random",
                   help="'random' = new noise per frame (boiling flicker), 'fixed' = same noise every frame")
    p.add_argument("--fps", type=float, help="processing/output fps (default: source fps; e.g. 12 = 'on twos')")
    p.add_argument("--width", type=int, default=640, help="generation width; output keeps source resolution")
    p.add_argument("--preview", type=int, metavar="N", help="only stylise N evenly spaced frames, no video")
    return p.parse_args()


def run_key(args: argparse.Namespace, fps: float) -> str:
    """Hash of everything that affects stylised output, so cached frames are never stale."""
    style_files = sorted(str(f) for f in style_paths(args))
    keys = ["prompt", "model", "lora", "control", "control_scale", "strength", "ip_scale",
            "ip_mode", "steps", "guidance", "seed", "seed_mode", "width"]
    blob = json.dumps({k: getattr(args, k) for k in keys} | {"style": style_files, "fps": fps}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def style_paths(args: argparse.Namespace) -> list[Path]:
    if not args.style_dir:
        return []
    return sorted(f for f in args.style_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    args = parse_args()
    fps = args.fps or frames.probe_fps(args.video)
    work = args.work_dir / args.video.stem
    src_frames = frames.extract(args.video, work / f"frames_{fps:g}fps", fps)

    if args.preview:
        step = max(1, len(src_frames) // args.preview)
        todo = list(enumerate(src_frames))[::step][: args.preview]
    else:
        todo = list(enumerate(src_frames))

    styles = [Image.open(f).convert("RGB") for f in style_paths(args)]
    if not styles and not args.prompt and not args.lora:
        raise SystemExit("Give at least one of --style-dir, --prompt or --lora")

    out_dir = work / f"styled_{run_key(args, fps)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))

    pending = [(i, f) for i, f in todo if not (out_dir / f.name).exists()]
    if pending:
        stylizer = build_pipeline(args.model, args.control, lora=args.lora,
                                  ip_adapter="ip-adapter-plus_sd15.safetensors" if styles else None)
        for i, f in tqdm(pending, desc="stylising"):
            seed = args.seed + i if args.seed_mode == "random" else args.seed
            img = stylize_frame(
                stylizer, Image.open(f), styles, args.prompt,
                width=args.width, strength=args.strength, control_scale=args.control_scale,
                ip_scale=args.ip_scale, ip_mode=args.ip_mode, steps=args.steps,
                guidance=args.guidance, seed=seed,
            )
            img.save(out_dir / f.name)

    if args.preview:
        for _, f in todo:
            a, b = Image.open(f), Image.open(out_dir / f.name)
            pair = Image.new("RGB", (a.width * 2, a.height))
            pair.paste(a, (0, 0))
            pair.paste(b, (a.width, 0))
            pair.save(out_dir / f"compare_{f.name}")
        print(f"Preview frames (original | styled): {out_dir}/compare_*.png")
        return

    out_video = args.out or args.video.with_name(f"{args.video.stem}_roto.mp4")
    frames.assemble(out_dir, out_video, fps, audio_from=args.video)
    print(f"Wrote {out_video}")


if __name__ == "__main__":
    main()
