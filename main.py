"""Rotoscope a video: restyle every frame on fal.ai, then reassemble.

Backends:
    edit  instruction-based image editing (default: Nano Banana 2); frame + style refs + instruction
    flux  FLUX img2img + ControlNet + IP-Adapter; finer control, weaker style transfer

Needs a fal.ai API key in the FAL_KEY environment variable.

Example:
    uv run python main.py clip.mp4 --style-dir styles/ --prompt "watercolour painting" --preview 3
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from rotoscope import frames
from rotoscope import edit
from rotoscope.stylize import CONTROLS, build_pipeline, stylize_frame

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
FLUX_USD_PER_FRAME = 0.075  # fal-ai/flux-general bills per started megapixel; frames are kept under 1MP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--backend", choices=["edit", "flux"], default="edit")
    p.add_argument("--style-dir", type=Path,
                   help="folder of reference images in the target style; frames cycle through them")
    p.add_argument("--refs", type=int, default=3,
                   help="[edit] style references sent with each frame (rotating through --style-dir)")
    p.add_argument("--edit-model", default=edit.DEFAULT_MODEL, help="[edit] fal.ai image-editing endpoint")
    p.add_argument("--resolution", choices=list(edit.USD_PER_IMAGE), default="1K", help="[edit] output size")
    p.add_argument("--prompt", default="", help="text description of the style/content")
    p.add_argument("--out", type=Path, help="output video (default: <video>_roto.mp4)")
    p.add_argument("--work-dir", type=Path, default=Path("work"))
    p.add_argument("--lora", help="[flux] URL or Hugging Face path of a FLUX style LoRA")
    p.add_argument("--control", nargs="*", choices=CONTROLS, default=["canny"],
                   help="[flux] structure guidance; pass none to disable")
    p.add_argument("--control-scale", type=float, default=0.7, help="[flux] how tightly structure is held")
    p.add_argument("--strength", type=float, default=0.75, help="[flux] 0 = original frame, 1 = fully regenerated")
    p.add_argument("--ip-scale", type=float, default=0.7, help="[flux] strength of the reference-image style")
    p.add_argument("--steps", type=int, default=28, help="[flux]")
    p.add_argument("--guidance", type=float, default=3.5, help="[flux]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-mode", choices=["random", "fixed"], default="random",
                   help="'random' = new noise per frame (boiling flicker), 'fixed' = same noise every frame")
    p.add_argument("--fps", type=float, help="processing/output fps (default: source fps; e.g. 12 = 'on twos')")
    p.add_argument("--width", type=int, default=768,
                   help="[flux] generation width (capped to stay under 1MP); output keeps source resolution")
    p.add_argument("--workers", type=int, default=4, help="frames processed concurrently on fal.ai")
    p.add_argument("--preview", type=int, metavar="N", help="only stylise N evenly spaced frames, no video")
    return p.parse_args()


def run_key(args: argparse.Namespace, fps: float) -> str:
    """Hash of everything that affects stylised output, so cached frames are never stale."""
    style_files = sorted(str(f) for f in style_paths(args))
    if args.backend == "edit":
        keys = ["backend", "prompt", "refs", "edit_model", "resolution", "seed", "seed_mode"]
    else:
        keys = ["backend", "prompt", "lora", "control", "control_scale", "strength", "ip_scale",
                "steps", "guidance", "seed", "seed_mode", "width"]
    blob = json.dumps({k: getattr(args, k) for k in keys} | {"style": style_files, "fps": fps}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def style_paths(args: argparse.Namespace) -> list[Path]:
    if not args.style_dir:
        return []
    return sorted(f for f in args.style_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    args = parse_args()
    if not os.environ.get("FAL_KEY"):
        raise SystemExit("Set FAL_KEY to your fal.ai API key (https://fal.ai/dashboard/keys)")
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
    if args.backend == "edit" and not styles:
        raise SystemExit("The edit backend needs reference images in --style-dir")

    out_dir = work / f"styled_{run_key(args, fps)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))

    pending = [(i, f) for i, f in todo if not (out_dir / f.name).exists()]
    if pending:
        per_frame = edit.USD_PER_IMAGE[args.resolution] if args.backend == "edit" else FLUX_USD_PER_FRAME
        print(f"{len(pending)} frames to stylise with the {args.backend} backend, "
              f"est. ${len(pending) * per_frame:.2f}")
        if args.backend == "edit":
            stylizer = edit.EditStylizer(args.edit_model, args.resolution)
        else:
            stylizer = build_pipeline(args.control, lora=args.lora)

        def work_one(i: int, f: Path) -> None:
            seed = args.seed + i if args.seed_mode == "random" else args.seed
            if args.backend == "edit":
                refs = [styles[(i + k) % len(styles)] for k in range(min(args.refs, len(styles)))]
                img = stylizer.stylize_frame(Image.open(f), refs, args.prompt, seed=seed)
            else:
                img = stylize_frame(
                    stylizer, Image.open(f), styles[i % len(styles)] if styles else None, args.prompt,
                    width=args.width, strength=args.strength, control_scale=args.control_scale,
                    ip_scale=args.ip_scale, steps=args.steps, guidance=args.guidance, seed=seed,
                )
            img.save(out_dir / f.name)

        # Upload style images up front so worker threads don't race to upload the same one.
        for img in styles:
            stylizer.upload_style(img)
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(work_one, i, f) for i, f in pending]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="stylising"):
                fut.result()

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
