# img2img-rotoscope

Restyle a video frame by frame with an image model on [fal.ai](https://fal.ai), then reassemble it, keeping the
original composition while changing colours, texture and detail. Each frame is generated independently, so the
result "boils" from frame to frame like hand-drawn rotoscoping; that flicker is intentional.

```
video ──ffmpeg──▶ frames ──fal.ai (frame + style refs + prompt)──▶ styled frames ──ffmpeg──▶ video (+ original audio)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/), ffmpeg (`brew install ffmpeg`) and a fal.ai API key.

```sh
uv sync
echo "FAL_KEY=your-key-here" > .env   # from https://fal.ai/dashboard/keys; .env is gitignored
```

## Usage

1. Put a few reference images in the target style in a folder (e.g. stills from a clip you like).
2. Preview a handful of frames and compare them with the originals:

   ```sh
   uv run --env-file .env python main.py clip.mp4 \
     --style-dir references/my-style \
     --prompt-file styles/crayon-pastel.txt \
     --fps 15 --preview 5
   ```

   This writes `work/clip/styled_<hash>/compare_*.png` (original | styled side by side).
3. Tune the prompt and options, then drop `--preview` to render the whole clip to `clip_roto.mp4`.

The script prints an estimated cost before it starts generating.

## Backends

| `--backend` | How it works | Cost per frame |
|---|---|---|
| `edit` (default) | An instruction-based image-editing model gets the frame plus `--refs` style references and is told to redraw the frame in their style. Strongest style transfer; composition is held by the instruction, so expect occasional added objects or small framing drift. | `seedream-4` (default) $0.03, `nano-banana-2` $0.08 at 1K |
| `flux` | FLUX img2img with a ControlNet (`canny`/`tile`) for structure and an IP-Adapter for style. Tighter structure and more knobs, but much weaker style transfer in testing. | ~$0.075 (kept under 1MP) |

Choose the edit model with `--edit-model`. In testing, `seedream-4` gave the most childlike crayon look, and
`nano-banana-2` held composition more tightly with richer texture.

## Options worth knowing

- `--fps` – processing/output frame rate. Hand-drawn animation is often 12–15 fps; lower also means cheaper.
- `--refs` – style references sent per frame (edit backend), rotating through `--style-dir`. 5 (the default) worked better than 8.
- `--prompt` / `--prompt-file` – style description appended to the built-in redraw instruction. Short prompts
  work better than long, literal ones: the model takes descriptive details too literally. Reusable prompts live
  in `styles/`; `styles/crayon-pastel.txt` is the best-performing one so far.
- `--seed-mode` – `random` (default) uses a new seed per frame for maximum flicker; `fixed` reuses one seed.
- `--workers` – frames submitted to fal.ai at once (default 40). fal runs up to your account's concurrency
  limit in parallel (new accounts: 2, rising automatically to 40 with credit purchases; see the dashboard's
  Concurrency page) and queues the rest. Queued requests still run, and are billed, if the script is stopped.
- `--preview N` – only stylise N evenly spaced frames and write comparison images; no video.

Run `uv run python main.py --help` for the full list.

## Caching and output

- Extracted frames go to `work/<video>/frames_<fps>fps/`.
- Styled frames go to `work/<video>/styled_<hash>/`, where the hash covers every setting that affects the output
  (backend, model, prompt, references, fps, seed, ...). Re-running with the same settings only generates
  missing frames, so interrupted renders resume and previews are reused by the full render.
- `args.json` in each styled folder records the exact settings used.
- `work/`, `references/`, `.env` and rendered `*_roto.mp4` files are gitignored.

## Layout

```
main.py               CLI: extract frames, stylise, reassemble
rotoscope/frames.py   ffmpeg helpers (extract, assemble with audio)
rotoscope/edit.py     edit backend (Seedream 4 / Nano Banana 2) and the redraw instruction
rotoscope/stylize.py  flux backend (FLUX + ControlNet + IP-Adapter)
styles/               reusable style prompts
```

## Notes

- Running locally (SD1.5 on a 16GB Apple Silicon Mac) was tried first and abandoned: ~25 s per frame at best,
  and portrait frames ran out of memory. All generation now happens on fal.ai.
- Only use reference footage you have the right to use; `references/` is kept out of git for that reason.
