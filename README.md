# Scribblescope

Restyle a video frame by frame with an image model on [fal.ai](https://fal.ai), then reassemble it, keeping the
original composition while changing colours, texture and detail. Each frame is generated independently, so the
result "boils" from frame to frame like hand-drawn rotoscoping; that flicker is intentional.

```
video ──ffmpeg──▶ frames ──line art──▶ fal.ai (map + style refs + prompt + palette) ──▶ styled frames ──QC──▶ video (+ audio)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/), ffmpeg (`brew install ffmpeg`) and a fal.ai API key.

```sh
uv sync
echo "FAL_KEY=your-key-here" > .env   # from https://fal.ai/dashboard/keys; .env is gitignored
```

## Usage

1. Put a few reference images in the target style in a folder (e.g. stills from a clip you like).
2. Render a short run of consecutive frames and check the motion:

   ```sh
   uv run --env-file .env python main.py clip.mp4 \
     --style-dir references/my-style \
     --structure lineart --refs 2 \
     --prompt-file styles/crayon-pastel-varied.txt \
     --palettes styles/crayon-pastel-palettes.txt \
     --fps 15 --range 15:27

   uv run python inspect_run.py work/clip/styled_<hash> --count 12
   ```

   `inspect_run.py` writes a filmstrip, onion skins and frame diffs to `work/clip/styled_<hash>/inspect/` and
   prints per-frame structure/stability scores (see [Inspecting motion](#inspecting-motion)).
3. Tune the prompt and options, then drop `--range` to render the whole clip to `clip_roto.mp4`. Frames already
   rendered with the same settings are reused.

`--preview N` renders N evenly spaced frames with side-by-side comparison images instead, which is quicker for
judging the look but says nothing about motion. The script prints an estimated cost before it starts generating.

## Recommended settings

These gave the best result so far (stable composition, fully drawn frames, natural colour variation):

- `--backend edit --edit-model seedream-4` (the defaults), $0.03 per frame.
- `--structure lineart` – each frame is first turned into a line drawing (free on fal.ai), and the model colours
  that in. Sending the photo itself lets photographic colour and texture leak through; depth maps lose too much
  detail.
- `--refs 2` – fewer references means less of their content (rather than style) leaking into frames.
- `--palettes styles/crayon-pastel-palettes.txt` with `--prompt-file styles/crayon-pastel-varied.txt` – each
  frame gets a randomly chosen colour/lighting description, so colour varies from drawing to drawing at
  generation time. The palettes are mostly orange and green, mixed with yellow,
  blue, red and plenty of near-black.

## Backends

| `--backend` | How it works | Cost per frame |
|---|---|---|
| `edit` (default) | An instruction-based image-editing model gets the frame (or its line art) plus `--refs` style references and is told to redraw it in their style. | `seedream-4` (default) $0.03, `seedream-5-lite` $0.035, `seedream-4.5` $0.04, `nano-banana-2` $0.08 at 1K |
| `flux` | FLUX img2img with a ControlNet (`canny`/`tile`) for structure, an IP-Adapter and optional LoRA for style. Weak style transfer in testing. | ~$0.075 (kept under 1MP) |

In testing, `seedream-4` gave the most childlike crayon look; `seedream-4.5` was more polished, `seedream-5-lite`
left backgrounds photographic, and `nano-banana-2` was more painterly and less childlike.

## Options worth knowing

- `--fps` – processing/output frame rate. Hand-drawn animation is often 12–15 fps; lower also means cheaper.
- `--structure` – what the edit model sees of each frame: `photo`, `lineart` or `depth`.
- `--refs` – style references sent per frame, rotating through `--style-dir`.
- `--prompt` / `--prompt-file` – style description appended to the built-in instruction. Short prompts work
  better than long, literal ones, and colour words naming objects (e.g. "brick red") can add those objects.
- `--palettes` – file of colour/lighting lines; one is picked at random per frame. Describe colours and light
  only, never objects.
- `--retries` – after a render, frames whose layout breaks from their neighbours are regenerated with a new
  seed, up to this many rounds (default 2). `qc.json` in the run folder records the attempts. It catches
  redrawn scenes and content copied from the references reliably, single objects changing size less so.
- `--seed-mode` – `random` (default) uses a new seed per frame; `fixed` reuses one seed.
- `--workers` – frames submitted to fal.ai at once (default 8). fal runs up to your account's concurrency
  limit in parallel (new accounts: 2, rising automatically to 40 with credit purchases; see the dashboard's
  Concurrency page) and queues the rest. Submitted requests still run, and are billed, if the script is
  stopped, so the default is kept low to cap the cost of a cancelled run; raise it for long renders.
- `--range START:END` / `--preview N` – render only part of the clip; no video.

Run `uv run python main.py --help` for the full list.

## Inspecting motion

`uv run python inspect_run.py <styled folder> [--start N --count N]` measures a run without watching it:

- `strip.png` – consecutive original | styled pairs; `onion.png` – consecutive frames blended, so jumps show as
  ghosted double edges; `diff.png` – what changed between frames.
- Per frame: `structure` (share of the source's edges that reappear), `stability` (frame-to-frame edge agreement
  relative to the footage's own), `fig_size` / `fig_shift` (the figure's size and position relative to the
  source; rough, since a shadow can be counted as part of the figure). The worst frames are flagged.

## Caching and output

- Extracted frames go to `work/<video>/frames_<fps>fps/`.
- Styled frames go to `work/<video>/styled_<hash>/`, where the hash covers every setting that affects the output
  (backend, model, prompt, palettes, references, structure, fps, seed, built-in instruction, ...). Re-running
  with the same settings only generates missing frames, so interrupted renders resume and partial renders are
  reused by the full render.
- `args.json` in each styled folder records the exact settings used.
- `work/`, `references/`, `.env` and rendered `*_roto.mp4` files are gitignored.

## Layout

```
main.py                CLI: extract frames, stylise, quality pass, reassemble
inspect_run.py         motion inspection (filmstrips, onion skins, per-frame scores)
rotoscope/frames.py    ffmpeg helpers (extract, assemble with audio)
rotoscope/edit.py      edit backend (Seedream / Nano Banana), structure maps and instructions
rotoscope/quality.py   detects frames whose layout breaks from their neighbours
rotoscope/stylize.py   flux backend (FLUX + ControlNet + IP-Adapter + LoRA)
styles/                reusable style prompts and palettes
```

## Notes

- Running locally (SD1.5 on a 16GB Apple Silicon Mac) was tried first and abandoned: ~25 s per frame at best,
  and portrait frames ran out of memory. All generation now happens on fal.ai.
- Midjourney has no official API and its terms forbid automated access, so it isn't supported.

## License

Copyright 2026 Val Enfys. Licensed under the [Apache License 2.0](LICENSE).
