"""Inspect a stylised run for motion problems without watching the video.

Writes, into <styled_dir>/inspect/:
    strip.png   consecutive original | styled frames side by side
    onion.png   each group of --onion consecutive styled frames blended; jumps show up as ghosted double edges
    diff.png    change between each styled frame and the next (bright = changed)
    metrics.csv per-frame structure and figure measurements

and prints a per-frame table with the worst frames flagged.

Metrics (computed on downscaled, blurred frames so crayon texture doesn't dominate):
    structure  share of the source frame's edges that reappear (within a few pixels) in the styled frame
    stability  edge agreement between a styled frame and the previous styled frame, relative to the same
               agreement between the source frames (1.0 = moves no more than the footage does)
    fig_size   height of the figure relative to the source frame's (1.0 = same size); the source figure is the
               largest tall near-black shape, and the styled one is looked for around it whatever its colour
    fig_shift  distance between the figure's centre in the styled and source frame, as a share of frame height

Example:
    uv run python inspect_run.py work/clip/styled_abc123 --start 20 --count 10
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ANALYSIS_WIDTH = 270
EDGE_TOLERANCE_PX = 4
# Dark pixels that can belong to a silhouette; the figure is the largest such shape that is taller than wide.
DARK_THRESHOLD = 60
MIN_FIGURE_AREA = 0.002


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("styled_dir", type=Path)
    p.add_argument("--source-dir", type=Path, help="source frames (default: sibling frames_<fps>fps folder)")
    p.add_argument("--start", type=int, default=0, help="index of the first frame in the strip/onion/diff images")
    p.add_argument("--count", type=int, default=10, help="frames shown in the strip/onion/diff images")
    p.add_argument("--onion", type=int, default=3, help="consecutive frames blended per onion-skin image")
    p.add_argument("--flag", type=int, default=5, help="number of worst frames to flag")
    return p.parse_args()


def source_dir_for(styled_dir: Path) -> Path:
    fps = json.loads((styled_dir / "args.json").read_text()).get("fps")
    candidates = sorted(styled_dir.parent.glob("frames_*fps"))
    if fps:
        match = [c for c in candidates if c.name == f"frames_{float(fps):g}fps"]
        if match:
            return match[0]
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit("Can't tell which frames_* folder is the source; pass --source-dir")


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h = round(img.shape[0] * ANALYSIS_WIDTH / img.shape[1])
    return cv2.resize(img, (ANALYSIS_WIDTH, h), interpolation=cv2.INTER_AREA)


def edges(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150) > 0


def edge_recall(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Share of `reference` edge pixels that have a `candidate` edge pixel within the tolerance."""
    if not reference.any():
        return 1.0
    dist = cv2.distanceTransform((~candidate).astype(np.uint8), cv2.DIST_L2, 3)
    return float((dist[reference] <= EDGE_TOLERANCE_PX).mean())


def edge_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return (edge_recall(a, b) + edge_recall(b, a)) / 2


def _largest_tall_shape(mask: np.ndarray, min_area: float) -> tuple[int, int, int, int] | None:
    """Bounding box (x, y, w, h) of the largest connected shape in `mask` that is taller than wide."""
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area and h >= w and (best is None or area > best[0]):
            best = (area, (x, y, w, h))
    return best[1] if best else None


def source_figure(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """The figure in the source footage: the largest tall near-black shape (a dark coat on grey pavement)."""
    return _largest_tall_shape(gray < DARK_THRESHOLD, MIN_FIGURE_AREA * gray.size)


def styled_figure(gray: np.ndarray, src_box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """The figure in a styled frame, found near the source figure's position whatever its colour: within a window
    around the source box, the largest tall shape on the dark side of an Otsu split."""
    x, y, w, h = src_box
    x0, y0 = max(0, x - w), max(0, y - h // 2)
    x1, y1 = min(gray.shape[1], x + 2 * w), min(gray.shape[0], y + h + h // 2)
    window = gray[y0:y1, x0:x1]
    _, dark = cv2.threshold(window, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    box = _largest_tall_shape(dark, 0.25 * w * h)
    return (box[0] + x0, box[1] + y0, box[2], box[3]) if box else None


def pair_image(a: Path, b: Path, height: int) -> Image.Image:
    imgs = [Image.open(p).convert("RGB") for p in (a, b)]
    imgs = [i.resize((i.width * height // i.height, height)) for i in imgs]
    out = Image.new("RGB", (imgs[0].width + imgs[1].width, height))
    out.paste(imgs[0], (0, 0))
    out.paste(imgs[1], (imgs[0].width, 0))
    return out


def labelled_row(images: list[Image.Image], labels: list[str], gap: int = 8) -> Image.Image:
    label_h = 22
    row = Image.new("RGB", (sum(i.width for i in images) + gap * (len(images) - 1), images[0].height + label_h),
                    "white")
    draw = ImageDraw.Draw(row)
    x = 0
    for img, label in zip(images, labels):
        draw.text((x + 4, 4), label, fill="black")
        row.paste(img, (x, label_h))
        x += img.width + gap
    return row


def main() -> None:
    args = parse_args()
    styled = sorted(p for p in args.styled_dir.glob("[0-9]*.png"))
    if not styled:
        raise SystemExit(f"No styled frames in {args.styled_dir}")
    src_dir = args.source_dir or source_dir_for(args.styled_dir)
    out_dir = args.styled_dir / "inspect"
    out_dir.mkdir(exist_ok=True)

    rows = []
    prev = None
    for f in styled:
        src_gray, sty_gray = load_gray(src_dir / f.name), load_gray(f)
        src_e, sty_e = edges(src_gray), edges(sty_gray)
        row = {"frame": f.stem, "structure": edge_recall(src_e, sty_e), "stability": None,
               "fig_size": None, "fig_shift": None}
        # Stability only means something between frames that are adjacent in the source.
        if prev and int(f.stem) - int(prev["name"]) == 1:
            src_move = edge_agreement(prev["src_e"], src_e)
            row["stability"] = edge_agreement(prev["sty_e"], sty_e) / src_move if src_move else None
        src_fig = source_figure(src_gray)
        sty_fig = styled_figure(sty_gray, src_fig) if src_fig else None
        if src_fig and sty_fig:
            h_img = src_gray.shape[0]
            centre = lambda b: np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
            row["fig_size"] = sty_fig[3] / src_fig[3]
            row["fig_shift"] = float(np.linalg.norm(centre(sty_fig) - centre(src_fig)) / h_img)
        rows.append(row)
        prev = {"name": f.stem, "src_e": src_e, "sty_e": sty_e}

    with open(out_dir / "metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Worst frames: low structure or stability, or a figure that changed size or moved.
    def badness(r: dict) -> float:
        score = (1 - r["structure"]) * 2
        if r["stability"] is not None:
            score += max(0.0, 1 - r["stability"]) * 2
        if r["fig_size"] is not None:
            score += abs(np.log(r["fig_size"])) + r["fig_shift"] * 5
        else:
            score += 1  # figure not found: probably merged with something or missing
        return score

    worst = {r["frame"] for r in sorted(rows, key=badness, reverse=True)[: args.flag]}
    fmt = lambda v, spec: "   -  " if v is None else format(v, spec)
    print(f"{'frame':>7} {'structure':>9} {'stability':>9} {'fig_size':>8} {'fig_shift':>9}")
    for r in rows:
        flag = "  <-- worst" if r["frame"] in worst else ""
        print(f"{r['frame']:>7} {r['structure']:>9.2f} {fmt(r['stability'], '9.2f')} "
              f"{fmt(r['fig_size'], '8.2f')} {fmt(r['fig_shift'], '9.3f')}{flag}")
    found = [r for r in rows if r["fig_size"] is not None]
    stab = [r["stability"] for r in rows if r["stability"] is not None]
    print(f"\nmean structure {np.mean([r['structure'] for r in rows]):.2f}"
          + (f", mean stability {np.mean(stab):.2f}" if stab else "")
          + f", figure found in {len(found)}/{len(rows)} frames"
          + (f", figure size range {min(r['fig_size'] for r in found):.2f}-{max(r['fig_size'] for r in found):.2f}"
             if found else ""))

    window = styled[args.start: args.start + args.count]
    strip = [pair_image(src_dir / f.name, f, 360) for f in window]
    labelled_row(strip, [f.stem + (" *" if f.stem in worst else "") for f in window]).save(out_dir / "strip.png")

    full = [np.asarray(Image.open(f).convert("RGB").resize((360, 640)), dtype=np.float32) for f in window]
    onions = [Image.fromarray(np.mean(full[i: i + args.onion], axis=0).astype(np.uint8))
              for i in range(0, max(1, len(full) - args.onion + 1), args.onion)]
    labelled_row(onions, [f"{window[i].stem}+{args.onion - 1}" for i in range(0, len(onions) * args.onion,
                                                                              args.onion)]).save(out_dir / "onion.png")
    if len(full) > 1:
        diffs = [Image.fromarray(np.clip(np.abs(full[i + 1] - full[i]).mean(axis=2) * 2, 0, 255).astype(np.uint8))
                 for i in range(len(full) - 1)]
        labelled_row(diffs, [f"{window[i].stem}>{window[i + 1].stem}" for i in range(len(diffs))]).save(
            out_dir / "diff.png")
    print(f"\nImages and metrics.csv written to {out_dir}/")


if __name__ == "__main__":
    main()
