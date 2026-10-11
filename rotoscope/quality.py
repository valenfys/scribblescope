"""Detect failed frames: ones whose overall layout doesn't match the rest of the clip.

Each frame is reduced to a coarse, normalised map of its brightness and edges, which ignores colour, texture and
global brightness but keeps where the big shapes are. A frame whose map agrees much less with its neighbours than
is typical for the clip has usually redrawn the scene (a figure changing size, content copied from the style
references, a different viewpoint). Comparing with neighbours rather than the whole clip keeps camera moves from
making the start or end of a clip look like outliers.
"""

from pathlib import Path

import cv2
import numpy as np

ANALYSIS_WIDTH = 270
GRID = (27, 48)
# A frame is an outlier when its median similarity to the frames within WINDOW of it is this far below the
# clip's median.
WINDOW = 4
OUTLIER_MARGIN = 0.15


def _layout(path: Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (ANALYSIS_WIDTH, round(gray.shape[0] * ANALYSIS_WIDTH / gray.shape[1])),
                      interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150).astype(np.float32)
    parts = []
    for channel in (gray.astype(np.float32), edges):
        small = cv2.resize(cv2.GaussianBlur(channel, (0, 0), 4), GRID, interpolation=cv2.INTER_AREA).ravel()
        parts.append((small - small.mean()) / (small.std() + 1e-6))
    v = np.concatenate(parts)
    return v / np.linalg.norm(v)


def layout_scores(frames: list[Path]) -> list[float]:
    """For each frame, its median layout similarity to the frames within WINDOW of it."""
    feats = np.stack([_layout(f) for f in frames])
    sims = feats @ feats.T
    scores = []
    for i in range(len(frames)):
        near = [j for j in range(max(0, i - WINDOW), min(len(frames), i + WINDOW + 1)) if j != i]
        scores.append(float(np.median(sims[i, near])))
    return scores


def outliers(frames: list[Path]) -> list[Path]:
    if len(frames) < 5:
        return []  # too few frames to know what "typical" looks like
    scores = layout_scores(frames)
    cutoff = float(np.median(scores)) - OUTLIER_MARGIN
    return [f for f, s in zip(frames, scores) if s < cutoff]
