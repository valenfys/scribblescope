"""Video <-> frame-sequence conversion via ffmpeg."""

import json
import subprocess
from pathlib import Path


def probe_fps(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    num, den = json.loads(out)["streams"][0]["r_frame_rate"].split("/")
    return float(num) / float(den)


def has_audio(video: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    return bool(out.strip())


def extract(video: Path, out_dir: Path, fps: float) -> list[Path]:
    """Dump frames as numbered PNGs, resampled to `fps`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not any(out_dir.glob("*.png")):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps}",
             str(out_dir / "%06d.png")],
            check=True,
        )
    return sorted(out_dir.glob("*.png"))


def assemble(frames_dir: Path, out_video: Path, fps: float, audio_from: Path | None = None) -> None:
    """Encode numbered PNGs into an H.264 video, copying audio from `audio_from` if present."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(frames_dir / "%06d.png")]
    if audio_from is not None and has_audio(audio_from):
        cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out_video)]
    subprocess.run(cmd, check=True)
